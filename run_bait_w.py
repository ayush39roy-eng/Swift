"""End-to-end swift pipeline: train victim -> compute confusion-zone
geometry -> train backdoored model -> evaluate attack + defenses.

Usage:
    python run_bait_w.py --config configs/cifar10.yaml
    python run_bait_w.py --config configs/cifar100.yaml
"""

import argparse
import yaml
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.models import ResNet18
from src.trigger import Trigger
from src.geometry import (
    extract_class_features,
    sliced_wasserstein_matrix,
    hybrid_distance,
    select_confusion_zones,
    soft_labels_from_zones,
)
from src.semantic import (
    cifar100_superclass_distance,
    word_embedding_distance,
    semantic_fallback_cifar10,
)
from src.train import train_victim, train_bait_w
from src.evaluate import evaluate_attack, zone_accuracy_and_concentration
from src.defenses import run_neural_cleanse, run_strip, run_fine_pruning


def get_dataloaders(cfg, device):
    mean = torch.tensor(cfg["normalize"]["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["normalize"]["std"]).view(1, 3, 1, 1)
    normalize_fn = lambda x: (x - mean.to(x.device)) / std.to(x.device)

    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tf_test = T.ToTensor()

    if cfg["dataset"] == "cifar10":
        ds_cls = torchvision.datasets.CIFAR10
    elif cfg["dataset"] == "cifar100":
        ds_cls = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"unsupported dataset: {cfg['dataset']}")

    train_set = ds_cls("./data", train=True, download=True, transform=tf_train)
    test_set = ds_cls("./data", train=False, download=True, transform=tf_test)

    train_loader = DataLoader(train_set, cfg["victim"]["batch_size"], shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, 256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader, normalize_fn


def get_semantic_distance(cfg, class_names):
    source = cfg["geometry"]["semantic_source"]
    if source == "cifar100_superclass":
        return cifar100_superclass_distance(cfg["num_classes"])
    if source == "word_embedding":
        try:
            return word_embedding_distance(class_names)
        except RuntimeError:
            print("word embeddings unavailable, using CIFAR-10 fallback grouping")
            return semantic_fallback_cifar10()
    if source == "fallback":
        return semantic_fallback_cifar10()
    raise ValueError(f"unsupported semantic_source: {source}")


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["num_classes"]

    train_loader, test_loader, normalize_fn = get_dataloaders(cfg, device)
    trigger = Trigger(*cfg["input_shape"][1:], cfg["input_shape"][0],
                       cfg["trigger"]["blend_alpha"], device)

    # --- Step 1: train the clean reference / baseline model -----------------
    print("=== Training victim (reference / baseline) model ===")
    victim = ResNet18(num_classes).to(device)
    train_victim(victim, train_loader, test_loader, device,
                 cfg["victim"]["epochs"], cfg["victim"]["lr"],
                 cfg["victim"]["weight_decay"], cfg["victim"]["momentum"],
                 normalize_fn)

    # --- Steps 2-4: confusion-zone geometry ---------------------------------
    print("=== Computing confusion-zone geometry ===")
    victim_norm = lambda x: victim(normalize_fn(x))  # not used directly; kept for clarity
    barycenters, class_features = extract_class_features(
        victim, ((normalize_fn(x), y) for x, y in train_loader), device,
        num_classes, 512, cfg["geometry"]["samples_per_class"],
    )
    sw_matrix = sliced_wasserstein_matrix(
        class_features, num_classes, cfg["geometry"]["n_projections"],
    )
    class_names = [str(i) for i in range(num_classes)]  # replace with real names
    sem_matrix = get_semantic_distance(cfg, class_names)
    hybrid = hybrid_distance(sw_matrix, sem_matrix, cfg["geometry"]["alpha"])
    nearest_classes = select_confusion_zones(hybrid, num_classes, cfg["geometry"]["k"])
    soft_labels = soft_labels_from_zones(nearest_classes, num_classes,
                                          cfg["geometry"]["k"], device)

    # --- Step 5: train the backdoored model ---------------------------------
    print("=== Training swift backdoored model ===")
    model_bait = ResNet18(num_classes).to(device)
    train_bait_w(model_bait, train_loader, test_loader, device, num_classes,
                 soft_labels, trigger, cfg["bait_w"]["epochs"],
                 cfg["bait_w"]["poison_rate"], cfg["bait_w"]["lr"],
                 cfg["bait_w"]["weight_decay"], cfg["bait_w"]["momentum"],
                 normalize_fn)

    # --- Evaluation ----------------------------------------------------------
    print("=== Evaluation ===")
    clean_acc, asr, ds, _ = evaluate_attack(
        model_bait, test_loader, device, trigger, num_classes, normalize_fn
    )
    zone_acc, max_conc, _ = zone_accuracy_and_concentration(
        model_bait, test_loader, device, trigger, num_classes,
        nearest_classes, normalize_fn,
    )
    print(f"Clean Accuracy:  {clean_acc:.2f}%")
    print(f"ASR:             {asr:.2f}%")
    print(f"DS:              {ds:.3f}")
    print(f"Zone Accuracy:   {zone_acc:.1f}%")
    print(f"Max Concentration: {max_conc:.1f}%")

    print("=== Defense evaluation ===")
    _, nc_anomaly, _ = run_neural_cleanse(model_bait, test_loader, device,
                                           num_classes, normalize_fn,
                                           input_shape=cfg["input_shape"])
    print(f"Neural Cleanse anomaly index: {nc_anomaly:.3f} "
          f"({'evaded' if nc_anomaly < 2.0 else 'detected'})")

    strip_result = run_strip(model_bait, test_loader, device, trigger, normalize_fn)
    print(f"STRIP flagged rate: {strip_result['flagged_rate']:.1f}% "
          f"({'evaded' if strip_result['evaded'] else 'detected'})")

    fp_results = run_fine_pruning(model_bait, test_loader, device, trigger,
                                   num_classes, normalize_fn)
    print("Fine-pruning:")
    for r in fp_results:
        print(f"  prune={r['prune_rate']*100:>3.0f}%  "
              f"clean={r['clean_acc']:.2f}  asr={r['asr']:.2f}  ds={r['ds']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
