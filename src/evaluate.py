"""Evaluation metrics: Clean Accuracy, ASR, Dispersibility Score,
Zone Accuracy, and Maximum Class Concentration."""

import math
import numpy as np
import torch


def dispersibility_score(hist):
    """DS = 1 - sqrt( sum_j (p_j - 1/|H|)^2 / |H| ), from CUBA (Xue et al.).
    1.0 = perfectly uniform dispersion; ~0.7 (for |H|=10) = full collapse
    onto a single class."""
    p = hist / max(hist.sum(), 1)
    h = len(p)
    return float(1 - math.sqrt(((p - 1 / h) ** 2).sum() / h))


@torch.no_grad()
def evaluate_attack(model, loader, device, trigger, num_classes, normalize_fn=None):
    """Returns (clean_accuracy, ASR, DS, prediction_histogram)."""
    model.eval()
    mis = tot = 0
    ben_correct = ben_total = 0
    hist = np.zeros(num_classes)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xn = normalize_fn(x) if normalize_fn else x
        out_clean = model(xn)
        ben_correct += (out_clean.argmax(1) == y).sum().item()
        ben_total += len(y)

        xt = trigger(x)
        xtn = normalize_fn(xt) if normalize_fn else xt
        out_trig = model(xtn)
        pred = out_trig.argmax(1)
        wrong = pred != y
        mis += wrong.sum().item()
        tot += len(y)
        for c in pred[wrong].cpu().numpy():
            hist[c] += 1

    clean_acc = 100 * ben_correct / ben_total
    asr = 100 * mis / max(tot, 1)
    ds = dispersibility_score(hist)
    return clean_acc, asr, ds, hist


@torch.no_grad()
def zone_accuracy_and_concentration(model, loader, device, trigger, num_classes,
                                     nearest_classes, normalize_fn=None,
                                     samples_per_class=100):
    """Zone Accuracy: fraction of successful misclassifications landing inside
    the geometrically-predicted confusion zone (novel metric, this work).
    Max Concentration: largest single-class share of misclassifications
    (checks for the collapse failure mode CUBA identifies in prior work)."""
    model.eval()
    total_zone_hits = total_misclassified = 0
    hist = np.zeros(num_classes)

    for target_class in range(num_classes):
        imgs = []
        for x, y in loader:
            mask = y == target_class
            imgs.append(x[mask])
            if sum(len(b) for b in imgs) >= samples_per_class:
                break
        if not imgs:
            continue
        imgs = torch.cat(imgs)[:samples_per_class].to(device)

        xt = trigger(imgs)
        xtn = normalize_fn(xt) if normalize_fn else xt
        pred = model(xtn).argmax(1)
        wrong = pred != target_class
        wrong_preds = pred[wrong].cpu().tolist()

        total_misclassified += len(wrong_preds)
        total_zone_hits += sum(1 for p in wrong_preds if p in nearest_classes[target_class])
        for p in wrong_preds:
            hist[p] += 1

    zone_accuracy = 100 * total_zone_hits / max(total_misclassified, 1)
    max_concentration = 100 * (hist / max(hist.sum(), 1)).max()
    return zone_accuracy, max_concentration, hist
