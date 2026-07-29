"""STRIP (Gao et al., 2019): superimposes randomly sampled clean images onto
a test input and flags inputs whose prediction remains abnormally confident
(low entropy) after perturbation. Dispersive untargeted attacks tend to
evade this by construction, since their triggered predictions are already
high-entropy (spread across the confusion zone) rather than confidently
concentrated on one class."""

import numpy as np
import torch


def prediction_entropy(logits):
    p = torch.softmax(logits, dim=-1)
    return -(p * (p + 1e-8).log()).sum(-1)


@torch.no_grad()
def run_strip(model, loader, device, trigger, normalize_fn=None,
              n_superimpose=100, n_test=200):
    model.eval()
    all_imgs = []
    for x, _ in loader:
        all_imgs.append(x)
    all_imgs = torch.cat(all_imgs).to(device)

    clean_entropy, trig_entropy = [], []
    indices = torch.randperm(len(all_imgs))[:n_test]

    for idx in indices:
        base = all_imgs[idx:idx + 1]
        sup = all_imgs[torch.randperm(len(all_imgs))[:n_superimpose]]

        blended_clean = 0.5 * base + 0.5 * sup
        out_c = model(normalize_fn(blended_clean) if normalize_fn else blended_clean)
        clean_entropy.append(prediction_entropy(out_c).mean().item())

        blended_trig = 0.5 * trigger(base) + 0.5 * sup
        out_t = model(normalize_fn(blended_trig) if normalize_fn else blended_trig)
        trig_entropy.append(prediction_entropy(out_t).mean().item())

    clean_entropy = np.array(clean_entropy)
    trig_entropy = np.array(trig_entropy)
    threshold = clean_entropy.mean() - 2 * clean_entropy.std()
    flagged_rate = 100 * (trig_entropy < threshold).mean()

    return {
        "clean_entropy_mean": float(clean_entropy.mean()),
        "trig_entropy_mean": float(trig_entropy.mean()),
        "threshold": float(threshold),
        "flagged_rate": float(flagged_rate),
        "evaded": flagged_rate < 10.0,
    }
