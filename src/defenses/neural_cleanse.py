"""Neural Cleanse (Wang et al., 2019): reverse-engineers a minimal trigger
per output class and flags classes whose reconstructed trigger is
anomalously small (MAD-based anomaly index, threshold 2.0)."""

import numpy as np
import torch
import torch.nn.functional as F


def run_neural_cleanse(model, loader, device, num_classes, normalize_fn=None,
                        steps=300, lr=0.1, mask_penalty=0.03, input_shape=(3, 32, 32)):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    xs, _ = next(iter(loader))
    xs = xs[:128].to(device)
    c, h, w = input_shape
    mask_norms = []

    for target in range(num_classes):
        mask = torch.zeros(1, 1, h, w, device=device, requires_grad=True)
        pattern = torch.zeros(1, c, h, w, device=device, requires_grad=True)
        opt = torch.optim.Adam([mask, pattern], lr=lr)
        tgt = torch.full((len(xs),), target, device=device)

        for _ in range(steps):
            m = torch.sigmoid(mask)
            p_ = torch.sigmoid(pattern)
            adv = (1 - m) * xs + m * p_
            out = model(normalize_fn(adv) if normalize_fn else adv)
            loss = F.cross_entropy(out, tgt) + mask_penalty * m.abs().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            mask_norms.append(torch.sigmoid(mask).abs().sum().item())

    mask_norms = np.array(mask_norms)
    median = np.median(mask_norms)
    mad = np.median(np.abs(mask_norms - median)) + 1e-8
    anomaly_index = (median - mask_norms) / (1.4826 * mad)

    for p in model.parameters():
        p.requires_grad_(True)

    return mask_norms, float(anomaly_index.max()), anomaly_index
