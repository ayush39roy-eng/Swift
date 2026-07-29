"""Training routines: (1) clean reference/victim model, (2) BAIT-W poisoned model."""

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


def train_victim(model, train_loader, test_loader, device, epochs, lr=0.1,
                  weight_decay=5e-4, momentum=0.9, normalize_fn=None, log_every=10):
    """Standard clean training. This model is used only to extract feature
    geometry for confusion-zone computation, and doubles as the epoch-matched
    baseline for reporting clean accuracy without any backdoor."""
    opt = torch.optim.SGD(model.parameters(), lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = GradScaler("cuda")

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
                out = model(normalize_fn(x) if normalize_fn else x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        if (ep + 1) % log_every == 0 or ep == 0:
            acc = evaluate_clean(model, test_loader, device, normalize_fn)
            print(f"  [victim] epoch {ep + 1:3d}  clean_acc={acc:.2f}%")
    return model


@torch.no_grad()
def evaluate_clean(model, loader, device, normalize_fn=None):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(normalize_fn(x) if normalize_fn else x)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return 100 * correct / total


def train_bait_w(model, train_loader, test_loader, device, num_classes,
                  soft_labels, trigger, epochs, poison_rate=0.10, lr=0.1,
                  weight_decay=5e-4, momentum=0.9, normalize_fn=None, log_every=5):
    """BAIT-W poisoned training: a single cross-entropy loss over a mixed
    batch of clean (one-hot label) and triggered (soft geometric label)
    samples. No auxiliary losses, generators, or alternating optimization.
    """
    opt = torch.optim.SGD(model.parameters(), lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = GradScaler("cuda")

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = len(x)
            poison_mask = (torch.rand(batch_size) < poison_rate).to(device)

            x_mixed = x.clone()
            x_mixed[poison_mask] = trigger(x[poison_mask])

            y_soft = F.one_hot(y, num_classes).float()
            y_soft[poison_mask] = soft_labels[y[poison_mask]]

            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
                out = model(normalize_fn(x_mixed) if normalize_fn else x_mixed)
                loss = -(y_soft * F.log_softmax(out, dim=1)).sum(1).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        if (ep + 1) % log_every == 0 or ep == 0:
            clean, asr, ds, _ = evaluate_attack(
                model, test_loader, device, trigger, num_classes, normalize_fn
            )
            print(f"  [bait-w] epoch {ep + 1:3d}  clean={clean:.2f}  "
                  f"ASR={asr:.2f}  DS={ds:.3f}")
    return model
