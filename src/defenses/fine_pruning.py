"""Fine-Pruning (Liu et al., 2018): removes (zeroes) the channels with
lowest mean activation on clean data, under the assumption that
backdoor-specific channels are dormant on clean inputs. If ASR remains
coupled to clean accuracy as pruning increases, the backdoor pathway is
entangled with normal classification rather than isolated in prunable,
dormant neurons."""

import copy
import numpy as np
import torch

from ..evaluate import dispersibility_score


@torch.no_grad()
def _channel_activation(model, loader, device, normalize_fn=None):
    acts = torch.zeros(512)
    n_batches = 0
    hook_out = {}

    def hook(_, __, output):
        hook_out["act"] = output.detach()

    handle = model.layer4.register_forward_hook(hook)
    for x, _ in loader:
        x = x.to(device)
        model(normalize_fn(x) if normalize_fn else x)
        acts += hook_out["act"].mean(dim=(0, 2, 3)).cpu()
        n_batches += 1
    handle.remove()
    return acts / n_batches


def run_fine_pruning(model, loader, device, trigger, num_classes,
                      normalize_fn=None, prune_rates=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9)):
    activations = _channel_activation(model, loader, device, normalize_fn)
    order = torch.argsort(activations)  # ascending: lowest-activation first
    results = []

    for rate in prune_rates:
        pruned = copy.deepcopy(model)
        k = int(rate * 512)
        kill = order[:k]
        with torch.no_grad():
            pruned.layer4[-1].c2.weight[kill] = 0
            pruned.layer4[-1].b2.weight[kill] = 0
            pruned.layer4[-1].b2.bias[kill] = 0

        pruned.eval()
        mis = tot = ben_correct = ben_total = 0
        hist = np.zeros(num_classes)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                xn = normalize_fn(x) if normalize_fn else x
                out_c = pruned(xn)
                ben_correct += (out_c.argmax(1) == y).sum().item()
                ben_total += len(y)

                xt = trigger(x)
                xtn = normalize_fn(xt) if normalize_fn else xt
                out_t = pruned(xtn)
                pred = out_t.argmax(1)
                wrong = pred != y
                mis += wrong.sum().item()
                tot += len(y)
                for c in pred[wrong].cpu().numpy():
                    hist[c] += 1

        results.append({
            "prune_rate": rate,
            "clean_acc": 100 * ben_correct / ben_total,
            "asr": 100 * mis / max(tot, 1),
            "ds": dispersibility_score(hist),
        })

    return results
