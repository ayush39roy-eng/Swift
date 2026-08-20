# SWIFT: Sliced Wasserstein Informed Feature Space Triggering for Untargeted Backdoor Attacks
An untargeted backdoor attack that derives its dispersion targets directly from
a victim classifier's own feature-space geometry, rather than through
engineered loss functions or external vision-language models.

Instead of forcing triggered inputs toward an arbitrary set of wrong classes
(as in prior loss-engineered untargeted attacks), swift identifies each
class's *naturally confusable* classes by combining:

- **Sliced Wasserstein distance** between penultimate-layer feature
  distributions of a clean reference model — capturing the full
  distributional geometry of how the model separates classes, not just
  centroid proximity.
- **Semantic distance** between class labels — preventing geometrically
  "attractor" classes from dominating unrelated confusion zones.

A backdoored model is then trained with a **single cross-entropy loss** over
soft labels derived from these confusion zones, with only 10% of training
data poisoned. No auxiliary losses, trigger generators, or alternating
optimization are required.

## Results summary

| Dataset | Model | K | Clean Acc. | ASR | DS | Zone Acc. |
|---|---|---|---|---|---|---|
| CIFAR-10 | ResNet-18 | 5 | 94.6% | 99.8% | 0.961 | 96.9% |
| CIFAR-100 | ResNet-18 | 8 | 76.2% | 95.7% | 0.992 | 86.6% |
| GTSRB | PreActResNet18 | 5 | 97.3% | 99.8% | 0.979 | 99.4% |
| MNIST | SimpleCNN | 3 | 98.7% | 99.6% | 0.957 | 99.3% |

swift evades Neural Cleanse, STRIP, and SentiNet, and resists fine-pruning
even at 90% channel removal (see paper for full defense evaluation).

## Repository structure

```
swift-repo/
├── run_bait_w.py              # end-to-end pipeline entry point
├── configs/
│   ├── cifar10.yaml
│   └── cifar100.yaml
├── src/
│   ├── models.py               # ResNet-18
│   ├── trigger.py              # globally-blended sinusoidal trigger
│   ├── geometry.py              # Sliced Wasserstein + confusion-zone selection
│   ├── semantic.py              # semantic distance sources
│   ├── train.py                 # victim + swift training loops
│   ├── evaluate.py              # CA, ASR, DS, Zone Accuracy, Max Concentration
│   └── defenses/
│       ├── neural_cleanse.py
│       ├── strip.py
│       └── fine_pruning.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md   # optional, for word-embedding semantic distance
```

## Usage

```bash
python run_bait_w.py --config configs/cifar10.yaml
python run_bait_w.py --config configs/cifar100.yaml
```

This trains the reference model, computes confusion-zone geometry, trains the
backdoored model, and reports attack metrics plus Neural Cleanse / STRIP /
fine-pruning results.

## Metrics

- **Clean Accuracy (CA)** — accuracy on untriggered inputs.
- **Attack Success Rate (ASR)** — misclassification rate on triggered inputs
  (restricted to inputs correctly classified in clean form).
- **Dispersibility Score (DS)** — from CUBA (Xue et al.); measures how evenly
  misclassifications spread across the target set.
- **Zone Accuracy** *(this work)* — fraction of successful misclassifications
  landing inside the geometrically-predicted confusion zone, rather than
  scattering arbitrarily. Random baseline: `K / (num_classes - 1)`.
- **Max Class Concentration** *(this work)* — largest single-class share of
  misclassifications; checks for the collapse-to-one-class failure mode.

## Citation

```bibtex
@article{baitw2026,
  title   = {swift: Geometry-Guided Untargeted Backdoor Attacks via Hybrid Wasserstein-Semantic Confusion Zones},
  author  = {[Author names]},
  year    = {2026},
  note    = {Working draft, AIMS-DTU}
}
```

## License

For research purposes only.
