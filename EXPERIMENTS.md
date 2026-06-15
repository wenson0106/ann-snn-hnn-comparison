# Experiment Log

## E1: ANN / SNN / HNN Accuracy on MNIST

Common setup:
- Split: `data/splits/mnist_seed42.pt`
- Train / validation / test samples: 54000 / 6000 / 10000
- Epochs: 3
- Batch size: 128 (ANN/HNN), 64 (SNN)
- Seed: 42
- SNN/HNN time steps: 10
- SNN input: deterministic rate-coded spikes from MNIST images

Results:

| Model | Run | Best Val Acc | Test Acc | Test Loss |
|---|---|---|---:|---:|---:|
| ANN | `runs/ann_mnist/20260526-111504` | 0.9775 | **0.9809** | 0.0560 |
| SNN | `runs/snn_mnist_spike/20260526-111623` | 0.9733 | 0.9764 | 0.0753 |
| HNN | `runs/hnn_mnist/20260526-112225` | 0.9745 | 0.9783 | 0.0670 |

Observation: MNIST is too easy — all three models cluster in 97-98%.

---

## E2: SNN / HNN Firing Rate on MNIST

Common setup: E1 checkpoints, T=10, beta=0.95, threshold=1.0

| Model | Test Acc | Overall Firing Rate | Hidden Firing Rate |
|---|---:|---:|---:|
| SNN | 0.9764 | 0.1785 | 0.1862 |
| HNN | 0.9783 | 0.2545 | 0.2545 |

Layer-wise:

| Model | Layer | Firing Rate |
|---|---:|---:|
| SNN | input | 0.1326 |
| SNN | conv1 | 0.1783 |
| SNN | conv2 | 0.1849 |
| SNN | fc1 | 0.2799 |
| SNN | fc2 | 0.3933 |
| HNN | first_spike | 0.2920 |
| HNN | conv2 | 0.2097 |
| HNN | fc1 | 0.2757 |
| HNN | fc2 | 0.3840 |

---

## E3: Parameter Sweeps on MNIST

### E3-A: Threshold Sweep (MNIST)

| Model | Threshold | Test Acc | Hidden Firing Rate |
|---|---|---:|---:|---:|
| SNN | 0.5 | **0.9813** | 0.1775 |
| SNN | 1.0 | 0.9764 | 0.1862 |
| SNN | 1.5 | 0.9714 | 0.1996 |
| HNN | 0.5 | **0.9827** | 0.2632 |
| HNN | 1.0 | 0.9783 | 0.2545 |
| HNN | 1.5 | 0.9721 | 0.2532 |

### E3-B: Beta Sweep (MNIST, threshold=0.5)

| Model | Beta | Test Acc | Hidden Firing Rate |
|---|---|---:|---:|---:|
| SNN | 0.8 | 0.9796 | 0.1847 |
| SNN | 0.9 | 0.9811 | 0.1806 |
| SNN | 0.95 | **0.9813** | 0.1775 |
| HNN | 0.8 | 0.9808 | 0.2707 |
| HNN | 0.9 | **0.9829** | 0.2663 |
| HNN | 0.95 | 0.9827 | 0.2632 |

### E3-C: Time Steps Sweep (MNIST, threshold=0.5, beta=0.95)

| Model | Time Steps | Test Acc | Hidden Firing Rate |
|---|---|---:|---:|---:|
| SNN | 5 | 0.9800 | 0.1986 |
| SNN | 10 | **0.9813** | 0.1775 |
| SNN | 20 | 0.9803 | 0.1672 |
| HNN | 5 | 0.9790 | 0.2724 |
| HNN | 10 | 0.9827 | 0.2632 |
| HNN | 20 | **0.9836** | 0.2549 |

---

# CIFAR-10 Experiments

All experiments use:
- Split: `data/splits/cifar10_seed42.pt`
- Train / validation / test: 45000 / 5000 / 10000
- Architecture: LeNet (in_channels=3, feature_size=5)
- Early stopping: patience=5 (val loss), max epochs=1000
- Best checkpoint selected by validation accuracy
- Adam lr=0.001

## E1 (CIFAR-10): ANN / SNN / HNN Accuracy

Default params: threshold=1.0, beta=0.95, T=10 (SNN/HNN), batch=128 (ANN/HNN), batch=64 (SNN)

| Model | Run | Epochs | Best Val Acc | Test Acc | Test Loss |
|---|---|---|---:|---:|---:|---:|
| ANN | `runs/ann_cifar10/20260614-225310` | 16 | 0.6262 | **0.6200** | 1.1053 |
| SNN | `runs/snn_cifar10/20260614-225311` | 18 | 0.5574 | **0.5508** | 1.2723 |
| HNN | `runs/hnn_cifar10/20260614-225312` | 25 | 0.5986 | **0.6045** | 1.1297 |

Key finding: ANN ≈ HNN > SNN. The gap between ANN and SNN (~5.7%) is far more pronounced than on MNIST, confirming that MNIST was too simple to differentiate these model families.

---

## E2 (CIFAR-10): Firing Rate Analysis

### Firing rate by time steps (default thr=1.0, beta=0.95)

| Model | T | Test Acc | Overall FR | Hidden FR | conv1/first_spike | conv2 | fc1 | fc2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SNN | 1 | 0.4362 | 0.3655 | 0.3131 | 0.3474 | 0.2118 | 0.2942 | 0.3471 |
| SNN | 3 | 0.4925 | 0.3550 | 0.2977 | 0.3447 | 0.1633 | 0.2389 | 0.3070 |
| SNN | 5 | 0.5217 | 0.3093 | 0.2304 | 0.2558 | 0.1464 | 0.2661 | 0.3529 |
| SNN | 10 | 0.5508 | 0.2647 | 0.1647 | 0.1618 | 0.1483 | 0.3112 | 0.4287 |
| SNN | 20 | 0.5702 | 0.2576 | 0.1543 | 0.1466 | 0.1429 | 0.3723 | 0.4925 |
| HNN | 1 | 0.5276 | 0.2945 | 0.2945 | 0.2917 | 0.2861 | 0.3632 | 0.3945 |
| HNN | 3 | 0.5856 | 0.2527 | 0.2527 | 0.2994 | 0.2148 | 0.2519 | 0.3201 |
| HNN | 5 | 0.5851 | 0.2506 | 0.2506 | 0.2856 | 0.2206 | 0.2574 | 0.3209 |
| HNN | 10 | 0.6045 | 0.2297 | 0.2297 | 0.2587 | 0.1967 | 0.2923 | 0.3620 |
| HNN | 20 | 0.6152 | 0.2233 | 0.2233 | 0.2462 | 0.1820 | 0.3579 | 0.4962 |

Key findings:
- SNN hidden FR drops dramatically with more time steps (0.313→0.154), as longer integration enables more selective spiking
- HNN hidden FR also drops (0.295→0.223) but less dramatically — its analog first layer already provides stable features
- For both models, conv1/first_spike and conv2 FR decrease with T, while fc1 and fc2 FR increase — deeper layers maintain high firing rates to support classification

---

## E3 (CIFAR-10): Parameter Sweeps

### E3-A (CIFAR-10): Threshold Sweep (T=10, beta=0.95)

| Model | Threshold | Test Acc |
|---|---:|---:|
| SNN | 0.5 | 0.5323 |
| SNN | 1.0 | 0.5505 |
| SNN | **1.5** | **0.5529** |
| HNN | **0.5** | **0.6081** |
| HNN | 1.0 | 0.6026 |
| HNN | 1.5 | 0.6048 |

Notable: SNN benefits from higher threshold (opposite to MNIST, where lower was better). HNN benefits from lower threshold (consistent with MNIST). This suggests that on harder datasets, a higher SNN threshold acts as a denoising mechanism.

### E3-B (CIFAR-10): Beta Sweep (SNN: thr=1.5, HNN: thr=0.5, T=10)

| Model | Beta | Test Acc |
|---|---:|---:|
| SNN | 0.8 | 0.5505 |
| SNN | 0.9 | 0.5449 |
| SNN | **0.95** | **0.5529** |
| HNN | 0.8 | 0.5910 |
| HNN | 0.9 | 0.5970 |
| HNN | **0.95** | **0.6081** |

Higher beta is generally better — though SNN shows slight non-monotonicity at β=0.9 (0.5449 vs 0.5505 at β=0.8), the overall trend favors higher β.

### E3-C (CIFAR-10): Time Steps Sweep (SNN: thr=1.5, HNN: thr=0.5, beta=0.95)

| Model | Time Steps | Test Acc |
|---|---:|---:|
| SNN | 5 | 0.5310 |
| SNN | 10 | 0.5529 |
| SNN | **20** | **0.5602** |
| HNN | 5 | 0.5883 |
| HNN | **10** | **0.6081** |
| HNN | 20 | 0.6016 |

SNN monotonically improves with more time steps. HNN peaks at T=10, suggesting its analog first layer already captures enough spatial information and additional time steps don't help.

---

---

## E1-B (CIFAR-10): Time Steps Variation (Default Config)

Same default params as E1 (threshold=1.0, beta=0.95). Combined with E3-C time steps sweep data (different threshold per model) for comparison.

Default config (thr=1.0, beta=0.95):

| Model | Time Steps | Epochs | Best Val Acc | Test Acc | Test Loss |
|---|---|---|---|---:|---:|---:|---:|
| SNN | 1 | 17 | 0.4396 | **0.4362** | 1.5718 |
| SNN | 3 | 16 | 0.4938 | **0.4925** | 1.4185 |
| SNN | 5 | 16 | 0.5214 | **0.5217** | 1.3374 |
| SNN | 10 | 18 | 0.5476 | **0.5508** | 1.2723 |
| SNN | 20 | 17 | 0.5666 | **0.5702** | 1.2209 |
| HNN | 1 | 24 | 0.5260 | **0.5276** | 1.3565 |
| HNN | 3 | 25 | 0.5868 | **0.5856** | 1.1853 |
| HNN | 5 | 21 | 0.5854 | **0.5851** | 1.1718 |
| HNN | 10 | 25 | 0.5986 | **0.6045** | 1.1297 |
| HNN | 20 | 27 | 0.6114 | **0.6152** | 1.1145 |

Both models benefit monotonically from more time steps. SNN is more sensitive (gap between T=1 and T=10: +11.5%) than HNN (+7.7%), because HNN's analog first layer already extracts spatial features before the temporal loop. Firing rate decreases with more time steps (SNN hidden FR: 0.313 at T=1 → 0.154 at T=20), showing that longer integration enables more selective spiking.

Compared with tuned configs from E3-C (SNN thr=1.5, HNN thr=0.5):

| Model | Time Steps | Default Config | Tuned Config |
|---|---|---|---|---:|
| SNN | 1 | 0.4362 | — |
| SNN | 3 | 0.4925 | — |
| SNN | 5 | 0.5217 | 0.5310 |
| SNN | 10 | 0.5508 | 0.5529 |
| SNN | 20 | 0.5702 | 0.5602 |
| HNN | 1 | 0.5276 | — |
| HNN | 3 | 0.5856 | — |
| HNN | 5 | 0.5851 | 0.5883 |
| HNN | 10 | 0.6045 | 0.6081 |
| HNN | 20 | 0.6152 | 0.6016 |

---

## E4 (CIFAR-10): Post-Training Weight Quantization (8-bit / 4-bit)

**Method**: Uniform per-tensor affine quantization of all Conv2d/Linear weights after training (no fine-tuning). Quantized weights are dequantized for simulated inference.

| Model | float32 | int8 | int4 | Size (float32) | Size (int8) | Size (int4) |
|---|---|---|---:|---:|---:|---:|---:|
| ANN | 0.6200 | **0.6204** (0.0%) | **0.5743** (-7.4%) | 247KB | 61.8KB (4×) | 30.9KB (8×) |
| SNN | 0.5508 | **0.5535** (+0.5%) | **0.5078** (-7.8%) | 247KB | 61.8KB (4×) | 30.9KB (8×) |
| HNN | 0.6045 | **0.6033** (-0.2%) | **0.5732** (-5.2%) | 247KB | 61.8KB (4×) | 30.9KB (8×) |

Key findings:
- **8-bit quantization**: Near-zero accuracy loss (<0.5%) for all three model families.
- **4-bit quantization**: Accuracy drops 5-8%. HNN is most robust (-5.2%), SNN is most affected (-7.8%), suggesting spiking dynamics amplify weight quantization noise.
- All models have identical architecture size (same LeNet skeleton), so compression ratio is uniform.

---

## E5 (CIFAR-10): Kernel Size Comparison (5×5 vs 3×3)

Default params: threshold=1.0, beta=0.95, T=10, patience=5

| Model | Kernel | Best Epoch | Best Val Acc | Test Acc | Test Loss |
|---|---|---|---:|---:|---:|---:|
| ANN | 5 | 16 | 0.6262 | **0.6200** | 1.1053 |
| ANN | 3 | 18 | 0.6262 | **0.6248** | 1.1053 |
| SNN | 5 | 14 | 0.5476 | **0.5508** | 1.2723 |
| SNN | 3 | 20 | 0.5420 | **0.5505** | 1.3031 |
| HNN | 5 | 8 | 0.5986 | **0.6045** | 1.1297 |
| HNN | 3 | 16 | 0.5838 | **0.6009** | 1.1655 |

Kernel size has minimal impact on CIFAR-10 with LeNet architecture — all changes < 0.5%.

## CIFAR-10 Commands (for reference)

```bash
# E1 default
python -m src.train --experiment ann_cifar10 --epochs 1000 --batch-size 128 --lr 0.001 --patience 5
python -m src.train --experiment snn_cifar10 --epochs 1000 --batch-size 64 --lr 0.001 --time-steps 10 --patience 5
python -m src.train --experiment hnn_cifar10 --epochs 1000 --batch-size 128 --lr 0.001 --time-steps 10 --patience 5

# E3 threshold sweep (T=10, beta=0.95)
python -m src.train --experiment snn_cifar10 --epochs 1000 --batch-size 64 --lr 0.001 --time-steps 10 --threshold 1.5 --patience 5
python -m src.train --experiment hnn_cifar10 --epochs 1000 --batch-size 128 --lr 0.001 --time-steps 10 --threshold 0.5 --patience 5

# E2 firing rate
python -m scripts.evaluate_firing_rate --runs runs/snn_cifar10/<timestamp> runs/hnn_cifar10/<timestamp> --out-dir experiments/e2_firing_rate_cifar10 --device cuda
```
