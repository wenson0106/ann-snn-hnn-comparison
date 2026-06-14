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
|---|---|---:|---:|---:|
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
| ANN | `runs/ann_cifar10/20260526-144318` | 18 | 0.6112 | **0.6077** | 1.1593 |
| SNN | `runs/snn_cifar10/20260526-145103` | 18 | 0.5574 | 0.5505 | 1.2704 |
| HNN | `runs/hnn_cifar10/20260526-151814` | 24 | 0.5998 | **0.6026** | 1.1241 |

Key finding: ANN ≈ HNN > SNN. The gap between ANN and SNN (~5.7%) is far more pronounced than on MNIST, confirming that MNIST was too simple to differentiate these model families.

---

## E2 (CIFAR-10): Firing Rate Analysis

### Default E1 checkpoints (T=10, thr=1.0, beta=0.95)

| Model | Test Acc | Overall Firing Rate | Hidden Firing Rate |
|---|---|---:|---:|---:|
| SNN | 0.5505 | 0.2746 | 0.1793 |
| HNN | 0.6026 | 0.2336 | 0.2336 |

Layer-wise (default E1):

| Model | Layer | Firing Rate |
|---|---:|---:|
| SNN | input | 0.4766 |
| SNN | conv1 | 0.1762 |
| SNN | conv2 | 0.1632 |
| SNN | fc1 | 0.3290 |
| SNN | fc2 | 0.4479 |
| HNN | first_spike | 0.2654 |
| HNN | conv2 | 0.1988 |
| HNN | fc1 | 0.2864 |
| HNN | fc2 | 0.3736 |

### Best configuration checkpoints

| Model | Config | Test Acc | Overall Firing Rate | Hidden Firing Rate |
|---|---|---|---:|---:|---:|
| SNN | T=20, thr=1.5, beta=0.95 | **0.5602** | 0.2758 | 0.1810 |
| HNN | T=10, thr=0.5, beta=0.95 | **0.6081** | 0.2744 | 0.2744 |

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

Higher beta is better for both models — consistent with MNIST.

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
|---|---|---|---:|---:|---:|---:|
| SNN | 1 | 17 | 0.4396 | **0.4362** | 1.5718 |
| SNN | 3 | 16 | 0.4938 | **0.4925** | 1.4185 |
| SNN | 10 | 18 | 0.5476 | **0.5508** | 1.2723 |
| HNN | 1 | 24 | 0.5260 | **0.5276** | 1.3565 |
| HNN | 3 | 25 | 0.5868 | **0.5856** | 1.1853 |
| HNN | 10 | 25 | 0.5986 | **0.6045** | 1.1297 |

Both models benefit monotonically from more time steps. SNN is more sensitive (gap between T=1 and T=10: +11.5%) than HNN (+7.7%), because HNN's analog first layer already extracts spatial features before the temporal loop.

Compared with tuned configs from E3-C (SNN thr=1.5, HNN thr=0.5):

| Model | Time Steps | Default Config | Tuned Config |
|---|---|---|---:|
| SNN | 1 | 0.4362 | — |
| SNN | 3 | 0.4925 | — |
| SNN | 5 | — | 0.5310 |
| SNN | 10 | 0.5508 | 0.5529 |
| SNN | 20 | — | 0.5602 |
| HNN | 1 | 0.5276 | — |
| HNN | 3 | 0.5856 | — |
| HNN | 5 | — | 0.5883 |
| HNN | 10 | 0.6045 | 0.6081 |
| HNN | 20 | — | 0.6016 |

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
