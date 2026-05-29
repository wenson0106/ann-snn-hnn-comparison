# Experiment Log

## E1: ANN / SNN / HNN Accuracy on MNIST

Goal: compare classification accuracy across the three model families using the same original MNIST train/validation/test split.

Common setup:

- Split: `data/splits/mnist_seed42.pt`
- Train / validation / test samples: 54000 / 6000 / 10000
- Epochs: 3
- Batch size: 128
- Seed: 42
- Device: CPU
- SNN/HNN time steps: 10
- SNN input: deterministic rate-coded spikes from MNIST images
- HNN input: MNIST image input, LIF spikes after the first layer

Results:

| Model | Run | Best Val Acc | Test Acc | Test Loss |
|---|---|---:|---:|---:|
| ANN | `runs/ann_mnist/20260526-111504` | 0.9775 | 0.9809 | 0.0560 |
| SNN | `runs/snn_mnist_spike/20260526-111623` | 0.9733 | 0.9764 | 0.0753 |
| HNN | `runs/hnn_mnist/20260526-112225` | 0.9745 | 0.9783 | 0.0670 |

Summary files:

- `runs/exp1_accuracy_mnist_20260526-112225.json`
- `runs/exp1_accuracy_mnist_20260526-112225.csv`

Initial observation:

ANN is the strongest after 3 epochs. HNN is slightly below ANN and slightly above SNN in this run. The gap is small enough that follow-up firing-rate analysis is meaningful.

## E2: SNN / HNN Firing Rate on MNIST

Goal: compare spike activity between the best SNN and HNN checkpoints from E1.

Common setup:

- Test split: `data/splits/mnist_seed42.pt`
- Test samples: 10000
- Checkpoints:
  - SNN: `runs/snn_mnist_spike/20260526-111623/best.pt`
  - HNN: `runs/hnn_mnist/20260526-112225/best.pt`
- Time steps: 10
- Beta: 0.95
- Threshold: 1.0

Results:

| Model | Test Acc | Overall Firing Rate | Hidden Firing Rate |
|---|---:|---:|---:|
| SNN | 0.9764 | 0.1785 | 0.1862 |
| HNN | 0.9783 | 0.2545 | 0.2545 |

Layer-wise firing rate:

| Model | Layer | Firing Rate |
|---|---|---:|
| SNN | input | 0.1326 |
| SNN | conv1 | 0.1783 |
| SNN | conv2 | 0.1849 |
| SNN | fc1 | 0.2799 |
| SNN | fc2 | 0.3933 |
| HNN | first_spike | 0.2920 |
| HNN | conv2 | 0.2097 |
| HNN | fc1 | 0.2757 |
| HNN | fc2 | 0.3840 |

Summary files:

- `experiments/e2_firing_rate_mnist/firing_rate_results.json`
- `experiments/e2_firing_rate_mnist/layer_firing_rate.csv`

Initial observation:

HNN is slightly more accurate in this checkpoint pair, but it also has a higher hidden firing rate. The largest difference appears immediately after the HNN first analog layer, where `first_spike` fires at about 0.2920.

## E3: Threshold Sweep on SNN / HNN

Goal: study how an internal LIF parameter affects accuracy and firing rate.

Common setup:

- Dataset: MNIST
- Split: `data/splits/mnist_seed42.pt`
- Test samples: 10000
- Models: SNN and HNN
- Epochs: 3
- Batch size: 128
- Seed: 42
- Time steps: 10
- Beta: 0.95
- Threshold values: 0.5, 1.0, 1.5
- Threshold 1.0 reuses the E1/E2 checkpoints.

Main results:

| Model | Threshold | Test Acc | Hidden Firing Rate |
|---|---:|---:|---:|
| SNN | 0.5 | 0.9813 | 0.1775 |
| SNN | 1.0 | 0.9764 | 0.1862 |
| SNN | 1.5 | 0.9714 | 0.1996 |
| HNN | 0.5 | 0.9827 | 0.2632 |
| HNN | 1.0 | 0.9783 | 0.2545 |
| HNN | 1.5 | 0.9721 | 0.2532 |

Summary files:

- `experiments/e3_param_sweep_mnist/threshold_sweep_summary.csv`
- `experiments/e3_param_sweep_mnist/accuracy_threshold_sweep.csv`
- `experiments/e3_param_sweep_mnist/accuracy_threshold_sweep.json`
- `experiments/e3_param_sweep_mnist/firing_rate_threshold_sweep/firing_rate_results.json`
- `experiments/e3_param_sweep_mnist/firing_rate_threshold_sweep/layer_firing_rate.csv`

Initial observation:

In this retrained sweep, lower threshold gives better accuracy for both SNN and HNN. Firing rate is not strictly monotonic with threshold because each model is retrained from scratch and can compensate through its weights. In this first sweep, HNN remains more accurate than SNN at the same threshold, but its hidden firing rate is also higher.

### E3-B: Beta Sweep

Common setup:

- Threshold: 0.5
- Time steps: 10
- Beta values: 0.8, 0.9, 0.95

| Model | Beta | Test Acc | Hidden Firing Rate |
|---|---:|---:|---:|
| SNN | 0.8 | 0.9796 | 0.1847 |
| SNN | 0.9 | 0.9811 | 0.1806 |
| SNN | 0.95 | 0.9813 | 0.1775 |
| HNN | 0.8 | 0.9808 | 0.2707 |
| HNN | 0.9 | 0.9829 | 0.2663 |
| HNN | 0.95 | 0.9827 | 0.2632 |

Summary files:

- `experiments/e3_param_sweep_mnist/beta_sweep_summary.csv`
- `experiments/e3_param_sweep_mnist/accuracy_beta_sweep.csv`
- `experiments/e3_param_sweep_mnist/firing_rate_beta_sweep/firing_rate_results.json`
- `experiments/e3_param_sweep_mnist/firing_rate_beta_sweep/layer_firing_rate.csv`

Initial observation:

Higher beta generally lowers hidden firing rate while preserving or slightly improving accuracy in this sweep.

### E3-C: Time Steps Sweep

Common setup:

- Threshold: 0.5
- Beta: 0.95
- Time steps: 5, 10, 20

| Model | Time Steps | Test Acc | Hidden Firing Rate |
|---|---:|---:|---:|
| SNN | 5 | 0.9800 | 0.1986 |
| SNN | 10 | 0.9813 | 0.1775 |
| SNN | 20 | 0.9803 | 0.1672 |
| HNN | 5 | 0.9790 | 0.2724 |
| HNN | 10 | 0.9827 | 0.2632 |
| HNN | 20 | 0.9836 | 0.2549 |

Summary files:

- `experiments/e3_param_sweep_mnist/time_steps_sweep_summary.csv`
- `experiments/e3_param_sweep_mnist/accuracy_time_steps_sweep.csv`
- `experiments/e3_param_sweep_mnist/firing_rate_time_steps_sweep/firing_rate_results.json`
- `experiments/e3_param_sweep_mnist/firing_rate_time_steps_sweep/layer_firing_rate.csv`

Initial observation:

More time steps reduce average per-step hidden firing rate. HNN benefits more clearly from 20 time steps in accuracy, while SNN accuracy is similar across 5, 10, and 20 time steps.
