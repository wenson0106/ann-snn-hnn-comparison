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
