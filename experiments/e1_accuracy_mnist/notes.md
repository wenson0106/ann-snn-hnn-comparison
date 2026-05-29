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
