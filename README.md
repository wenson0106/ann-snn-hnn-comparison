# ANN / SNN / HNN Comparison on MNIST & CIFAR-10

## 實驗目標

比較三種神經網路架構在圖像分類任務上的表現差異：

| 模型 | 全稱 | 說明 |
|---|---|---|
| **ANN** | Artificial Neural Network | 標準類比神經網路（ReLU activation），作為 baseline |
| **SNN** | Spiking Neural Network | 全脈衝神經網路（LIF neuron），輸入為 rate-coded spike train |
| **HNN** | Hybrid Neural Network | 第一層為類比 Conv 其餘為脈衝層，介於 ANN 與 SNN 之間的過渡架構 |

所有模型共用 LeNet 骨架，確保比較公平性。涵蓋三個實驗：

- **E1 — 準確率比較**：在 MNIST 與 CIFAR-10 上比較 ANN / SNN / HNN
- **E2 — Firing Rate 分析**：分析 SNN / HNN 各層的 spike 放電率
- **E3 — 超參數掃描**：Threshold、Beta、Time Steps 對 SNN / HNN 的影響

詳細結果與討論請見 [`report/report.md`](report/report.md)（含圖表）。

---

## 環境安裝 (Conda)

```powershell
conda create -n snn python=3.9 -y
conda activate snn
pip install -r requirements.txt
```

如需 GPU 加速（RTX 3070 等 CUDA 卡）：

```powershell
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

---

## 資料準備

```powershell
# MNIST + CIFAR-10
python scripts/prepare_data.py --dataset mnist
python scripts/prepare_data.py --dataset cifar10
```

MNIST 由 `torchvision` 自動下載。SNN 版本的 MNIST 使用相同的 split，在訓練時即時轉換為 rate-coded spike train。

---

## 執行實驗

### MNIST

```powershell
python -m src.train --experiment ann_mnist --epochs 3 --batch-size 128 --device cuda
python -m src.train --experiment snn_mnist_spike --epochs 3 --batch-size 64 --device cuda --time-steps 10
python -m src.train --experiment hnn_mnist --epochs 3 --batch-size 128 --device cuda --time-steps 10
```

### CIFAR-10（含 Early Stopping）

```powershell
python -m src.train --experiment ann_cifar10 --epochs 1000 --batch-size 128 --lr 0.001 --patience 5 --device cuda
python -m src.train --experiment snn_cifar10 --epochs 1000 --batch-size 64 --lr 0.001 --time-steps 10 --patience 5 --device cuda
python -m src.train --experiment hnn_cifar10 --epochs 1000 --batch-size 128 --lr 0.001 --time-steps 10 --patience 5 --device cuda
```

### 超參數掃描

```powershell
# Threshold sweep (T=10, beta=0.95)
python -m src.train --experiment snn_cifar10 --epochs 1000 --batch-size 64 --lr 0.001 --time-steps 10 --threshold 0.5 --patience 5 --device cuda
python -m src.train --experiment snn_cifar10 --epochs 1000 --batch-size 64 --lr 0.001 --time-steps 10 --threshold 1.5 --patience 5 --device cuda

# Beta sweep
python -m src.train --experiment snn_cifar10 --epochs 1000 --batch-size 64 --lr 0.001 --time-steps 10 --threshold 1.5 --beta 0.8 --patience 5 --device cuda

# Time steps sweep
python -m src.train --experiment snn_cifar10 --epochs 1000 --batch-size 64 --lr 0.001 --time-steps 20 --threshold 1.5 --patience 5 --device cuda
```

---

## Firing Rate 分析

```powershell
python -m scripts.evaluate_firing_rate --runs runs/snn_cifar10/<timestamp> runs/hnn_cifar10/<timestamp> --out-dir experiments/e2_firing_rate_cifar10 --device cuda
```

---

## 專案結構

```text
.
+-- configs/                 # Default experiment arguments
+-- data/
|   +-- splits/              # Deterministic train/val/test split files
|   +-- raw/                 # Downloaded datasets
+-- runs/                    # Checkpoints and training logs
+-- experiments/             # Experiment-level summaries (E1, E2, E3)
+-- report/                  # Results report with figures
|   +-- images/              # Generated charts
|   +-- report.md            # Full results & discussion
+-- scripts/
|   +-- prepare_data.py      # Download datasets and create split files
|   +-- summarize_accuracy.py
|   +-- evaluate_firing_rate.py
|   +-- make_figures.py      # Generate report figures
+-- src/
    +-- data/                # Dataset builders
    +-- models/              # ANN, SNN, HNN model definitions
    +-- utils/               # Training utilities
    +-- train.py             # Shared training entry point
```
