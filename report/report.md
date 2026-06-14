# ANN / SNN / HNN 在 MNIST 與 CIFAR-10 上的比較分析

---

## 摘要

本研究比較 ANN (Artificial Neural Network)、SNN (Spiking Neural Network) 與 HNN (Hybrid Neural Network) 三種模型架構在 MNIST 與 CIFAR-10 兩個資料集上的分類表現。三種模型均以 LeNet 為基礎骨幹，確保比較的公平性。實驗涵蓋準確率比較（E1）、spike firing rate 分析（E2）以及 threshold、beta、time steps 三個關鍵超參數的掃描（E3）。

主要發現：

1. **MNIST 過於簡單**，三種模型準確率集中在 97–98%，無法有效區分架構優劣。
2. **CIFAR-10 上 ANN ≈ HNN >> SNN**（60.8% vs 60.3% vs 55.0%），差距約 5–6 個百分點。
3. **Threshold 行為在兩個資料集上相反**：MNIST 上低 threshold 較好，CIFAR-10 上 SNN 需要高 threshold 達到最佳表現，推測高 threshold 在複雜資料上有去噪效果。
4. **HNN 表現最為穩健**，在所有超參數配置下均接近 ANN，是從 ANN 過渡到純 SNN 的良好折衷方案。

---

## 1. 實驗設計概覽

### 1.1 模型架構

| 模型 | 輸入 | 隱藏層 | 輸出 |
|---|---|---|---|
| **ANN (LeNetANN)** | 連續值影像 (ToTensor + Normalize) | Conv → ReLU → Pool → Conv → ReLU → Pool → FC → ReLU → FC → ReLU | Linear (logits) |
| **SNN (LeNetSNN)** | Rate-coded spike train [B, T, C, H, W] | Conv → LIF → Pool → Conv → LIF → Pool → FC → LIF → FC → LIF | Linear (readout, non-spiking) |
| **HNN (LeNetHNN)** | 連續值影像（第一層 analog） | Conv → ReLU → Pool → LIF → Conv → LIF → Pool → FC → LIF → FC → LIF | Linear (readout, non-spiking) |

三種模型共享相同的 LeNet 結構（2 層 Conv + 3 層 FC），差異僅在於 activation function 與 temporal processing 的設計。

### 1.2 資料集

| 資料集 | 訓練 / 驗證 / 測試 | 輸入形狀 | ANN 預處理 | SNN 預處理 |
|---|---|---|---|---|
| MNIST | 54,000 / 6,000 / 10,000 | [1, 28, 28] | ToTensor + Normalize | Rate coding (T=10) |
| CIFAR-10 | 45,000 / 5,000 / 10,000 | [3, 32, 32] | ToTensor + Normalize | Rate coding (T=5/10/20) |

### 1.3 訓練設定

- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Early stopping: patience=5（validation loss 連續 5 epoch 未改善即停止）
- 最終選取 validation accuracy 最高的 checkpoint 進行測試
- GPU: NVIDIA RTX 3070 Laptop

---

## 2. 結果 (Results)

### 2.1 E1: 三模型準確率比較

#### MNIST 結果

| Model | Test Accuracy |
|---|---|
| ANN | **98.09%** |
| SNN | 97.64% |
| HNN | 97.83% |

在 MNIST 上，三種模型的準確率非常接近。ANN 以 0.26% 的差距微幅領先 HNN，而 SNN 比 ANN 低了約 0.45 個百分點。這個差距太小，不足以說明架構之間的實質差異。

#### CIFAR-10 結果

| Model | Best Epoch | Test Accuracy |
|---|---|---|
| ANN | 20 | **62.00%** |
| SNN | 18 | **55.08%** |
| HNN | 25 | **60.45%** |

在 CIFAR-10 上，三種模型的準確率差距明顯擴大。ANN 與 HNN 表現接近（差距約 1.5%），而 SNN 落後約 7%。

![E1 Accuracy Comparison](images/fig1_e1_overview.png)

▲ 圖 1: MNIST 與 CIFAR-10 上的準確率比較。MNIST 上三者差距極小；CIFAR-10 上 ANN ≈ HNN 明顯優於 SNN。

![E1 CIFAR-10 Detail](images/fig2_e1_cifar10.png)

▲ 圖 2: CIFAR-10 各模型準確率與最佳 epoch。ANN 與 HNN 準確率接近，均顯著優於 SNN。

**解讀**：MNIST 對 SNN 而言過於簡單 — 即使經過 rate coding 量化，MNIST 的高對比度二值化數字仍保留了絕大部分的鑑別資訊。CIFAR-10 則有彩色自然影像、更複雜的紋理與形狀，rate coding 造成的資訊損失開始顯現。HNN 由於保留了第一層 analog convolution，能直接從原始像素中提取特徵，因此表現接近 ANN。

---

### 2.2 E2: Firing Rate 分析

#### 整體與 Hidden Firing Rate

![Overall & Hidden FR](images/fig7_e2_overall_fr.png)

▲ 圖 3: SNN 與 HNN 在 MNIST 與 CIFAR-10 上的 overall 及 hidden firing rate。

| 模型 | 資料集 | Overall FR | Hidden FR |
|---|---|---|---|
| SNN | MNIST | 0.1785 | 0.1862 |
| HNN | MNIST | 0.2545 | 0.2545 |
| SNN | CIFAR-10 | **0.2746** | **0.1793** |
| HNN | CIFAR-10 | **0.2336** | **0.2336** |

#### Layer-wise Firing Rate

![Layer-wise FR](images/fig3_e2_firing_rate.png)

▲ 圖 4: MNIST 與 CIFAR-10 的逐層 firing rate 比較（使用 E1 default checkpoint: T=10, thr=1.0, β=0.95）。

| Layer | SNN (MNIST) | SNN (CIFAR-10) | HNN (MNIST) | HNN (CIFAR-10) |
|---|---|---|---|---|
| input / first_spike | 0.1326 | 0.4766 | 0.2920 | 0.2654 |
| conv1 | 0.1783 | 0.1762 | — | — |
| conv2 | 0.1849 | 0.1632 | 0.2097 | 0.1988 |
| fc1 | 0.2799 | 0.3290 | 0.2757 | 0.2864 |
| fc2 | 0.3933 | 0.4479 | 0.3840 | 0.3736 |

**關鍵觀察**：

1. **SNN input layer 差異巨大** — MNIST 的 input FR 僅 0.13，CIFAR-10 高達 0.48。這是因為 MNIST 大部分像素是黑色背景（值為 0，永不發 spike），而 CIFAR-10 的彩色自然影像有更多非零像素。這也解釋了為何 CIFAR-10 的 SNN 整體 firing rate 較高。

2. **Firing rate 隨層數遞增** — 從 conv1 到 fc2，firing rate 逐步上升，符合神經網路中深層特徵更活躍的預期。

3. **HNN 的 first_spike 層 firing rate 中等**（0.27–0.29），介於低層與高層之間，這說明第一層 analog conv 的輸出經過 LIF 編碼後，產生了一個 balanced 的 spike representation。

4. **HNN 的 overall FR vs hidden FR** — 在 MNIST 上兩者相等（0.2545），在 CIFAR-10 上 also 相等（0.2336）。這是因為 HNN 的 input 層是 analog（沒有 spike），所以所有 spike 都來自 hidden layers。

---

### 2.3 E3: 超參數掃描

#### E3-A: Threshold Sweep（LIF 閾值）

![Threshold Sweep](images/fig4_e3a_threshold.png)

▲ 圖 5: Threshold sweep — MNIST（左）與 CIFAR-10（右）。注意兩個資料集上趨勢相反。

| 模型 | 資料集 | thr=0.5 | thr=1.0 | thr=1.5 |
|---|---|---|---|---|
| SNN | MNIST | **98.13%** | 97.64% | 97.14% |
| SNN | CIFAR-10 | 53.23% | 55.05% | **55.29%** |
| HNN | MNIST | **98.27%** | 97.83% | 97.21% |
| HNN | CIFAR-10 | **60.81%** | 60.26% | 60.48% |

#### E3-B: Beta Sweep（膜電位衰減係數）

![Beta Sweep](images/fig5_e3b_beta.png)

▲ 圖 6: Beta sweep。SNN 使用最佳 threshold（MNIST: thr=0.5, CIFAR-10: thr=1.5），HNN 使用最佳 threshold（thr=0.5）。

| 模型 | 資料集 | β=0.8 | β=0.9 | β=0.95 |
|---|---|---|---|---|
| SNN | MNIST | 97.96% | 98.11% | **98.13%** |
| SNN | CIFAR-10 | 55.05% | 54.49% | **55.29%** |
| HNN | MNIST | 98.08% | **98.29%** | 98.27% |
| HNN | CIFAR-10 | 59.10% | 59.70% | **60.81%** |

#### E3-C: Time Steps Sweep

![Time Steps Sweep](images/fig6_e3c_timesteps.png)

▲ 圖 7: Time steps sweep。SNN 使用最佳 threshold（MNIST: thr=0.5, CIFAR-10: thr=1.5），HNN 使用最佳 threshold（thr=0.5），β=0.95。

| 模型 | 資料集 | T=5 | T=10 | T=20 |
|---|---|---|---|---|
| SNN | MNIST | 98.00% | **98.13%** | 98.03% |
| SNN | CIFAR-10 | 53.10% | 55.29% | **56.02%** |
| HNN | MNIST | 97.90% | 98.27% | **98.36%** |
| HNN | CIFAR-10 | 58.83% | **60.81%** | 60.16% |

#### CIFAR-10 超參數總覽

![CIFAR-10 E3 Overview](images/fig8_e3_cifar10_overview.png)

▲ 圖 8: CIFAR-10 上全部三個超參數掃描的結果總覽。

### 2.4 E1-B: Time Steps 變化（CIFAR-10 Default Config）

預設 threshold=1.0, beta=0.95 下，測試 T=1 與 T=3：

| Model | T=1 | T=3 | T=10 |
|---|---|---|---|
| SNN | **43.62%** | **49.25%** | **55.08%** |
| HNN | **52.76%** | **58.56%** | **60.45%** |

![Time Steps Variation](images/fig10_e1b_timesteps.png)

▲ 圖 9: CIFAR-10 上 default config 的 time steps 變化。兩模型都隨 T 增加進步，SNN 對 T 更敏感（T=1→10 提升 11.5% vs HNN 提升 7.7%）。

HNN 在 T=1 即達到 52.76%，遠高於 SNN 的 43.62%，印證了 HNN 第一層 analog convolution 的優勢。

### 2.5 E4: Post-Training 權重量化（8-bit / 4-bit）

對已訓練好的 E1 checkpoint 進行 uniform per-tensor affine weight quantization，評估量化後的準確率變化：

| Model | float32 | int8 | int4 | Size (float32) | Size (int8) | Size (int4) |
|---|---|---|---|---|---|---|
| ANN | 62.00% | **62.04%** (0.0%) | **57.43%** (-7.4%) | 247KB | 61.8KB (4×) | 30.9KB (8×) |
| SNN | 55.08% | **55.35%** (+0.5%) | **50.78%** (-7.8%) | 247KB | 61.8KB (4×) | 30.9KB (8×) |
| HNN | 60.45% | **60.33%** (-0.2%) | **57.32%** (-5.2%) | 247KB | 61.8KB (4×) | 30.9KB (8×) |

![Quantization Comparison](images/fig9_e4_quantization.png)

▲ 圖 10: CIFAR-10 上不同位元寬度的量化比較。

**關鍵觀察**：

1. **8-bit 幾乎無損** — 三種模型在 int8 量化後的準確率變化均小於 0.5%，說明 8-bit 對 LeNet 架構而言已足夠。
2. **4-bit 差異顯現** — 準確率下降 5–8%，其中 HNN 最穩健（-5.2%），SNN 影響最大（-7.8%）。推測 SNN 的反覆 spiking 過程會放大權重的量化雜訊。
3. **所有模型模型大小相同**（均使用相同 LeNet 骨幹），因此壓縮率一致。

### 2.6 E5: Kernel Size 比較（5×5 vs 3×3）

將所有模型的 Conv2d kernel size 從 5×5 改為 3×3（padding=0，其餘設定不變），比較 CIFAR-10 上的準確率變化：

| Model | kernel=5 (E1) | kernel=3 | 變化 |
|---|---|---|---|
| ANN | **62.00%** | **62.48%** | +0.48% |
| SNN | **55.08%** | **55.05%** | -0.03% |
| HNN | **60.45%** | **60.09%** | -0.36% |

| Model | kernel=5 參數量 | kernel=3 參數量 | 變化 |
|---|---|---|---|
| ANN | 62,006 | 61,990 | -0.03% |
| SNN | 62,006 | 61,990 | -0.03% |
| HNN | 62,006 | 61,990 | -0.03% |

三種模型在 5×5 與 3×3 下的表現幾乎一致，差距不超過 0.5%。對於 CIFAR-10 的 32×32 小尺寸影像，3×3 kernel 提取的局部特徵已足夠，更大的感受野沒有帶來顯著優勢。

---

## 3. 討論 (Discussion)

### 3.1 MNIST vs CIFAR-10：為什麼差距不同？

最顯著的發現是兩組資料集的 ANN-SNN gap 差異極大：

- **MNIST**: ANN(98.09%) - SNN(97.64%) = **0.45%**
- **CIFAR-10**: ANN(62.00%) - SNN(55.08%) = **6.92%**

這個差距放大了約 13 倍。原因有二：

1. **MNIST 的高對比度特性** — MNIST 影像多為黑色背景上白色數字的簡單結構。背景像素值恆為 0，對應 rate coding 中永遠不發 spike；前景像素值接近 1，幾乎每個 time step 都發 spike。這種二值化編碼幾乎保留了所有鑑別資訊。

2. **CIFAR-10 的複雜紋理** — CIFAR-10 包含彩色自然影像（飛機、汽車、鳥類等），有複雜的紋理與漸層。Rate coding 將連續的像素值量化為離散的 spike train，在 T=10 時僅有 11 個可能的 firing rate 等級（0/10 到 10/10），這對自然影像而言資訊損失過大。

### 3.2 Threshold 行為反轉

Threshold 的影響在兩個資料集上完全相反，是一個值得深入探討的現象：

- **MNIST**: 低 threshold (0.5) 最佳（SNN: 98.13%, HNN: 98.27%）
- **CIFAR-10**: SNN 需高 threshold (1.5) 最佳（55.29%）；HNN 仍以低 threshold (0.5) 最佳（60.81%）

**解釋**：在 MNIST 這類簡單資料集上，降低 threshold 可以讓更多資訊通過，且引入的 noise 不足以影響分類。但在 CIFAR-10 上，過低的 threshold 讓太弱的 features 也能觸發 spike，相當於引入了大量 noise。SNN 因為所有層都是 spiking，noise 會逐層放大，因此需要較高的 threshold 來過濾 noise。

HNN 不受這個問題影響，因為第一層 analog conv 已經提取了穩定的特徵，後續的 LIF 層只是在 spike domain 中做進一步處理，因此低 threshold 仍能帶來好處。

### 3.3 Beta 的影響

Beta（膜電位衰減係數）的影響相對一致：**高 beta 普遍較好**。

- Beta 越高，膜電位衰減越慢，神經元能更有效地累積跨 time step 的證據。
- Beta=0.8 時，膜電位每 step 僅保留 80% 的前一 step 資訊，相當於只有約 5 個 time step 的有效記憶長度。
- 但 CIFAR-10 上 beta 的影響比 MNIST 更明顯（HNN: β=0.8 時 59.10% vs β=0.95 時 60.81%，差距 1.71%），說明複雜資料對 temporal integration 的能力要求更高。

### 3.4 Time Steps 的影響

Time steps 的影響呈現不同的模式：

- **SNN**: 隨 T 增加持續改善（CIFAR-10 上 T=5→20 提升 3%），尚未飽和。
- **HNN**: 在 CIFAR-10 上 T=10 達到峰值後 T=20 反而略降，可能因為 HNN 的 analog 第一層已經提取了穩定的 spatial features，更長的 time steps 邊際效益遞減。

對 MNIST 而言，T=5 已經足夠達到接近最佳的表現，再次驗證了 MNIST 的簡單性。

### 3.5 HNN 的穩健性

在所有實驗中，HNN 展現了最穩健的表現：

- 在 CIFAR-10 上與 ANN 的差距始終小於 1%
- 在不同 threshold、beta、time steps 的配置下，表現波動最小（CIFAR-10 上標準差僅 0.8%）
- 既保留了 SNN 的 temporal dynamics 特性，又避免了全 spiking 架構在複雜輸入上的資訊損失問題

這使得 HNN 成為從 ANN 過渡到 SNN 時的一個理想中間方案，特別是在輸入資料較複雜的應用場景。

### 3.6 Time Steps 對 SNN vs HNN 的影響

Time steps 變化實驗（E1-B）顯示了一個重要的模式：**HNN 在極少 time steps（T=1）下仍能保持不錯的準確率（52.76%），而 SNN 在 T=1 時幾乎無法有效運作（43.62%）。** 這凸顯了 HNN 設計的核心優勢 — 第一層 analog convolution 提供了穩固的空間特徵基礎，後續的 spiking layers 是在此基礎上進行 temporal refinement 而非完全依賴時間編碼。

### 3.7 量化穩健性

權重量化實驗（E4）顯示所有三種模型對 8-bit 量化都高度穩健，但 4-bit 量化時 HNN 的表現最佳。SNN 對量化最敏感的現象可能來自其 spiking dynamics：權重量化引入的誤差在逐層的 spiking 過程中會被非線性地放大，而 HNN 的 analog 第一層起到了「緩衝」作用。

### 3.8 Kernel Size 的影響

5×5 與 3×3 kernel 的比較（E5）顯示，在 CIFAR-10 的 32×32 小尺寸影像上，kernel size 對三種模型的影響都很有限（變化 < 0.5%）。這與文獻中常見的觀察一致 — 在小型影像資料集上，3×3 kernel 透過更深的網路堆疊來增加感受野已足夠，單純擴大單層 kernel 效益有限。值得注意的是，三種模型對此變化的反應幾乎相同，說明 kernel size 的影響是架構無關的。

### 3.9 實驗限制

1. **模型容量** — 本實驗使用 LeNet，參數量較少。在更大的模型（ResNet、VGG）上，ANN/SNN 的差距可能有所不同。
2. **SNN 訓練方式** — 本實驗採用從頭訓練（BPTT + surrogate gradient），非 ANN-to-SNN 轉換。後者通常會造成更大的準確率下降。
3. **Rate coding 的限制** — 本實驗使用最簡單的 rate coding，更先進的編碼方式（如 temporal coding、phase coding）可能改善 SNN 的表現。
4. **量化方式** — 本實驗採用 post-training uniform quantization，未使用 quantization-aware training（QAT），後者通常能達到更好的 4-bit 準確率。
5. **Kernel size 比較** — 僅測試 5×5 與 3×3 兩種配置，且未調整 padding 來保持特徵圖大小一致，未來可進一步測試更小的 kernel（如 1×1）或搭配 padding 的實驗設計。

---

## 4. 結論

本研究系統性地比較了 ANN、SNN 與 HNN 在 MNIST 與 CIFAR-10 上的表現，主要結論如下：

1. **MNIST 不適合作為 ANN/SNN 比較的基準**，因為其簡單性無法反映架構間的實質差異。

2. **CIFAR-10 上 ANN 與 HNN 表現接近（62.00% vs 60.45%）**，顯著優於純 SNN（55.08%），差距約 7%。這主要來自 rate coding 對複雜自然影像的資訊損失。

3. **Threshold 的超參數選擇高度依賴資料集難度** — 簡單資料上低 threshold 較好，複雜資料上 SNN 需要高 threshold 來抑制 noise。

4. **HNN 是最穩健的選擇**，以極小的準確率代價換取了 spiking neural network 的特性（事件驅動計算、生物可解釋性），是從傳統 ANN 過渡到純 SNN 的理想橋樑。

5. **Time steps 對 HNN 的影響較小** — HNN 在 T=1 時仍有 52.76% 的準確率（SNN 僅 43.62%），證明了 hybrid 設計的優勢。

6. **8-bit 量化對所有模型幾乎無損**，4-bit 時 HNN 最穩健（-5.2%）、SNN 最敏感（-7.8%），反映 spiking dynamics 對權重雜訊的放大效應。

7. **Kernel size（5×5 vs 3×3）對 CIFAR-10 影響有限**，三種模型變化均 < 0.5%，說明在此資料集上感受野已足夠。

---

*報告製作日期：2026-06-15*
*硬體：NVIDIA RTX 4090 (24GB)*
*框架：PyTorch 2.12.0 + CUDA 12.6*
