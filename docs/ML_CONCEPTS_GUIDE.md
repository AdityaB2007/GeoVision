# GeoVision Deep Learning & Computer Vision Curriculum

This guide walks through every machine learning and deep learning concept powering **GeoVision**, linking theory directly to the exact modules and design choices in this repository.

---

## Module 1: Image Tensors, Normalization & Data Leakage
**Code:** [`src/data/preprocessing.py`](../src/data/preprocessing.py), [`src/data/dataset.py`](../src/data/dataset.py)

### What a Satellite Patch Looks Like to PyTorch
An `EuroSAT_RGB` image is a `64 × 64` grid of `3` color channels (**Red = B04, Green = B03, Blue = B02**) stored as 8-bit integers in `[0, 255]`. `torchvision.transforms.ToTensor()` converts a `PIL.Image` into a `float32` tensor shaped `(C, H, W) = (3, 64, 64)` scaled to `[0.0, 1.0]`.

### Why Z-Score Standardization Matters
Before feeding tensors into convolutional layers, we standardize each channel $c \in \{R, G, B\}$:

$$x'_c = \frac{x_c - \mu_c}{\sigma_c}$$

- **Optimization conditioning:** Centering inputs near zero mean and unit variance prevents large positive inputs from saturating activations or creating elongated loss-surface valleys that slow down gradient descent.
- **Preventing Data Leakage (Scratch CNN):** When training `GeoVisionCNN` from scratch, `compute_channel_stats()` computes $(\mu, \sigma)$ strictly on `splits["train"]` — never on `val` or `test`. Computing dataset statistics over all 27,000 images leaks test-set brightness distribution into training.
- **Transfer Learning Alignment:** When fine-tuning `ResNet-18` (pre-trained on ImageNet), we intentionally normalize with `IMAGENET_MEAN` and `IMAGENET_STD` so EuroSAT pixel intensities align with the exact scale the pre-trained first-layer edge filters expect.

### Spatial Leakage in Geospatial Splits
Standard random splits can place two neighboring `640 m × 640 m` Sentinel-2 patches from the same farm valley into `train` and `test`. GeoVision therefore supports both `official` (`eurosat-*.txt`) and `spatial` (`eurosat-spatial-*.txt`) TorchGeo splits, plus `assert_disjoint_splits()` to verify zero file overlap.

---

## Module 2: Remote Sensing Data Augmentation ($D_4$ Symmetry)
**Code:** [`RandomRotate90` & `build_transforms`](../src/data/preprocessing.py)

In ground-level datasets (e.g., ImageNet), flipping a dog upside down creates an unrealistic scene. Satellite imagery, however, is captured at **nadir** (straight down): a forest or orchard rotated by $90^\circ$, $180^\circ$, $270^\circ$, or mirrored horizontally/vertically is still 100% physically valid.

This forms the **Dihedral Group $D_4$** (8 exact label-preserving symmetries):
- `RandomHorizontalFlip(p=0.5)`
- `RandomVerticalFlip(p=0.5)`
- `RandomRotate90()` (exact integer multiples of $90^\circ$ — no interpolation blur or black border triangles)
- Subtle `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02)` simulating atmospheric haze and seasonal sun angles.

Crucially, `is_train=False` disables all random augmentations during validation, testing, and FastAPI inference so predictions are deterministic.

---

## Module 3: Why Build a Classical Baseline First?
**Code:** [`src/models/baseline.py`](../src/models/baseline.py)

Before investing in deep neural networks, `BaselineClassifier` extracts **122 handcrafted features** per image (`extract_handcrafted_features()`):
1. **48 features:** Per-channel RGB 16-bin normalized histograms (global color palette)
2. **48 features:** Per-channel HSV 16-bin normalized histograms (separates Hue/Saturation from brightness)
3. **24 features:** 2×2 spatial quadrant RGB means and standard deviations (coarse layout)
4. **2 features:** Mean and standard deviation of luminance gradient magnitude (roughness / urban edges)

Fed into `StandardScaler -> LogisticRegression`, this baseline establishes how far global color statistics go. For instance, `SeaLake` (uniform dark blue) and `Forest` (uniform dark green) are fairly easy via color histograms, whereas `Residential` vs `Industrial` vs `Highway` depend heavily on fine **spatial geometry** that global histograms discard!

---

## Module 4: Convolutional Neural Networks from First Principles
**Code:** [`GeoVisionCNN`](../src/models/custom_cnn.py)

Why do CNNs outperform handcrafted histograms?
1. **Local Receptive Fields & Weight Sharing:** A `3 × 3` convolutional kernel slides across every spatial location, detecting edges, crop-row periodicity, road stripes, or building corners regardless of where they appear in the patch (**translation equivariance**).
2. **Hierarchical Spatial Compression (`64 -> 32 -> 16 -> 8 -> 4`):**
   - **Block 1 (`3 -> 32 ch`, `64 -> 32`):** Primitive color contrasts & edges.
   - **Block 2 (`32 -> 64 ch`, `32 -> 16`):** Textures, canopy grain, road edges.
   - **Block 3 (`64 -> 128 ch`, `16 -> 8`):** Mid-level structures (rooftops, highway interchanges, orchard rows).
   - **Block 4 (`128 -> 256 ch`, `8 -> 4`):** Scene-level semantic composition across the `640 m × 640 m` patch.
3. **BatchNorm2d (`bias=False` before BN):** Normalizes layer activations across each mini-batch, stabilizing deeper gradient flow. We omit convolutional bias before BatchNorm because BatchNorm's learnable shift parameter $\beta$ absorbs any constant offset.
4. **Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`):** Instead of flattening `(256, 4, 4)` into a 4,096-element vector (which would create over 500,000 fragile dense parameters), GAP averages each `4 × 4` feature map into one number (`256` features total). That slashes overfitting and lets the CNN process arbitrary input resolutions.

---

## Module 5: Transfer Learning & Two-Stage Fine-Tuning
**Code:** [`GeoVisionTransferNet`](../src/models/transfer.py)

Even though ImageNet contains everyday photos (dogs, cars, buildings) rather than satellites, early and mid-level CNN layers learn **universal visual primitives** (Gabor edges, corners, textures, color blobs).

`GeoVisionTransferNet` (supporting `resnet18`, `resnet50`, `efficientnet_b0`) trains in two deliberate stages:
1. **Stage 1 — Linear Probing (`freeze_backbone()`):** All pre-trained convolutional weights have `requires_grad = False`. Only the newly initialized classification head trains. Why? If we updated the backbone immediately, large noisy gradients from the random head would corrupt delicate pre-trained ImageNet filters.
2. **Stage 2 — Selective Fine-Tuning (`unfreeze_top_layers()`):** We unfreeze ResNet's top semantic block (`layer4`) along with the head, using **differential learning rates** (`backbone_lr = 1e-4`, `head_lr = 5e-4`): early layers keep general edge filters intact while top layers specialize for overhead Sentinel-2 land cover.

---

## Module 6: Training Dynamics & Optimization
**Code:** [`src/training/train.py`](../src/training/train.py)

- **Cross-Entropy Loss with Label Smoothing (`0.05`):** Softens hard one-hot targets (`1.0` -> `0.955`, `0.0` -> `0.005`), discouraging infinitely confident logits, reducing overfitting, and improving probability calibration.
- **AdamW Optimizer:** Unlike classic Adam (which entangles L2 penalties with adaptive momentum estimates), AdamW decouples weight decay so regularization applies uniformly across weights.
- **Cosine Annealing LR Schedule:** Smoothly decays the learning rate along a cosine curve toward zero so weights settle into wide, flat loss minima.
- **Gradient Clipping (`max_norm=1.0`):** Bounds the global L2 gradient norm per step to prevent sudden gradient spikes.

---

## Module 7: Quantitative Evaluation, Calibration & Error Analysis
**Code:** [`src/evaluation/evaluate.py`](../src/evaluation/evaluate.py)

Accuracy alone hides failure modes. GeoVision computes:
1. **Top-1 & Top-3 Accuracy:** Checks both exact top prediction and whether the true class appears in the top 3.
2. **Macro vs Weighted F1:** EuroSAT has mild class imbalance (`Pasture` = 2,000 patches vs `Forest`/`SeaLake` = 3,000). **Macro-F1** treats all 10 classes equally so errors on smaller classes aren't masked.
3. **Confusion Matrix (Counts & Row-%):** Reveals physically understandable confusions (e.g., `AnnualCrop` vs `PermanentCrop`, `HerbaceousVegetation` vs `Pasture`, `Highway` vs `River`).
4. **One-vs-Rest Macro ROC-AUC:** Threshold-independent ranking quality across all 10 classes.
5. **Expected Calibration Error (ECE) & Reliability Diagram:** Bins predictions by confidence (0–10%, 10–20%, ..., 90–100%) and checks whether predicted softmax probabilities match true empirical accuracy.
6. **Top-15 Confident Misclassifications:** Saves exact filenames where the model was confidently wrong — ideal for inspecting noisy or borderline patches (e.g. a highway passing right beside a river).

---

## Module 8: Production ML Serving with FastAPI
**Code:** [`backend/app.py`](../backend/app.py), [`backend/inference.py`](../backend/inference.py), [`backend/schemas.py`](../backend/schemas.py)

1. **Lifespan Model Loading:** Loads the `.pt` checkpoint **once** at server startup (`@asynccontextmanager lifespan`), avoiding multi-second disk reads per HTTP request.
2. **Zero Train/Serve Skew:** Checkpoints store `input_size`, `mean`, `std`, and `class_names`. `GeoVisionInferenceEngine` reconstructs the exact deterministic transform automatically.
3. **`torch.inference_mode()`:** Disables autograd graph tracking and version counters during serving for maximum throughput and minimal memory footprint.
4. **Input Guards:** Validates empty uploads (`400`), corrupt/non-image payloads (`400`), and >10 MB files (`413`) before running model inference.

---

## Module 9: Visual Explainability with Grad-CAM
**Code:** [`src/explainability/gradcam.py`](../src/explainability/gradcam.py), [`src/explainability/visualization.py`](../src/explainability/visualization.py), [`src/explainability/generate_gallery.py`](../src/explainability/generate_gallery.py)

High accuracy alone does not prove a satellite classifier learned physically meaningful cues — a model could cheat by memorizing border artifacts or background haze. **Grad-CAM (Gradient-weighted Class Activation Mapping)** opens the CNN "black box" for any target class $c$:

1. **Channel Importance Weights ($\alpha_k^c$):** Take the pre-softmax logit $y^c$ and backpropagate gradients into the final spatial convolutional feature maps $A^k \in \mathbb{R}^{H' \times W'}$ (`net.layer4[-1]` at $4 \times 4$ for ResNet-18, or `features[-2]` at $8 \times 8$ for `GeoVisionCNN`), then global-average-pool across spatial positions:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A^k_{i,j}}$$
2. **Positive Evidence Filter ($\text{ReLU}$):** Multiply each feature map by its importance weight and discard negative contributions:
   $$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$
   Why `ReLU`? Negative sums correspond to features supporting *other* land-cover classes. Keeping only positive activations highlights pixels that actively increase class $c$'s score.
3. **Why the Final Convolutional Layer?** Early conv layers have crisp resolution but only detect generic edges; global pooling destroys all spatial layout. The final conv block preserves coarse spatial coordinates while encoding high-level concepts (highway strips, rooftops, river curves).
4. **Counterfactual Grad-CAM:** By passing `target_class="Highway"` (even if the top prediction is `"River"`), you can ask: *"Which pixels made the model think Highway?"*
5. **10-Class Gallery ([`gradcam_gallery.png`](../outputs/explainability/gradcam_gallery.png)):** Generated across one held-out test patch per class (`9/10` predicted correctly in [`gradcam_samples.json`](../outputs/explainability/gradcam_samples.json), while the lone low-confidence `23.9%` miss on `PermanentCrop_100.jpg` demonstrates how Grad-CAM flags ambiguous linear patterns!).

---

## Observed GeoVision Empirical Benchmark Results

All three models were evaluated on the exact same held-out **5,400-image** TorchGeo official test split (`eurosat-test.txt`):

| Model | Train Samples | Input Size | Top-1 Accuracy | Top-3 Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC (OvR) | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_logreg`** (122 features) | 16,200 | 64×64 | 85.00% | 97.70% | 84.35% | 84.82% | 0.9831 | **0.0262** |
| **`custom_cnn`** (`GeoVisionCNN`) | 4,000 | 64×64 | 87.43% | 98.78% | 86.95% | 87.24% | 0.9900 | 0.0641 |
| **`transfer_resnet18`** (`ResNet-18`, 2-stage) | 2,500 | 128×128 | **93.28%** | **99.46%** | **93.13%** | **93.28%** | **0.9966** | 0.0506 |

### Key Takeaways from the Real Numbers
1. **Spatial Convolutions Beat Color Histograms (Even With 4× Less Data):** On `Highway`, the color/edge baseline achieved only `63.19%` F1 (frequently confusing `River` ↔ `Highway`: 65 + 57 errors) because linear water channels and asphalt corridors look similar in global histograms. `GeoVisionCNN` jumped to `75.11%` F1 (+11.9 pts) and `ResNet-18` reached `88.59%` F1 (+25.4 pts over baseline) despite training on only 2,500 images vs 16,200.
2. **Pre-Trained Visual Hierarchies Dominate:** Stage 1 linear probing (`5,130` trainable head params) reached `79.76%` val accuracy in 2 epochs; unfreezing `layer4` in Stage 2 (`8.4M` trainable params) propelled val accuracy to `93.37%` and full 5,400-image test accuracy to **`93.28%`** (`99.46%` Top-3!).
3. **Top Semantic Confusion Pairs (`transfer_resnet18` on 5,400 test patches):**
   - `River` → `Highway` (59 errors): narrow linear rivers/canals bordered by embankments resemble road geometry at 10 m/px.
   - `PermanentCrop` → `HerbaceousVegetation` (30 errors) & `AnnualCrop` → `PermanentCrop` (20 errors): orchard/vineyard textures overlap with seasonal fields and natural meadows in RGB (where Sentinel-2 Red-Edge/NIR bands B05–B08 help further!).
4. **Accuracy vs. Calibration:** Notice `baseline_logreg` achieved the lowest ECE (`0.0262`) even with lower accuracy, whereas `ResNet-18` achieved `0.0506` ECE with `99.36%` empirical accuracy in its `[0.90, 1.00]` confidence bin (`3,592 / 5,400` test patches).
