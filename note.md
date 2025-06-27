### 20250616-212748
- CWoLa split by jet flavor: `2q0g`, `1q1g`, and `2g0q`.
- No preprocessing and no data augmentation.
- Maiximum 100 epochs, with early stopping and lr scheduler.
- Monitored with `valid_auc` for checkpointing, early stopping, and lr scheduler.
- Model: (`lr`, `batch_size_per_step`, `batch_accumulated`) => (`test_accuracy`, `test_auc`, `early_stopped_epochs`)
  - CNN_Baseline: (1e-5, 64, 8) => (0.630, 0.674, 47)  # smooth enough
  - CNN_Light: (5e-5, 64, 8) => (0.615, 0.654, 66)  # more zig-zag than baseline
  - ParT_Baseline: (1e-5, 64, 8) => (0.685, 0.752, 94)  # try slightly higher lr
  - ParT_Medium: (1e-5, 64, 8) => (0.685, 0.748, 100)  # try higher lr
  - ParT_Light:  (1e-4, 64, 8) => (0.686, 0.754, 90)  # try higher lr
  - ParT_SuperLight: (1e-4, 64, 8) => (0.696, 0.767, 83)  # try higher lr
  - ParT_ExtremeLight: (5e-4, 64, 8) => (0.682, 0.746, 100)  # try higher lr

### 20250617-092802
- Following the setup in `20250616-212748`.
- **Only tuned the learning rate**, with 5 or 10 times larger.
- Model: (`lr`, `batch_size_per_step`, `batch_accumulated`) => (`test_accuracy`, `test_auc`, `early_stopped_epochs`)
  - ParT_Baseline: (5e-5, 64, 8) => (0.696, 0.760, 86)  # better performance and faster convergence
  - ParT_Medium: (5e-5, 64, 8) => (0.683, 0.759, 77)  # could try higher lr
  - ParT_Light:  (1e-3, 64, 8) => (0.698, 0.768, 55)  # better performance
  - ParT_SuperLight: (1e-3, 64, 8) => (0.684, 0.754, 79)  # more unstable
  - ParT_ExtremeLight: (5e-3, 64, 8) => (0.685, 0.748, 100)  # performance not improved too much but training more unstable

### ~~20250617-223410~~
- **The calculation of phi in center-of-pt frame has a bug (#1):**
  ```python
  # The `cop` in this version is calculated with:
  phi = phi - pt * phi / (np.sum(pt, axis=-1, keepdims=True) + eps)

  # The `cop` should be corrected as:
  phi = phi - np.sum(pt * phi, axis=-1, keepdims=True) / (np.sum(pt, axis=-1, keepdims=True) + eps)
  ```
- Basic settings are the same as above.
- **Add `center of pt (cop)` preprocessing**.
- Model: (`lr`, `batch_size_per_step`, `batch_accumulated`) => (`test_accuracy`, `test_auc`, `early_stopped_epochs`)
  - CNN_Baseline: (1e-5, 64, 8) => (0.613, 0.656, 51)  # smooth enough
  - CNN_Light: (5e-4, 64, 8) => (0.615, 0.667, 30)  # smoother than before
  - ParT_Baseline: (5e-5, 64, 8) => (0.684, 0.754, 75)  # better convergence
  - ParT_Medium: (1e-4, 64, 8) => (0.684, 0.758, 62)  # better convergence
  - ParT_Light:  (5e-4, 64, 8) => (0.675, 0.766, 53)  # better convergence
  - ParT_SuperLight: (5e-4, 64, 8) => (0.678, 0.746, 100)  # try higher lr
  - ParT_ExtremeLight: (1e-3, 64, 8) => (0.691, 0.756, 100)  # try higher lr

### 20250618-095124 (original) & ~~20250618-144632 (cop)~~
- **The cop has a bug (see #1).**
- **Change random seed from `42` to `43`**.
- Model: (`lr`, `batch_size_per_step * batch_accumulated`) => (`test_accuracy`, `test_auc`, `early_stopped_epochs`)
  - CNN_Baseline: (lr1e-05, b64x8) => (0.635, 0.685, 49) # original
  - CNN_Baseline: (lr1e-05, b64x8) => (0.609, 0.652, 50) # cop
  - CNN_Light: (lr5e-04, b64x8) => (0.618, 0.661, 29) # original
  - CNN_Light: (lr5e-04, b64x8) => (0.605, 0.651, 34) # cop
  - ParT_Baseline: (lr5e-05, b64x8) => (0.692, 0.755, 84) # original
  - ParT_Baseline: (lr5e-05, b64x8) => (0.680, 0.757, 68) # cop
  - ParT_Medium: (lr1e-04, b64x8) => (0.682, 0.753, 86) # original
  - ParT_Medium: (lr1e-04, b64x8) => (0.679, 0.755, 71) # cop
  - ParT_Light: (lr5e-04, b64x8) => (0.689, 0.754, 68) # original
  - ParT_Light: (lr5e-04, b64x8) => (0.694, 0.760, 61) # cop
  - ParT_SuperLight: (lr1e-03, b64x8) => (0.685, 0.745, 80) # original
  - ParT_SuperLight: (lr1e-03, b64x8) => (0.667, 0.737, 93) # cop
  - ParT_ExtremeLight: (lr5e-03, b64x8) => (0.674, 0.741, 100) # original
  - ParT_ExtremeLight: (lr5e-03, b64x8) => (0.675, 0.740, 92) # cop

### Test phi-augmentation
- Same settings as above (20250618-095124), but with augmentation. Related versions:
  - 20250620-155542 (aug_uni_5)
  - 20250621-151229 (aug_uni_10)
  - 20250622-142505 (aug_rand_5)
  - 20250623-011501 (aug_rand_10)
- Test **uniform (uni)** or **random (rand)** augmentation for training data.
- Model: (`lr`, `batch_size_per_step * batch_accumulated`) => (`test_accuracy`, `test_auc`, `early_stopped_epochs`)
  - CNN_Baseline (no_aug     ): (lr1e-05, b64x8) => (0.635, 0.685, 49)
  - CNN_Baseline (aug_uni_5  ): (lr1e-05, b64x8) => (0.635, 0.687, 49)
  - CNN_Baseline (aug_uni_10 ): (lr1e-05, b64x8) => (0.636, 0.686, 49)
  - CNN_Baseline (aug_rand_5 ): (lr1e-05, b64x8) => (0.633, 0.685, 49)
  - CNN_Baseline (aug_rand_10): (lr1e-05, b64x8) => (0.635, 0.686, 49)
  - CNN_Light (no_aug     ): (lr5e-04, b64x8) => (0.618, 0.661, 29)
  - CNN_Light (aug_uni_5  ): (lr5e-04, b64x8) => (0.616, 0.658, 33)
  - CNN_Light (aug_uni_10 ): (lr5e-04, b64x8) => (0.618, 0.660, 33)
  - CNN_Light (aug_rand_5 ): (lr5e-04, b64x8) => (0.619, 0.662, 29)
  - CNN_Light (aug_rand_10): (lr5e-04, b64x8) => (0.619, 0.661, 33)
  - ParT_Baseline (no_aug     ): (lr5e-05, b64x8) => (0.692, 0.755, 84)
  - ParT_Baseline (aug_uni_5  ): (lr5e-05, b64x8) => (0.703, 0.777, 76)
  - ParT_Baseline (aug_uni_10 ): (lr5e-05, b64x8) => (0.703, 0.777, 76)
  - ParT_Baseline (aug_rand_5 ): (lr5e-05, b64x8) => (0.692, 0.755, 84)
  - ParT_Baseline (aug_rand_10): (lr5e-05, b64x8) => (0.703, 0.777, 76)
  - ParT_Medium (no_aug     ): (lr1e-04, b64x8) => (0.682, 0.753, 86)
  - ParT_Medium (aug_uni_5  ): (lr1e-04, b64x8) => (0.692, 0.771, 88)
  - ParT_Medium (aug_uni_10 ): (lr1e-04, b64x8) => (0.692, 0.771, 88)
  - ParT_Medium (aug_rand_5 ): (lr1e-04, b64x8) => (0.682, 0.753, 86)
  - ParT_Medium (aug_rand_10): (lr1e-04, b64x8) => (0.692, 0.771, 88)
  - ParT_Light (no_aug     ): (lr5e-04, b64x8) => (0.689, 0.754, 68)
  - ParT_Light (aug_uni_5  ): (lr5e-04, b64x8) => (0.677, 0.754, 62)
  - ParT_Light (aug_uni_10 ): (lr5e-04, b64x8) => (0.677, 0.754, 62)
  - ParT_Light (aug_rand_5 ): (lr5e-04, b64x8) => (0.689, 0.754, 68)
  - ParT_Light (aug_rand_10): (lr5e-04, b64x8) => (0.677, 0.754, 62)
  - ParT_SuperLight (no_aug     ): (lr1e-03, b64x8) => (0.685, 0.745, 80)
  - ParT_SuperLight (aug_uni_5  ): (lr1e-03, b64x8) => (0.694, 0.765, 68)
  - ParT_SuperLight (aug_uni_10 ): (lr1e-03, b64x8) => (0.694, 0.765, 68)
  - ParT_SuperLight (aug_rand_5 ): (lr1e-03, b64x8) => (0.685, 0.745, 80)
  - ParT_SuperLight (aug_rand_10): (lr1e-03, b64x8) => (0.694, 0.765, 68)
  - ParT_ExtremeLight (no_aug     ): (lr5e-03, b64x8) => (0.674, 0.741, 100)
  - ParT_ExtremeLight (aug_uni_5  ): (lr5e-03, b64x8) => (0.678, 0.746, 100)
  - ParT_ExtremeLight (aug_uni_10 ): (lr5e-03, b64x8) => (0.678, 0.746, 100)
  - ParT_ExtremeLight (aug_rand_5 ): (lr5e-03, b64x8) => (0.674, 0.741, 100)
  - ParT_ExtremeLight (aug_rand_10): (lr5e-03, b64x8) => (0.678, 0.746, 100)

### ~~Test phi-augmentation with cop-preprocessing~~
- **These versions have a common bug (see #1):**
  Related versions:
  - 20250626-124209 (aug_uni_5)
  - 20250623-164121 (aug_uni_10)
  - 20250624-181244 (aug_rand_5)
  - 20250625-144350 (aug_rand_10)
- Same settings as above (20250618-144632), but with augmentation.
- Test **uniform** or **random** augmentation for training data.
- Model: (`lr`, `batch_size_per_step * batch_accumulated`) => (`test_accuracy`, `test_auc`, `early_stopped_epochs`)
  - CNN_Baseline (no_aug     ): (lr1e-05, b64x8) => (0.609, 0.652, 50)
  - CNN_Baseline (aug_uni_5  ): (lr1e-05, b64x8) => (0.608, 0.651, 50)
  - CNN_Baseline (aug_uni_10 ): (lr1e-05, b64x8) => (0.610, 0.652, 50)
  - CNN_Baseline (aug_rand_5 ): (lr1e-05, b64x8) => (0.609, 0.653, 50)
  - CNN_Baseline (aug_rand_10): (lr1e-05, b64x8) => (0.608, 0.651, 50)
  - CNN_Light (no_aug     ): (lr5e-04, b64x8) => (0.605, 0.651, 34)
  - CNN_Light (aug_uni_5  ): (lr5e-04, b64x8) => (0.608, 0.652, 34)
  - CNN_Light (aug_uni_10 ): (lr5e-04, b64x8) => (0.608, 0.652, 34)
  - CNN_Light (aug_rand_5 ): (lr5e-04, b64x8) => (0.609, 0.652, 34)
  - CNN_Light (aug_rand_10): (lr5e-04, b64x8) => (0.608, 0.650, 34)
  - ParT_Baseline (no_aug     ): (lr5e-05, b64x8) => (0.680, 0.757, 68)
  - ParT_Baseline (aug_uni_5  ): (lr5e-05, b64x8) => (0.680, 0.757, 68)
  - ParT_Baseline (aug_uni_10 ): (lr5e-05, b64x8) => (0.680, 0.757, 68)
  - ParT_Baseline (aug_rand_5 ): (lr5e-05, b64x8) => (0.680, 0.757, 68)
  - ParT_Baseline (aug_rand_10): (lr5e-05, b64x8) => (0.680, 0.757, 68)
  - ParT_Medium (no_aug     ): (lr1e-04, b64x8) => (0.679, 0.755, 71)
  - ParT_Medium (aug_uni_5  ): (lr1e-04, b64x8) => (0.679, 0.755, 71)
  - ParT_Medium (aug_uni_10 ): (lr1e-04, b64x8) => (0.679, 0.755, 71)
  - ParT_Medium (aug_rand_5 ): (lr1e-04, b64x8) => (0.679, 0.755, 71)
  - ParT_Medium (aug_rand_10): (lr1e-04, b64x8) => (0.679, 0.755, 71)
  - ParT_Light (no_aug     ): (lr5e-04, b64x8) => (0.694, 0.760, 61)
  - ParT_Light (aug_uni_5  ): (lr5e-04, b64x8) => (0.694, 0.760, 61)
  - ParT_Light (aug_uni_10 ): (lr5e-04, b64x8) => (0.694, 0.760, 61)
  - ParT_Light (aug_rand_5 ): (lr5e-04, b64x8) => (0.694, 0.760, 61)
  - ParT_Light (aug_rand_10): (lr5e-04, b64x8) => (0.694, 0.760, 61)
  - ParT_SuperLight (no_aug     ): (lr1e-03, b64x8) => (0.667, 0.737, 93)
  - ParT_SuperLight (aug_uni_5  ): (lr1e-03, b64x8) => (0.667, 0.737, 93)
  - ParT_SuperLight (aug_uni_10 ): (lr1e-03, b64x8) => (0.667, 0.737, 93)
  - ParT_SuperLight (aug_rand_5 ): (lr1e-03, b64x8) => (0.667, 0.737, 93)
  - ParT_SuperLight (aug_rand_10): (lr1e-03, b64x8) => (0.667, 0.737, 93)
  - ParT_ExtremeLight (no_aug     ): (lr5e-03, b64x8) => (0.675, 0.740, 92)
  - ParT_ExtremeLight (aug_uni_5  ): (lr5e-03, b64x8) => (0.675, 0.740, 92)
  - ParT_ExtremeLight (aug_uni_10 ): (lr5e-03, b64x8) => (0.675, 0.740, 92)
  - ParT_ExtremeLight (aug_rand_5 ): (lr5e-03, b64x8) => (0.675, 0.740, 92)
  - ParT_ExtremeLight (aug_rand_10): (lr5e-03, b64x8) => (0.675, 0.740, 92)

### Test phi-augmentation with cop-preprocessing
- Related versions:
  - 20250627-005917 (no_aug)
  - 20250626-173605 (aug_uni_5)
  - 20250627-051649 (aug_uni_10)
  - 20250627-112145 (aug_rand_5)
  - 20250627-160540 (aug_rand_10)
- Same settings as above (20250618-144632), but with augmentation and preprocessing.
- Test **uniform** or **random** augmentation for training data.
- Model: (`lr`, `batch_size_per_step * batch_accumulated`) => (`test_accuracy`, `test_auc`, `early_stopped_epochs`)
  - CNN_Baseline (no_aug     ): (lr1e-05, b64x8) => (0.633, 0.686, 49)
  - CNN_Baseline (aug_uni_5  ): (lr1e-05, b64x8) => (0.621, 0.665, 46)
  - CNN_Baseline (aug_uni_10 ): (lr1e-05, b64x8) => (0.636, 0.686, 49)
  - CNN_Baseline (aug_rand_5 ): (lr1e-05, b64x8) => (0.636, 0.686, 49)
  - CNN_Baseline (aug_rand_10): (lr1e-05, b64x8) => (0.637, 0.686, 49)
  - CNN_Light (no_aug     ): (lr5e-04, b64x8) => (0.618, 0.661, 33)
  - CNN_Light (aug_uni_5  ): (lr5e-04, b64x8) => (0.622, 0.660, 31)
  - CNN_Light (aug_uni_10 ): (lr5e-04, b64x8) => (0.618, 0.660, 33)
  - CNN_Light (aug_rand_5 ): (lr5e-04, b64x8) => (0.618, 0.659, 33)
  - CNN_Light (aug_rand_10): (lr5e-04, b64x8) => (0.619, 0.660, 33)
  - ParT_Baseline (no_aug     ): (lr5e-05, b64x8) => (0.703, 0.777, 76)
  - ParT_Baseline (aug_uni_5  ): (lr5e-05, b64x8) => (0.692, 0.759, 67)
  - ParT_Baseline (aug_uni_10 ): (lr5e-05, b64x8) => (0.703, 0.777, 76)
  - ParT_Baseline (aug_rand_5 ): (lr5e-05, b64x8) => (0.703, 0.777, 76)
  - ParT_Baseline (aug_rand_10): (lr5e-05, b64x8) => (0.703, 0.777, 76)
  - ParT_Medium (no_aug     ): (lr1e-04, b64x8) => (0.692, 0.771, 88)
  - ParT_Medium (aug_uni_5  ): (lr1e-04, b64x8) => (0.683, 0.759, 77)
  - ParT_Medium (aug_uni_10 ): (lr1e-04, b64x8) => (0.692, 0.771, 88)
  - ParT_Medium (aug_rand_5 ): (lr1e-04, b64x8) => (0.692, 0.771, 88)
  - ParT_Medium (aug_rand_10): (lr1e-04, b64x8) => (0.692, 0.771, 88)
  - ParT_Light (no_aug     ): (lr5e-04, b64x8) => (0.677, 0.754, 62)
  - ParT_Light (aug_uni_5  ): (lr5e-04, b64x8) => (0.688, 0.758, 63)
  - ParT_Light (aug_uni_10 ): (lr5e-04, b64x8) => (0.677, 0.754, 62)
  - ParT_Light (aug_rand_5 ): (lr5e-04, b64x8) => (0.677, 0.754, 62)
  - ParT_Light (aug_rand_10): (lr5e-04, b64x8) => (0.677, 0.754, 62)
  - ParT_SuperLight (no_aug     ): (lr1e-03, b64x8) => (0.694, 0.765, 68)
  - ParT_SuperLight (aug_uni_5  ): (lr1e-03, b64x8) => (0.673, 0.741, 79)
  - ParT_SuperLight (aug_uni_10 ): (lr1e-03, b64x8) => (0.694, 0.765, 68)
  - ParT_SuperLight (aug_rand_5 ): (lr1e-03, b64x8) => (0.694, 0.765, 68)
  - ParT_SuperLight (aug_rand_10): (lr1e-03, b64x8) => (0.694, 0.765, 68)
  - ParT_ExtremeLight (no_aug     ): (lr5e-03, b64x8) => (0.678, 0.746, 100)
  - ParT_ExtremeLight (aug_uni_5  ): (lr5e-03, b64x8) => (0.689, 0.757, 100)
  - ParT_ExtremeLight (aug_uni_10 ): (lr5e-03, b64x8) => (0.678, 0.746, 100)
  - ParT_ExtremeLight (aug_rand_5 ): (lr5e-03, b64x8) => (0.678, 0.746, 100)
  - ParT_ExtremeLight (aug_rand_10): (lr5e-03, b64x8) => (0.678, 0.746, 100)
