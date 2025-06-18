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

### 20250617-223410
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

### 20250618-095124 (original) & 20250618-144632 (cop)
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