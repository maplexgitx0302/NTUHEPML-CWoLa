# Training Results

This markdown file records the training result logs. The symbols and abbreviations used are described below:
- `xqyg`: $x$ quark jets and $y$ gluon jets.
- $\eta$: Initial learning rate.
- $N_\text{step}$: Batch size in one step.
- $N_\text{acc}$: Accumulated steps for one update of parameters.
- $ACC$: Accuracy.
- $AUC$: Area under ROC curve.
- $E_\text{early}$: Early stopped epoch.
- `cop`: Center of $p_T$ frame.


### Pure Training with CWoLa Setup
- CWoLa split by jet flavor: `2q0g`, `1q1g`, and `2g0q`.
- No preprocessing and no data augmentation.
- Maiximum 100 epochs, with early stopping and lr scheduler.
- Monitored with `valid_auc` for checkpointing, early stopping, and lr scheduler.
- First training conducted in version 20250616-212748, and 20250617-092802 slightly tuned some of the lr.
- Model: $(\eta, N_\text{step}, N_\text{accum}) \Rightarrow (ACC_\text{test}, AUC_\text{test}, E_\text{early})$
  - CNN_Baseline: (lr1e-05, b64x8) $\Rightarrow$ (0.630, 0.674, 47) # (20250616-212748) lr-baseline
  - CNN_Light: (lr5e-05, b64x8) $\Rightarrow$ (0.615, 0.654, 66) # (20250616-212748) lr-baseline
  - ParT_Baseline: (lr1e-05, b64x8) $\Rightarrow$ (0.685, 0.752, 94) # (20250616-212748) lr-baseline
  - ParT_Baseline: (lr5e-05, b64x8) $\Rightarrow$ (0.696, 0.760, 86) # (20250617-092802) tuned lr
  - ParT_Medium: (lr1e-05, b64x8) $\Rightarrow$ (0.685, 0.748, 100) # (20250616-212748) lr-baseline
  - ParT_Medium: (lr5e-05, b64x8) $\Rightarrow$ (0.683, 0.759, 77) # (20250617-092802) tuned lr
  - ParT_Light: (lr1e-04, b64x8) $\Rightarrow$ (0.686, 0.754, 90) # (20250616-212748) lr-baseline
  - ParT_Light: (lr1e-03, b64x8) $\Rightarrow$ (0.698, 0.768, 55) # (20250617-092802) tuned lr
  - ParT_SuperLight: (lr1e-04, b64x8) $\Rightarrow$ (0.696, 0.767, 83) # (20250616-212748) lr-baseline
  - ParT_SuperLight: (lr1e-03, b64x8) $\Rightarrow$ (0.684, 0.754, 79) # (20250617-092802) tuned lr
  - ParT_ExtremeLight: (lr5e-04, b64x8) $\Rightarrow$ (0.682, 0.746, 100) # (20250616-212748) lr-baseline
  - ParT_ExtremeLight: (lr5e-03, b64x8) $\Rightarrow$ (0.685, 0.748, 100) # (20250617-092802) tuned lr


### Data Preprocessing with `cop`
- Change random seed from `42` to `43`.
- Related versions:
  - 20250618-095124: Tuned some of the learning rates compared to before.
  - 20250627-005917: Add data preprocessing of $\phi$-rotations in the center of $p_T$ frame (`cop`).
- Model: $(\eta, N_\text{step}, N_\text{accum}) \Rightarrow (ACC_\text{test}, AUC_\text{test}, E_\text{early})$
  - CNN_Baseline: (lr1e-05, b64x8) $\Rightarrow$ (0.635, 0.685, 49) # original
  - CNN_Baseline: (lr1e-05, b64x8) $\Rightarrow$ (0.633, 0.686, 49) # cop     
  - CNN_Light: (lr5e-04, b64x8) $\Rightarrow$ (0.618, 0.661, 29) # original
  - CNN_Light: (lr5e-04, b64x8) $\Rightarrow$ (0.618, 0.661, 33) # cop     
  - ParT_Baseline: (lr5e-05, b64x8) $\Rightarrow$ (0.692, 0.755, 84) # original
  - ParT_Baseline: (lr5e-05, b64x8) $\Rightarrow$ (0.703, 0.777, 76) # cop     
  - ParT_Medium: (lr1e-04, b64x8) $\Rightarrow$ (0.682, 0.753, 86) # original
  - ParT_Medium: (lr1e-04, b64x8) $\Rightarrow$ (0.692, 0.771, 88) # cop     
  - ParT_Light: (lr5e-04, b64x8) $\Rightarrow$ (0.689, 0.754, 68) # original
  - ParT_Light: (lr5e-04, b64x8) $\Rightarrow$ (0.677, 0.754, 62) # cop     
  - ParT_SuperLight: (lr1e-03, b64x8) $\Rightarrow$ (0.685, 0.745, 80) # original
  - ParT_SuperLight: (lr1e-03, b64x8) $\Rightarrow$ (0.694, 0.765, 68) # cop     
  - ParT_ExtremeLight: (lr5e-03, b64x8) $\Rightarrow$ (0.674, 0.741, 100) # original
  - ParT_ExtremeLight: (lr5e-03, b64x8) $\Rightarrow$ (0.678, 0.746, 100) # cop


### Data Preprocessing with `cop` and Augmentation with $\phi$-rotations
- Same settings as 20250627-005917, but with augmentation.
- Test **uniform** or **random** augmentation for training data.
