# Training results with bugs

### Wrong implemetation of `cop`
- Related versions:
  - 20250617-223410, 20250618-144632
  - 20250626-124209 (aug_uni_5), 20250623-164121 (aug_uni_10), 20250624-181244 (aug_rand_5), 20250625-144350 (aug_rand_10). Also contain bugs in [issue #1](https://github.com/maplexgitx0302/NTUHEPML-CWoLa/issues/1).
- The calculation of phi in center-of-pt frame has a bug:
  ```python
  # The `cop` in this version is calculated with:
  phi = phi - pt * phi / (np.sum(pt, axis=-1, keepdims=True) + eps)

  # The `cop` should be corrected as:
  phi = phi - np.sum(pt * phi, axis=-1, keepdims=True) / (np.sum(pt, axis=-1, keepdims=True) + eps)
  ```

### Augmentation with respect to wrong dim for sequences
- Related versions: 
  - `original`: 20250620-155542 (aug_uni_5), 20250621-151229 (aug_uni_10), 20250622-142505 (aug_rand_5), 20250623-011501 (aug_rand_10)
  - `cop`: 20250627-005917 (no_aug), 20250626-173605 (aug_uni_5), 20250627-051649 (aug_uni_10), 20250627-112145 (aug_rand_5), 20250627-160540 (aug_rand_10), 20250628-024824 (aug_rand_10)
- See [issue #1](https://github.com/maplexgitx0302/NTUHEPML-CWoLa/issues/1) for further detail.