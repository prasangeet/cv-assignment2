# CV Assignment 2 — Cost-Efficient Salient Object Segmentation

A lightweight binary salient-object segmentation system implemented in PyTorch.

## Pipeline

- **Model:** SwiftMask-inspired nested residual U-block encoder-decoder with depthwise-separable convolutions.
- **Training data:** DUTS-TR only.
- **Validation:** deterministic 90/10 split of DUTS-TR.
- **Final test:** DUTS-TE, held out from training and model selection.
- **Training loss:** 0.5 BCE-with-logits + 0.5 soft Dice.
- **Optimizer:** AdamW with cosine learning-rate annealing.
- **Default model input:** 128 × 128. Evaluation scores predictions against the original 256 × 256 masks.

## Dataset layout

Place the extracted dataset under `data/`:

```text
data/
├── DUTS-TR/
│   ├── DUTS-TR-Image/
│   └── DUTS-TR-Mask/
└── DUTS-TE/
    ├── DUTS-TE-Image/
    └── DUTS-TE-Mask/
```

Each image is paired with a PNG mask having the same filename stem.

## Environment

The repository uses `uv`. Install/sync dependencies:

```bash
uv sync
```

## Training

Run from the repository root:

```bash
uv run python -m src.train \
  --data data \
  --epochs 40 \
  --batch-size 16 \
  --image-size 128 \
  --output-dir experiments/swiftmask_v2
```

A quick smoke test can use a subset of DUTS-TR:

```bash
uv run python -m src.train --limit 128 --epochs 1 --batch-size 8 --output-dir experiments/smoke_test
```

Training writes the configuration, history, loss/IoU plots, and best/latest checkpoints into the selected experiment directory. Existing experiments are not overwritten unless `--overwrite` is supplied.

## Final evaluation

Evaluate only after training:

```bash
uv run python -m src.evaluate \
  --data data \
  --split DUTS-TE \
  --checkpoint experiments/swiftmask_v2/model_best.pt
```

Evaluation writes `results.json` and `segmentation_examples.png` alongside the checkpoint by default. The evaluator reports foreground IoU, background IoU, dataset-level mIoU, per-image Boundary F1, model inference time, and checkpoint size.

## Notes

- Do not use DUTS-TE for training, validation, threshold selection, or checkpoint selection.
- Compare models using identical DUTS-TE images, preprocessing, threshold, and metric implementations.
- The 128 × 128 training resolution is the reference-inspired efficiency setting. Use `--image-size 256` for a separate higher-resolution experiment; do not compare the two without reporting the resolution.
