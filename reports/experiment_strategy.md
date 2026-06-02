# Adaptive Experiment Strategy

The goal is to beat the observed public rank 1 score of `0.703047`, while still
protecting against private leaderboard overfitting.

## Operating Rules

- Run local validation before every Zindi submission.
- Submit only when a run has a defensible change and a plausible validation gain.
- Keep one config change small enough to explain.
- Use `reports/experiment_log.csv` after every run.
- Save leaderboard screenshots after every scored submission.

## Adaptive Loop

1. Start with `configs/baseline.yaml`.
2. Run `train-generate` in Colab.
3. Append the local score and public score to `reports/experiment_log.csv`.
4. Generate the next candidate:

```bash
python -m health_qa.cli suggest-next \
  --base-config configs/baseline.yaml \
  --history reports/experiment_log.csv \
  --output configs/next.yaml
```

5. Review the suggested change before running.

## Manual Overrides

Use judgment when the adaptive suggestion is too conservative. Priority
directions:

- Better model family if baseline is far below `0.70`.
- Language-aware prompting if weak languages lag.
- Beam and length tuning if answers are fluent but lexically mismatched.
- More epochs or LoRA capacity if validation loss underfits.
- Lower learning rate if validation score is unstable.

## Submission Budget

The challenge allows 5 submissions/day and 50 total. Do not use all available
daily submissions unless validation strongly supports it.
