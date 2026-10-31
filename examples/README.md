# Inference example

`learner_features.csv` contains three illustrative feature rows, not observed learners or evaluation data. They demonstrate the input contract for `predict.py`.

Run from the repository root:

```bash
python predict.py --input examples/learner_features.csv --output outputs/ranked_learners.csv
```

The command refits the frozen method on the historical training cohort, ranks the incoming batch, and selects the top `ceil(n * fraction)` records. It does not retrain on incoming records. The feature CSV alone cannot prove that upstream event extraction respected the observation cutoff; that contract remains the caller's responsibility.

The score is a research ranking score, not a calibrated probability. With three example rows and a 10% capacity, rounding selects one row. A batch rank depends on the other records in the batch.
