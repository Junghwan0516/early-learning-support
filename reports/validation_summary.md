## Validation findings

The validation-selected method is **Logistic regression / 2**, capturing 41 of 233 later withdrawals within a budget of 179 reviewed learners. Recall@10% is **17.6%**, precision@10% **22.9%**, and average precision **0.178**.

The selected method minus the low-activity rule has an observed validation recall difference of 2.1%, with an exploratory paired-bootstrap interval of -0.4% to 4.3%. This interval is conditional on the fitted methods and does not account for configuration selection. It is not a final-test performance claim.

MLP early stopping, finite configuration budgets, and seed dependence limit universal model comparisons. Warnings, all configurations, feature ablations, and execution versions are retained in `reports/`.
