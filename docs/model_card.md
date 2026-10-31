# Model and intended-use note

Selected method: Logistic regression / 2. Input: eight day-28 engagement features. Training cohort: BBB 2013B. Selection cohort: BBB 2013J with repeated learners excluded. Frozen settings: reports/frozen_selection.json.

Validation Recall@10%: 0.176; precision: 0.229; average precision: 0.178. These are development estimates. The selected method captures 41 withdrawals in 179 reviewed records, while 192 of 233 withdrawals are missed and 138 reviewed records do not withdraw. The capacity constraint and limited signal matter to practical usefulness.

The exploratory difference from the low-activity rule is uncertain and its bootstrap interval crosses zero. No final-test or prospective institutional performance has been established. The score must not be treated as a calibrated probability, causal explanation, disciplinary assessment, or automatic decision about a learner.

The batch CLI validates feature completeness and aggregate consistency. Upstream observation timing cannot be verified from aggregate features alone. It refits the frozen method on training data, not on submitted records. A deployed interface, drift monitoring, and intervention evaluation are future work.

Tested environment: {"python": "3.12.14", "numpy": "2.3.5", "pandas": "2.2.3", "matplotlib": "3.10.8", "scikit-learn": "1.8.0"}.
