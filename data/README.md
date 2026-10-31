# Data documentation

## Source and attribution

Original data: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). Open University Learning Analytics dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5KK69

Dataset description: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

Source page: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

`features_day28.csv` is a transformed subset, not the original dataset. The data authors do not endorse this project. Preserve attribution and identify further modifications when redistributing.

## Preparation recorded in provenance.json

- Retain BBB presentations 2013B, 2013J, and 2014J.
- Link student information and registration records by module, presentation, and learner.
- Retain learners registered by course start and not withdrawn before day 28.
- Exclude Withdrawn records lacking withdrawal dates and inconsistent outcome/withdrawal records.
- Aggregate positive recorded activity on days 0–27 into weekly clicks, active-day count, and recency.
- Keep learners without observed activity, assigning zero clicks, zero active days, and recency 28.
- Set target 1 for later Withdrawn outcomes; set 0 for Fail, Pass, and Distinction.
- Keep identifiers for linkage and split isolation, not as model inputs.

The CSV has 5,062 rows before repeated learners are excluded from later splits. It is not a table of unique people. No raw archive or private teaching records are included.

## Fields

| Field | Meaning |
|---|---|
| code_module | Anonymized module code, BBB |
| code_presentation | Course presentation code |
| id_student | Anonymized learner identifier |
| clicks_w1–clicks_w4 | Click totals on days 0–6, 7–13, 14–20, and 21–27 |
| active_days | Number of days with positive activity |
| days_since_activity | 27 minus last active day; 28 for no activity |
| last_week_change | clicks_w4 minus clicks_w3 |
| total_clicks | Sum of the four weekly totals |
| target | Later recorded withdrawal: 1; other retained final outcomes: 0 |

Current analysis reproduces from this prepared CSV. `00_data_preparation.ipynb` and `prepare_data.py` reproduce the prepared table from the official archive. The reconstruction was executed and matched every supplied feature value and row order.
