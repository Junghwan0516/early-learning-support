# Analysis design and interpretation boundaries

## The decision

At the start of day 28, rank learners still enrolled for a hypothetical review capacity of 10%. Both the cutoff and capacity are explicit project assumptions. Earlier withdrawals and late registrants are outside the target cohort.

## Stages

1. Rebuild the day-28 table and verify source and feature checksums.
2. Explore only the earliest training cohort.
3. Compare predeclared configurations and representations on a later validation cohort.
4. Freeze the selected configuration, then evaluate the final test once.
5. Build an inference interface after the intended use and performance limitations are documented.

## Why these figures exist

| View | Question | Main limitation |
|---|---|---|
| Outcome distribution | How common is the event? | Accuracy hides minority-event failures |
| Activity quantiles with Wilson intervals | Where is the withdrawal-rate contrast? | Bins hide within-group variation |
| Volume/active-day heatmap | Does consistency add a descriptive signal? | Coarse strata do not fully adjust volume |
| Weekly medians and IQRs | Is change shared across outcomes? | Group medians are not individual trajectories |
| Recency groups | Does long inactivity coincide with withdrawal? | Zero activity is a distinct category |
| Early–late patterns | Is disengagement more than a low total? | Pattern labels depend on thresholds |
| Density and ECDF | How much do outcome distributions overlap? | Descriptive, not classifier performance |
| Recency/volume heatmap | Is recency associated within volume groups? | Sparse cells restrict comparison |
| Early–late scatter | How heterogeneous are learners? | Overplotting and coarse click measurement |
| Feature correlation | Which features share information? | Correlation is not model importance |
| Zero-activity weeks | How common is weekly nonparticipation? | No activity does not imply no learning |
| Individual normalized profiles | What does group averaging conceal? | Normalization discards absolute volume |
| Threshold sensitivity | Are pattern conclusions definition-dependent? | Exploratory, not a threshold search |
| Model recall comparison | Which method captures withdrawals within 10%? | Validation selection can be optimistic |
| PR and review-capacity curves | How does ranking change with capacity? | Curves must not redefine the primary objective afterward |
| Fit-time comparison | Is complexity useful in this environment? | Hardware-specific timing |
| Feature ablation | Do added representations improve the same model? | Alternative sums introduce no new observations |
| Error profiles | How do captured and missed groups differ? | Does not explain individual predictions |
| Paired bootstrap | How variable is the selected-method/rule contrast? | Does not include model-selection uncertainty |

## Statistical language

Wilson intervals are pointwise binomial approximations, not simultaneous bands or individual probability intervals. IQR shading describes dispersion, not uncertainty in a median. Subgroup and bootstrap intervals do not correct for all repeated comparisons. Associations must not be reported as causal intervention effects.

The validation-selected method can be only marginally better than a rule. Negative or uncertain findings should remain visible. Additional charts are useful when they test a question, not when they merely increase figure count.

## Before a deployment-realistic temporal claim

Verify exact course-end dates, result finalization, and what information would be available at each historical prediction time. Presentation order by itself is insufficient. Assess institution shift, accessibility, and the consequences of missed and unnecessary outreach before using predictions with real learners.
