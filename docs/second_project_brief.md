# Project 2 proposal: Science Answer Review

## Purpose

Extend the portfolio from tabular learning analytics to educational NLP. The first project ranks engagement records; this second project would analyze the content of a short student answer.

**Question:** Can a model distinguish correct, partially correct, contradictory, and other short science answers on questions it has not seen during training?

This is a proposal, not an implemented or evaluated project. Keep it in a separate repository when development begins.

## Data candidate and sources

- SemEval-2013 Task 7 Student Response Analysis, specifically SciEntsBank.
- Primary paper: Dzikovska et al. (2013), https://aclanthology.org/S13-2045/.
- Author-hosted distribution: https://github.com/myrosia/semeval-2013-task7.
- The author-hosted repository's LICENSE states CC BY-SA 3.0. Retain its attribution and license with any permitted redistributed data or derivatives. Verify the exact archive and accompanying documentation before fixing label mappings.

The task provides a question, reference answer, and learner answer. Its labels describe the relationship between an answer and the expected content. They do not justify diagnosing a learner's overall understanding or assigning high-stakes grades.

## First version

1. Parse one documented SciEntsBank task variant and retain its original labels.
2. Audit question IDs, duplicates, label imbalance, answer length, and reference overlap.
3. Respect official unseen-answer, unseen-question, and unseen-domain splits where applicable. Do not randomly split near-identical answers to the same question across partitions and call that unseen-question generalization.
4. Compare a majority baseline, TF-IDF plus a linear classifier, sentence-pair embeddings, and one compact transformer after a reproducible baseline exists.
5. Report macro-F1, class-level precision/recall, confusion matrices, and performance across question familiarity.
6. Add an uncertainty-based review option, measuring coverage versus error rather than claiming every answer can be graded automatically.
7. Build a small interface accepting question, reference answer, and student answer. Show the predicted answer category and review flag.

## Planned notebooks

| Notebook | Deliverable |
|---|---|
| 01_corpus_audit.ipynb | Reproducible parsing, label and question audits |
| 02_baselines.ipynb | Baseline comparisons with question-aware evaluation |
| 03_semantic_models.ipynb | Pair representations, tuning, and error analysis |
| 04_review_demo.ipynb | Coverage/error trade-off and inference examples |

## Definition of done

A reproducible parser, explicit split design, comparative results, error cases, and a working local inference path. Model selection and label mapping must be fixed before final testing. A visually elaborate interface is secondary to evaluation quality.

## Portfolio connection

Together, the projects would show tabular ML, NLP, leakage-aware evaluation, uncertainty analysis, and implementation of usable prediction functions. Finish the first project's final evaluation and basic inference interface before expanding the second project's scope.
