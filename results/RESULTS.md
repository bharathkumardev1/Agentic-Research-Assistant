# Results

## QASPER dev: 150 questions (seed 0)

`research-assistant eval --split dev --sample 150 --cap 5` at commit `daac2a1` on 2026-10-28. Answers and reviews by `gpt-6-luna`, grading by `gpt-6.1-sol` (reasoning effort low), top 5 passages per search, all-MiniLM-L6-v2 embeddings. Recorded spend $0.21 of a $5 cap.

| System | Answer F1 (95% CI) | Evidence F1 (95% CI) | Evidence P / R | Judged correct (95% CI) | $ per question | p50 / p95 seconds | Calls per question |
|---|---|---|---|---|---|---|---|
| single pass | 0.337 (0.284 to 0.392) | 0.447 (0.377 to 0.517) | 0.47 / 0.47 | 0.493 (0.413 to 0.573) | 0.0010 | 1.4 / 2.9 | 1.0 |
| loop | 0.349 (0.296 to 0.405) | 0.455 (0.386 to 0.525) | 0.47 / 0.48 | 0.507 (0.427 to 0.587) | 0.0012 | 1.8 / 10.8 | 2.5 |

Loop minus single pass, paired by question (10,000 bootstrap resamples):

| Metric | Difference | 95% CI | Loop better | Single pass better |
|---|---|---|---|---|
| answer_f1 | +0.012 | -0.011 to +0.037 | 13 | 7 |
| evidence_f1 | +0.009 | -0.016 to +0.036 | 5 | 5 |
| judge_accuracy | +0.013 | -0.020 to +0.047 | 4 | 2 |

Where answers went wrong (first failing stage):

| System | answered an unanswerable question | cited the evidence, answer wrong | correct | declined an answerable question | evidence not retrieved | evidence retrieved, not cited |
|---|---|---|---|---|---|---|
| single pass | 4 | 22 | 74 | 24 | 22 | 4 |
| loop | 3 | 20 | 76 | 22 | 23 | 6 |

## QASPER test: 150 questions (seed 0)

`research-assistant eval --split test --sample 150 --cap 5` at commit `daac2a1` on 2026-10-28. Answers and reviews by `gpt-6-luna`, grading by `gpt-6.1-sol` (reasoning effort low), top 5 passages per search, all-MiniLM-L6-v2 embeddings. Recorded spend $0.39 of a $5 cap.

| System | Answer F1 (95% CI) | Evidence F1 (95% CI) | Evidence P / R | Judged correct (95% CI) | $ per question | p50 / p95 seconds | Calls per question |
|---|---|---|---|---|---|---|---|
| single pass | 0.435 (0.374 to 0.497) | 0.563 (0.491 to 0.634) | 0.58 / 0.60 | 0.560 (0.480 to 0.640) | 0.0009 | 1.4 / 2.2 | 1.0 |
| loop | 0.441 (0.380 to 0.504) | 0.587 (0.517 to 0.656) | 0.59 / 0.63 | 0.587 (0.507 to 0.667) | 0.0011 | 1.9 / 11.0 | 2.5 |

Loop minus single pass, paired by question (10,000 bootstrap resamples):

| Metric | Difference | 95% CI | Loop better | Single pass better |
|---|---|---|---|---|
| answer_f1 | +0.007 | -0.015 to +0.029 | 9 | 9 |
| evidence_f1 | +0.024 | -0.003 to +0.051 | 10 | 2 |
| judge_accuracy | +0.027 | +0.000 to +0.060 | 5 | 1 |

Where answers went wrong (first failing stage):

| System | cited the evidence, answer wrong | correct | declined an answerable question | evidence not retrieved | evidence retrieved, not cited |
|---|---|---|---|---|---|
| single pass | 17 | 84 | 26 | 22 | 1 |
| loop | 16 | 88 | 23 | 20 | 3 |

The judge agreed with the 40 labels in `results/qasper/labels.csv` on 87.5% of answers (Cohen's kappa 0.74); it passed 4 the labels failed and failed 1 the labels passed.

How the labels were made: an AI-assisted draft, not independent human annotation. An independent grade is pending.
