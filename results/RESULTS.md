# Results

## QASPER dev: 150 questions (seed 0)

`research-assistant eval --split dev --cap 5 --matched` at commit `aac699e` on 2026-11-02. Answers and reviews by `gpt-6-luna`, grading by `gpt-6.1-sol` (reasoning effort low), top 5 passages per search, all-MiniLM-L6-v2 embeddings. Recorded spend so far (a running total over every recorded call, not this split alone): $0.65 of a $5 cap.

| System | Answer F1 (95% CI) | Evidence F1 (95% CI) | Evidence P / R | Judged correct (95% CI) | $ per question | p50 / p95 seconds | Calls per question | Passages per question |
|---|---|---|---|---|---|---|---|---|
| single pass | 0.337 (0.284 to 0.392) | 0.447 (0.377 to 0.517) | 0.47 / 0.47 | 0.493 (0.413 to 0.573) | 0.0010 | 1.4 / 2.9 | 1.0 | 5.0 |
| loop | 0.349 (0.296 to 0.405) | 0.455 (0.386 to 0.525) | 0.47 / 0.48 | 0.507 (0.427 to 0.587) | 0.0012 | 1.8 / 10.8 | 2.5 | 5.6 |
| single pass, top 10 | 0.414 (0.359 to 0.471) | 0.518 (0.450 to 0.587) | 0.53 / 0.56 | 0.580 (0.500 to 0.660) | 0.0011 | 1.3 / 2.2 | 1.0 | 10.0 |
| single pass, top 15 | 0.419 (0.367 to 0.474) | 0.559 (0.493 to 0.626) | 0.55 / 0.62 | 0.640 (0.567 to 0.713) | 0.0011 | 1.3 / 2.1 | 1.0 | 14.9 |

Loop minus single pass, paired by question (10,000 bootstrap resamples):

| Metric | Difference | 95% CI | Loop better | Single pass better | Sign-test p |
|---|---|---|---|---|---|
| answer_f1 | +0.012 | -0.011 to +0.037 | 13 | 7 | 0.263 |
| evidence_f1 | +0.009 | -0.016 to +0.036 | 5 | 5 | 1.000 |
| judge_accuracy | +0.013 | -0.020 to +0.047 | 4 | 2 | 0.688 |

Loop minus single pass, top 10 (one call over a wider first retrieval; the loop's own passage count is in the table above), paired by question:

| Metric | Difference | 95% CI | Loop better | Baseline better | Sign-test p |
|---|---|---|---|---|---|
| answer_f1 | -0.065 | -0.106 to -0.025 | 29 | 47 | 0.050 |
| evidence_f1 | -0.063 | -0.119 to -0.006 | 15 | 29 | 0.049 |
| judge_accuracy | -0.073 | -0.133 to -0.013 | 5 | 16 | 0.027 |

Loop minus single pass, top 15 (one call over a wider first retrieval; the loop's own passage count is in the table above), paired by question:

| Metric | Difference | 95% CI | Loop better | Baseline better | Sign-test p |
|---|---|---|---|---|---|
| answer_f1 | -0.070 | -0.109 to -0.033 | 31 | 58 | 0.006 |
| evidence_f1 | -0.104 | -0.163 to -0.046 | 25 | 38 | 0.130 |
| judge_accuracy | -0.133 | -0.200 to -0.073 | 3 | 23 | 0.000 |

Where answers went wrong (first failing stage):

| System | answered an unanswerable question | cited the evidence, answer wrong | correct | declined an answerable question | evidence not retrieved | evidence retrieved, not cited |
|---|---|---|---|---|---|---|
| single pass | 4 | 22 | 74 | 24 | 22 | 4 |
| loop | 3 | 20 | 76 | 22 | 23 | 6 |
| single pass, top 10 | 4 | 26 | 87 | 13 | 13 | 7 |
| single pass, top 15 | 4 | 22 | 96 | 10 | 7 | 11 |

## QASPER test: 150 questions (seed 0)

`research-assistant eval --split test --cap 5 --matched` at commit `aac699e` on 2026-11-02. Answers and reviews by `gpt-6-luna`, grading by `gpt-6.1-sol` (reasoning effort low), top 5 passages per search, all-MiniLM-L6-v2 embeddings. Recorded spend so far (a running total over every recorded call, not this split alone): $0.89 of a $5 cap.

| System | Answer F1 (95% CI) | Evidence F1 (95% CI) | Evidence P / R | Judged correct (95% CI) | $ per question | p50 / p95 seconds | Calls per question | Passages per question |
|---|---|---|---|---|---|---|---|---|
| single pass | 0.435 (0.374 to 0.497) | 0.563 (0.491 to 0.634) | 0.58 / 0.60 | 0.560 (0.480 to 0.640) | 0.0009 | 1.4 / 2.2 | 1.0 | 5.0 |
| loop | 0.441 (0.380 to 0.504) | 0.587 (0.517 to 0.656) | 0.59 / 0.63 | 0.587 (0.507 to 0.667) | 0.0011 | 1.9 / 11.0 | 2.5 | 5.6 |
| single pass, top 10 | 0.473 (0.416 to 0.532) | 0.590 (0.524 to 0.659) | 0.57 / 0.66 | 0.680 (0.600 to 0.753) | 0.0010 | 1.4 / 2.2 | 1.0 | 10.0 |
| single pass, top 15 | 0.475 (0.419 to 0.532) | 0.582 (0.516 to 0.648) | 0.56 / 0.68 | 0.680 (0.607 to 0.753) | 0.0011 | 1.4 / 2.3 | 1.0 | 15.0 |

Loop minus single pass, paired by question (10,000 bootstrap resamples):

| Metric | Difference | 95% CI | Loop better | Single pass better | Sign-test p |
|---|---|---|---|---|---|
| answer_f1 | +0.007 | -0.015 to +0.029 | 9 | 9 | 1.000 |
| evidence_f1 | +0.024 | -0.003 to +0.051 | 10 | 2 | 0.039 |
| judge_accuracy | +0.027 | +0.000 to +0.060 | 5 | 1 | 0.219 |

Loop minus single pass, top 10 (one call over a wider first retrieval; the loop's own passage count is in the table above), paired by question:

| Metric | Difference | 95% CI | Loop better | Baseline better | Sign-test p |
|---|---|---|---|---|---|
| answer_f1 | -0.032 | -0.080 to +0.016 | 28 | 52 | 0.010 |
| evidence_f1 | -0.003 | -0.058 to +0.051 | 27 | 24 | 0.780 |
| judge_accuracy | -0.093 | -0.167 to -0.020 | 9 | 23 | 0.020 |

Loop minus single pass, top 15 (one call over a wider first retrieval; the loop's own passage count is in the table above), paired by question:

| Metric | Difference | 95% CI | Loop better | Baseline better | Sign-test p |
|---|---|---|---|---|---|
| answer_f1 | -0.034 | -0.080 to +0.014 | 28 | 51 | 0.013 |
| evidence_f1 | +0.005 | -0.051 to +0.062 | 27 | 23 | 0.672 |
| judge_accuracy | -0.093 | -0.167 to -0.020 | 10 | 24 | 0.024 |

Where answers went wrong (first failing stage):

| System | answered an unanswerable question | cited the evidence, answer wrong | correct | declined an answerable question | evidence not retrieved | evidence retrieved, not cited |
|---|---|---|---|---|---|---|
| single pass | 0 | 17 | 84 | 26 | 22 | 1 |
| loop | 0 | 16 | 88 | 23 | 20 | 3 |
| single pass, top 10 | 3 | 14 | 102 | 16 | 11 | 4 |
| single pass, top 15 | 2 | 18 | 102 | 12 | 8 | 8 |

The judge agreed with the 40 labels in `results/qasper/labels-independent.csv` on 87.5% of answers (Cohen's kappa 0.75); it passed 2 the labels failed and failed 3 the labels passed.

How the labels were made: graded by me, Bharath Mandha, on 2026-11-02, without AI tools and without seeing the judge's verdict on any of these answers. One grader and one pass, so there is no inter-rater agreement. An earlier AI-assisted draft sits in `labels.csv` and is no longer scored.
