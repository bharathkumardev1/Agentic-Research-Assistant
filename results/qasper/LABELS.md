# The 40 judge labels

Status: graded by me, Bharath Mandha, on 2026-11-02, without AI tools and without seeing the judge's verdict on any of these answers. One grader and one pass, so there is no inter-rater agreement. An earlier AI-assisted draft sits in `labels.csv` and is no longer scored.

## The independent grade

`labels-independent.csv` holds 40 judged test answers, sampled with seed 0 from the test answers that went to the judge (exact matches skip it). I graded each from three things only: the question, the annotators' answers and the answer, using the judge's rubric as written. I had not seen any per-answer judge verdict. I had seen the aggregate judged accuracy in `RESULTS.md`.

The sheet was made after the matched-context run, so the judged pool includes the two matched single-pass systems. The sample is 40 rows over 37 distinct questions: 12 single pass, 10 single pass top 10, 11 single pass top 15 and 7 loop. It shares only 3 rows with the earlier draft's 40, so the two sets are not comparable row by row. Because the same question can appear under several systems with near-identical answers, the 40 labels carry less information than 40 independent questions would.

`research-assistant judge-agreement --results results/qasper-test.json --labels results/qasper/labels-independent.csv` writes agreement and Cohen's kappa into `results/qasper-test.json` and `RESULTS.md`. That replaces the figure computed from the draft.

To regrade, make a fresh sheet and score it:

```
research-assistant label-judge --results results/qasper-test.json --out results/qasper/labels-independent.csv
# fill in your_verdict with correct or incorrect, without looking at the judge's verdicts
research-assistant judge-agreement --results results/qasper-test.json --labels results/qasper/labels-independent.csv
```

## The earlier AI-assisted draft

`labels.csv` holds 40 judged test answers sampled earlier from the single-pass and loop rows only, with the `your_verdict` column filled in from an AI-assisted grading draft. Against the judge it gave 87.5% agreement and a kappa of 0.74, which is agreement between two model-assisted graders and says nothing about people. It covers 36 distinct questions and applies a stricter standard than the judge's rubric in several rows. Ten rows were borderline under the judge's rubric: the five where the draft and the judge disagreed, and five where they agreed but the answer is arguable.

| Question | System | Label | Judge | Why it needs a second look |
|---|---|---|---|---|
| Do they evaluate which compression method yields the most gains? | loop | incorrect | correct | the answer "No" matches one of four annotators |
| How big are improvements on these tasks? | loop | incorrect | correct | one annotator said Unanswerable and the answer says the passages give no numbers |
| Which models do they use for NMT? | loop | incorrect | correct | "attention-based transformer" matches the second annotator |
| What are the baseline models? | loop | correct | incorrect | the answer merges BERT-Cased and BERT-Uncased into BERT |
| What social-network features are used? | single pass | incorrect | correct | the answer holds the listed features plus extra ones |
| How is the remotely sensed data annotated? | loop | incorrect | incorrect | "experts visually inspect it" against "architects based on satellite and Street View images" |
| What were the evaluation metrics? | loop | incorrect | incorrect | "BLEU" plus an added human faithfulness rating |
| What is the performance of the best model? | single pass | incorrect | incorrect | names the model an annotator named, without the number |
| What contributes to improve the accuracy on the legal question answering task? | single pass | incorrect | incorrect | "auxiliary features" against "more features" |
| What contributes to improve the accuracy on the legal question answering task? | loop | incorrect | incorrect | the same answer as above |
