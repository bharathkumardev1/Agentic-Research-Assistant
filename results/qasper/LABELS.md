# The 40 judge labels

Status: an AI-assisted draft, not independent human annotation. An independent grade is pending.

`labels.csv` holds 40 judged test answers, sampled with seed 0 from the 232 test answers that went to the judge (exact matches skip it). The `your_verdict` column was filled in from an AI-assisted grading draft, so agreement with the judge here is agreement between two model-assisted graders. It does not show that the judge matches people.

Three things limit what the 40 labels can show:

- Four questions appear twice, once per system, with near-identical answers, so the sample is 36 distinct questions.
- Several labels apply a stricter standard than the judge's rubric, which accepts an answer that matches any one annotator whatever its extra detail.
- Ten rows are borderline under the judge's rubric and need a second look: the five where the judge and the label disagree, and five where they agree but the answer is arguable. They are in the table below.

To replace this with an independent grade, make a blank sheet of the same 40 answers without overwriting `labels.csv`, grade it blind with the judge's rubric next to you, and score it:

```
research-assistant label-judge --results results/qasper-test.json --out results/qasper/labels-independent.csv
# fill in your_verdict with correct or incorrect, without looking at the judge's verdicts
research-assistant judge-agreement --results results/qasper-test.json --labels results/qasper/labels-independent.csv
```

`judge-agreement` writes its result into `results/qasper-test.json` and regenerates `RESULTS.md`, so commit that as a separate change, and update the status line above to say who graded.

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
