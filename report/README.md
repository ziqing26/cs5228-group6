# CS5228 English LaTeX Report Structure

Open `report_structure.tex` as the main document. `IEEEtran.cls` is copied
unchanged from the course template supplied by the team.

This is an editable writing structure, not a completed final report.
The source, visible prompts, comments, headings and table labels are English.
No final model or final score has been assumed.

## Main structure

1. Introduction
2. Data and Problem Formulation
3. Exploratory Data Analysis and Preprocessing
4. Data Mining Methods and Model Development
   - Reference Predictor and Ridge Baseline
   - Candidate Models
   - Completed Model Improvements
   - Final Model Specification
5. Experimental Setup
6. Final Results and Discussion
   - Overall Model Comparison
   - Performance across Time Periods
   - Ablation Studies
   - Error Analysis and Model Interpretation
   - Final Test Predictions and Submission
   - Limitations of the Final Findings
7. Conclusion
8. References
9. Appendix: Team Workload Breakdown and Supplementary Details

There is no future-outlook section. The model-development prompts are for
describing experiments completed by final submission, not planned work.
Delete any optional subsection or table row whose experiment was not completed.
The limitations subsection assesses the scope of the completed evidence.

## Editing

- Replace each blue `Writing prompt` with scientific prose.
- Fill each `TBD` using the frozen final experiment results.
- Replace figure boxes with exported plots using `\includegraphics`.
- Replace bibliography placeholders with verified references and cite them.
- Confirm author names, IDs, team name and actual workload.
- All principal results and the final model definition belong in the main text.
- Do not mix progress-slide scores, collapsed-target scores and final scores.
- `\outlinefalse` hides drafting prompts, but does not replace their missing prose
  or remove other placeholders. Use it only after completing the content.

## Page budget

| Component | Approximate pages |
|---|---:|
| Title and abstract | 0.4 |
| Introduction | 0.6 |
| Data | 0.4 |
| EDA and preprocessing | 1.4 |
| Methods and completed model improvements | 1.5 |
| Experimental setup | 1.0 |
| Final results and discussion | 2.0 |
| Conclusion | 0.3 |
| **Target main-text total** | **7.6** |

The eight-page limit includes title, abstract, tables, plots and figures.
References and the appendix begin after explicit page breaks. The rendered
outline length is not the eventual completed-paper length. Verify the final
main-text page count after replacing the prompts and inserting real figures.
Preserve the course template's fonts, margins and column layout.

## Compilation

Upload both files to Overleaf, set `report_structure.tex` as the main file and
use pdfLaTeX. Alternatively, with a local TeX installation:

```bash
pdflatex report_structure.tex
pdflatex report_structure.tex
```

Two passes resolve table, figure and section cross-references. No external
figure files are required to compile the initial skeleton.
