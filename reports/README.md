# CS5228 rental prediction report

`final_report_group6.tex` contains the main report, abstract, conclusion and AI
declaration. `final_report_group6.pdf` is its compiled version. `IEEEtran.cls`
comes from the supplied course template; figures live in `figures/`.

From the repository root, regenerate figures with
`python scripts/create_report_figures.py`. Compile with
`tectonic reports/final_report_group6.tex`, or run pdfLaTeX twice from `reports/`.
Upload the whole report directory to Overleaf when compiling there.

The root README documents environment setup, prerequisite preprocessing,
experiment execution and the result files supporting each report section.
Main-text tables are placed at their relevant discussions. References and the
appendix start on separate pages, outside the eight-page main-text limit.

Appendix A records the team's shared discussion/outline work and individual
project and report responsibilities. Appendix B includes the complete CatBoost
parameter grid, MLP seed-stability results and selected-model flat-type errors
from the saved notebook experiment outputs.
