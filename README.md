# FINM 36700 Portfolio & Risk Management — Group Assignments

Each assignment lives in its own folder with a self-contained notebook (the submission file) and its own `data/`.

| # | Case | Folder | Status |
|---|---|---|---|
| 1 | [Harvard's Endowment](https://markhendricks.github.io/finm-portfolio/case_studies/Harvard%27s%20Endowment.html) | [`hw1_harvard_endowment/`](hw1_harvard_endowment/) | ✅ done |
| 2 | [ProShares Replication](https://markhendricks.github.io/finm-portfolio/case_studies/ProShares%20Replication.html) | [`hw2_proshares_replication/`](hw2_proshares_replication/) | ✅ done |
| 3 | [Barnstable and Long-Run Risk](https://markhendricks.github.io/finm-portfolio/case_studies/Barnstable%20and%20Long-Run%20Risk.html) | [`hw3_barnstable/`](hw3_barnstable/) | ✅ done |
| 4 | [DFA and Factor Investing](https://markhendricks.github.io/finm-portfolio/case_studies/DFA%20and%20Factor%20Investing.html) | [`hw4_dfa/`](hw4_dfa/) | 🚧 in progress |

## Conventions (all assignments)
- One notebook per assignment, **self-contained** aside from standard packages (don't import helpers from other folders).
- Read data with relative paths (`data/...`) and run the notebook from its own folder.
- Annualize **statistics only** (mean ×12, vol ×√12), never the raw time series.
- Before committing: *Restart & Run All*. Before starting work: `git pull`. Don't edit the same notebook at the same time.

## Setup
```bash
pip install -r requirements.txt
```
