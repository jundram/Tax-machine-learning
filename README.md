# Tax Machine Learning — Peer-Group Anomaly Review

Unsupervised detection of unusual individual tax returns, explained with SHAP
and served through a local review dashboard.

Every return is first assigned to an income-composition **peer group** (K-Means
on income shares and scale), then scored by that group's own **Isolation
Forest**. A rental investor is therefore judged against rental investors and a
wage earner against wage earners, rather than against the whole population.
Local **SHAP** values explain each score, and a small rule-based research agent
links the strongest drivers to official ATO guidance and reviewer questions.

The fitted artifacts in `models/` were trained on the **ATO 2022–23 individual
sample file** (a 2 % sample of individual returns). The data file itself is
confidential and is not, and must never be, committed to this repository.

> **Interpretive rule.** An anomaly flag is a statistical signal that a return
> is unusual relative to its peer group. It is not evidence of non-compliance or
> fraud, and this prototype must not be used to make decisions about real
> taxpayers without human review.

## What's in the repository

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: enter or load a record, see the verdict, score gauge, peer percentile, SHAP drivers, ATO guidance and a review assistant |
| `anomaly_service.py` | Loads the fitted pipeline and scores one record: peer group → anomaly score → threshold → SHAP drivers and protective factors |
| `record_builder.py` | Turns the dashboard's plain-language fields (salary, work-related expenses, rental income and deductions, …) into the itemised ATO record the model expects, deriving total income, total deductions and taxable income |
| `research_agent.py` | Deterministic retrieval over `ato_knowledge_base.json`; writes the review narrative and answers questions without calling an external model |
| `taxpayer_framework.py` | The full research pipeline (segmentation, model selection, contamination sweep, supervised comparison, SHAP) that produced the artifacts |
| `tools/export_segment_profiles.py` | Summarises the row-level results into aggregate per-segment statistics (`segment_profiles.json`) and population-typical itemisation proportions (`input_splits.json`) for the dashboard |
| `models/` | Fitted scalers, K-Means, per-segment Isolation Forests, `model_metadata.json`, `segment_profiles.json`, `input_splits.json` |
| `figures/`, `tables/`, `results/summary_public.json` | Aggregate outputs of the training run (see *Data confidentiality*) |
| `test_*.py` | Unit tests |

## Run the dashboard

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the address Streamlit prints (usually <http://localhost:8501>). Load one
of the example scenarios in the sidebar — typical wage earner, inflated
work-related claims, negatively geared investor, sole trader, retiree — and
select **Analyse taxpayer**.

Run the tests with:

```bash
python -m unittest -v test_anomaly_service.py test_record_builder.py test_research_agent.py
```

## Retrain the models

Place the ATO sample file at `../data/2023_sample_file_SA4.csv` (one directory
above this repository), then:

```bash
python taxpayer_framework.py                 # ~15 min; writes models/, figures/, tables/, results/
python tools/export_segment_profiles.py      # aggregate profiles, input splits and the public summary
```

If the data file is absent, the framework generates a schema-matched synthetic
dataset so the pipeline still runs end to end; `model_metadata.json` records
which source was used.

## Data confidentiality

The ATO sample file is never committed: `.gitignore` excludes `data/` and every
`*.csv` outside `tables/`. The training run also produces outputs that quote
individual records, and these are excluded as well:

| Excluded | Why |
|---|---|
| `results/taxpayer_scores.csv` | one row per taxpayer record |
| `results/summary.json`, `results/run_log.txt` | quote example records (`Ind`, amounts) |
| `tables/table_top15_review_list.csv`, `tables/table_local_explanations.csv` | per-record listings |
| `figures/fig10*_waterfall*.png` | local SHAP plots titled with a record id |

What is committed contains only fitted parameters and aggregates: the model
artifacts, `models/segment_profiles.json` (counts, medians and score quantiles
per peer group), `models/input_splits.json` (population-level proportions),
`results/summary_public.json` (the run summary with the per-record
`local_cases` key removed) and the aggregate tables and figures. After any new
training run, re-run `tools/export_segment_profiles.py` and check `git status`
before committing.

## Method summary

| Stage | Method |
|---|---|
| 1 · Segmentation | K-Means on seven income-composition shares plus log gross activity and log super balance; *k* chosen by silhouette subject to a minimum segment size |
| 2 · Anomaly detection | One Isolation Forest per segment (200 trees) on 35 log-amount and behavioural-ratio features; 2 % review budget per segment |
| 3 · Supervised comparison | Random Forest and Histogram Gradient Boosting trained on documented injected perturbations, used only to benchmark the unsupervised design |
| 4 · Explainability | `shap.TreeExplainer` on each segment's forest for local drivers; surrogate model for segment membership |

Developed as a Master of Data Science capstone project. The scikit-learn
version is pinned in `requirements.txt` because the pickled artifacts are
tied to it.
