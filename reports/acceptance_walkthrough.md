# Acceptance walkthrough

Every PRD v1.0 requirement, checked against its evidence. **Met** means it was verified. **Partly** means part of it wasn't verified or isn't delivered, and the note says which part. **Not met** means a target was missed.

Checked on 3 November 2026, on Windows 11 with an RTX 4070.

## Success criteria (PRD 3.3)

| Criterion | Target | Result | Status | Evidence |
|---|---|---:|---|---|
| Volume forecast accuracy | MAPE ≤ 12% | 8.55% | Met | `reports/final_metrics.json` |
| Congestion classification | Macro-F1 ≥ 0.80 | 0.774 (interval 0.750 to 0.798) | **Not met** | `reports/final_report.md`, section 6 |
| Accident-risk ranking | ROC-AUC ≥ 0.75 | 0.643 (interval 0.608 to 0.683) | **Not met** | `reports/final_report.md`, section 6 |
| DL vs ML benchmark | Sequence model beats best classical on RMSE | 54.27 against 54.62 | Met, narrowly | `reports/benchmark.md` |
| Reproducibility | Reruns from raw inputs with one command | Identical results on rebuild | Met | `reports/reproducibility.md` |
| Dashboard | All nine views render on real outputs | All render | Met | `tests/test_dashboard.py`, browser check |

## Functional requirements (PRD 4)

| ID | Requirement | Status | Where |
|---|---|---|---|
| FR-01 | Ingest raw sensor, weather and calendar tables | Met | `src/ingest.py` |
| FR-02 | Validate schema, types and ranges, and log rejects | Met | `src/ingest.py`, quarantine log |
| FR-03 | Resolve missing readings, duplicates and outliers | Met | `src/clean.py`, `reports/data_quality.md` |
| FR-04 | Merge into one segment-window table | Met | `src/clean.py` |
| FR-05 | Engineer features | Met | `src/features.py`, with leakage tests |
| FR-06 | Versioned dataset and data-quality report | Met | Dataset named by content hash; `reports/data_quality.md` |
| FR-07 | Linear regression, tree, forest and XGBoost on regression targets | Met | `reports/scoreboard_volume.csv`, `reports/scoreboard_travel_time.csv` |
| FR-08 | Classifiers for congestion and accident risk, with an SVM | Met | `reports/scoreboard_congestion.csv`, `reports/scoreboard_accident.csv` |
| FR-09 | LSTM/GRU trained from scratch, no pretrained weights | Met | `src/dl_model.py`; model cards record this |
| FR-10 | Volume, congestion, travel-time and accident forecasts per segment and window | Met | `data/processed/predictions.parquet` |
| FR-11 | Confidence on each prediction | Met | Conformal ranges and class probabilities |
| FR-12 | Persist models and reload without retraining | Met | `models/`, `src/serve.py` |
| FR-13 | Near-term predictions per segment | Met | Live prediction page, which replays past dates |
| FR-14 | Historical trends, heatmaps and road comparison | Met | Those three pages |
| FR-15 | Performance metrics, importance and forecast charts | Met | Model performance, What drives the forecasts, Forecasts vs what happened |
| FR-16 | Weather-vs-traffic view | Met | Weather vs traffic page |
| FR-17 | Upload a file and trigger a prediction run | Met | Upload data page; forecasts reach the other forecast pages |
| FR-18 | Export a summary of a time range | Met | Reports and insights page. The downloads were generated, but not opened |

## Non-functional requirements (PRD 5)

| Category | Target | Status | Notes |
|---|---|---|---|
| Performance | Batch inference ≤ 30 s | Met | About 3 to 5 s per horizon, for 172,700 rows |
| Scalability | ≥ 150k rows on a 16 GB machine | Partly | 181,200 rows handled. Memory was about 1 GB when sampled during classical training, but it hasn't been run on a 16 GB machine or profiled at its peak |
| Reproducibility | Seeded, and a rerun reproduces metrics | Met | Zero difference on rebuild, same machine and device. CPU and GPU differ slightly in floating point |
| Maintainability | Modular, PEP 8 and docstrings | Met | Lint enforces both |
| Usability | ≤ 3 clicks to any view; usable without training | Partly | Every page is one click from the sidebar. It hasn't been tried with an actual operator |
| Reliability | Bad rows quarantined, 0 silent drops | Met | Tested in `tests/test_ingest.py` |
| Portability | Windows, macOS or Linux, 1 setup command | Partly | Full pipeline on Windows only. On Linux, CI runs the fast tests and rebuilds the dataset with a matching hash. Model training hasn't run there, and macOS is untested |
| Transparency | Every prediction traceable | Met | Each forecast carries its model version and input hash, and the Model records page shows each card |

## Deliverables (PRD 18)

| Deliverable | Status | Where |
|---|---|---|
| Source repository with the PRD 16 layout | Met | https://github.com/Naman6019/flowcast (private) |
| Processed dataset and cleaning pipeline | Met | `data/processed/flowcast_v*.parquet`, named by a hash of its contents, and `src/ingest.py`, `src/clean.py` |
| Data-quality report | Met | `reports/data_quality.md`, and the Data quality page |
| EDA notebook | Met | `notebooks/01_eda.ipynb` |
| Classical models with a scoreboard | Met | `reports/scoreboard_*.csv`, `reports/model_selection.md`. Trained model files are in the submission zip. The repository leaves them out, and the pipeline rebuilds them |
| Deep model, training curves and benchmark | Met | `reports/figures/deep_loss_curves.png`, `reports/benchmark.md` |
| Model cards for every model | Met | `models/**/model_card.json`, enforced by a test |
| Streamlit dashboard: nine views and modules | Met | `dashboard/`. All nine views, upload, training, prediction and report modules |
| Final technical report | Met | `reports/final_report.md`, `reports/final_report.pdf` |
| README with setup and reproduction | Met | `README.md` |

## Definition of done (PRD, final section)

| Step | Status | Notes |
|---|---|---|
| A reviewer can clone the repository | Met | It's private, so they need access |
| Run one setup command | Met | README |
| Reproduce every metric from the raw data | Met | The raw CSVs are in the repository. `python -m src.pipeline all` rebuilds everything, verified here with no differences. From a fresh clone, only the dataset rebuild was run (hash matches, on Windows and on Linux in CI); the full rebuild was run in the working copy |
| Open a dashboard showing all nine views | Met | After running the pipeline |

## Beyond the PRD

Real-data validation on I-94 traffic: see `reports/real_data.md`. It isn't a PRD requirement, and it tests volume only.
