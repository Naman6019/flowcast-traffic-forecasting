# FlowCast

Short-horizon traffic forecasting for the 25-segment Northline Corridor. For every segment and 30-minute window, FlowCast forecasts volume, speed, travel time, congestion level and accident risk 30 to 120 minutes ahead, each with a confidence estimate, and shows the results in a Streamlit dashboard.

This is an internship project built from a client-style brief. Meridian Mobility Systems is a scenario in that brief, and the delivered dataset looks synthetic: congestion is an exact function of volume, and travel time is an exact function of speed. The brief and the data files delivered with it aren't included in this public copy (see Setup).

The results, including the targets that were missed and why, are in [reports/final_report.md](reports/final_report.md).

## Setup

You need Python 3.11. Clone the repository and run the one setup command for your system.

The three CSV files delivered with the brief (`traffic_sensor_log.csv`, `weather_observations.csv` and `calendar_events.csv`) aren't redistributed here. To rerun the Northline stages, put them in `data/raw/`. The `real` stage needs none of them: it downloads the public I-94 dataset. The reports, figures and model cards from the full run are included, so the results can be read without rerunning anything.

On Windows:

```bash
python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Then activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS or Linux).

That gives you CPU-only PyTorch on Windows, which works but is slower. To train the deep models on an NVIDIA GPU, also install a CUDA build from the [PyTorch site](https://pytorch.org/get-started/locally/) at the same version. `deep.device: auto` in `config.yaml` then uses the GPU when it finds one.

Trained models and forecasts aren't included, because the pipeline rebuilds them.

## Reproduce everything

One command rebuilds every dataset, model, score, figure and notebook from the raw files:

```bash
python -m src.pipeline all
```

This took about 80 minutes on a 14-core laptop with an RTX 4070. A GPU isn't required, but the deep stage will take longer without one. All randomness is seeded from `config.yaml`, and a rerun on the same kind of device reproduces the metrics. Floating-point results differ slightly between CPU and GPU, so each deep model card records the device it was trained on. You can also run one stage at a time, in this order:

If a long run is interrupted, add `--resume` to continue from the last finished trial, for example `python -m src.pipeline classical --resume`. Seeds are set per run, so resuming doesn't change the results.

| Stage | What it does |
|---|---|
| `data` | Validates, cleans and merges the raw files. Writes the versioned dataset, the quarantine log and `reports/data_quality.md` |
| `features` | Builds leakage-safe features and targets for each of the 4 horizons |
| `classical` | Tunes and compares the classical models on time-series CV and validation only |
| `deep` | Trains LSTM, GRU and BiLSTM from scratch, on validation only |
| `final` | Fixes the production choices, then scores the test window once. Writes the scoreboards, benchmark and forecasts |
| `real` | Downloads a public I-94 traffic dataset and validates the approach on it. Skipped with a warning if the download isn't possible |
| `notebooks` | Rebuilds and runs the three notebooks |
| `report` | Exports `reports/final_report.md` to PDF |

## Open the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard reads saved outputs and never trains on load. **If you have the submission zip, the trained models and forecasts are already included, so you can open it straight after setup.** From a clone of the repository, run the pipeline first to build them.

It opens on an overview that explains the project and links to each task. The pages are grouped by the question they answer:

| Section | Pages |
|---|---|
| What happens next? | Live prediction (a replay of past dates), accident risk, request a forecast |
| Start here | Overview |
| What has been happening? | Congestion heatmap, road comparison, historical trends, weather vs traffic |
| Can I trust it? | Forecasts vs what happened, prediction confidence, model performance, real-data check, what drives the forecasts, data quality, model records |
| Tools | Upload data, retrain a model, reports and insights |

The overview also maps the PRD's nine required views to these pages.

To try the upload module, upload a CSV in the same format as `traffic_sensor_log.csv`. Any defective rows land in quarantine. An upload's forecasts then appear on the forecast pages until you remove them on the Upload data page. The accuracy figures don't change, because they describe the evaluated models.

## Where the outputs live

| Path | Contents |
|---|---|
| `data/processed/flowcast_v<hash>.parquet` | The analysis-ready dataset, named by a hash of its contents |
| `data/processed/predictions.parquet` | Forecasts for every segment, window and horizon in validation and test, with the model version and input hash |
| `data/interim/quarantine.parquet` | Every rejected row and changed value, with its rule and old and new values |
| `models/<target>/<model>/<version>/` | Each model with its `model_card.json`. `models/registry.json` names the ones in service |
| `reports/data_quality.md` | Every defect found and how it was resolved |
| `reports/model_selection.md`, `reports/benchmark.md` | Why each model was chosen, and the classical-vs-deep verdict |
| `reports/final_report.md` | The technical report |
| `reports/acceptance_walkthrough.md` | Every PRD requirement checked against its evidence |
| `reports/real_data.md` | Results on real I-94 traffic |

## Project layout

```
src/ingest.py        M1  load, validate, quarantine
src/clean.py         M2  clean and merge
src/features.py      M3  features and targets
src/eda.py           M4  analysis tables
src/scratch.py           linear regression, metrics and scaling written in NumPy
src/ml_models.py     M5  classical models
src/dl_model.py      M6  recurrent models, no pretrained weights
src/evaluate.py          metrics, time-series folds, error breakdowns
src/uncertainty.py       conformal intervals and class probabilities
src/final.py             the only code that scores the test window
src/serve.py             inference and the upload path
dashboard/           M7  Streamlit app
tests/                   run with: pytest
```

## Tests

```bash
pytest -m "not slow"
```

These need no data files and include the leakage tests. To also check the real data, trained models and upload path, run `pytest` after the pipeline. Without the brief's data files in `data/raw/`, those slow tests are skipped.

## License

Code: MIT, see [LICENSE](LICENSE). The I-94 data is the UCI Metro Interstate Traffic Volume dataset (id 492), CC BY 4.0.
