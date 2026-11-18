# Haven – AI-Powered House Price Prediction & Real Estate Analytics

Haven is the upgraded continuation of the supplied FastAPI + HTML/CSS/JavaScript project. It retains the existing Ames Housing dataset, fitted ExtraTrees regression model, feature engineering, and green Haven dashboard identity. The backend now provides model inference, transparent future-value scenarios, financial calculators, comparison, persistent SQLite prediction history, and downloadable reports.

> **Important market-data limitation:** the included model and dataset concern historical Ames, Iowa home sales (Kaggle House Prices). Predictions are model estimates in the dataset's SalePrice currency (USD), not live Indian property valuations. Model prediction values are displayed in USD to match the source target. The finance calculators accept user-entered amounts and display INR by default; no currency conversion is performed. For real-world INR valuation, retrain on suitable INR-denominated local market data and document its provenance. Future values are user-assumption scenarios, not forecasts or guarantees.

## Run on Windows (PowerShell)

```powershell
# 1. Extract the ZIP and open this folder in VS Code
cd .\Haven_House_Price_Prediction_Complete

# 2. Create and activate an isolated environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation for this terminal only:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the existing trained-model API (model/model.joblib is included)
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Interactive API documentation is at **http://127.0.0.1:8000/docs**. `run_haven.bat` remains available for double-click startup.

If you intentionally need to retrain, run `python train.py`. Training compares Gradient Boosting, Random Forest, and ExtraTrees using a held-out validation split and saves the best model. Do not retrain merely to start the application.

## Architecture

```text
Haven/
├── app.py                       # FastAPI routes, validation, calculations, persistence, PDF/CSV
├── src/inference.py             # shared feature engineering and retained training helpers
├── model/model.joblib           # existing trained ExtraTrees model + preprocessor + metadata
├── data/train.csv               # Ames training data (1,460 labeled rows)
├── data/test.csv                # Kaggle competition test rows
├── static/index.html            # responsive Haven single-page dashboard
├── tests/                       # API and calculation tests
├── docs/                        # project methodology and analysis documents
├── outputs/                     # existing experiment/model artifacts
├── results/, visualizations/    # existing project outputs
├── requirements.txt
├── requirements-serve.txt
└── Dockerfile
```

## Machine-learning methodology

- **Retained estimator:** the included `model/model.joblib` reports `ExtraTrees` and includes the fitted preprocessing transformer, estimator, feature-column contract, baseline defaults, validation metrics, training row count, and training timestamp.
- **Inference:** the API maps the supported UI inputs into the Ames raw feature schema, fills unprovided columns from training medians/modes, calls the same `add_features` routine, aligns to the saved feature columns, applies the fitted preprocessor, and predicts log1p(SalePrice). It reverses the transform with `expm1`.
- **Range:** the low/high interval is an indicative log-RMSE band derived from the saved validation RMSE; it is not a calibrated guarantee interval.
- **Metrics:** `GET /model/metrics` returns the saved held-out validation MAE in price units, RMSE in log-price space, R² in log-price space, model name, and training metadata. The metric units are explicitly named to avoid implying price-space RMSE/R².
- **Feature importance:** `GET /model/feature-importance` and `GET /insights` expose fitted ExtraTrees impurity-based global importance after preprocessing. These are global model-reliance indicators, not causal effects or a faithful per-record attribution. A SHAP/permutation local explanation is not claimed.
- **No leakage claim:** the saved metadata originates from the existing training workflow's held-out split. The training script compares candidates on the same fixed split; cross-validation has not been added to this delivered model selection workflow.

## Numerical validation

Numerical areas and counts are no longer capped at the Kaggle training extrema. Negative areas/counts, non-finite values, invalid quality values, impossible remodel chronology, and inconsistent garage configuration are rejected. `year_built` supports future construction years; `/future/construction` explicitly marks those results as scenario estimates. Historical training coverage is not treated as a hard user-input ceiling.

## Implemented features and APIs

| Feature | Endpoint |
|---|---|
| Existing trained-model prediction + price range + tips | `POST /predict` |
| Future construction year using existing model pipeline | `POST /future/construction` |
| Future property value, three compound scenarios | `POST /future/value` |
| EMI, total interest, total payment | `POST /finance/emi` |
| Appreciation, gross/net rental yield, ROI, cash flow | `POST /investment/analyze` |
| Affordability and indicative loan budget | `POST /finance/affordability` |
| Two- or three-property prediction comparison | `POST /properties/compare` |
| Historical training-data matches (not listings) | `GET /recommendations` |
| Global feature importance and model metadata | `GET /insights`, `GET /model/metrics`, `GET /model/feature-importance` |
| Persistent history search/list/delete | `GET /history`, `DELETE /history/{id}` |
| History CSV | `GET /history/export.csv` |
| Prediction PDF | `GET /reports/prediction/{id}.pdf` |
| Health and API docs | `GET /health`, `/docs` |

The dashboard adds working future-value, investment, affordability, EMI, and what-if controls. Main predictions and what-if scenarios use the actual fitted model. Prediction history is stored in `haven_history.sqlite3` rather than only browser localStorage. Existing profile/theme preferences and comparison display remain browser-local.

## API examples

```powershell
# Health
Invoke-RestMethod http://127.0.0.1:8000/health

# Future property value scenarios
$body = @{ current_value=7500000; current_year=2026; target_year=2035; conservative_rate=0.02; moderate_rate=0.05; optimistic_rate=0.08 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/future/value -ContentType 'application/json' -Body $body

# Prediction history
Invoke-RestMethod http://127.0.0.1:8000/history
```

## Tests

```powershell
python -m pytest -v --tb=short
```

Tests cover model health and inference, invalid input, future construction, compound growth, EMI, investment yields, affordability, comparison, persistent history, PDF, and CSV export.

## Docker

```powershell
docker build -t haven-house-price .
docker run --rm -p 8000:8000 -v haven-db:/data haven-house-price
```

For production, mount a dedicated persistent volume for `haven_history.sqlite3`, use a non-root container user, terminate TLS at a trusted reverse proxy, configure backups, and add authentication/authorization before exposing multi-user data.

## Data, scope, and responsible use

- Dataset: Kaggle House Prices – Advanced Regression Techniques, Ames, Iowa. `data/train.csv` contains the labeled training data; see `data/data_description.txt` for feature definitions.
- The reference dataset is historical and location-specific. `GET /recommendations` filters actual historical rows and labels them as training-data matches; it does not fabricate active listings.
- Location-specific adjustment is not automatically inferred because no reliable live market feed is included.
- Financial calculators exclude taxes, insurance, transaction charges, vacancy, lending fees, and many other costs unless the user incorporates them in assumptions.
- No output is financial, legal, lending, or property-buying advice.

## Future scope

Add authenticated multi-user workspaces, a local-market dataset with licensed provenance, currency-aware valuation, temporal/geographic validation, cross-validation and calibration, SHAP/permutation local explanations, richer charts, and complete multi-property PDF reports.
