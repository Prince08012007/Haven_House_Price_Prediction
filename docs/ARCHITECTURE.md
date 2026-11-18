# Haven architecture

Haven is a single FastAPI service serving the existing static dashboard and the model API. `app.py` owns HTTP contracts, Pydantic validation, financial scenario functions, SQLite persistence, and report streaming. `src/inference.py` retains the project's feature engineering and training helpers. `model/model.joblib` contains the serialized preprocessor, fitted ExtraTrees estimator, input feature schema, defaults, and held-out validation metadata. The browser dashboard calls same-origin endpoints; prediction history is server-side SQLite, while display preferences and the small comparison shortlist remain local browser preferences.

Request path for model predictions: JSON -> `HouseFeatures` validation -> Ames baseline defaults -> `to_ames_row` mapping -> `add_features` -> saved preprocessor -> ExtraTrees log-price prediction -> `expm1` -> indicative validation band -> SQLite record. Future value and financial tools are independent scenario calculations and never masquerade as ML model predictions.

SQLite path defaults to `haven_history.sqlite3` in the project root. Set `HAVEN_DB_PATH` to choose a persistent database location. The Docker image defaults this to `/data/haven_history.sqlite3` and declares `/data` as a volume.
