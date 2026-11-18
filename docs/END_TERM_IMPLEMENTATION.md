# End-Term implementation notes

This repository extends the existing Kaggle Ames Housing project instead of replacing it. Original notebooks, feature-engineering scripts, Kaggle result files, visualizations and documentation are retained.

## Runtime flow
Browser (`static/index.html`) → FastAPI (`app.py`) → Pydantic validation → Ames input mapping + training defaults → shared feature engineering (`src/inference.py`) → serialized preprocessing/model (`model/model.joblib`) → predicted SalePrice + indicative range + rule-based tips → append-only JSONL prediction log.

## Reproducibility
Run `python train.py` to regenerate model comparison metrics and the deployable model bundle. MLflow is configured to use `sqlite:///mlflow.db`. Four distinct training configurations are run, spanning three algorithms.

## Evidence to capture before submission
1. Haven UI with successful prediction.
2. Swagger `/docs` showing GET `/health` and POST `/predict`.
3. MLflow experiment page showing at least three completed runs.
4. `pytest -v` output with all tests passing.
5. Docker build and running container.
6. GitHub Actions latest green workflow.
7. Public deployment URL and sample `logs/predictions.jsonl`.
