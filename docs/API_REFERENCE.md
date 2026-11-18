# Haven API reference

Open `/docs` while the app is running for interactive schemas and try-it-out.

- `GET /health`: model and database readiness.
- `POST /predict`: model prediction. Input uses `overall_qual`, `neighborhood`, `year_built`, `house_style`, `living_area`, `lot_area`, `bedrooms`, `bathrooms`, `basement_area`, `garage_cars`, `central_air`, `garage_area`, and optional `year_remodeled`. Output price values are USD because the retained Ames SalePrice target is USD.
- `POST /future/construction`: same property input; requires `year_built` later than current year. Runs the saved model and labels its extrapolation as a scenario.
- `POST /future/value`: `current_value`, `current_year`, `target_year`, and three annual rates as decimals (e.g. 0.05 for 5%).
- `POST /finance/emi`: `property_price`, `down_payment`, `annual_interest_rate` (percent), `loan_years`.
- `POST /investment/analyze`: purchase price, down payment, annual appreciation decimal, duration, rent, maintenance, and other annual expenses.
- `POST /finance/affordability`: income, expenses, existing obligations, down-payment budget, interest rate, tenure, and optional debt-to-income ratio.
- `POST /properties/compare`: 2–3 `properties`, `target_year`, appreciation, loan, and rent assumptions.
- `GET /recommendations`: filters actual historical Ames rows; this is not a listing feed.
- `GET /insights`, `/model/metrics`, `/model/feature-importance`: model diagnostics.
- `GET /history?q=...&limit=...`, `DELETE /history/{id}`, `GET /history/export.csv`.
- `GET /reports/prediction/{id}.pdf`: PDF for a saved prediction.
