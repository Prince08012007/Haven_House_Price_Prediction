# Haven user guide

1. Start the app using the Windows PowerShell commands in the root README.
2. On **Predict House Price**, complete the property details and generate an estimate. The result is produced by the included ExtraTrees model. Its source dataset is historical Ames, Iowa sales and its prices are USD.
3. Use **Future Value** for a future construction-year estimate or for compound-growth projections. A future-built house uses the ML model with the requested construction year; existing-property future value uses user-selected annual growth assumptions.
4. Use **What-If Analysis** to compare a baseline property against modified living area, lot area, garage area, and quality through the same model API.
5. Use **EMI Calculator**, **Investment Analyzer**, and **Affordability** for assumption-based financial calculations. Enter monetary amounts in the currency you intend to plan with; these calculators do not fetch exchange rates or bank offers.
6. Use **Compare Properties** to retain up to three model-predicted scenarios in the browser shortlist. For API-based comparisons use `/properties/compare`.
7. Use **Prediction History** to search saved backend predictions, delete an individual record, download a CSV, or open its PDF report.
8. **Smart Insights** shows global feature importance from the fitted estimator. Importance is not causality or a property-specific SHAP explanation.

All projections are estimates, not guarantees, offers, valuations, or loan approvals.
