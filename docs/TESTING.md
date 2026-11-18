# Testing and QA

Run from the project root:

```powershell
python -m pytest -v --tb=short
```

The test suite checks health and UI startup, the retained model prediction, validation, model metadata and feature importance, future construction, compound future value, EMI, rental yields, affordability, property comparison, persistent prediction history, PDF output, and CSV output. A separate JavaScript syntax check can be run by extracting the inline `<script>` content from `static/index.html` and running `node --check` against it.

A successful local validation run is recorded in the delivery notes. Production QA should additionally test supported Windows/Python versions, persistent Docker volume behavior, browser accessibility, and load/security requirements.
