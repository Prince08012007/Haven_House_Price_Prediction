from fastapi.testclient import TestClient
from app import app, DB_PATH
client=TestClient(app)
BASE={"overall_qual":7,"neighborhood":"NAmes","year_built":2005,"house_style":"1Story","living_area":1800,"lot_area":8500,"bedrooms":3,"bathrooms":2,"basement_area":900,"garage_cars":2,"central_air":"Y","garage_area":480}

def test_model_metadata_and_importance():
    assert client.get('/model/metrics').json()['model']=='ExtraTrees'
    assert client.get('/model/feature-importance').json()['features']

def test_future_construction_supports_out_of_training_year():
    r=client.post('/future/construction',json={**BASE,'year_built':2035})
    assert r.status_code==200 and r.json()['predicted_price']>0

def test_future_value_compound_scenarios():
    d=client.post('/future/value',json={'current_value':1000000,'current_year':2026,'target_year':2030,'conservative_rate':.02,'moderate_rate':.05,'optimistic_rate':.08}).json()
    assert d['scenarios']['moderate']['future_value']==1215506.25

def test_emi_zero_rate():
    d=client.post('/finance/emi',json={'property_price':1200000,'down_payment':200000,'annual_interest_rate':0,'loan_years':10}).json()
    assert d['monthly_emi']==pytest.approx(8333.33, rel=1e-4)

def test_investment_yield():
    d=client.post('/investment/analyze',json={'purchase_price':1000000,'down_payment':200000,'annual_appreciation':.05,'years':5,'monthly_rent':10000,'annual_maintenance':12000,'annual_other_expenses':0}).json()
    assert d['gross_rental_yield_pct']==pytest.approx(12)
    assert d['net_rental_yield_pct']==pytest.approx(10.8)

def test_affordability():
    d=client.post('/finance/affordability',json={'monthly_income':100000,'monthly_expenses':30000,'existing_loan_obligations':5000,'down_payment_budget':500000,'annual_interest_rate':8,'loan_years':20}).json()
    assert d['affordable_emi']>0 and d['affordable_property_budget']>500000

def test_compare_2_properties():
    d=client.post('/properties/compare',json={'properties':[BASE,{**BASE,'living_area':2200}], 'target_year':2035,'appreciation_rate':.05})
    assert d.status_code==200 and len(d.json()['properties'])==2

def test_invalid_negative_and_future_years():
    assert client.post('/predict',json={**BASE,'living_area':-1}).status_code==422
    assert client.post('/future/construction',json=BASE).status_code==422

def test_history_and_pdf_export():
    d=client.post('/predict',json=BASE).json(); ident=d['history_id']
    assert any(x['id']==ident for x in client.get('/history').json()['items'])
    assert client.get(f'/reports/prediction/{ident}.pdf').status_code==200
    assert 'text/csv' in client.get('/history/export.csv').headers['content-type']
    assert client.delete(f'/history/{ident}').status_code==200

import pytest

def test_recommendations_are_historical_dataset_records():
    d=client.get('/recommendations',params={'max_budget':300000,'min_living_area':1000,'min_bedrooms':2})
    assert d.status_code==200
    assert 'not live property listings' in d.json()['source']


def test_non_finite_input_rejected():
    r=client.post('/predict',content='{"overall_qual":7,"neighborhood":"NAmes","year_built":2005,"house_style":"1Story","living_area":NaN,"lot_area":8500,"bedrooms":3,"bathrooms":2,"basement_area":900,"garage_cars":2,"central_air":"Y","garage_area":480}',headers={'Content-Type':'application/json'})
    assert r.status_code==422

def test_prediction_currency_is_explicitly_dataset_usd():
    d=client.post('/predict',json=BASE).json()
    assert d['currency']=='USD'
