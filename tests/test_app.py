from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
VALID={"overall_qual":7,"neighborhood":"NAmes","year_built":2005,"house_style":"1Story","living_area":1800,"lot_area":8500,"bedrooms":3,"bathrooms":2,"basement_area":900,"garage_cars":2,"central_air":"Y","garage_area":480}
def test_health_endpoint():
    r=client.get('/health'); assert r.status_code==200; assert r.json()['model_ready'] is True
def test_ui_loads():
    r=client.get('/'); assert r.status_code==200; assert 'Haven' in r.text
def test_prediction_returns_positive_price():
    r=client.post('/predict',json=VALID); assert r.status_code==200; assert r.json()['predicted_price']>0
def test_boundary_rejects_quality_out_of_range():
    bad={**VALID,'overall_qual':11}; r=client.post('/predict',json=bad); assert r.status_code==422
def test_wrong_category_rejected():
    bad={**VALID,'central_air':'sometimes'}; r=client.post('/predict',json=bad); assert r.status_code==422
def test_domain_logic_higher_quality_not_lower_prediction():
    low={**VALID,'overall_qual':3}; high={**VALID,'overall_qual':9}
    a=client.post('/predict',json=low).json()['predicted_price']; b=client.post('/predict',json=high).json()['predicted_price']; assert b>a
