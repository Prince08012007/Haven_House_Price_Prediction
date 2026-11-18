"""Haven – AI-powered house price prediction and real-estate decision support."""
from __future__ import annotations
import csv, io, json, logging, math, sqlite3, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from src.inference import add_features

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'model'/'model.joblib'; LOG_DIR=ROOT/'logs'; LOG_DIR.mkdir(exist_ok=True)
DB_PATH=Path(os.getenv('HAVEN_DB_PATH', str(ROOT/'haven_history.sqlite3')))
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s'); logger=logging.getLogger('haven')

NEIGHBORHOODS=["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","NAmes","NPkVill","NWAmes","NoRidge","NridgHt","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"]
STYLES=["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"]
class HouseFeatures(BaseModel):
    model_config=ConfigDict(extra='forbid')
    overall_qual:int=Field(7,ge=1,le=10)
    neighborhood:Literal["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","NAmes","NPkVill","NWAmes","NoRidge","NridgHt","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"]='NAmes'
    year_built:int=Field(2005,ge=1800,le=2200)
    house_style:Literal["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"]='1Story'
    living_area:float=Field(1800,ge=1)
    lot_area:float=Field(8500,ge=1)
    bedrooms:int=Field(3,ge=0)
    bathrooms:float=Field(2,ge=0)
    basement_area:float=Field(900,ge=0)
    garage_cars:int=Field(2,ge=0)
    central_air:Literal['Y','N']='Y'
    garage_area:float=Field(480,ge=0)
    year_remodeled:Optional[int]=Field(None,ge=1800,le=2200)
    @model_validator(mode='after')
    def logical(self):
        if self.year_remodeled is not None and self.year_remodeled<self.year_built: raise ValueError('Year remodeled cannot be earlier than year built.')
        if self.basement_area>0 and self.basement_area>self.lot_area*5: raise ValueError('Basement area is implausibly large relative to lot area.')
        if self.garage_cars==0 and self.garage_area>0: raise ValueError('Garage area requires at least one garage space.')
        if self.garage_cars>0 and self.garage_area==0: raise ValueError('Garage capacity requires a positive garage area.')
        if self.bathrooms>20 or self.bedrooms>30 or self.garage_cars>20: raise ValueError('Room or garage count is not logically plausible.')
        return self

class FutureValueRequest(BaseModel): current_value:float=Field(gt=0); current_year:int=Field(ge=1800,le=2200); target_year:int=Field(ge=1800,le=2200); conservative_rate:float=Field(0.02,ge=-0.5,le=1); moderate_rate:float=Field(0.05,ge=-0.5,le=1); optimistic_rate:float=Field(0.08,ge=-0.5,le=1)
class EMIRequest(BaseModel): property_price:float=Field(gt=0); down_payment:float=Field(ge=0); annual_interest_rate:float=Field(ge=0,le=100); loan_years:float=Field(gt=0,le=50)
class InvestmentRequest(BaseModel): purchase_price:float=Field(gt=0); down_payment:float=Field(ge=0); annual_appreciation:float=Field(-0.5,le=1); years:int=Field(ge=0,le=100); monthly_rent:float=Field(ge=0); annual_maintenance:float=Field(ge=0); annual_other_expenses:float=Field(ge=0); loan_amount:Optional[float]=Field(None,ge=0)
class AffordabilityRequest(BaseModel): monthly_income:float=Field(gt=0); monthly_expenses:float=Field(ge=0); existing_loan_obligations:float=Field(ge=0); down_payment_budget:float=Field(ge=0); annual_interest_rate:float=Field(ge=0,le=100); loan_years:int=Field(gt=0,le=50); debt_to_income:float=Field(.4,gt=0,le=.8)
class CompareRequest(BaseModel):
    properties:list[HouseFeatures]=Field(min_length=2,max_length=3)
    target_year:int=Field(ge=1800,le=2200)
    appreciation_rate:float=Field(.05,ge=-.5,le=1)
    down_payment_pct:float=Field(.2,ge=0,lt=1)
    annual_interest_rate:float=Field(8.5,ge=0,le=100)
    loan_years:int=Field(20,gt=0,le=50)
    monthly_rent:float=Field(0,ge=0)

bundle=None
def db():
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row
    con.execute('''CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, inputs TEXT NOT NULL, result TEXT NOT NULL)'''); con.commit(); return con
def load_bundle():
    global bundle
    if bundle is None:
        if not MODEL_PATH.exists(): raise RuntimeError(f'Model file missing: {MODEL_PATH}. Run python train.py first.')
        bundle=joblib.load(MODEL_PATH)
    return bundle

def to_ames_row(payload:HouseFeatures,defaults:dict)->pd.DataFrame:
    row=defaults.copy(); v=payload.model_dump()
    mapping={'overall_qual':'OverallQual','neighborhood':'Neighborhood','year_built':'YearBuilt','house_style':'HouseStyle','living_area':'GrLivArea','lot_area':'LotArea','bedrooms':'BedroomAbvGr','bathrooms':'FullBath','basement_area':'TotalBsmtSF','garage_cars':'GarageCars','central_air':'CentralAir','garage_area':'GarageArea'}
    for src,dst in mapping.items(): row[dst]=v[src]
    built=v['year_built']; row['YearRemodAdd']=v['year_remodeled'] or max(built,int(row.get('YearRemodAdd',built)))
    # For future construction, align sale year to construction year so engineered age is not negative.
    row['YrSold']=max(int(row.get('YrSold',built)),built)
    if row.get('GarageYrBlt') is not None and not pd.isna(row.get('GarageYrBlt')) and v['garage_cars']>0: row['GarageYrBlt']=built
    return pd.DataFrame([row])
def infer(p:HouseFeatures):
    b=load_bundle(); raw=to_ames_row(p,b['baseline_defaults']); feat=add_features(raw).reindex(columns=b['feature_columns']); trans=b['preprocessor'].transform(feat)
    logp=float(b['model'].predict(trans)[0]); price=max(float(np.expm1(logp)),1.0); err=float(b.get('validation_rmse_log',.13))
    return {'predicted_price':round(price,2),'currency':'USD','price_low':round(max(float(np.expm1(logp-err)),1),2),'price_high':round(float(np.expm1(logp+err)),2),'model_name':b.get('model_name','Existing trained model'),'tips':smart_tips(p),'model_metrics':metrics(),'feature_importance':feature_importance()[:12]}
def smart_tips(p):
    tips=[]
    if p.overall_qual>=8: tips.append('High overall material and finish quality is a positive signal in the trained model.')
    elif p.overall_qual<=4: tips.append('Quality and finish are relatively low; upgrades may improve buyer appeal.')
    if p.living_area<1200: tips.append('The living area is compact; additional usable space may improve appeal.')
    if p.garage_cars==0: tips.append('No garage capacity is specified.')
    if p.year_built>datetime.now().year: tips.append('Future construction year: treat this as a model-based scenario; training examples do not establish future market conditions.')
    return tips or ['Compare this estimate with local comparable sales and inspect property condition.']
def metrics():
    b=load_bundle(); return {'mae':b.get('validation_mae_price'),'rmse_log':b.get('validation_rmse_log'),'r2_log':b.get('validation_r2_log'),'model':b.get('model_name'),'trained_at':b.get('trained_at'),'training_rows':b.get('training_rows')}
def feature_importance():
    b=load_bundle(); model=b['model']; imp=getattr(model,'feature_importances_',None)
    if imp is None:return []
    try: names=b['preprocessor'].get_feature_names_out()
    except Exception: names=[f'feature_{i}' for i in range(len(imp))]
    vals=sorted([{'feature':str(n).replace('numeric__','').replace('categorical__',''),'importance':float(x)} for n,x in zip(names,imp)],key=lambda x:x['importance'],reverse=True)
    return vals

def future_value(value,years,rate): return round(float(value)*((1+float(rate))**int(years)),2)
def emi(price,down,rate,years):
    if down>=price: raise HTTPException(422,'Down payment must be less than property price.')
    principal=price-down; n=int(round(years*12)); r=rate/1200
    monthly=principal/n if r==0 else principal*r*(1+r)**n/((1+r)**n-1)
    return {'loan_amount':round(principal,2),'monthly_emi':round(monthly,2),'total_payment':round(monthly*n,2),'total_interest':round(monthly*n-principal,2),'payments':n}
def save_history(inputs,result):
    con=db(); cur=con.execute('INSERT INTO predictions(created_at,inputs,result) VALUES(?,?,?)',(datetime.now(timezone.utc).isoformat(),json.dumps(inputs),json.dumps(result))); con.commit(); ident=cur.lastrowid; con.close(); return ident

app=FastAPI(title='Haven – AI Real Estate Decision Support',version='2.0.0',description='Existing Ames-trained ML inference plus transparent financial and scenario analysis.')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
def _safe_error_value(value):
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    if isinstance(value,dict): return {str(k):_safe_error_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_safe_error_value(v) for v in value]
    return value
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request:Request, exc:RequestValidationError):
    errors=[]
    for err in exc.errors():
        errors.append({k:_safe_error_value(v) for k,v in err.items() if k in {'type','loc','msg','input','ctx'}})
    return JSONResponse(status_code=422,content={'detail':errors})
@app.on_event('startup')
def startup():
    try: load_bundle()
    except Exception: logger.exception('Model startup check failed; API will report model unavailable.')
@app.get('/',include_in_schema=False)
def home(): return FileResponse(ROOT/'static'/'index.html')
@app.get('/health')
def health():
    try: load_bundle(); ready=True
    except Exception: ready=False
    return {'status':'ok' if ready else 'model_missing','model_ready':ready,'model_file':MODEL_PATH.name,'history_database':DB_PATH.name}
@app.get('/model/metrics')
def model_metrics(): return metrics()
@app.get('/model/feature-importance')
def model_features(): return {'features':feature_importance()}
@app.get('/insights')
def insights():
    items=feature_importance(); return {'top_features':items[:15],'explanation':'These are global model feature importances from the fitted ExtraTrees estimator after preprocessing. They describe overall model reliance, not causal effects or a per-property causal explanation.','model':load_bundle().get('model_name')}
@app.get('/recommendations')
def recommendations(max_budget:float=Query(...,gt=0),min_living_area:float=Query(0,ge=0),min_bedrooms:int=Query(0,ge=0),limit:int=Query(10,ge=1,le=50)):
    path=ROOT/'data'/'train.csv'
    if not path.exists(): raise HTTPException(503,'Reference training dataset is unavailable.')
    df=pd.read_csv(path); df=df[(df['SalePrice']<=max_budget)&(df['GrLivArea']>=min_living_area)&(df['BedroomAbvGr']>=min_bedrooms)].sort_values('SalePrice',ascending=False).head(limit)
    cols=['Id','SalePrice','GrLivArea','LotArea','BedroomAbvGr','FullBath','GarageArea','OverallQual','YearBuilt','Neighborhood']
    return {'items':df[cols].astype(object).where(pd.notna(df[cols]),None).to_dict(orient='records'),'source':'Historical Ames training dataset sales, not live property listings or current market offers.','count':int(len(df))}
@app.post('/predict')
def predict(payload:HouseFeatures):
    try:
        result=infer(payload); ident=save_history(payload.model_dump(),result); result['history_id']=ident
        logger.info('prediction_completed model=%s price=%.2f',result['model_name'],result['predicted_price']); return result
    except HTTPException: raise
    except Exception as exc: logger.exception('prediction_failed'); raise HTTPException(500,'Prediction failed. Check model bundle and server logs.') from exc
@app.post('/future/construction')
def future_construction(payload:HouseFeatures):
    if payload.year_built<=datetime.now().year: raise HTTPException(422,'Future construction mode requires a construction year after the current year.')
    r=infer(payload); r['mode']='future_construction'; r['construction_year']=payload.year_built; r['disclaimer']='Model estimate for a future-built home; this is not a guaranteed future market valuation.'; return r
@app.post('/future/value')
def future_property(req:FutureValueRequest):
    years=req.target_year-req.current_year
    if years<0: raise HTTPException(422,'Target year must be the current year or later.')
    scenarios={k:{'annual_rate':rate,'future_value':future_value(req.current_value,years,rate),'appreciation':round(future_value(req.current_value,years,rate)-req.current_value,2)} for k,rate in [('conservative',req.conservative_rate),('moderate',req.moderate_rate),('optimistic',req.optimistic_rate)]}
    return {'current_value':req.current_value,'current_year':req.current_year,'target_year':req.target_year,'years':years,'scenarios':scenarios,'disclaimer':'Scenario-based compound-growth estimates using user assumptions, not guaranteed market predictions.'}
@app.post('/finance/emi')
def emi_api(req:EMIRequest): return emi(req.property_price,req.down_payment,req.annual_interest_rate,req.loan_years)
@app.post('/investment/analyze')
def investment(req:InvestmentRequest):
    if req.down_payment>req.purchase_price: raise HTTPException(422,'Down payment cannot exceed purchase price.')
    fv=future_value(req.purchase_price,req.years,req.annual_appreciation); annual_rent=req.monthly_rent*12; net_rent=annual_rent-req.annual_maintenance-req.annual_other_expenses; invested=req.down_payment or req.purchase_price
    return {'future_value':fv,'appreciation':round(fv-req.purchase_price,2),'gross_rental_yield_pct':round(annual_rent/req.purchase_price*100,2),'net_rental_yield_pct':round(net_rent/req.purchase_price*100,2),'annual_rental_income':annual_rent,'net_annual_rental_income':round(net_rent,2),'total_investment_basis':round(invested+req.annual_maintenance*req.years+req.annual_other_expenses*req.years,2),'roi_pct':round((fv-req.purchase_price+net_rent*req.years)/invested*100,2),'annual_cash_flow':round(net_rent,2),'scenarios':{k:{'rate':r,'future_value':future_value(req.purchase_price,req.years,r)} for k,r in [('conservative',max(-.5,req.annual_appreciation-.02)),('moderate',req.annual_appreciation),('optimistic',min(1,req.annual_appreciation+.03))]},'assumptions':'Appreciation and rent are user assumptions; taxes, vacancy, transaction costs and financing are excluded unless entered.'}
@app.post('/finance/affordability')
def affordability(req:AffordabilityRequest):
    available=max(0,req.monthly_income*req.debt_to_income-req.existing_loan_obligations); disposable=max(0,req.monthly_income-req.monthly_expenses-req.existing_loan_obligations); allowed=min(available,disposable); n=req.loan_years*12; r=req.annual_interest_rate/1200; factor=n if r==0 else ((1+r)**n-1)/(r*(1+r)**n); loan=allowed*factor; budget=loan+req.down_payment_budget
    return {'affordable_emi':round(allowed,2),'estimated_loan_eligibility':round(loan,2),'affordable_property_budget':round(budget,2),'down_payment_budget':req.down_payment_budget,'monthly_income_after_expenses_and_emi':round(disposable-allowed,2),'disclaimer':'Indicative affordability estimate only; not a bank approval or lending commitment.'}
@app.post('/properties/compare')
def compare(req:CompareRequest):
    out=[]
    for i,p in enumerate(req.properties,1):
        r=infer(p); years=max(0,req.target_year-datetime.now().year); loan=emi(r['predicted_price'],r['predicted_price']*req.down_payment_pct,req.annual_interest_rate,req.loan_years); out.append({'label':f'Property {i}','inputs':p.model_dump(),'predicted_price':r['predicted_price'],'future_value':future_value(r['predicted_price'],years,req.appreciation_rate),'price_low':r['price_low'],'price_high':r['price_high'],'monthly_emi':loan['monthly_emi'],'gross_rental_yield_pct':round(req.monthly_rent*12/r['predicted_price']*100,2),'assumptions':{'down_payment_pct':req.down_payment_pct,'annual_interest_rate':req.annual_interest_rate,'loan_years':req.loan_years,'monthly_rent':req.monthly_rent}})
    return {'properties':out,'target_year':req.target_year,'appreciation_rate':req.appreciation_rate,'best_value':'Lowest predicted price per living-area sq ft (not a quality-adjusted recommendation).','disclaimer':'Future values are assumption-based projections.'}
@app.get('/history')
def history(q:str='',limit:int=Query(200,ge=1,le=1000)):
    con=db(); rows=con.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?',(limit,)).fetchall(); con.close(); data=[]
    for x in rows:
        item={'id':x['id'],'created_at':x['created_at'],'inputs':json.loads(x['inputs']),'result':json.loads(x['result'])}
        if q and q.lower() not in json.dumps(item).lower(): continue
        data.append(item)
    return {'items':data,'total':len(data)}
@app.delete('/history/{record_id}')
def delete_history(record_id:int):
    con=db(); cur=con.execute('DELETE FROM predictions WHERE id=?',(record_id,)); con.commit(); con.close()
    if not cur.rowcount: raise HTTPException(404,'History record not found.')
    return {'deleted':record_id}
@app.get('/history/export.csv')
def export_history():
    con=db(); rows=con.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall(); con.close(); s=io.StringIO(); w=csv.writer(s); w.writerow(['id','created_at','predicted_price','price_low','price_high','inputs'])
    for x in rows:
        res=json.loads(x['result']); w.writerow([x['id'],x['created_at'],res.get('predicted_price'),res.get('price_low'),res.get('price_high'),x['inputs']])
    return StreamingResponse(iter([s.getvalue()]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=haven_history.csv'})
@app.get('/reports/prediction/{record_id}.pdf')
def prediction_pdf(record_id:int):
    try: from reportlab.lib.pagesizes import A4; from reportlab.pdfgen import canvas
    except ImportError: raise HTTPException(501,'Install reportlab to enable PDF reports.')
    con=db(); row=con.execute('SELECT * FROM predictions WHERE id=?',(record_id,)).fetchone(); con.close()
    if not row: raise HTTPException(404,'History record not found.')
    result=json.loads(row['result']); inputs=json.loads(row['inputs']); buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=A4); y=800
    lines=['HAVEN | PROPERTY VALUATION REPORT','AI-powered House Price Prediction & Real Estate Analytics','',f"Generated: {row['created_at']}",f"Predicted price: USD {result['predicted_price']:,.0f}",f"Indicative range: USD {result['price_low']:,.0f} – {result['price_high']:,.0f}",f"Model: {result['model_name']}",'','GLOBAL FEATURE IMPORTANCE (not a causal explanation)']+[f"{x['feature']}: {x['importance']:.5f}" for x in feature_importance()[:8]]+['','PROPERTY INPUTS']+[f'{k}: {v}' for k,v in inputs.items()]+['','MODEL METRICS',f"MAE: {metrics().get('mae')}",f"Log RMSE: {metrics().get('rmse_log')}",f"Log R²: {metrics().get('r2_log')}",'','Assumptions: Model estimates are not valuations or guaranteed sale prices. Future projections depend on user-selected assumptions.']
    for line in lines:
        c.drawString(48,y,line[:110]); y-=18
        if y<50: c.showPage(); y=800
    c.save(); buf.seek(0); return StreamingResponse(buf,media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename=haven_report_{record_id}.pdf'})
