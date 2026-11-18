"""Train and compare multiple regression models; log experiments to MLflow."""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.inference import add_features, compute_baseline_defaults
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"/"train.csv"; MODEL_DIR=ROOT/"model"; OUT=ROOT/"outputs"

def build_preprocessor(frame):
    num=frame.select_dtypes(include="number").columns.tolist(); cat=frame.select_dtypes(exclude="number").columns.tolist()
    numeric=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())])
    categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2))])
    return ColumnTransformer([("numeric",numeric,num),("categorical",categorical,cat)])

def candidates(args):
    return [
      ("GradientBoosting",{"n_estimators":args.n_estimators,"learning_rate":0.04,"max_depth":2,"loss":"huber","random_state":42},GradientBoostingRegressor(n_estimators=args.n_estimators,learning_rate=0.04,max_depth=2,loss="huber",random_state=42)),
      ("GradientBoosting",{"n_estimators":args.n_estimators,"learning_rate":0.035,"max_depth":3,"loss":"huber","random_state":42},GradientBoostingRegressor(n_estimators=args.n_estimators,learning_rate=0.035,max_depth=3,loss="huber",random_state=42)),
      ("RandomForest",{"n_estimators":args.rf_trees,"max_depth":None,"min_samples_leaf":2,"random_state":42},RandomForestRegressor(n_estimators=args.rf_trees,max_depth=None,min_samples_leaf=2,n_jobs=-1,random_state=42)),
      ("ExtraTrees",{"n_estimators":args.rf_trees,"min_samples_leaf":2,"random_state":42},ExtraTreesRegressor(n_estimators=args.rf_trees,min_samples_leaf=2,n_jobs=-1,random_state=42)),
    ]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n_estimators',type=int,default=250);ap.add_argument('--rf_trees',type=int,default=300);args=ap.parse_args()
    if not DATA.exists(): raise FileNotFoundError(f"Dataset missing: {DATA}")
    raw=pd.read_csv(DATA)
    if len(raw)<1000 or 'SalePrice' not in raw: raise ValueError('Expected >=1,000 training rows and SalePrice target.')
    y=np.log1p(raw.SalePrice.astype(float)); X=add_features(raw.drop(columns=['Id','SalePrice']))
    Xtr,Xval,ytr,yval=train_test_split(X,y,test_size=.2,random_state=42)
    try:
      import mlflow
      mlflow.set_tracking_uri(f"sqlite:///{(ROOT/'mlflow.db').as_posix()}");mlflow.set_experiment('Haven House Price Regression')
    except ImportError:
      mlflow=None;print('MLflow is not installed in this environment; metrics will still be saved. Install requirements.txt for experiment tracking.')
    rows=[]
    for name,params,est in candidates(args):
      pre=build_preprocessor(Xtr); A=pre.fit_transform(Xtr); B=pre.transform(Xval); est.fit(A,ytr); pred=est.predict(B)
      rmse=float(np.sqrt(mean_squared_error(yval,pred))); mae=float(mean_absolute_error(np.expm1(yval),np.expm1(pred)));r2=float(r2_score(yval,pred))
      rows.append({'algorithm':name,'params':params,'rmse_log':rmse,'mae_price':mae,'r2_log':r2})
      if mlflow:
       with mlflow.start_run(run_name=f'{name}_{len(rows)}'):
        mlflow.log_param('algorithm',name);mlflow.log_params(params);mlflow.log_metrics({'rmse_log':rmse,'mae_price':mae,'r2_log':r2})
      print(f"{name} run {len(rows)} | RMSE(log)={rmse:.5f} | MAE(price)={mae:,.0f} | R2(log)={r2:.4f}")
    results=pd.DataFrame([{k:v for k,v in r.items() if k!='params'} for r in rows]).sort_values('rmse_log')
    OUT.mkdir(exist_ok=True);MODEL_DIR.mkdir(exist_ok=True);results.to_csv(OUT/'mlflow_model_comparison.csv',index=False)
    best= min(rows,key=lambda r:r['rmse_log']); best_est=None
    for name,params,est in candidates(args):
      if name==best['algorithm'] and params==best['params']: best_est=est;break
    Xall=add_features(raw.drop(columns=['Id','SalePrice'])); prep=build_preprocessor(Xall); Z=prep.fit_transform(Xall);best_est.fit(Z,y)
    bundle={'preprocessor':prep,'model':best_est,'model_name':best['algorithm'],'feature_columns':Xall.columns.tolist(),'baseline_defaults':compute_baseline_defaults(raw),'validation_rmse_log':best['rmse_log'],'validation_mae_price':best['mae_price'],'validation_r2_log':best['r2_log'],'training_rows':len(raw),'trained_at':pd.Timestamp.utcnow().isoformat()}
    joblib.dump(bundle,MODEL_DIR/'model.joblib',compress=3)
    print('\nBest model:',best['algorithm'],'| validation RMSE(log):',round(best['rmse_log'],5));print('Saved:',MODEL_DIR/'model.joblib')
if __name__=='__main__':main()
