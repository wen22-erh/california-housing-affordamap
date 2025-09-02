import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,roc_auc_score
from sklearn.cluster import KMeans
import os
import numpy as np
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
df=pd.read_csv('data/housing.csv')
gdf_pts=gpd.GeoDataFrame(
    df.copy(),
    geometry=gpd.points_from_xy(df['longitude'],df['latitude']),
    crs='EPSG:4326'
)
counties=gpd.read_file(r'data\california_counties.geojson').to_crs('EPSG:4326')

proj='EPSG:3310'
gdf_proj=gdf_pts.to_crs(proj)
coords=np.c_[gdf_proj.geometry.x.values,gdf_proj.geometry.y.values]

kmeans=KMeans(n_clusters=15,n_init='auto',random_state=42)
gdf_pts['cluster_k15']=kmeans.fit_predict(coords)

for col in ['total_bedrooms']:
    if col in gdf_pts.columns:
        gdf_pts[col] = gdf_pts[col].fillna(gdf_pts[col].median())
        
if 'ocean_proximity' in gdf_pts.columns:
    oh = pd.get_dummies(gdf_pts['ocean_proximity'], prefix='ocean', drop_first=False)
else:
    oh = pd.DataFrame(index=gdf_pts.index)
feature_cols = [
    'longitude','latitude','housing_median_age','total_rooms','total_bedrooms',
    'population','households','median_income','cluster_k15'
]
feature_cols = [c for c in feature_cols if c in gdf_pts.columns]
X_all = pd.concat([gdf_pts[feature_cols], oh], axis=1).fillna(0)
# ---- 防呆：清理欄位名，避免 XGBoost 拒收特殊字元 ----
X_all.columns = (
    X_all.columns.astype(str)
    .str.replace(r"[\[\]<>]", "_", regex=True)  # 把 [, ], <, > 換掉
    .str.replace(r"\s+", "_", regex=True)       # 空白改底線（可要可不要，但更穩）
)


def print_reg_metrics(tag, y_true_tr, y_pred_tr, y_true_te, y_pred_te):
    print(f"[{tag}] R²  train = {r2_score(y_true_tr, y_pred_tr):.3f} | test = {r2_score(y_true_te, y_pred_te):.3f}")
    print(f"[{tag}] MAE train = {mean_absolute_error(y_true_tr, y_pred_tr):.0f} | test = {mean_absolute_error(y_true_te, y_pred_te):.0f}")
    rmse_tr = mean_squared_error(y_true_tr, y_pred_tr) ** 0.5   
    rmse_te = mean_squared_error(y_true_te, y_pred_te) ** 0.5   
    print(f"[{tag}] RMSE train = {rmse_tr:.0f} | test = {rmse_te:.0f}")

# # ==== 4A. 迴歸：預測房價（畫預測房價地圖） ====
y = gdf_pts['median_house_value'].astype(float)
Xtr, Xte, ytr, yte = train_test_split(X_all, y, test_size=0.2, random_state=42)
regr = RandomForestRegressor(n_estimators=1000,max_depth=None,min_samples_leaf=10, random_state=42, n_jobs=-1)
regr.fit(Xtr, ytr)
y_pred_te = regr.predict(Xte)
y_pred_tr_rf = regr.predict(Xtr)


use_device_kw = hasattr(XGBRegressor(), "device")  # 粗暴檢查    
xgb = XGBRegressor(# R²  train = 0.859 | test = 0.817
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    reg_alpha=1,
    random_state=42,
    **({"device": "cuda"} if use_device_kw else {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": 0})
)
xgb.fit(Xtr,ytr)
xgb_tr=xgb.score(Xtr,ytr)
xgb_te=xgb.score(Xte,yte)
y_pred_tr_xgb = xgb.predict(Xtr)
y_pred_te_xgb = xgb.predict(Xte)
print_reg_metrics("XGBR", ytr, y_pred_tr_xgb, yte, y_pred_te_xgb)
print_reg_metrics("RF", ytr, y_pred_tr_rf, yte, y_pred_te)

# 可選：log 目標比較
use_log = True
yy = np.log1p(y) if use_log else y
r2_tr=regr.score(Xtr,ytr)
r2_te=regr.score(Xte,yte)
print(f'random forest train r2:{r2_tr}')
print(f'random forest test r2:{r2_te}')

# 定義 5 折
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# 做 5 折 R² 評估
scores = cross_val_score(xgb, X_all, y, cv=cv, scoring='r2')
print("每折 R²:", scores)
print("平均 R²:", np.mean(scores))


BUDGET = 100
y_cls = (y <= BUDGET).astype(int)

# 建議：類別不平衡時算一下 scale_pos_weight（負類/正類）
pos = (y_cls == 1).sum()
neg = (y_cls == 0).sum()
spw = neg / pos if pos > 0 else 1.0

Xtr, Xte, ytr, yte = train_test_split(
    X_all, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# 如果你的 XGB 版本 >= 2.0 用 device='cuda'；舊版用 tree_method='gpu_hist'
use_device_kw = hasattr(XGBClassifier(), "device")

clf = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    reg_alpha=1,
    random_state=42,
    scale_pos_weight=spw, 
    objective="binary:logistic",
    eval_metric="auc",
    **({"device": "cuda"} if use_device_kw else {"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
)

# 早停（可關，建議開）
clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

proba_te = clf.predict_proba(Xte)[:, 1]

print(f"[XGB Classify] AUC (affordable@${BUDGET:,}) = {roc_auc_score(yte, proba_te):.3f}")
result_dir=r'c:\Users\USER\Desktop\machine_learning_project\reports'
os.makedirs(result_dir,exist_ok=True)
y_pred_all = xgb.predict(X_all)
fig,ax=plt.subplots(figsize=(9,9))
counties.boundary.plot(ax=ax,linewidth=0.6,edgecolor='grey')
sc1 = ax.scatter(gdf_pts.geometry.x,gdf_pts.geometry.y,c=y_pred_all,  s=6, alpha=0.6,cmap='viridis')
cbar = plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Predicted median house value ($)')
ax.set_title('Predicted house value (XGBoost)'); ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
plt.tight_layout(); plt.savefig(os.path.join(result_dir, 'predict_price.png'), dpi=200)
# 把點掛到縣，算每縣的中位預測價與可負擔率
y_pred_all = xgb.predict(X_all)
afford_prob_all = clf.predict_proba(X_all.astype(np.float32))[:, 1]
gdf_pts['pred_price']  = y_pred_all
gdf_pts['afford_prob'] = afford_prob_all
pts_with_cty = gpd.sjoin(gdf_pts[['pred_price','afford_prob','geometry']], 
                         counties[['NAME','geometry']], how='left', predicate='within')
agg = pts_with_cty.groupby('NAME').agg(
    median_pred_price=('pred_price','median'),
    afford_prob_mean=('afford_prob','mean'),
    n=('pred_price','size')
).reset_index()

cty_view = counties.merge(agg, on='NAME', how='left')

fig, ax = plt.subplots(figsize=(9,9))
cty_view.plot(ax=ax, column='afford_prob_mean', cmap='YlGnBu',
              legend=True, edgecolor='grey', linewidth=0.4,
              missing_kwds={'color':'lightgrey','hatch':'///','label':'No data'})
ax.set_title(f'Affordability probability by county (budget=${BUDGET:,})')
ax.set_axis_off()
plt.tight_layout(); plt.savefig(os.path.join(result_dir, 'xg_afford_prob_by_county.png'), dpi=200)

