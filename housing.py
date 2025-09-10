import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.special import expit as sigmoid
import matplotlib.patheffects as pe
import folium
import joblib, json, os
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

X_all.columns = (
    X_all.columns.astype(str)
    .str.replace(r"[\[\]<>]", "_", regex=True) 
    .str.replace(r"\s+", "_", regex=True)      
)


def print_reg_metrics(tag, y_true_tr, y_pred_tr, y_true_te, y_pred_te):
    print(f"[{tag}] R²  train = {r2_score(y_true_tr, y_pred_tr):.3f} | test = {r2_score(y_true_te, y_pred_te):.3f}")
    print(f"[{tag}] MAE train = {mean_absolute_error(y_true_tr, y_pred_tr):.0f} | test = {mean_absolute_error(y_true_te, y_pred_te):.0f}")
    rmse_tr = mean_squared_error(y_true_tr, y_pred_tr) ** 0.5   
    rmse_te = mean_squared_error(y_true_te, y_pred_te) ** 0.5   
    print(f"[{tag}] RMSE train = {rmse_tr:.0f} | test = {rmse_te:.0f}")


X_all = X_all.astype(np.float32)
y = gdf_pts['median_house_value'].astype(float)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_all, y, test_size=0.4, random_state=42)
X_val, X_te,  y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)  # 0.2/0.2

use_device_kw = hasattr(XGBRegressor(), "device")
xgb = XGBRegressor(
    n_estimators=1500, learning_rate=0.03, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2, reg_alpha=1,
    random_state=42,
    **({"device": "cuda"} if use_device_kw else {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": 0})
)
xgb.fit(X_tr, y_tr)

# 訓練/測試分數
y_pred_tr = xgb.predict(X_tr)
y_pred_te = xgb.predict(X_te)
print_reg_metrics("XGBR", y_tr, y_pred_tr, y_te, y_pred_te)

#-----------存模型--------------
best_params = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "random_state": 42,
}

use_device_kw = hasattr(XGBRegressor(), "device")
device_params = ({"device": "cuda"} if use_device_kw
                 else {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": 0})

xgb = XGBRegressor(**best_params, **device_params)
xgb.fit(X_tr, y_tr)



models_dir = r"c:\Users\USER\Desktop\machine_learning_project\models"
os.makedirs(models_dir, exist_ok=True)

xgb.save_model(os.path.join(models_dir, "xgb_model.json"))   
joblib.dump(xgb, os.path.join(models_dir, "xgb_model.pkl")) 


artifacts = {
    "feature_cols_base": feature_cols,                   # 你數值/連續欄位 + cluster
    "ohe_columns": list(oh.columns),                     # one-hot 後的欄位
    "all_feature_columns": list(X_all.columns),          # 真的用來訓練的欄位順序
    "kmeans": kmeans,                                    # 已 fit 的 KMeans 物件
    "crs_pts": "EPSG:4326",                              # 讓後端知道需要用 WGS84
    "best_params": best_params,                          # 留作紀錄（可有可無）
}
joblib.dump(artifacts, os.path.join(models_dir, "preprocess_artifacts.pkl"))


BUDGET=1
B = BUDGET  
yhat_val = xgb.predict(X_val).astype(float)
z_val = (y_val.astype(float) <= float(B)).astype(int)
p_emp = z_val.mean()

if p_emp == 0.0:
    def afford_prob_from_pred(pred_price):
        return np.zeros_like(pred_price, dtype=float)
elif p_emp == 1.0:
    def afford_prob_from_pred(pred_price):
        return np.ones_like(pred_price, dtype=float)
else:
    # 特徵 = (B - ŷ)，只擬合 1 維 + 截距
    feat = (B - yhat_val).reshape(-1, 1)
    lr = LogisticRegression(
         fit_intercept=True, solver="lbfgs", max_iter=1000, C=1e6
    )
    lr.fit(feat, z_val)
    a = lr.coef_[0, 0]          # 斜率 = 1/τ
    b = lr.intercept_[0]        # 截距

    def afford_prob_from_pred(pred_price):
        d = (B - pred_price).reshape(-1, 1)
        return sigmoid(a * d + b).ravel()
    
y_pred_all = xgb.predict(X_all).astype(float)
afford_prob_all = afford_prob_from_pred(y_pred_all)

gdf_pts["pred_price"]  = y_pred_all
gdf_pts["afford_prob"] = afford_prob_all


pts_with_cty = gpd.sjoin(
    gdf_pts[['pred_price','afford_prob','geometry']],
    counties[['NAME','geometry']], how='left', predicate='within'
)
agg = pts_with_cty.groupby('NAME').agg(
    median_pred_price=('pred_price','median'),
    afford_prob_mean=('afford_prob','mean'),
    n=('pred_price','size')
).reset_index()
cty_view = counties.merge(agg, on='NAME', how='left')

fig, ax = plt.subplots(figsize=(9,9))
cty_view.plot(
    ax=ax, column='afford_prob_mean', cmap='YlGnBu',
    legend=True, edgecolor='grey', linewidth=0.4,
    legend_kwds={'label':'Affordability probability', 'format':'{x:.0%}'},
    missing_kwds={'color':'lightgrey','hatch':'///','label':'No data'}
)

ax.set_title(f'Affordability probability by county (budget=${BUDGET:,})')
ax.set_axis_off()

proj='EPSG:3310'
_city_label = counties.to_crs(proj).copy()
_city_label['label_pt'] = _city_label.representative_point()
city_label = _city_label.set_geometry('label_pt').to_crs('EPSG:4326')
for _, row in city_label.iterrows():
    pt = row['label_pt']
    ax.text(
        pt.x, pt.y, row['NAME'],
        fontsize=6, color='black', weight='bold',
        path_effects=[pe.withStroke(linewidth=1, foreground="white")]
    )

result_dir = r'c:\Users\USER\Desktop\machine_learning_project\reports'
os.makedirs(result_dir, exist_ok=True)
plt.tight_layout(); plt.savefig(os.path.join(result_dir, 'xg_afford_prob_by_county.png'), dpi=200)
plt.show()

#Folium 部分

counties_wgs84 = counties.to_crs(4326)
cty_view = counties_wgs84.merge(agg, on='NAME', how='left').copy()

cty_view['afford_prob_pct'] = (cty_view['afford_prob_mean'] * 100).round(1).astype('float')

pts_wgs84 = gdf_pts.to_crs(4326)
center = [pts_wgs84.geometry.y.mean(), pts_wgs84.geometry.x.mean()]

m = folium.Map(location=center, zoom_start=6, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=cty_view.to_json(),
    data=cty_view[['NAME','afford_prob_mean']].dropna(),
    columns=['NAME','afford_prob_mean'],
    key_on='feature.properties.NAME',
    fill_color='YlGnBu',
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 明確分級，易讀
    fill_opacity=0.85,
    line_opacity=0.4,
    nan_fill_color='lightgray',
    legend_name='Affordability probability',
    name='Affordability (prob.)'
).add_to(m)

folium.GeoJson(
    cty_view,
    name='County outlines & tooltip',
    style_function=lambda f: {'fillColor':'transparent','color':'#666','weight':0.6},
    tooltip=folium.GeoJsonTooltip(
        fields=['NAME','median_pred_price','afford_prob_pct','n'],
        aliases=['County','Median Pred Price','Affordable Rate (%)','N Points'],
        localize=True,
        sticky=False
    )
).add_to(m)

folium.Choropleth(
    geo_data=cty_view.to_json(),
    data=cty_view[['NAME','median_pred_price']].dropna(),
    columns=['NAME','median_pred_price'],
    key_on='feature.properties.NAME',
    fill_color='YlOrRd',
    bins=9, 
    fill_opacity=0.85,
    line_opacity=0.4,
    nan_fill_color='lightgray',
    legend_name='Median predicted price ($)',
    name='Median price'
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

m.save(os.path.join(result_dir, "affordability_price_map.html"))

