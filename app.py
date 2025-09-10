import os
import time
import traceback
import threading
import logging
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from backend.preprocess import Preprocessor
import geopandas as gpd
import folium


# 環境設定
ENV: str = os.getenv("ENV", "dev").lower()  # dev / prod / staging ...
ADMIN_TOKEN: Optional[str] = os.getenv("ADMIN_TOKEN")
SHOW_TRACE: bool = (ENV == "dev")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("housing-api")

# 路徑
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# FastAPI + CORS
app = FastAPI(title="Housing Affordability API", version="0.1.0")

# CORS：dev 全開；prod 收斂到環境變數 CORS_ORIGINS
if ENV == "dev":
    allow_origins = ["*", "null", "http://127.0.0.1:8010", "http://localhost:8010"]
else:
    cors_env = os.getenv("CORS_ORIGINS", "")
    allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()] or ["https://your-frontend.example"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Housing Affordability API</h2>
    <ul>
      <li><a href="/docs">/docs (Swagger UI)</a></li>
      <li><a href="/ping">/ping</a></li>
      <li><a href="/health">/health</a></li>
      <li><a href="/map?budget=120000">/map?budget=120000</a></li>
    </ul>
    """


# 全域狀態 & 鎖
xgb = None
pp  = None
model_error: str | None = None

# 地圖 / 校準用全域
GDF_POINTS = None
COUNTIES_WGS84 = None
POINT2COUNTY = None
Y_TRUE_ALL = None
YHAT_ALL = None
VAL_IDX = None

# 併發鎖：確保載入/重載一致性
_load_lock = threading.Lock()


# 初始化與載入
def _try_load_model():
    """嘗試載入模型與前處理；失敗時把完整 traceback 存進 model_error。"""
    global xgb, pp, model_error
    try:
        model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
        art_path   = os.path.join(MODELS_DIR, "preprocess_artifacts.pkl")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.isfile(art_path):
            raise FileNotFoundError(f"Artifacts file not found: {art_path}")

        xgb = joblib.load(model_path)
        pp  = Preprocessor(art_path)
        model_error = None
        log.info("[startup] Model & artifacts loaded OK.")
    except Exception:
        model_error = traceback.format_exc()
        xgb = None
        pp  = None
        log.error("[startup] FAILED to load model/artifacts:\n%s", model_error)

# 路徑基底
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # ...\machine_learning_project
DATA_DIR_CAND = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(ROOT_DIR, "data"),
]

def _pick_file(*relpath):
    """從候選資料夾中挑第一個存在的檔案"""
    for d in DATA_DIR_CAND:
        p = os.path.join(d, *relpath)
        if os.path.isfile(p):
            return p
    return None

def _prepare_map_data():
    """
    1) 讀 housing.csv、縣界
    2) 照 artifacts/Preprocessor 的規則產出 X_all（欄位順序 = pp.all_cols）
    3) 取得 YHAT_ALL、Y_TRUE_ALL
    4) 固定 random_state=42 切出驗證集索引 VAL_IDX（與 housing.py 一致）
    5) 建立 GDF_POINTS / COUNTIES_WGS84 / POINT2COUNTY
    """
    global GDF_POINTS, COUNTIES_WGS84, POINT2COUNTY, Y_TRUE_ALL, YHAT_ALL, VAL_IDX

    if xgb is None or pp is None:
        log.warning("[startup] model not ready -> skip map data.")
        return

    csv_path = _pick_file("housing.csv")
    geo_path = _pick_file("california_counties.geojson")
    if not csv_path:
        raise FileNotFoundError(f"housing.csv not found in: {DATA_DIR_CAND}")
    if not geo_path:
        raise FileNotFoundError(f"california_counties.geojson not found in: {DATA_DIR_CAND}")

    df = pd.read_csv(csv_path)
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )


    proj = "EPSG:3310"
    gdf_proj = gdf.to_crs(proj)
    coords = np.c_[gdf_proj.geometry.x.values, gdf_proj.geometry.y.values]
    if getattr(pp, "kmeans", None) is not None:
        gdf["cluster_k15"] = pp.kmeans.predict(coords).astype("float32")
    else:
        gdf["cluster_k15"] = 0.0

    # one-hot（與 artifacts 對齊）
    ohe_cols = getattr(pp, "ohe_cols", None)
    if ohe_cols is None:
        ohe_cols = getattr(pp, "ohe_columns", [])
    if "ocean_proximity" in gdf.columns:
        oh = pd.get_dummies(gdf["ocean_proximity"], prefix="ocean", drop_first=False)
    else:
        oh = pd.DataFrame(index=gdf.index)

    # 以訓練欄位為準，建全 0 設計矩陣，再逐欄覆寫
    all_cols = list(dict.fromkeys(getattr(pp, "all_cols", [])))
    if not all_cols:
        raise RuntimeError("Preprocessor 沒有 all_cols，請檢查 artifacts/Preprocessor")
    X_all = pd.DataFrame(0, index=gdf.index, columns=all_cols, dtype="float32")

    # 數值/連續欄位（含 cluster）
    base_cols = getattr(pp, "feature_cols_base", [])
    for c in list(base_cols) + ["cluster_k15"]:
        if c in gdf.columns and c in X_all.columns:
            X_all[c] = gdf[c].astype("float32")

    # one-hot 欄位（只覆蓋訓練時存在者）
    for c in ohe_cols:
        if c in oh.columns and c in X_all.columns:
            X_all[c] = oh[c].astype("float32")

    # 5) 一次性推論所有點
    YHAT_ALL = xgb.predict(X_all).astype(float)
    Y_TRUE_ALL = df["median_house_value"].astype(float).values

    # 6) 產生驗證集索引
    n = len(Y_TRUE_ALL)
    idx_all = np.arange(n)
    tr_idx, tmp_idx, _, _ = train_test_split(idx_all, idx_all, test_size=0.4, random_state=42)
    val_idx, te_idx, _, _ = train_test_split(tmp_idx, tmp_idx, test_size=0.5, random_state=42)
    VAL_IDX = np.array(val_idx)

    # 7) Geo 與縣名對應
    GDF_POINTS = gpd.GeoDataFrame(
        pd.DataFrame({"pred_price": YHAT_ALL}),
        geometry=gdf.geometry, crs="EPSG:4326"
    )
    COUNTIES_WGS84 = gpd.read_file(geo_path).to_crs(4326)
    joined = gpd.sjoin(
        GDF_POINTS, COUNTIES_WGS84[["NAME", "geometry"]],
        how="left", predicate="within"
    )
    POINT2COUNTY = joined["NAME"].values

    log.info("[startup] map data ready: %d points, %d counties", len(GDF_POINTS), COUNTIES_WGS84.shape[0])

def _afford_prob_from_budget_calibrated(B: float):
    """
    校準流程（與 housing.py 一致）
    """
    y_true_val = Y_TRUE_ALL[VAL_IDX].astype(float)
    yhat_val   = YHAT_ALL[VAL_IDX].astype(float)

    z_val = (y_true_val <= float(B)).astype(int)
    p_emp = z_val.mean()

    if p_emp == 0.0:
        return np.zeros_like(YHAT_ALL, dtype=float)
    if p_emp == 1.0:
        return np.ones_like(YHAT_ALL, dtype=float)

    feat = (float(B) - yhat_val).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=True, solver="lbfgs", max_iter=1000, C=1e6)
    lr.fit(feat, z_val)
    a = lr.coef_[0, 0]
    b = lr.intercept_[0]

    from scipy.special import expit as sigmoid
    d = (float(B) - YHAT_ALL).reshape(-1, 1)
    return sigmoid(a * d + b).ravel()

def _make_map_html(budget: float) -> str:
    p = _afford_prob_from_budget_calibrated(budget)

    # 聚合到縣
    df_pts = pd.DataFrame({
        "county": POINT2COUNTY,
        "pred_price": GDF_POINTS["pred_price"].values,
        "p": p
    })
    agg = df_pts.groupby("county").agg(
        afford_prob_mean=("p", "mean"),
        median_pred_price=("pred_price", "median"),
        n=("p", "size")
    ).reset_index()

    cty_view = COUNTIES_WGS84.merge(agg, left_on="NAME", right_on="county", how="left")
    cty_view["afford_prob_pct"] = (cty_view["afford_prob_mean"] * 100).round(1)

    # Folium 地圖
    center = [GDF_POINTS.geometry.y.mean(), GDF_POINTS.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=cty_view.to_json(),
        data=cty_view[["NAME", "afford_prob_mean"]].dropna(),
        columns=["NAME", "afford_prob_mean"],
        key_on="feature.properties.NAME",
        fill_color="YlGnBu",
        bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        fill_opacity=0.85, line_opacity=0.4,
        nan_fill_color="lightgray",
        legend_name=f"Affordability probability (budget=${int(budget):,})",
        name="Affordability (prob.)"
    ).add_to(m)

    folium.GeoJson(
        cty_view,
        name="County outlines & tooltip",
        style_function=lambda f: {"fillColor": "transparent", "color": "#666", "weight": 0.6},
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME", "median_pred_price", "afford_prob_pct", "n"],
            aliases=["County", "Median Pred Price", "Affordable Rate (%)", "N Points"],
            localize=True, sticky=False
        )
    ).add_to(m)

    folium.Choropleth(
        geo_data=cty_view.to_json(),
        data=cty_view[["NAME", "median_pred_price"]].dropna(),
        columns=["NAME", "median_pred_price"],
        key_on="feature.properties.NAME",
        fill_color="YlOrRd",
        bins=9, fill_opacity=0.85, line_opacity=0.4,
        nan_fill_color="lightgray",
        legend_name="Median predicted price ($)",
        name="Median price"
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()


# Startup
@app.on_event("startup")
def _load_all():
    with _load_lock:
        _try_load_model()
        if model_error:
            log.warning("[startup] model error -> skip map data.")
        else:
            _prepare_map_data()

# Endpoints
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()
@app.get("/ping")
def ping():
    return {"ok": True, "ts": time.time()}

@app.get("/health")
def health():
    body = {
        "ok": model_error is None,
        "model_loaded": (xgb is not None) and (pp is not None),
    }
    if SHOW_TRACE:
        body["paths"] = {
            "model": os.path.join(MODELS_DIR, "xgb_model.pkl"),
            "artifacts": os.path.join(MODELS_DIR, "preprocess_artifacts.pkl"),
        }
        body["error"] = model_error
    else:
        # 生產不外洩堆疊
        body["error"] = None if model_error is None else "model load failed"
    return body

def _require_admin(x_admin_token: Optional[str]):
    """
    dev：若未設定 ADMIN_TOKEN，放行（方便本機調試）
    prod：必須設 ADMIN_TOKEN，且 header token 必須相符
    """
    if ENV == "dev":
        return
    # 非 dev（如 prod/staging）
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="admin token not configured")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

@app.post("/debug/reload")
def debug_reload(x_admin_token: Optional[str] = Header(default=None)):
    _require_admin(x_admin_token)
    with _load_lock:
        _try_load_model()
        if not model_error and (xgb is not None and pp is not None):
            try:
                _prepare_map_data()
            except Exception as e:
                # 這裡不要把堆疊外洩給客戶端
                log.exception("map data prepare failed after reload")
                if SHOW_TRACE:
                    raise HTTPException(status_code=500, detail=str(e))
                else:
                    raise HTTPException(status_code=500, detail="map data prepare failed")
    if model_error:
        # dev 回完整；prod 回簡訊息
        if SHOW_TRACE:
            raise HTTPException(status_code=500, detail=model_error)
        raise HTTPException(status_code=500, detail="model reload failed")
    return {"reloaded": True}

class RecommendIn(BaseModel):
    budget: float = Field(..., ge=0)
    usage: str = "中"

class RecommendOut(BaseModel):
    pred_price: float
    affordable: bool
    plan: str

@app.post("/recommend", response_model=RecommendOut)
def recommend(inp: RecommendIn):
    if model_error:
        raise HTTPException(status_code=503, detail=("Model not ready: " + model_error) if SHOW_TRACE else "Model not ready")
    if xgb is None or pp is None:
        raise HTTPException(status_code=503, detail="Model loading...")

    try:
        # 最簡單的 raw 特徵
        raw = {
            "housing_median_age": 15.0,
            "total_rooms": 3000.0,
            "total_bedrooms": 500.0,
            "population": 1000.0,
            "households": 300.0,
            "median_income": 5.0,
            "longitude": -119.0,
            "latitude": 36.5,
            "ocean_proximity": "<1H OCEAN",
        }
        row = pp.build_feature_row(raw)
        price = float(xgb.predict(row)[0])
        plan = "Basic" if price <= inp.budget else ("Pro" if price <= 1.2 * inp.budget else "Enterprise")
        return RecommendOut(pred_price=price, affordable=price <= inp.budget, plan=plan)
    except Exception as e:
        log.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(e) if SHOW_TRACE else "internal error")

@app.get("/map", response_class=HTMLResponse)
def map_html(budget: float):
    """
    例：GET /map?budget=120000
    畫出「依照 budget 校準後」的可負擔機率地圖
    """
    if (xgb is None or pp is None) or (GDF_POINTS is None or COUNTIES_WGS84 is None):
        return HTMLResponse("<h3>Model or map data not ready.</h3>", status_code=503)
    try:
        return HTMLResponse(_make_map_html(float(budget)), media_type="text/html")
    except Exception as e:
        log.exception("/map failed")
        msg = str(e) if SHOW_TRACE else "map failed"
        return HTMLResponse(f"<h3>{msg}</h3>", status_code=500)
