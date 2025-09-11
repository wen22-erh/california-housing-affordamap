# 房價預算估算器 
一個可以幫你判斷「在加州，這筆預算能買到多少房子」的互動式網站。  
輸入你的 **Budget**，就能即時看到：
- 預測房價
- 是否可負擔
- 一張縣市可負擔率地圖

---

## Demo
![demo screenshot](docs/demo.png)

---

## 功能
- 即時輸入預算 → 估算房價 & 可負擔性
- 一鍵切換地圖 → 顯示各縣市可負擔率
- 提供健康檢查 (`/ping`, `/health`) → 方便部署監控

---

## 安裝方式
git clone https://github.com/yourname/housing-estimator.git
cd housing-estimator
pip install -r requirements.txt

## 執行
uvicorn app:app --reload
開啟瀏覽器 → http://127.0.0.1:8000/
就能看到介面

## Dataset
我使用 Kaggle 上的 [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) 資料集。  
該資料集包含加州不同地區的房價與人口、收入、地理資訊等特徵，適合用來做回歸與房價預測相關的實驗。

## Model Performance (XGBoost Regressor)
- **R²**: 0.861 (train) | 0.821 (test)
- **MAE**: 29,858 (train) | 33,440 (test)
- **RMSE**: 43,011 (train) | 49,487 (test)

## 履歷描述
「先用回歸預測房價，再把(預測房價-驗證集的差值)丟進 Logistic Regression，學出最適合的 sigmoid 曲線」。
XGB 回歸 → 預測房價 

驗證集上算差值 

LogisticRegression 校準 
驗證集上的差值 與「真實是否買得起」的關係

Sigmoid 得到最終機率

再使用folium將資料視覺化

## 技術
FastAPI
XGBoost
Folium + GeoPandas
HTML / JS (簡單前端)

## 未來的方向
能夠支援台灣地區的資料
使用React寫一個更豐富的前端圖表

## 方法流程
1. **輸入特徵**  
   - 使用房屋特徵作為模型輸入

2. **回歸預測**  
   - 使用 `XGBRegressor` 預測房價

3. **分類嘗試（已棄用）**  
   - 透過「預算 vs. 實際價格」切分標籤 → `XGBClassifier`  
   - 可計算 ROC-AUC，但缺點是：  
     - 不同預算需 **重新訓練**  
     - 遇到極端預算（太低/太高）會報錯  
   - → **已棄用**

4. **現行流程：機率校準**  
   - 使用 `XGBRegressor` 輸出預測價格 \(\hat{y}\)  
   - 與使用者輸入的 Budget 比較，轉換為二元標籤（是否 ≤ 預算）  
   - 以 `LogisticRegression` 校準 → Sigmoid 函數 → 輸出**可負擔機率** \(p \in [0,1]\)  
   - 將結果彙整後，可視化為各縣市的**可負擔率地圖**

---

## 系統架構流程
1. **輸入**：使用者於前端輸入Budget  
2. **傳輸**：前端呼叫 FastAPI 後端  

### API 1: `POST /recommend` → 回傳 JSON
- **流程**：
  - `XGBoost Regressor` 預測單筆房價
  - `Logistic Regression` 校準計算可負擔性
- **回傳內容**：
  - 預測價格  
  - 是否可負擔  

### API 2: `GET /map?budget` → 回傳 HTML
- **流程**：
  - `XGBoost Regressor` 對整份資料集批次推論  
  - 使用 `GeoPandas` + `Folium` 進行縣市聚合  
  - 產生 Choropleth 地圖（可負擔率熱力圖）  

---

## 可視化範例
- **房價預測結果**：單筆 JSON 回傳  
- **可負擔地圖**：地圖上顯示各縣市的可負擔率  

---
