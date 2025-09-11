# 房價預算估算器 
一個可以幫你判斷「在加州，這筆預算能買到多少房子」的互動式網站。  
輸入你的 **Budget**，就能即時看到：
- 預測房價
- 是否可負擔
- 一張縣市可負擔率地圖

---

## Demo
![demo screenshot](doc/demo.png)

---

## 功能
- 即時輸入預算 → 估算房價 & 可負擔性
- 一鍵切換地圖 → 顯示各縣市可負擔率
- 提供健康檢查 (`/ping`, `/health`) → 方便部署監控

---

# 安裝方式
```bash
git clone https://github.com/yourname/housing-estimator.git
cd housing-estimator
pip install -r requirements.txt
```

## 執行
```bash
uvicorn app:app --reload
```
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

## 方法流程圖
```mermaid
flowchart TD
    A[輸入房屋特徵] --> B[XGBRegressor 預測房價]

    B -.嘗試過.-> C1[切分標籤:把預算和原資料集比較]
    C1 --> D1[XGBClassifier 訓練分類模型]
    D1 --> E1{極端預算?}
    E1 -->|是| F1[若預算太低或太高 → 報錯]
    E1 -->|否| G1[可輸出 ROC-AUC，但需每次重訓]
    F1 --> H1[棄用]

    B --> C2[(使用訓練完的 XGBRegressor 訓練驗證集)]
    C2 --> D2[LogisticRegression 校準]
    D2 --> E2[Sigmoid 函數]
    E2 --> F2["可負擔機率 (0~1)"]
    F2 --> G2[地圖可視化: 預估各縣可負擔率]

    style H1 fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    style F2 fill:#ccffcc,stroke:#00aa00,stroke-width:2px
```

```mermaid
flowchart LR
    A[使用者輸入 Budget] --> B[Frontend (HTML/JS)]
    B --> C[FastAPI Backend]
    C --> D1[POST /recommend<br/>回傳 JSON]
    C --> D2[GET /map?budget<br/>回傳地圖 HTML]

    D1 --> E1[XGBoost Regressor<br/>預測房價]
    E1 --> F1[Logistic Regression 校準<br/>計算可負擔性]
    F1 --> G1[回傳結果: 預測價格 / 是否可負擔 / Plan]

    D2 --> E2[XGBoost Regressor<br/>批次推論]
    E2 --> F2[GeoPandas + Folium<br/>縣市聚合]
    F2 --> G2[產生 Choropleth 地圖]

    style F2 fill:#ccffcc,stroke:#00aa00,stroke-width:2px
```