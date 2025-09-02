# machine_learning_project
機器學習的基礎技巧練習
所以我最後的成果可以是我依他的經緯度判斷牠的房價 之後對每個經緯度做map對應各個城市
地理特徵工程（強烈建議）：到海岸距離、到 LA/SF 中心距離、人口密度（population/households/cluster 面積）。
Problem：預測房價與可負擔區域，支援預算導向找區塊。

Data：California housing（欄位：lat/lon, income, rooms…），County GeoJSON。

Method：EPSG:3310 投影 → KMeans 空間特徵 → RF 回歸/分類 → sjoin → Choropleth。

Metrics：R²（price）、AUC（afford），含 baseline 對比與空間 CV。

Results：兩張核心圖 + SHAP；重點發現 3 點（例：灣區高價、中央谷地較可負擔…）。

Repro：python scripts/train.py、python scripts/plot_maps.py、環境需求。

+
Limitations：時間切片缺乏、特徵有限、KMeans 非鄰接、機率經校準後使用等。

Future Work：XGBoost/LightGBM、H3 網格、時序/鄰接特徵、交互式地圖。
模型比較表 (Linear vs RF vs XGBoost)

Feature Importance 圖表

地圖視覺化結果

GitHub 專案

notebooks/ → EDA + Modeling

src/ → Python 模型程式碼

app/ → Streamlit/FastAPI 小 Demo

履歷描述

Built a geospatial machine learning to predict housing prices.
Engineered location-based features (geo-clusters, distance to coast, household ratios), improving model RMSE by XX%.
Deployed an interactive dashboard visualizing predicted prices on a map using Folium.