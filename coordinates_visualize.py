from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
import numpy as np

# shp_path = r"C:\Users\USER\Downloads\tl_2024_us_county.zip"
# counties = gpd.read_file(shp_path)

# # 篩選加州 (STATEFP = '06')
# ca = counties[counties["STATEFP"] == "06"]

# # 輸出成 GeoJSON
# ca.to_file(r"C:\Users\USER\Desktop\california_counties.geojson", driver="GeoJSON")
# print("✅ 輸出完成：california_counties.geojson")


df=pd.read_csv('data/housing.csv')

X = df[['longitude','latitude']].copy()   
kmeans=KMeans(n_clusters=15,n_init='auto',random_state=42)
df['cluster_k15'] = kmeans.fit_predict(X)
centers=kmeans.cluster_centers_
#轉GeoDataFrame
gdf_pts=gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['longitude'],df['latitude']),
    crs='EPSG:4326'
)
counties=gpd.read_file(r"data\california_counties.geojson")
counties=counties.to_crs(gdf_pts.crs)#確保正確疊合繪圖

price=gdf_pts['median_house_value'].astype(float)
low,height=np.nanpercentile(price,[1,99])
price_clip=price.clip(low,height)

fig,ax=plt.subplots(figsize=(9,9))#fig整張圖axes單一子圖設定座標標籤圖例等
#底圖 縣界
counties.boundary.plot(ax=ax,linewidth=0.6)
norm=plt.Normalize(vmin=price_clip.min(),vmax=price_clip.max())
sc=ax.scatter(
    gdf_pts.geometry.x, gdf_pts.geometry.y,
    c=price_clip, s=6, alpha=0.6,
    cmap='viridis', norm=norm
)
cbar=plt.colorbar(sc,ax=ax,fraction=0.03,pad=0.02)
cbar.set_label('Median House Value')


    
#群中心
# ax.scatter(centers[:,0],centers[:,1],marker='+',s=120,linewidths=2,color='black')
# ax.set_xlabel('Longitude')  
# ax.set_ylabel('Latitude')
# ax.set_title('california housing - KMeans (k=15) over County Boundaries')

import os
result_dir=r"c:\Users\USER\Desktop\machine_learning_project\reports"
plt.savefig(os.path.join(result_dir,'housing_clustering.png'),dpi=200)
os.makedirs(result_dir,exist_ok=True)

plt.tight_layout()
plt.show()