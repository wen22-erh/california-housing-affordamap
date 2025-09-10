import pandas as pd
import numpy as np
import joblib
import geopandas as gpd

class Preprocessor:
    def __init__(self, art_path):
        art = joblib.load(art_path)
        self.all_cols = art["all_feature_columns"]
        self.kmeans = art.get("kmeans", None)
        self.crs_pts = art.get("crs_pts", "EPSG:4326")
        self.feature_cols_base = art.get("feature_cols_base", [])
        self.ohe_cols = art.get("ohe_columns", [])
        self.cluster_proj = "EPSG:3310"

    def build_feature_row(self, raw: dict) -> pd.DataFrame:
        df = pd.DataFrame([raw])

        if {"longitude","latitude"}.issubset(df.columns) and self.kmeans is not None:
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326"
            ).to_crs(self.cluster_proj)
            coords = np.c_[gdf.geometry.x.values, gdf.geometry.y.values]
            df["cluster_k15"] = self.kmeans.predict(coords).astype(np.float32)

        if "ocean_proximity" in df.columns and len(self.ohe_cols):
            oh = pd.get_dummies(df["ocean_proximity"], prefix="ocean", drop_first=False)
            df = pd.concat([df, oh], axis=1)

        cols = list(set(self.feature_cols_base + self.ohe_cols + ["cluster_k15"]))
        row = df.reindex(columns=cols, fill_value=0).astype(np.float32)

        row = row.reindex(columns=self.all_cols, fill_value=0).astype(np.float32)
        return row
