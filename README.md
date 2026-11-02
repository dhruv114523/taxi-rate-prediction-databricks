# NYC Taxi Fare Prediction on Databricks

An end-to-end fare prediction pipeline built on Databricks: raw trip data is loaded into a Delta table, cleaned and feature-engineered with **PySpark** across ~54M records, explored with **Spark SQL**, and used to train an **XGBoost** regression model tracked with **MLflow**.

This is a personal project. It rebuilds an earlier local pandas/XGBoost solution on Databricks to get hands-on with Spark, Delta Lake and MLflow.

## Results

| Metric | Train | Validation |
|---|---|---|
| MAE (USD) | 2.034 | **2.047** |
| R² | 0.817 | **0.807** |

- Train and validation scores are close, so the model is not overfitting.
- The model was trained on a 10% sample (~5.4M rows, split 80/20). The earlier local run on the full dataset reached an MAE of about $2.04, so sampling cost very little accuracy.

## Pipeline

```
train.csv (Kaggle)
   |  converted locally to Parquet (chunked, selected columns)
   v
Parquet file in a Unity Catalog volume
   |  spark.read.parquet -> Delta
   v
Bronze Delta table   (raw trips, untouched)
   |  PySpark cleaning + feature engineering
   v
Silver Delta table   (54,026,339 rows)
   |  Spark SQL exploration
   |  10% sample -> pandas
   v
XGBoost regressor + MLflow tracking
```

### 1. Bronze table
The raw Parquet file is loaded into a Delta table with no changes. Keeping raw data separate means cleaning rules can be changed and re-run without re-uploading anything.

### 2. Cleaning (PySpark)
- Coordinates and fares cast to `double` so trigonometry keeps its precision
- NYC bounding box filter on pickup location (lon -74.3 to -73.6, lat 40.4 to 41.0)
- Rows with nulls in any critical column dropped
- Fares of zero or below removed
- Outlier filters: fare between $2.50 and $200, haversine distance 0 to 100 km, 1 to 6 passengers

### 3. Feature engineering (PySpark)
The pandas/NumPy helper functions were rewritten with `pyspark.sql.functions` so they run distributed.

| Feature | Description |
|---|---|
| `haversine_km` | Great-circle distance between pickup and dropoff |
| `manhattan_km` | Grid-style distance (north-south leg plus east-west leg) |
| `bearing_deg` | Compass direction of the trip |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour of day |
| `day_sin`, `day_cos` | Cyclical encoding of day of week |
| `is_weekend` | Saturday/Sunday flag |
| `month` | Month of year |
| `passenger_count` | Passengers per trip |

Spark numbers weekdays Sun=1 to Sat=7, while pandas uses Mon=0 to Sun=6. The notebook converts to the pandas convention so results match the original local code.

### 4. Spark SQL exploration
Two queries run on the silver table:
- Average fare and trip count by hour of day
- Average fare and distance, weekday vs weekend

Findings (hours are in UTC):
- Trip volume is lowest around 05:00 UTC (~528K trips) and highest in the 18:00 to 21:00 UTC range (over 3M trips per hour).
- Average fare peaks at about $15.14 at 05:00 UTC, the lowest-volume hour. The cheapest hours are around $10.55 to $10.90.
- Weekend and weekday fares are nearly the same ($11.27 vs $11.34). Weekend trips are slightly longer (3.43 km vs 3.27 km on average).

### 5. Modeling
- Features: `haversine_km`, `manhattan_km`, `bearing_deg`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `is_weekend`, `passenger_count`, `month`
- Target: `fare_amount`
- A 10% sample of the silver table (seed 123) is pulled into pandas, then split 80/20 (4,320,314 train rows, 1,080,079 validation rows)
- Model: `XGBRegressor` (`hist` tree method, CPU)

| Parameter | Value |
|---|---|
| `n_estimators` | 600 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `random_state` | 123 |

Parameters, metrics and the trained model are logged with MLflow (run name `xgb_taxi_sample`).

## Tech stack
Databricks, PySpark, Spark SQL, Delta Lake, Unity Catalog, MLflow, XGBoost, scikit-learn, pandas, NumPy

## Repo contents
- `nyc_taxi_databricks.ipynb`: the full Databricks notebook (cleaning, features, SQL, training)
- `README.md`: this file

## Data
The data comes from the Kaggle competition [New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) (`train.csv`, about 55M rows). It is not included in this repo because of its size. To reproduce:

1. Download `train.csv` from Kaggle.
2. Convert it to Parquet, keeping only the columns used here (`fare_amount`, `pickup_datetime`, pickup and dropoff longitude/latitude, `passenger_count`). Parquet keeps the file under the 2 GB browser upload limit.
3. Upload the Parquet file to a Unity Catalog volume and load it into a Delta table.
4. Update the `BRONZE` and `SILVER` table names at the top of the notebook, then run the cells in order.

## Limitations and next steps
- The pickup location is bounded to NYC, but the dropoff location is not. The 0 to 100 km distance filter catches most bad dropoffs, but a dropoff bounding box would be stricter.
- Training used a 10% sample to fit within Databricks Free Edition memory limits. A larger cluster or distributed XGBoost could train on the full dataset.
- Possible improvements: hyperparameter tuning, airport and landmark proximity features, and a gradient boosting comparison with LightGBM.
- Hour-of-day analysis is in UTC. Converting to New York local time would make the hourly patterns easier to interpret.
