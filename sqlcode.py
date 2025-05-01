
### 5. `sqlcode.py`  

#!/usr/bin/env python
import os
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# 1) Load .env
load_dotenv(find_dotenv())

# 2) Get DB creds
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("One or more DB_* environment variables missing")

# 3) Build connection URL & engine
DATABASE_URL = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
print("DEBUG: DATABASE_URL =", DATABASE_URL)
engine = create_engine(DATABASE_URL, echo=False)

# 4) Query function
def load_snapshot_data():
    sql = """
    WITH params AS (
      SELECT INTERVAL '6 months' AS horizon
    ),
    order_totals AS (
      SELECT
        o."OrderID"    AS order_id,
        o."CustomerID" AS customer_id,
        o."OrderDate"  AS order_date,
        SUM(od."UnitPrice" * od."Quantity" * (1 - od."Discount")) AS order_value
      FROM "Orders" o
      JOIN "Order Details" od
        ON o."OrderID" = od."OrderID"
      GROUP BY o."OrderID", o."CustomerID", o."OrderDate"
    ),
    snapshots AS (
      SELECT
        ot.*,
        SUM(order_value) OVER (PARTITION BY customer_id ORDER BY order_date) AS total_spent,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date)     AS total_orders,
        AVG(order_value) OVER (PARTITION BY customer_id ORDER BY order_date) AS avg_order_value,
        EXTRACT(
          DAY FROM (
            ot.order_date
            - LAG(ot.order_date) OVER (PARTITION BY customer_id ORDER BY order_date)
          )
        ) AS days_since_prev_order,
        EXTRACT(MONTH FROM order_date) AS order_month
      FROM order_totals ot
    )
    SELECT
      s.customer_id,
      s.order_id       AS snapshot_id,
      s.order_date     AS snapshot_date,
      s.total_spent,
      s.total_orders,
      s.avg_order_value,
      COALESCE(s.days_since_prev_order, 999) AS days_since_prev_order,
      s.order_month,
      CASE
        WHEN EXISTS (
          SELECT 1 FROM order_totals fut, params p
          WHERE fut.customer_id = s.customer_id
            AND fut.order_date > s.order_date
            AND fut.order_date ≤ s.order_date + p.horizon
        ) THEN 1 ELSE 0
      END AS reorder_6m
    FROM snapshots s
    ORDER BY s.customer_id, s.order_date;
    """
    return pd.read_sql_query(sql, engine, parse_dates=['snapshot_date'])

# 5) Main
def main():
    df = load_snapshot_data()
    FEATURES = ['total_spent','total_orders','avg_order_value',
                'days_since_prev_order','order_month']
    X, y = df[FEATURES].values, df['reorder_6m'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test).ravel()
    print(f"Test Acc: {acc:.3f}, ROC-AUC: {roc_auc_score(y_test,y_pred):.3f}")
    print(classification_report(y_test, (y_pred>0.5).astype(int)))

    sample = np.array([[1000,5,200,30,7]])
    p = model.predict(scaler.transform(sample))[0][0]
    print(f"6-month reorder probability: {p:.2%}")

if __name__=="__main__":
    main()
