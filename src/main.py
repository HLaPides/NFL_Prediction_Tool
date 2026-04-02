import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from backend.engine.model import load_training_data, FEATURE_COLS

df = load_training_data()
train = df[df["season"] < 2025]

X = train[FEATURE_COLS].fillna(0)
y = train["result"]

model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.01,
                     subsample=0.8, colsample_bytree=1.0, random_state=42, verbosity=0)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
print(importances.sort_values(ascending=False).to_string())
print(f"\nTotal features: {len(FEATURE_COLS)}")