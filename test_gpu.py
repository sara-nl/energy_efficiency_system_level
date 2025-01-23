import xgboost as xgb
import numpy as np



print(xgb.build_info())


params = {"device": "cuda"}
# Create a small dummy dataset
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.rand(100)      # 100 target values

# Convert to DMatrix
dtrain = xgb.DMatrix(data=X, label=y)

# Try training a simple model
model = xgb.train(params, dtrain)




# import lightgbm as lgb
# import numpy as np

# # Check LightGBM version
# print(f"LightGBM version: {lgb.__version__}")

# params = {"device": "cuda"}  # Use GPU for training
# # Create a small dummy dataset
# X = np.random.rand(100, 10)  # 100 samples, 10 features
# y = np.random.rand(100)      # 100 target values

# # Convert to LightGBM Dataset
# dtrain = lgb.Dataset(data=X, label=y)

# # Train a simple model
# model = lgb.train(params, dtrain)
