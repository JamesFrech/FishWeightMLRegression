import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')
data.drop('Species',axis=1,inplace=True) # Dont use species

# Select model inputs
target=['Weight']
inputs=['Length1','Length2','Length3','Height','Width']

# Split the train/test data
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

param_grid = {'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
              'min_samples_leaf':[2,3,4,5,10,15],
              'max_features':[i+1 for i in range(len(inputs))]}

# Initialize random forest classifier
rf = RandomForestRegressor()

# Initialize Kfold object with 5 folds
kfold = KFold(5,
              random_state=0,
              shuffle=True)

# Get the grid of hyperparameters for the model
grid = RandomizedSearchCV(rf,
                          param_grid,
                          refit=True,
                          cv=kfold,
                          n_iter = 100,
                          verbose = 5,
                          scoring='neg_root_mean_squared_error')

# Fit the model
grid.fit(X_train, y_train.values.ravel())

# Give best parameters and compare to others
print(grid.best_params_)
print(grid.cv_results_[('mean_test_score')])

# Fit to testing data
best_rf = grid.best_estimator_

# Predict classes
train_pred = best_rf.predict(X_train)
test_pred = best_rf.predict(X_test)

train_rmse=np.sqrt(np.mean((train_pred-y_train.values.squeeze())**2))
test_rmse=np.sqrt(np.mean((test_pred-y_test.values.squeeze())**2))

print('Train RMSE:',train_rmse)
print('Test RMSE:',test_rmse)
print('R2:',best_rf.score(X_test,y_test.values))

metrics = pd.DataFrame([['RandomForest',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/RandomForest.csv',index=False)
