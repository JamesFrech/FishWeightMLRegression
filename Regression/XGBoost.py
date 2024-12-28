import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
import xgboost as xgb

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')
data.drop('Species',axis=1,inplace=True) # Dont use species

# Select model inputs
target=['Weight']
inputs=['Length1','Length2','Length3','Height','Width']

# Split the train/test data
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# Initialize random forest classifier
xgb_forest = xgb.XGBRegressor(importance_type="gain")

# Initialize Kfold object with 5 folds
kfold = KFold(5,
              random_state=0,
              shuffle=True)

# Get the grid of hyperparameters for the model
n_estimators = [400, 500, 600, 700, 900, 1000, 1100,]
max_depth = [2, 3, 4, 5, 10, 15]
learning_rate=[0.01, 0.02, 0.03, 0.05,0.1]
min_child_weight=[1,2,3,4,5,6]
gamma=[0.5, 0.75, 1, 1.25, 1.5]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'objective':["reg:squarederror"],
    'gamma':gamma,
    }

# Initialize random search CV
random_cv = RandomizedSearchCV(estimator=xgb_forest,
            param_distributions=hyperparameter_grid,
            cv=5,
            n_iter=100,
            scoring = 'neg_root_mean_squared_error',
            verbose = 5,
            return_train_score = True,
            random_state=42)

# Run the random search CV
random_cv.fit(X_train,y_train.values.squeeze())

# Give best parameters and compare to others
print(random_cv.best_params_)
print(random_cv.cv_results_[('mean_test_score')])

# Fit to testing data
best_xgb = random_cv.best_estimator_

# Predict classes
train_pred = best_xgb.predict(X_train)
test_pred = best_xgb.predict(X_test)

train_rmse=np.sqrt(np.mean((train_pred-y_train.values.squeeze())**2))
test_rmse=np.sqrt(np.mean((test_pred-y_test.values.squeeze())**2))

print('Train RMSE:',train_rmse)
print('Test RMSE:',test_rmse)

metrics = pd.DataFrame([['XGBoost',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/XGBoost.csv',index=False)
