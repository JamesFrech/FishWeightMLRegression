import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')
data.drop('Species',axis=1,inplace=True) # Dont use species

# Select model inputs
target=['Weight']
inputs=['Length1','Length2','Length3','Height','Width']

# Split the train/test data
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train=pd.DataFrame(scaler.transform(X_train),columns=inputs,index=X_train.index)
X_test=pd.DataFrame(scaler.transform(X_test),columns=inputs,index=X_test.index)

param_grid = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}

# Initialize random forest classifier
knn = KNeighborsRegressor()

# Initialize Kfold object with 5 folds
kfold = KFold(5,
              random_state=42,
              shuffle=True)

# Get the grid of hyperparameters for the model
grid = GridSearchCV(knn,
                    param_grid,
                    refit=True,
                    cv=kfold,
                    verbose = 5,
                    scoring='neg_root_mean_squared_error')

# Fit the model
grid.fit(X_train, y_train.values.ravel())

# Give best parameters and compare to others
print(grid.best_params_)
print(grid.cv_results_[('mean_test_score')])

# Fit to testing data
best_knn = grid.best_estimator_

# Predict classes
train_pred = best_knn.predict(X_train)
test_pred = best_knn.predict(X_test)

train_rmse=np.sqrt(np.mean((train_pred-y_train.values.squeeze())**2))
test_rmse=np.sqrt(np.mean((test_pred-y_test.values.squeeze())**2))

print('Train RMSE:',train_rmse)
print('Test RMSE:',test_rmse)

metrics = pd.DataFrame([['KNN',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/KNN.csv',index=False)
