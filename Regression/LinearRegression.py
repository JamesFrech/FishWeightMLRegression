import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')
onehot = pd.get_dummies(data['Species'],dtype='int') # One hot encoding
data = pd.concat([data,onehot],axis=1)
data.drop('Species',axis=1,inplace=True)

# Select model inputs
target=['Weight']
inputs=['Length1','Length3','Height'] # Only variables with significant p-value
#inputs=data.columns.drop(target).values # All columns except target
#inputs=onehot.columns.tolist()+['Length1','Length3','Height'] # Use length 1, length 3, and height

# Split the train/test data
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# Fit a linear regression (ordinary least squares)
LR = sm.OLS(y_train,X_train)
results = LR.fit()

print(results.summary())
print('R^2:',results.rsquared)

train_pred = results.predict(X_train)
test_pred = results.predict(X_test)

train_rmse=np.sqrt(np.mean((train_pred.values-y_train.values.squeeze())**2))
test_rmse=np.sqrt(np.mean((test_pred.values-y_test.values.squeeze())**2))

print(train_rmse)
print(test_rmse)

metrics = pd.DataFrame([['LinearRegression',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/LinearRegression.csv',index=False)

# Get 95% prediction intervals
pred_ints=results.get_prediction(X_test).conf_int(obs=True, alpha=0.05)
print('Corresponding 95% prediction interval:',pred_ints)
distances=[i[1]-i[0] for i in pred_ints]
print(distances)
