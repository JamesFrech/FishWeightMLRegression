import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor
import matplotlib.pyplot as plt

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')
data.drop('Species',axis=1,inplace=True) # Dont use species

# Select model inputs
target=['Weight']
inputs=['Length1','Length2','Length3','Height','Width']

# Split the train/test data
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# Use best values from random forest and fit the model
qrf = RandomForestQuantileRegressor(n_estimators=800,
                                    min_samples_leaf=2,
                                    max_features=4)

qrf.fit(X_train, y_train.values.ravel())

# Get mean value like random forest
train_pred = qrf.predict(X_train, quantiles='mean')
test_pred = qrf.predict(X_test, quantiles='mean')
# Get values for 95% prediction interval
y_pred_int = qrf.predict(X_test, quantiles=[0.025, 0.975])
interval_size=[i[1]-i[0] for i in y_pred_int]
low_bnd = [i[0] for i in y_pred_int]
up_bnd = [i[1] for i in y_pred_int]

# Put the predictions, true values, intervals, and errors in a dataframe to check
test_predictions = pd.DataFrame()
test_predictions['Pred'] = test_pred
test_predictions['True'] = y_test.values.squeeze()
test_predictions['Int_Size'] = interval_size
test_predictions['AbsError'] = abs(test_pred - y_test.values.squeeze())
test_predictions['Lower_Bound'] = low_bnd
test_predictions['Upper_Bound'] = up_bnd
print(test_predictions)

# Get rmse
train_rmse=np.sqrt(np.mean((train_pred-y_train.values.squeeze())**2))
test_rmse=np.sqrt(np.mean((test_pred-y_test.values.squeeze())**2))

print('Train RMSE:',train_rmse)
print('Test RMSE:',test_rmse)

metrics = pd.DataFrame([['QuantileForest',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/QuantileForest.csv',index=False)

test_predictions.sort_values('Int_Size',inplace=True)
test_predictions.reset_index(inplace=True,drop=True)
print(test_predictions)

test_predictions.plot(y=['Pred','True','Lower_Bound','Upper_Bound'])
plt.title('True and Predicted values with\n95% Upper and Lower Prediction Interval Bounds')
plt.savefig('images/QuantileForestPredictionIntervals.png',bbox_inches='tight')
plt.close()

test_predictions.plot(y=['AbsError','Int_Size'])
plt.title('Prediction Interval Size vs Absolute Error')
plt.savefig('images/QuantileForest_AbsoluteError_IntervalSize.png',bbox_inches='tight')
plt.close()
