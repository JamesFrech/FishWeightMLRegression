import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')

# Select model inputs
inputs=['Length1','Length2','Length3','Height','Width']
target=['Weight']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)
train=pd.concat([X_train,y_train],axis=1)

# Split data again for validation set
X_train, X_val, y_train, y_val = train_test_split(
    train[inputs], train[target], test_size=0.2, random_state=42)

# scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train=pd.DataFrame(scaler.transform(X_train),columns=inputs,index=X_train.index)
X_val=pd.DataFrame(scaler.transform(X_val),columns=inputs,index=X_val.index)
X_test=pd.DataFrame(scaler.transform(X_test),columns=inputs,index=X_test.index)

# Add constant
X_train=sm.add_constant(X_train)
X_val=sm.add_constant(X_val)
X_test=sm.add_constant(X_test)
inputs=['const']+inputs

rmse_lasso=[]
min_lambda_lasso=100
for penalty in np.linspace(0,100,1000):
    # Fit model and predict on validation set
    lasso = sm.OLS(y_train, X_train)
    results_stats_lasso = lasso.fit_regularized(alpha=penalty, L1_wt=1)
    pred_lasso = results_stats_lasso.predict(X_val)
    # Compute RMSE
    rmse = np.sqrt(np.mean((pred_lasso.values.squeeze() - y_val.values.squeeze())**2))
    rmse_lasso.append(rmse)
    if rmse <= min(rmse_lasso): # Save best value
        min_err_lasso = rmse
        min_lambda_lasso = penalty

plt.title("Lasso Validation RMSE")
plt.plot(np.linspace(0,100,1000),rmse_lasso)
plt.xlabel("$alpha$")
plt.ylabel("RMSE")
plt.savefig("images/lasso_rmse.png",bbox_inches='tight')
plt.close()

best_lasso = sm.OLS(y_train, X_train)
results_best_lasso = best_lasso.fit_regularized(alpha=min_lambda_lasso, L1_wt=1)

#print(results_best_lasso.summary())
print("Standardized LASSO coeff:\n", results_best_lasso.params)
print("Best lambda:",min_lambda_lasso)

train_pred = results_best_lasso.predict(X_train)
test_pred = results_best_lasso.predict(X_test)

train_rmse=np.sqrt(np.mean((train_pred.values-y_train.values.squeeze())**2))
test_rmse=np.sqrt(np.mean((test_pred.values-y_test.values.squeeze())**2))

#print('R^2:',results_best_lasso.rsquared)
print(train_rmse)
print(test_rmse)

metrics = pd.DataFrame([['Lasso',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/Lasso.csv',index=False)
