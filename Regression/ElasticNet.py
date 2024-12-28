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

rmse_elastic_net=[]
min_lambda_elastic_net=100
param_space=pd.DataFrame()
for l1 in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
    for penalty in np.linspace(0,100,100):
        # Fit model and predict on validation set
        elastic_net = sm.OLS(y_train, X_train)
        results_stats_elastic_net = elastic_net.fit_regularized(alpha=penalty, L1_wt=l1)
        pred_elastic_net = results_stats_elastic_net.predict(X_val)
        # Compute RMSE
        rmse = np.sqrt(np.mean((pred_elastic_net.values.squeeze() - y_val.values.squeeze())**2))
        rmse_elastic_net.append(rmse)
        rmse_df = pd.DataFrame([[l1,penalty,rmse]],columns=['L1_wt','Penalty','RMSE'])
        param_space = pd.concat([param_space,rmse_df])
        if rmse <= min(rmse_elastic_net): # Save best value
            min_err_elastic_net = rmse
            min_lambda_elastic_net = penalty
            min_l1_weight = l1

print(param_space)

plt.title("Elastic Net Validation RMSE")
im = plt.scatter(param_space['Penalty'],param_space['L1_wt'],
                 c=param_space['RMSE'],marker='s')
plt.xlabel("$alpha$")
plt.ylabel("L1_wt")
plt.colorbar(im)
plt.savefig("images/elastic_net_rmse.png",bbox_inches='tight')
plt.close()

best_elastic_net = sm.OLS(y_train, X_train)
results_best_elastic_net = best_elastic_net.fit_regularized(alpha=min_lambda_elastic_net, L1_wt=1)

#print(results_best_lasso.summary())
print("Standardized Elastic Net coeff:\n", results_best_elastic_net.params)
print("Best lambda:",min_lambda_elastic_net)
print("Best L1 Weight:",min_l1_weight)

train_pred = results_best_elastic_net.predict(X_train)
test_pred = results_best_elastic_net.predict(X_test)

train_rmse=np.sqrt(np.mean((train_pred.values-y_train.values.squeeze())**2))
test_rmse=np.sqrt(np.mean((test_pred.values-y_test.values.squeeze())**2))

#print('R^2:',results_best_lasso.rsquared)
print(train_rmse)
print(test_rmse)

metrics = pd.DataFrame([['ElasticNet',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/ElasticNet.csv',index=False)
