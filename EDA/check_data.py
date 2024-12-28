import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')

# Describe the data
print(data.describe())

inputs=['Species','Length1','Length2','Length3','Height','Width']
target=['Weight']

# Split the train/test data for EDA to avoid looking at the test set
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

train = pd.concat([X_train,y_train],axis=1)

# Plot histogram for each input
fig,axs=plt.subplots(2,3,figsize=(6,6))
k=0
for i in range(2):
    for j in range(3):
        axs[i,j].hist(train[inputs[k]],bins=20)
        axs[i,j].set_title(inputs[k])
        k+=1
plt.tight_layout()
plt.savefig('images/histograms.png',bbox_inches='tight')
plt.close()

# Plot scatter plots with color as area
fig,axs=plt.subplots(2,3,figsize=(6,6))
k=0
for i in range(2):
    for j in range(3):
        axs[i,j].scatter(train[inputs[k]],train[target])
        axs[i,j].set_xlabel(inputs[k])
        if i==0:
            axs[i,j].set_ylabel(target)
        k+=1
plt.tight_layout()
plt.savefig('images/scatter.png',bbox_inches='tight')
plt.close()

# Convert to categorical for correlation matrix
train['Species'] = train['Species'].astype('category')
train['Species'] = train['Species'].cat.codes
print(train)
# plot the correlation matrix
sns.heatmap(train.corr(),vmin=-1,vmax=1,cmap='bwr')
plt.savefig('images/correlation_matrix.png',bbox_inches='tight')
plt.close()

# plot the pair plots
sns.pairplot(train,vars=inputs,hue=target[0])
plt.savefig('images/pair_plots.png',bbox_inches='tight')
plt.close()
