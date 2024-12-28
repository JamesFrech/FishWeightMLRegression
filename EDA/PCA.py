import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Select model inputs, Don't use species in PCA
inputs=['Length1','Length2','Length3','Height','Width']
target=['Weight']

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')

# Split the train/test data for EDA to avoid looking at the test set
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)
# Put training data together for ease. Ignore test data in PCA analysis.
train = pd.concat([X_train,y_train],axis=1)

# Scale data
pca_scaler = StandardScaler()
pca_scaler.fit(train[inputs])
scaled_data = pca_scaler.transform(train[inputs])

# Fit PCA to the scaled data and transform the data
pca = PCA()
pca.fit(scaled_data)

scores = pca.transform(scaled_data)

# Make a skree plot
explained_var = pca.explained_variance_ratio_
print(explained_var)
plt.plot([i+1 for i in range(len(explained_var))],explained_var*100,marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Percent Variance Explained')
plt.savefig('images/skree_plot.png',bbox_inches='tight')
plt.close()

# Plot the PC1 and PC2 with each air quality a different color
scale_arrow = s_ = 10
i, j = 0, 1 # which components

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# Plot PC1 vs PC2
ax.scatter(scores[:,0],
           scores[:,1],
           s=2,
           c=train[target].values)
# Label axes
ax.set_xlabel(f'PC{i+1} {round(explained_var[i]*100,2)}%')
ax.set_ylabel(f'PC{j+1} {round(explained_var[j]*100,2)}%')
# Plot PCs for each input variable
for k in range(pca.components_.shape[1]):
  ax.arrow(0, 0, s_*pca.components_[i,k], s_*pca.components_[j,k])
  ax.text(s_*pca.components_[i,k],
          s_*pca.components_[j,k],
          inputs[k])
plt.savefig('images/PC1_PC2_scatter.png',bbox_inches='tight')
plt.close()
