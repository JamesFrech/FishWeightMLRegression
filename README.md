# Predicting Fish Weight using Machine Learning Regression Models.
This repo seeks to predict the weight of a fish using height, width, and different length variables. The purpose of this project is just for practicing using many different machine learning models for a regression task on a tabular dataset. The data for this project is taken from kaggle ("https://www.kaggle.com/datasets/tarunkumar1912/fish-dataset1")

---

## Subdirectories

### EDA
This subdirectory checks distributions and correlations of variables in the dataset and also runs a PCA. Length1 and Length2 variables show almost perfect correlation so both may not be needed. The PCA also shows over 90% of the variance in the inputs is explained by the first two PCs, so they could potentially be used as inputs if dimension reduction is needed. However, there are only 5 variables used as inputs, so this would be unnecessary.

---

### Regression
This subdirectory tests many different machine learning models to predict the weight of fish. Models used include linear Regression, Lasso, Elastic Net, KNN, Random Forest, Quantile Forest, XGBoost and a simple Neural Network with one hidden layer consisting of 5 nodes. Appropriate hyperparameters are tuned for all models (except linear regression).

The linear methods (linear regression, lasso, and elastic net) all show the worst performance which indicates more non-linear relationships between the input data and the target weight. However, it is clear regularization improves the model as lasso and elastic net far outperform linear regression. In addition, width and Length2 are removed from the linear regression due to not having significant p-values.  

As for the nonlinear methods, KNN is slightly outperformed by XGBoost, which is slightly outperformed by random forest/quantile forest, which are outperformed by the neural network. Interestingly, all nonlinear methods except for the neural network have much higher test accuracy than training accuracy, showing that these methods may be overfitting to the training data. The neural net however has comparable train and test RMSE showing that it not only performs the best, but is also not overfitting to training data. The tree based methods may be further improved and reduce overfitting by trying more hyperparameter combinations as they were tuned using a randomized grid search CV for computational efficiency. In addition, feature selection using feature importance and permutation importance may further improve the models.  

![alt text](images/Model_RMSE_Comparison.png "RMSE Comparison Between Models")

Both random forest and quantile forest were used, however are very similar. The difference is that quantile forest keeps information about the distribution of data on the leaf nodes which allows for computation of quantiles. This gives the model the capability to give prediction intervals in addition to just predicting the mean. Since the quantile forest is an extension of the random forest, it makes sense that their train/test errors are almost identical given the same set of hyperparameters. Results show that the true value (and prediction) are almost always within the prediction interval. In addition, test points that have higher errors show to generally have larger prediction intervals.

![alt text](images/QuantileForestPredictionIntervals.png "Quantile Forest Prediction Intervals")
![alt text](images/QuantileForest_AbsoluteError_IntervalSize.png "Quantile Forest Absolute Error vs Interval Size")
