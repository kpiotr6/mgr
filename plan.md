# What needs to be done
First of all - separate on folders.
## 1. Analyze the data and preprocess it.
 1. Try different subsets of features
 2. Try following:
    - Normalization
    - Log transformation
    - Differencing
 3. Check for correlation between features

## 2. Implement models from DARTS and maybe some other models.
 1. Use linear regression, DLinear, NLinear, Tranformer, TFT, TSMixer, RNN(?), NHits some model like prophet.
 2. Try different hyperparameters for each model (e.g. learning rate, batch size, number of layers, etc.)
 3. Use early stopping to prevent overfitting.
 4. Try models with different input and output lengths.
 5. Save models

## 3. Train the models and evaluate them.
 1. Use MAE, RMSE, MAPE, SMAPE as evaluation metrics.
 2. Use cross-validation to evaluate the models.
 3. Make sure that for each input length we test on the same dataset length.

## 4. Implement simulator and MPC controller.
## 5. Evaluate models with MPC controller.