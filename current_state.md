## 1. Machine Learning
 1. Usage of subset of features gives usually better results, yet different subsets are needed for different output vairables.
 2. Normalization improves results. Log transform and differencing don't work well.
 3. Using NeuralForecast models gives better resuts than DARTS models (they are not random)
 4. Models can be better than naive sesonal model but only for small output lenghts 30, 60. 120 works worse.
 5. Using MAPE as loss can give better results than default.

## 2. MPC
 1. Basic one is implemented.

## 3. Simulator
