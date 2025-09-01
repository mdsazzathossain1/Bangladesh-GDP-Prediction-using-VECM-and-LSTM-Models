# Bangladesh-GDP-Prediction-using-VECM-and-LSTM-Models
This repository contains a data-driven analysis and forecast of Bangladesh's Gross Domestic Product (GDP) using two powerful time-series modeling techniques: the Vector Error Correction Model (VECM) and the Long Short-Term Memory (LSTM) neural network.
The Jupyter Notebook is organized into several key scripts. Below is a detailed explanation of each part.
Script 1: VECM Forecasting Model
This script focuses on building and visualizing a 10-year GDP forecast using the Vector Error Correction Model (VECM). VECM is suitable for multivariate time series that have a long-run cointegrating relationship.
Step-by-Step Instructions:
Load Data: The bangladesh-gdp.csv dataset is loaded into a pandas DataFrame.
Preprocessing:
The General government final consumption expenditure column is dropped as it is deemed unnecessary.
The date column is converted to a datetime object and set as the DataFrame index.
Missing numerical values are filled using the median of their respective columns.
All features are normalized to a range of using MinMaxScaler to ensure stable model training.
VECM Modeling:
A VECM is initialized with the scaled data. A cointegration rank of 2 (coint_rank=2) is chosen, suggesting two long-term relationships between the variables.
The model is fitted to the entire dataset.
Forecasting:
The model predicts the values for the next 10 years.
The forecasted values are transformed back to their original scale using scaler.inverse_transform.
Post-processing:
A constraint is applied to ensure that the forecasted GDP never falls below the last known GDP value from 2022.
Visualization: The script generates a plot comparing the historical GDP with the 10-year forecasted GDP, providing a clear visual representation of the prediction.
Script 2: VECM Performance Evaluation
This script evaluates the VECM model's predictive accuracy on unseen data.
Step-by-Step Instructions:
Train-Test Split: The scaled dataset is split into a training set (first 80%) and a testing set (last 20%).
Model Training: The VECM is trained only on the training data.
Prediction: The trained model is used to make predictions on the time steps corresponding to the test set.
Metric Calculation: The script calculates key performance metrics (MSE, RMSE, MAE, and R-squared) by comparing the model's predictions with the actual values from the test set. The resulting R-squared value of approximately 0.895 indicates a strong fit.
Script 3: LSTM Forecasting Model
This script implements a Long Short-Term Memory (LSTM) network to provide an alternative, deep learning-based GDP forecast.
Step-by-Step Instructions:
Data Preparation:
The data is loaded and cleaned as in the VECM script.
To stabilize variance, the GDP column (GDP_BD) is log-transformed.
The log-transformed GDP is then normalized using MinMaxScaler.
Sequence Creation: A function creates a sequential dataset suitable for LSTMs, where a sequence of the last time_steps (e.g., 3 years) is used to predict the next value.
LSTM Model Architecture:
A Sequential Keras model is built with two LSTM layers (50 units each) and a final Dense output layer.
Training: The model is compiled with the 'adam' optimizer and 'mean_squared_error' loss function. It is trained on the training data.
Autoregressive Forecasting:
A loop generates a 10-year forecast. The model predicts one step ahead, and this prediction is then used as input for the next prediction.
Post-processing & Visualization:
Forecasted values are transformed back from the scaled and logged format to their original GDP values.
A constraint (np.maximum.accumulate) is applied to ensure the forecast is monotonically increasing.
A plot is generated showing the historical GDP against the LSTM's 10-year forecast.
Script 4: LSTM Performance Evaluation
This script assesses the performance of the trained LSTM model on the test data.
Step-by-Step Instructions:
Prediction on Test Set: The LSTM model predicts GDP values for the test set.
Inverse Transformation: Both the predicted values and the true test values are transformed back from their scaled and logged format.
Metric Calculation: The script calculates performance metrics (MSE, RMSE, MAE, and R-squared). The resulting R-squared of approximately 0.96 and a MAPE of ~6.9% demonstrate very high accuracy, outperforming the VECM on this task.
Script 5: Exploratory Analysis - Johansen Cointegration Test
This is an important preliminary script that provides the statistical justification for using a VECM.
Step-by-Step Instructions:
Johansen Test: The script performs the Johansen Cointegration Test on the key economic variables. This test determines if there is a stable, long-term relationship between them.
Trend Visualization: It plots the long-term trends of all time series features on a single graph, helping to visually identify potential cointegrating relationships.
📊 Results Summary
Both models provided strong forecasts, but the LSTM model showed superior performance on the test set.
Metric	VECM Model	LSTM Model
R-squared (R²)	0.895	0.961
Mean Absolute Error (MAE)	0.058 (scaled)	16.52 (original scale)
RMSE	0.081 (scaled)	21.83 (original scale)
Conclusion: The LSTM model, with an R² of 0.96, is the more accurate predictor for this specific GDP forecasting task. Its ability to capture complex non-linear patterns in the univariate time series proved more effective than the VECM's multivariate approach in this case.
