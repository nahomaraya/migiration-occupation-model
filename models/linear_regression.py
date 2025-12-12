
import pandas as pd
import statsmodels.api as sm
import numpy as np
from typing import Optional
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics

__all__ = [
    "train_linear_regression",
]


def train_linear_regression(X_train: pd.DataFrame,
                            X_val: pd.DataFrame,
                            y_train, y_val,
                            predict_feature_name: Optional[str] = None,
                            predict_feature_value: Optional[float] = None):
    """
    Train OLS linear regression and compute prediction at specified feature value.

    Returns:
        model, prediction_at_feature
    """
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary())

    # Build representative input using mean and modes
    mean_values = X_train.mean(numeric_only=True)
    # Safe mode computation for all columns
    mode_values = X_train.mode(dropna=True).iloc[0]
    input_data = mean_values.copy()
    
    # Predictions on validation data
    X2_val = sm.add_constant(X_val, has_constant='add')
    pred_val = model.predict(X2_val)
    pred_train = model.predict(X2_train)
    residuals_val = y_val - pred_val
    result_val = pd.DataFrame({'Predicted': pred_val, 'Actual': y_val, 'Residual': residuals_val})
    print(result_val.head(20))
    # Histogram of residuals
    plt.figure(figsize=(8,5))
    sns.histplot(residuals_val, kde=True, bins=20, color="skyblue")
    plt.title("Histogram of Residuals (Validation Set)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


    #Metrics
    print("Training MAPE:", metrics.mean_absolute_percentage_error(y_train, pred_train) * 100)
    print("Training MAE:", metrics.mean_absolute_error(y_train, pred_train))
    print("Training RMSE:", np.sqrt(metrics.mean_squared_error(y_train, pred_train)))
    E = pred_train - y_train
    ME = E.mean() 
    print("Mean Error : " , ME)

    ##MPE
    PE = (pred_train - y_train)/y_train *100
    MPE = PE.mean() 
    print("Mean Percentage Error", MPE)


     
    X_vif = sm.add_constant(X_train)

    vif_data = pd.DataFrame({

        'Feature': X_vif.columns,

        'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    })

    vif_data = vif_data[vif_data['Feature'] != 'const']

    print(vif_data)
    
    # Override specific feature if provided
    if predict_feature_name and predict_feature_name in X_train.columns and predict_feature_value is not None:
        input_data[predict_feature_name] = predict_feature_value
    
    # Set categorical-like one-hot groups to mode if present
    for col in X_train.columns:
        if col in mode_values.index and col not in input_data.index:
            input_data[col] = mode_values[col]

    input_df = pd.DataFrame([input_data], columns=X_train.columns)
    input_df = sm.add_constant(input_df, has_constant="add")
    prediction = float(model.predict(input_df)[0])
    return model, prediction