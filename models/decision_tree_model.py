import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def evaluate_model(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_valid, y_pred) * 100
    print(f"Model: {model.__class__.__name__}")
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    print("-" * 40)
    
def simulate_feature_effect_classification(model, X_train, feature_name, X_mean, feature_min, feature_max):
    X_sim_min = X_mean.copy()
    X_sim_max = X_mean.copy()
    X_sim_min[feature_name] = feature_min
    X_sim_max[feature_name] = feature_max
    prob_min = model.predict_proba([X_sim_min])[0][1]
    prob_max = model.predict_proba([X_sim_max])[0][1]
    return prob_min, prob_max

 
#param_grid_dt_tuned = {
#'max_depth': list(range(2, 8)), 
#'min_samples_split': list(range(2, 8)), 
#'min_impurity_decrease': [0, 0.000001, 0.0001], # 3 values
#}

#grid_search_dt_tuned = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid_dt_tuned, cv=5, n_jobs=-1)
#grid_search_dt_tuned.fit(X_train, Y_train.values.ravel())
   
def train_decision_tree(X_train: pd.DataFrame,
                        X_val: pd.DataFrame,
                        y_train,
                        y_val,
                        max_depth: int = 13,
                        min_samples_split: int = 20,
                        min_impurity_decrease: float = 0.001,
                        visualize: bool = False):
    best_params = {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_impurity_decrease': min_impurity_decrease,
    }
    model = DecisionTreeClassifier(random_state=0, **best_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    print("=== Decision Tree Results ===")
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

    feat_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("Feature Importance:\n", feat_importance)

    if visualize:
        plt.figure(figsize=(20,10))
        plot_tree(model, feature_names=X_train.columns, class_names=['No Injury', 'Injury'], filled=True, rounded=True)
        plt.title("Decision Tree Visualization")
        plt.show()

    X_mean_class = X_train.mean()
    for feature_name in X_train.columns:
        feature_min = X_train[feature_name].min()
        feature_max = X_train[feature_name].max()
        dt_prob_min, dt_prob_max = simulate_feature_effect_classification(model, X_train, feature_name, X_mean_class, feature_min, feature_max)
        print(f"Decision Tree: {feature_name} min={feature_min} -> prob={dt_prob_min:.2f}, max={feature_max} -> prob={dt_prob_max:.2f}")

    return model
   
