import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV


def simulate_feature_effect_rf(model, X_train, feature_name, X_mean, feature_min, feature_max):
    X_sim_min = X_mean.copy()
    X_sim_max = X_mean.copy()
    X_sim_min[feature_name] = feature_min
    X_sim_max[feature_name] = feature_max
    pred_min = model.predict(pd.DataFrame([X_sim_min]))[0]
    pred_max = model.predict(pd.DataFrame([X_sim_max]))[0]
    return pred_min, pred_max

def train_random_forest(X_train: pd.DataFrame,
                        X_val: pd.DataFrame,
                        y_train,
                        y_val,
                        n_estimators_grid = [100, 200, 300, 400],
                        visualize_tree: bool = False):
    param_grid_rf = {'n_estimators': n_estimators_grid}
    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=1), param_grid_rf, cv=5, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train.values.ravel())

    model = grid_search_rf.best_estimator_
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("=== Random Forest Results ===")
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    print("Classification Report:\n", classification_report(y_val, y_val_pred))

    rf_feat_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("Feature Importance:\n", rf_feat_importance)

    if visualize_tree:
        rf_tree = model.estimators_[0]
        plt.figure(figsize=(20,10))
        plot_tree(rf_tree, feature_names=X_train.columns, class_names=['No Injury', 'Injury'], filled=True, rounded=True, fontsize=4)
        plt.title("Random Forest - Single Tree Visualization")
        plt.show()

    X_mean_rf = X_train.mean()
    for feature_name in X_train.columns:
        feature_min = X_train[feature_name].min()
        feature_max = X_train[feature_name].max()
        rf_pred_min, rf_pred_max = simulate_feature_effect_rf(model, X_train, feature_name, X_mean_rf, feature_min, feature_max)
        print(f"Random Forest: {feature_name} min={feature_min} -> pred={rf_pred_min:.2f}, max={feature_max} -> pred={rf_pred_max:.2f}")

    print('RF score: ', grid_search_rf.best_score_)
    print('RF parameters: ', grid_search_rf.best_params_)

    return model