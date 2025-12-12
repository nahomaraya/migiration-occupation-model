import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


__all__ = [
    "train_logistic_regression",
]


def train_logistic_regression(X_train: pd.DataFrame,
                              X_val: pd.DataFrame,
                              y_train,
                              y_val,
                              penalty: str = "l2",
                              solver: str = "liblinear",
                              max_iter: int = 1000):
    logit = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
    logit.fit(X_train, y_train)

    # Statsmodels summary on same design matrix
    result = sm.Logit(y_train, X_train).fit(disp=False)
    print("=== Logistic Regression Results ===")
    print(result.summary())
    print('intercept ', logit.intercept_[0])
    print(pd.DataFrame({'coeff': logit.coef_[0]},  index=X_train.columns).transpose())

    y_val_pred = logit.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    print("Classification Report:\n", classification_report(y_val, y_val_pred))

    return logit, result
