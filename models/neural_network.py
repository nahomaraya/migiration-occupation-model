import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
from dmba import classificationSummary

from sklearn.inspection import partial_dependence, permutation_importance, PartialDependenceDisplay

def feature_importance_analysis(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Approximate Feature Importance ---")
    print(importance_df)

    plt.figure(figsize=(8, 4))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title("Permutation-Based Feature Importance")
    plt.xlabel("Mean Decrease in Model Accuracy")
    plt.show()

def plot_variable_impacts(model, X, feature_names):
    """
    Generate and interpret partial dependence plots for each predictor variable.
    """
    print("\n--- Neural Network Variable Impact Analysis ---\n")

    for feature in feature_names:
        fig, ax = plt.subplots(figsize=(6, 4))
        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            [feature],
            ax=ax,
            kind="average"
        )
        plt.title(f"Impact of {feature} on Predicted Accident Outcome")
        plt.grid(True)
        plt.show()

        print(f"Interpretation for {feature}:")
        print(" • The shape of this curve shows how changing this variable changes the model's output probability.")
        print(" • A rising curve means higher values increase accident severity probability,")
        print("   while a downward curve means the opposite.\n")

# Example usage:
# plot_variable_impacts(model, x_val, x_val.columns)


# ============================================
# STEP 1: Function to train and evaluate MLP Neural Network
# ============================================

def mlp_neural_network(x_train, x_val, y_train, y_val):
    # ============================================
    # PART A: Model Architecture Description
    # --------------------------------------------
    first_model = MLPClassifier(max_iter=500, random_state=42)
    # first_model = MLPClassifier(hidden_layer_sizes=(10,10),
    #                       activation='relu',
    #                       solver='adam',
    #                       max_iter=1000,  # using partial_fit loop manually
    #                       random_state=42)

    # ============================================
    # PART B: Track Training and Validation Loss
    # ============================================

    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (10, 10), (20, 10), (30, 15)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.05]
    }

    # Grid Search with 5-fold CV
    grid_search = GridSearchCV(first_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)

    print("\nBest Parameters Found:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}\n")

    # Use the best found model
    model = grid_search.best_estimator_

    train_losses = []
    val_losses = []
    # Initialize model for partial_fit (requires classes)
    classes = np.unique(y_train)

    # Early stopping configuration
    patience = 10
    best_val_loss = np.inf
    epochs_no_improve = 0
    max_epochs = 500


    print("\n--- Training the MLP Neural Network ---\n")

    for epoch in range(max_epochs):
        # Incremental training
        model.partial_fit(x_train, y_train, classes=classes)

        # Compute log loss for training and validation
        train_loss = log_loss(y_train, model.predict_proba(x_train))
        val_loss = log_loss(y_val, model.predict_proba(x_val))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}\n")
                break

    # ============================================
    # PART C: Plot Learning Curves (Loss vs Epoch)
    # ============================================

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Learning Curve: Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================================
    # PART D: Model Diagnostics and Performance Metrics
    # ============================================

    print("\n--- Model Diagnostics ---")

    # Predictions
    y_pred = model.predict(x_val)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Accuracy, Precision, Recall, F1
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Specificity (for binary case)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        print(f"Specificity: {specificity:.4f}")

    # DMBA summary for class-level metrics
    print("\nDetailed Classification Summary:")
    classificationSummary(y_val, y_pred)

    plot_variable_impacts(model, x_val, x_val.columns)
    feature_importance_analysis(model, x_val, y_val)
    #feature_importance_and_simulation(model, x_val, y_val)

    # ============================================
    # PART E: Return model and metrics
    # ============================================
    return model, {
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1
    }
