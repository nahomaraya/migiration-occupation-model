import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data_unsupervised(csv_path: str, verbose: bool = False):
    """
    Load and preprocess dataset for UNSUPERVISED learning (clustering).
    No target variable needed.

    Returns:
        x_train, x_val, x_test (all features, no labels)
    """
    data = pd.read_csv(csv_path)

    if verbose:
        print("=" * 80)
        print("DATA INSPECTION AND PREPROCESSING (UNSUPERVISED)")
        print("=" * 80)
        print(f"\nNumber of rows: {data.shape[0]}, Columns: {data.shape[1]}")
        print("\nData types of each column:")
        print(data.dtypes)
        print(data.describe())

    # Clean column names
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace(r"\s+", " ", regex=True)
    # Drop ID-like columns if they exist
    id_columns = [col for col in data.columns if col.upper() in ['ID#', 'ID', 'ID_NUM', 'IDENTIFIER']]
    if id_columns:
        data = data.drop(columns=id_columns)
        if verbose:
            print(f"Dropped ID column(s): {id_columns}")

    if verbose:
        print("\n" + "=" * 80)
        print("Generating correlation matrix...")


    corr_matrix = data.corr(numeric_only=True)
    print(corr_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                square=True, linewidths=0.5)
    plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Handle missing values
    if verbose:
        print("Handling missing values...")
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values per column:")
            print(missing_counts[missing_counts > 0])
        else:
            print("No missing values found.")

    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Only show boxplots if verbose is True and dataset is not too large
    if verbose and len(data.columns) <= 20:
        print("\n" + "=" * 80)
        print("Detecing outliers... (BOXPLOTS)")
        print("=" * 80)
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            sns.boxplot(x=data[col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')

        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    # Handle outliers for numeric columns
    if verbose:
        print("Clipping outliers to 5th-95th percentile...")

    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            data[col] = data[col].clip(
                lower=data[col].quantile(0.05),
                upper=data[col].quantile(0.95),
            )

    # One-hot encode categorical features
    categorical_cols = data.select_dtypes(include="object").columns

    if verbose and len(categorical_cols) > 0:
        print(f"Categorical columns found: {list(categorical_cols)}")
        print("Applying one-hot encoding...")

    if len(categorical_cols) > 0:
        data = pd.get_dummies(
            data,
            columns=categorical_cols,
            drop_first=True,
            dtype=int,
        )

    if verbose:
        print("Applying StandardScaler (mean=0, std=1)...")

    # Initialize and fit scaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data)

    # Put back into a DataFrame for convenience
    x_scaled = pd.DataFrame(x_scaled, columns=data.columns)

    if verbose:
        print("\nFirst 5 rows after preprocessing:")
        print(x_scaled.head())

    # Split data (no stratification needed for unsupervised)
    x_train, x_temp = train_test_split(
        x_scaled, test_size=0.3, random_state=42
    )
    x_val, x_test = train_test_split(
        x_temp, test_size=0.5, random_state=42
    )

    # Remove constant and highly correlated columns (based on training set only)
    if verbose:
        print("REMOVING REDUNDANT FEATURES")


    constant_cols = [col for col in x_train.columns if x_train[col].nunique() <= 1]
    if constant_cols:
        if verbose:
            print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        x_train = x_train.drop(columns=constant_cols)
        x_val = x_val.drop(columns=[c for c in constant_cols if c in x_val.columns])
        x_test = x_test.drop(columns=[c for c in constant_cols if c in x_test.columns])

    corr_matrix = x_train.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop:
        if verbose:
            print(f"Removing {len(to_drop)} highly correlated columns (>0.95): {to_drop}")
        x_train = x_train.drop(columns=to_drop)
        x_val = x_val.drop(columns=[c for c in to_drop if c in x_val.columns])
        x_test = x_test.drop(columns=[c for c in to_drop if c in x_test.columns])

    if verbose:
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("\n" + "=" * 80)
        print(f"Training set: {x_train.shape[0]} samples, {x_train.shape[1]} features")
        print(f"Validation set: {x_val.shape[0]} samples, {x_val.shape[1]} features")
        print(f"Test set: {x_test.shape[0]} samples, {x_test.shape[1]} features")

    return x_train, x_val, x_test
