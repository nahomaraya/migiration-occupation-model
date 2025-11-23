# main.py
from preprocessing.preprocess_data import preprocess_data_xgboost
from models.xgboost import train_xgboost_regressor, train_xgboost_classifier


def main():
    print("\n" + "=" * 80)
    print("IPUMS IMMIGRATION STUDY - XGBOOST PIPELINE")
    print("=" * 80)

    data_path = "ipums_data_ml_ready.parquet"

    # ===================================
    # MODEL A: OCCSCORE REGRESSION
    # ===================================
    print("\n\n" + "=" * 80)
    print("MODEL A: PREDICTING OCCUPATIONAL SCORE (OCCSCORE)")
    print("=" * 80)

    X_train_occ, X_test_occ, y_train_occ, y_test_occ, w_train_occ, w_test_occ = preprocess_data_xgboost(
        data_path=data_path,
        target_column='occscore',
        weight_column='perwt',
        exclude_columns=['year', 'sample', 'classwkr', 'hwsei', 'hhwt'],
        test_size=0.2,
        stratify_column='year',
        apply_scaling=False,  # XGBoost doesn't need scaling
        handle_outliers=False,  # Already handled in feature engineering
        verbose=True
    )

    results_regression = train_xgboost_regressor(
        X_train_occ, X_test_occ,
        y_train_occ, y_test_occ,
        w_train_occ, w_test_occ,
        params={
            'device': 'cuda',
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.1,
            'min_child_weight': 5,
        },
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose=True
    )

    # Save model
    results_regression['model'].save_model('xgboost_occscore_model.json')
    results_regression['feature_importance'].to_csv('feature_importance_occscore.csv', index=False)
    print("\n✓ Saved: xgboost_occscore_model.json")
    print("✓ Saved: feature_importance_occscore.csv")

    # ===================================
    # MODEL B: CLASSWKR CLASSIFICATION
    # ===================================
    print("\n\n" + "=" * 80)
    print("MODEL B: PREDICTING CLASS OF WORKER (CLASSWKR)")
    print("=" * 80)

    X_train_cls, X_test_cls, y_train_cls, y_test_cls, w_train_cls, w_test_cls = preprocess_data_xgboost(
        data_path=data_path,
        target_column='classwkr',
        weight_column='perwt',
        exclude_columns=['year', 'sample', 'occscore', 'hwsei', 'hhwt'],
        test_size=0.2,
        stratify_column='year',
        apply_scaling=False,
        handle_outliers=False,
        verbose=True
    )

    results_classification = train_xgboost_classifier(
        X_train_cls, X_test_cls,
        y_train_cls, y_test_cls,
        w_train_cls, w_test_cls,
        params=None,  # Use defaults
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose=True
    )

    # Save model
    results_classification['model'].save_model('xgboost_classwkr_model.json')
    print("\n✓ Saved: xgboost_classwkr_model.json")

    # ===================================
    # SUMMARY
    # ===================================
    print("\n\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nRegression Results (OCCSCORE):")
    print(f"  Test RMSE: {results_regression['metrics']['test_rmse']:.4f}")
    print(f"  Test R²:   {results_regression['metrics']['test_r2']:.4f}")

    print("\nTop 5 Features for OCCSCORE:")
    print(results_regression['feature_importance'].head())


if __name__ == "__main__":
    main()