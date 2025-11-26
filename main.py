# main.py
from preprocessing.preprocess_data import preprocess_data_xgboost
from models.xgboost import train_xgboost_classifier
from dml_pipeline.run_dml_pipeline import run_dml_from_file


def main():
    print("\n" + "=" * 80)
    print("IPUMS IMMIGRATION STUDY - XGBOOST PIPELINE")
    print("=" * 80)

    data_path = "ipums_data_ml_ready.parquet"

    # ===================================
    # MODEL A: OCCSCORE REGRESSION
    # ===================================
    print("\n\n" + "=" * 80)
    print("MODEL: PREDICTING OCCUPATIONAL SCORE (OCCSCORE)")
    print("=" * 80)

    results = run_dml_from_file(
        data_path="ipums_data_training_final.parquet",
        sample_fraction=0.1,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"\nCausal Effect of Immigration on OCCSCORE: {results['causal_effect']:.4f}")
    print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    print(f"Significant: {'YES' if results['significant'] else 'NO'}")
    print(f"Outcome Model R²: {results['outcome_metrics']['test_r2']:.4f}")
    print(f"Treatment Model Accuracy: {results['treatment_metrics']['test_accuracy']:.4f}")
    print(f"Causal Effect: {results['causal_effect']:.4f}")
    print(f"Optimal Clusters: {results['cluster_results']['optimal_k']}")

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


if __name__ == "__main__":
    main()