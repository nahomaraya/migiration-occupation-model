from dml_pipeline.dml_debug import run_leakage_debug


if __name__ == "__main__":
    # Run debug check
    is_safe= run_leakage_debug(
        data_path="ipums_data_ml_ready.parquet",
        treatment_col='is_immigrant',
        outcome_col='occscore',
        weight_col='perwt',
        sample_size=10000,
        exclude_cols=['year', 'sample', 'hwsei', 'hhwt']
    )
    # results = run_dml_experiment(
    #     data_path="ipums_data_ml_ready.parquet",
    #     sample_fraction=0.1,
    #     experiment_name="DML_Immigration_Study",
    #     run_name="debug_run",
    #     verbose=True
    # )

    # print("\n" + "=" * 80)
    # print("EXPERIMENT COMPLETE")
    # print("=" * 80)
    # print(f"\nCausal Effect: {results['causal_effect']:.4f}")
    # print(f"P-value: {results['p_value']:.6f}")
    #
    # if MLFLOW_AVAILABLE:
    #     print("\nView results in MLflow UI:")
    #     print("  $ mlflow ui")
    #     print("  → Open http://localhost:5000")


    if is_safe:
        print("\n✓ Ready to run full DML pipeline!")
    else:
        print("\n✗ Fix leakage issues before proceeding!")