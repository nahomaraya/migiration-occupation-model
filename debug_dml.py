from dml_pipeline.dml_debug import run_leakage_debug

if __name__ == "__main__":
    # Run debug check
    is_safe = run_leakage_debug(
        data_path="ipums_data_ml_ready.parquet",
        treatment_col='is_immigrant',
        outcome_col='occscore',
        weight_col='perwt',
        sample_size=10000,
        exclude_cols=['year', 'sample', 'classwkr', 'hwsei', 'hhwt']
    )

    if is_safe:
        print("\n✓ Ready to run full DML pipeline!")
    else:
        print("\n✗ Fix leakage issues before proceeding!")