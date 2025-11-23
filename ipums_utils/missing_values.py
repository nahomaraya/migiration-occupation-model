import polars as pl
import numpy as np
import gc

# Load the data
print("Loading data...")
df = pl.read_parquet("ipums_data.parquet")
print(f"Initial shape: {df.shape}")

# ==========================================
# STEP 1: DATA FILTERING
# ==========================================
print("\n" + "=" * 50)
print("STEP 1: FILTERING TO LABOR FORCE")
print("=" * 50)

if 'labforce' in df.columns:
    print("\nlabforce value counts (before filtering):")
    print(df.group_by('labforce').agg(pl.len()).sort('labforce'))

    # Filter to employed only
    df_filtered = df.filter(pl.col('empstat') == 1)

    print(f"\nRows before filter: {df.shape[0]:,}")
    print(f"Rows after filter: {df_filtered.shape[0]:,}")
    print(
        f"Rows removed: {df.shape[0] - df_filtered.shape[0]:,} ({100 * (df.shape[0] - df_filtered.shape[0]) / df.shape[0]:.1f}%)")

    # Free memory
    del df
    gc.collect()
else:
    print("WARNING: labforce column not found!")
    df_filtered = df

# ==========================================
# DEFINE VARIABLE CATEGORIES FOR MISSING VALUE STRATEGY
# ==========================================

# Variables that should NEVER be imputed (structural/identifiers)
IDENTIFIER_VARS = [
    'year', 'sample', 'serial', 'cbserial', 'pernum',
    'cluster', 'strata', 'famunit'
]

# Weights (never impute, drop row if missing)
WEIGHT_VARS = ['hhwt', 'perwt']

# Household-level variables (categorical, use sentinel -1)
HOUSEHOLD_CATEGORICAL = [
    'gq', 'hhtype', 'region', 'stateicp', 'countyicp', 'city',
    'cityerr', 'ownershp', 'ownershpd', 'mortgage', 'lingisol',
    'multgen', 'multgend'
]

# Household-level numeric (use median + indicator)
HOUSEHOLD_NUMERIC = [
    'citypop', 'nfams'
]

# Person demographics (categorical, use sentinel -1)
PERSON_CATEGORICAL = [
    'sex', 'race', 'raced', 'marst', 'relate', 'related',
    'school', 'schltype'
]

# Person demographics (numeric, use median + indicator)
PERSON_NUMERIC = [
    'age', 'birthyr', 'nchild', 'nsibs', 'eldch', 'yngch'
]

# Migration variables (SPECIAL HANDLING - already done in previous step)
MIGRATION_VARS = [
    'bpl', 'bpld', 'citizen', 'yrnatur', 'yrimmig',
    'ancestr1', 'ancestr1d', 'language', 'languaged', 'speakeng',
    'migrate1', 'migrate1d', 'migplac1', 'migcounty1', 'migmet131', 'movedin'
]

# Education variables (categorical but logically dependent)
EDUCATION_VARS = [
    'educ', 'educd', 'gradeatt', 'gradeattd', 'degfield', 'degfieldd'
]

# Employment/Occupation (categorical, use sentinel -1)
EMPLOYMENT_CATEGORICAL = [
    'empstat', 'empstatd',
    'wrklstwk', 'absent', 'looking', 'availble', 'wrkrecal', 'workedyr'
]

# Work location/transport (categorical or numeric depending on variable)
WORK_VARS = [
    'pwstate2', 'carpool', 'riders'
]

# Target variables (should not be imputed in training data)
TARGET_VARS = [
    'occscore', 'classwkr'  # These are outcomes
]

# ==========================================
# STEP 2: HANDLE STRUCTURAL MISSINGNESS (MIGRATION)
# ==========================================
print("\n" + "=" * 50)
print("STEP 2: STRUCTURAL MISSINGNESS - MIGRATION VARIABLES")
print("=" * 50)

# Process in-place to save memory
df_processed = df_filtered

# A. Create is_immigrant from bpl
if 'bpl' in df_processed.columns:
    print("\n[A] Processing bpl (Birthplace)...")
    print(f"  US-born (bpl <= 120): {df_processed.filter(pl.col('bpl') <= 120).shape[0]:,}")
    print(f"  Foreign-born (bpl > 120): {df_processed.filter(pl.col('bpl') > 120).shape[0]:,}")

    df_processed = df_processed.with_columns([
        (pl.col('bpl') > 120).cast(pl.Int8).alias('is_immigrant')
    ])
    print("  ✓ Created: is_immigrant")

# B. Create years_in_us
if all(col in df_processed.columns for col in ['yrimmig', 'year', 'age']):
    print("\n[B] Processing yrimmig → years_in_us...")

    df_processed = df_processed.with_columns([
        pl.when(pl.col('is_immigrant') == 1)
        .then(
            pl.when(pl.col('yrimmig') > 0)
            .then((pl.col('year') - pl.col('yrimmig')).clip(lower_bound=0))
            .otherwise(pl.lit(0))
        )
        .otherwise(pl.col('age'))
        .alias('years_in_us')
    ])
    print("  ✓ Created: years_in_us")

# C. Create citizenship_status
if 'citizen' in df_processed.columns:
    print("\n[C] Processing citizen → citizenship_status...")

    df_processed = df_processed.with_columns([
        pl.when(pl.col('is_immigrant') == 0)
        .then(pl.lit(0))
        .when(pl.col('citizen') == 2)
        .then(pl.lit(1))
        .when(pl.col('citizen') == 3)
        .then(pl.lit(2))
        .otherwise(pl.lit(0))
        .alias('citizenship_status')
    ])

    citizenship_dist = df_processed.group_by('citizenship_status').agg(pl.len()).sort('citizenship_status')
    print("  ✓ Created: citizenship_status")
    print("    0=Native, 1=Naturalized, 2=Non-Citizen")
    print(citizenship_dist)

# D. Handle YRNATUR (Year Naturalized) - OPTIMIZED
if 'yrnatur' in df_processed.columns:
    print("\n[D] Processing yrnatur (Year Naturalized)...")

    # First check null count
    null_count = df_processed['yrnatur'].null_count()
    print(f"  Nulls in yrnatur: {null_count:,}")

    # Create missingness indicator ONLY if there are nulls
    if null_count > 0:
        df_processed = df_processed.with_columns([
            pl.col('yrnatur').is_null().cast(pl.Int8).alias('yrnatur_missing')
        ])

    # Fill values efficiently without creating intermediate DataFrames
    df_processed = df_processed.with_columns([
        pl.when(pl.col('is_immigrant') == 0)
        .then(pl.lit(0))
        .when(pl.col('citizenship_status') == 2)
        .then(pl.lit(0))
        .when(pl.col('yrnatur').is_null())
        .then(pl.lit(0))
        .otherwise(pl.col('yrnatur'))
        .alias('yrnatur')
    ])

    print("  ✓ yrnatur processed: 0 for natives/non-citizens, actual year for naturalized")

# E. Handle other migration variables with sentinel values
migration_to_handle = [v for v in MIGRATION_VARS if v in df_processed.columns]
print(f"\n[E] Handling {len(migration_to_handle)} other migration variables...")

for col in migration_to_handle:
    if col in ['bpl', 'bpld', 'citizen', 'yrimmig', 'yrnatur']:
        continue

    null_count = df_processed[col].null_count()
    if null_count > 0:
        df_processed = df_processed.with_columns([
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing")
        ])

        df_processed = df_processed.with_columns([
            pl.col(col).fill_null(-1)
        ])

        print(f"  ✓ {col}: {null_count:,} nulls → sentinel (-1) + indicator")

# ==========================================
# STEP 3: HANDLE EDUCATION VARIABLES
# ==========================================
print("\n" + "=" * 50)
print("STEP 3: EDUCATION VARIABLES (LOGICAL FILLING)")
print("=" * 50)

education_to_handle = [v for v in EDUCATION_VARS if v in df_processed.columns]
print(f"\nProcessing {len(education_to_handle)} education variables...")

for col in education_to_handle:
    null_count = df_processed[col].null_count()
    if null_count > 0:
        df_processed = df_processed.with_columns([
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing")
        ])

        if 'degfield' in col:
            df_processed = df_processed.with_columns([
                pl.col(col).fill_null(0)
            ])
            print(f"  ✓ {col}: {null_count:,} nulls → 0 (No Degree) + indicator")
        elif 'gradeatt' in col:
            df_processed = df_processed.with_columns([
                pl.col(col).fill_null(0)
            ])
            print(f"  ✓ {col}: {null_count:,} nulls → 0 (Not in school) + indicator")
        else:
            df_processed = df_processed.with_columns([
                pl.col(col).fill_null(-1)
            ])
            print(f"  ✓ {col}: {null_count:,} nulls → sentinel (-1) + indicator")

# ==========================================
# STEP 4-6: CATEGORICAL VARIABLES (COMBINED FOR EFFICIENCY)
# ==========================================
print("\n" + "=" * 50)
print("STEP 4-6: CATEGORICAL VARIABLES")
print("=" * 50)

all_categorical = HOUSEHOLD_CATEGORICAL + PERSON_CATEGORICAL + EMPLOYMENT_CATEGORICAL
categorical_to_handle = [v for v in all_categorical if v in df_processed.columns]
print(f"\nProcessing {len(categorical_to_handle)} categorical variables...")

for col in categorical_to_handle:
    null_count = df_processed[col].null_count()
    if null_count > 0:
        df_processed = df_processed.with_columns([
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing"),
            pl.col(col).fill_null(-1).alias(col)
        ])
        print(f"  ✓ {col}: {null_count:,} nulls → sentinel (-1) + indicator")

# ==========================================
# STEP 7: WORK/TRANSPORT VARIABLES
# ==========================================
print("\n" + "=" * 50)
print("STEP 7: WORK/TRANSPORT VARIABLES")
print("=" * 50)

work_to_handle = [v for v in WORK_VARS if v in df_processed.columns]
print(f"\nProcessing {len(work_to_handle)} work/transport variables...")

for col in work_to_handle:
    null_count = df_processed[col].null_count()
    if null_count > 0:
        df_processed = df_processed.with_columns([
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing"),
            pl.col(col).fill_null(-1).alias(col)
        ])
        print(f"  ✓ {col}: {null_count:,} nulls → sentinel (-1) + indicator")

# ==========================================
# STEP 8: NUMERIC VARIABLES
# ==========================================
print("\n" + "=" * 50)
print("STEP 8: NUMERIC VARIABLES (MEDIAN IMPUTATION)")
print("=" * 50)

numeric_to_handle = HOUSEHOLD_NUMERIC + PERSON_NUMERIC
numeric_existing = [v for v in numeric_to_handle if v in df_processed.columns]
print(f"\nProcessing {len(numeric_existing)} numeric variables...")

for col in numeric_existing:
    null_count = df_processed[col].null_count()
    if null_count > 0:
        median_val = df_processed[col].median()
        df_processed = df_processed.with_columns([
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing"),
            pl.col(col).fill_null(median_val).alias(col)
        ])
        print(f"  ✓ {col}: {null_count:,} nulls → median ({median_val}) + indicator")

# ==========================================
# STEP 9: VERIFY & SAVE
# ==========================================
print("\n" + "=" * 50)
print("FINAL VERIFICATION")
print("=" * 50)

total_nulls = sum([df_processed[col].null_count() for col in df_processed.columns])
print(f"\nTotal remaining nulls across all columns: {total_nulls:,}")

if total_nulls > 0:
    print("\nColumns still with nulls:")
    for col in df_processed.columns:
        null_count = df_processed[col].null_count()
        if null_count > 0:
            print(f"  {col}: {null_count:,}")

missing_indicators = [col for col in df_processed.columns if col.endswith('_missing')]
engineered_features = ['is_immigrant', 'years_in_us', 'citizenship_status']

print(f"\n✓ Missingness indicators created: {len(missing_indicators)}")
print(f"✓ Engineered migration features: {len([f for f in engineered_features if f in df_processed.columns])}")
print(f"\nFinal shape: {df_processed.shape}")
print(f"Final memory: {df_processed.estimated_size('mb'):.1f} MB")

# Save
output_file = "../ipums_data_processed.parquet"
print(f"\nSaving to {output_file}...")
df_processed.write_parquet(output_file, compression='snappy')
print(f"✓ Saved successfully")

# Preview
print("\n" + "=" * 50)
print("SAMPLE OF PROCESSED DATA")
print("=" * 50)
key_cols = ['is_immigrant', 'years_in_us', 'citizenship_status', 'age', 'educ', 'sex', 'race']
available_key_cols = [c for c in key_cols if c in df_processed.columns]
print(df_processed.select(available_key_cols).head(10))

# CHECK TARGETS
print("\n" + "=" * 50)
print("TARGET VALIDATION")
print("=" * 50)

if 'occscore' in df_processed.columns:
    zeros = df_processed.filter(pl.col('occscore') == 0).shape[0]
    print(f"Rows with OCCSCORE = 0: {zeros:,}")
    if zeros > 0:
        print("WARNING: You have people with 0 score. Distribution by empstat:")
        print(df_processed.filter(pl.col('occscore') == 0).group_by('empstat').agg(pl.len()))

if 'classwkr' in df_processed.columns:
    print("\nCLASSWKR Distribution (Target):")
    print(df_processed.group_by('classwkr').agg(pl.len()).sort('classwkr'))