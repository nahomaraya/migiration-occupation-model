import polars as pl
import numpy as np
import gc

# Load your clean data
print("Loading processed data...")
df = pl.read_parquet("ipums_data_processed.parquet")
print(f"Starting shape: {df.shape}")
print(f"Memory usage: {df.estimated_size('mb'):.1f} MB")

print("\nStarting Feature Engineering...")

# ==========================================
# A. EDUCATION & SKILL BLOCK
# ==========================================
print("\n" + "=" * 50)
print("A. EDUCATION & SKILL FEATURES")
print("=" * 50)

# 1. GROUP DEGFIELD (Field of Degree)
if 'degfield' in df.columns:
    print("\n[1] Processing DEGFIELD → is_stem...")
    # STEM fields based on your codebook
    STEM_CODES = [
        11,  # Agriculture
        13,  # Environment and Natural Resources
        14,  # Architecture
        21,  # Computer and Information Sciences
        24,  # Engineering
        25,  # Engineering Technologies
        36,  # Biology and Life Sciences
        37,  # Mathematics and Statistics
        50,  # Physical Sciences
        51,  # Nuclear, Industrial Radiology, and Biological Technologies
    ]

    df = df.with_columns([
        pl.col('degfield').is_in(STEM_CODES).cast(pl.Int8).alias('is_stem')
    ])

    print("  ✓ is_stem flag created")
    print(f"    STEM codes: {STEM_CODES}")
    print(f"    STEM proportion: {df['is_stem'].mean():.2%}")

# 2. SPEAKENG → proficient_english
if 'speakeng' in df.columns:
    print("\n[2] Processing SPEAKENG → proficient_english...")

    # IPUMS SPEAKENG codes (typical):
    df = df.with_columns([
        pl.when(pl.col('speakeng').is_in([3, 4, 5]))
        .then(pl.lit(1))  # Proficient
        .otherwise(pl.lit(0))  # Not proficient (1, 6)
        .alias("proficient_english")
    ])

    print("  ✓ proficient_english created")
    print(f"    Proficient proportion: {df['proficient_english'].mean():.2%}")


# ==========================================
# B. ORIGIN REGION (Accurate Mapping)
# ==========================================
print("\n" + "=" * 50)
print("B. ORIGIN REGION FEATURES")
print("=" * 50)

if 'bpl' in df.columns:
    print("\n[1] Processing BPL → origin_region (detailed)...")

    df = df.with_columns([
        # US Native (0-120)
        pl.when(pl.col('bpl') <= 120)
        .then(pl.lit("US_Native"))

        # Canada (150)
        .when(pl.col('bpl') == 150)
        .then(pl.lit("Canada"))

        # Mexico (200)
        .when(pl.col('bpl') == 200)
        .then(pl.lit("Mexico"))

        # Central America (210)
        .when(pl.col('bpl') == 210)
        .then(pl.lit("Central_America"))

        # Caribbean (250-260)
        .when((pl.col('bpl') >= 250) & (pl.col('bpl') <= 260))
        .then(pl.lit("Caribbean"))

        # South America (300)
        .when(pl.col('bpl') == 300)
        .then(pl.lit("South_America"))

        # Northern Europe - Scandinavia (400-405)
        .when((pl.col('bpl') >= 400) & (pl.col('bpl') <= 405))
        .then(pl.lit("Northern_Europe"))

        # UK & Ireland (410-414)
        .when((pl.col('bpl') >= 410) & (pl.col('bpl') <= 414))
        .then(pl.lit("UK_Ireland"))

        # Western Europe (420-426)
        .when((pl.col('bpl') >= 420) & (pl.col('bpl') <= 426))
        .then(pl.lit("Western_Europe"))

        # Southern Europe (430-440)
        .when((pl.col('bpl') >= 430) & (pl.col('bpl') <= 440))
        .then(pl.lit("Southern_Europe"))

        # Central/Eastern Europe (450-459)
        .when((pl.col('bpl') >= 450) & (pl.col('bpl') <= 459))
        .then(pl.lit("Central_Eastern_Europe"))

        # Former USSR/Baltic (460-465)
        .when((pl.col('bpl') >= 460) & (pl.col('bpl') <= 465))
        .then(pl.lit("Former_USSR"))

        # East Asia - China, Japan, Korea (500-502)
        .when((pl.col('bpl') >= 500) & (pl.col('bpl') <= 502))
        .then(pl.lit("East_Asia"))

        # Southeast Asia (510-519)
        .when((pl.col('bpl') >= 510) & (pl.col('bpl') <= 519))
        .then(pl.lit("Southeast_Asia"))

        # South Asia - India, Afghanistan, etc. (520-524)
        .when((pl.col('bpl') >= 520) & (pl.col('bpl') <= 524))
        .then(pl.lit("South_Asia"))

        # Middle East (530-549)
        .when((pl.col('bpl') >= 530) & (pl.col('bpl') <= 549))
        .then(pl.lit("Middle_East"))

        # Africa (600)
        .when(pl.col('bpl') == 600)
        .then(pl.lit("Africa"))

        # Oceania - Australia, NZ, Pacific (700-710)
        .when((pl.col('bpl') >= 700) & (pl.col('bpl') <= 710))
        .then(pl.lit("Oceania"))

        # Catch-all
        .otherwise(pl.lit("Other"))
        .alias("origin_region")
    ])

    print("  ✓ origin_region created")

    df = df.with_columns([
        pl.col("origin_region").cast(pl.Categorical)
    ])

    # Show distribution
    region_dist = df.group_by("origin_region").agg(pl.len().alias("count")).sort("count", descending=True)
    print("\n  Origin region distribution:")
    print(region_dist)

    # Create simplified region groupings for economic development
    print("\n[2] Creating origin_development_level...")

    df = df.with_columns([
        # High-income Anglosphere & Western countries
        pl.when(pl.col('origin_region').is_in([
            "US_Native", "Canada", "UK_Ireland", "Northern_Europe",
            "Western_Europe", "Oceania"
        ]))
        .then(pl.lit("High_Income_Western"))

        # High-income Asia (Japan, Singapore mainly via East Asia)
        .when(pl.col('origin_region') == "East_Asia")
        .then(pl.lit("High_Income_Asia"))

        # Upper-middle income Europe
        .when(pl.col('origin_region').is_in([
            "Southern_Europe", "Central_Eastern_Europe", "Former_USSR"
        ]))
        .then(pl.lit("Upper_Middle_Europe"))

        # Latin America
        .when(pl.col('origin_region').is_in([
            "Mexico", "Central_America", "Caribbean", "South_America"
        ]))
        .then(pl.lit("Latin_America"))

        # Asia developing
        .when(pl.col('origin_region').is_in([
            "South_Asia", "Southeast_Asia"
        ]))
        .then(pl.lit("Developing_Asia"))

        # Middle East & Africa
        .when(pl.col('origin_region').is_in(["Middle_East", "Africa"]))
        .then(pl.lit("MENA_Africa"))

        .otherwise(pl.lit("Other"))
        .alias("origin_development_level")
    ])

    print("  ✓ origin_development_level created")

    dev_dist = df.group_by("origin_development_level").agg(pl.len().alias("count")).sort("count", descending=True)
    df = df.with_columns([
        pl.col("origin_development_level").cast(pl.Categorical)
    ])
    print("\n  Development level distribution:")
    print(dev_dist)

# ==========================================
# C. MIGRATION BLOCK
# ==========================================
print("\n" + "=" * 50)
print("C. MIGRATION FEATURES")
print("=" * 50)

# 1. years_since_immigration (already created in preprocessing, but verify)
if 'years_in_us' in df.columns:
    print("\n[1] ✓ years_in_us already exists")
    print(f"    Mean for immigrants: {df.filter(pl.col('is_immigrant') == 1)['years_in_us'].mean():.1f} years")

# 2. age_at_arrival
if all(col in df.columns for col in ['age', 'years_in_us', 'is_immigrant']):
    print("\n[2] Creating age_at_arrival...")

    df = df.with_columns([
        pl.when(pl.col('is_immigrant') == 1)
        .then(pl.col('age') - pl.col('years_in_us'))
        .otherwise(pl.lit(-1))  # N/A for natives (or could use 0)
        .clip(lower_bound=-1, upper_bound=100)  # Sanity check
        .alias('age_at_arrival')
    ])

    print("  ✓ age_at_arrival created")
    print(f"    Mean age at arrival: {df.filter(pl.col('age_at_arrival') > 0)['age_at_arrival'].mean():.1f} years")

# 3. is_naturalized (based on citizenship_status)
if 'citizenship_status' in df.columns:
    print("\n[3] Creating is_naturalized...")

    df = df.with_columns([
        (pl.col('citizenship_status') == 1).cast(pl.Int8).alias('is_naturalized')
    ])

    print("  ✓ is_naturalized created")
    print(f"    Naturalized proportion: {df.filter(pl.col('is_immigrant') == 1)['is_naturalized'].mean():.2%}")

# ==========================================
# D. FAMILY/HOUSEHOLD BLOCK
# ==========================================
print("\n" + "=" * 50)
print("D. FAMILY/HOUSEHOLD FEATURES")
print("=" * 50)

# 1. family_burden
if 'nchild' in df.columns:
    print("\n[1] Creating family_burden...")

    # Sum children and siblings
    family_cols = ['nchild']
    if 'nsibs' in df.columns:
        family_cols.append('nsibs')

    # Create sum expression
    sum_expr = pl.lit(0)
    for col in family_cols:
        sum_expr = sum_expr + pl.col(col).fill_null(0)

    df = df.with_columns([
        sum_expr.alias('family_burden')
    ])

    print(f"  ✓ family_burden created (summing: {family_cols})")
    print(f"    Mean family burden: {df['family_burden'].mean():.2f}")

# 2. is_married
if 'marst' in df.columns:
    print("\n[2] Creating is_married...")

    # IPUMS MARST codes (typical):
    # 1 = Married, spouse present
    # 2 = Married, spouse absent
    # 3 = Separated
    # 4 = Divorced
    # 5 = Widowed
    # 6 = Never married/single

    df = df.with_columns([
        pl.col('marst').is_in([1, 2]).cast(pl.Int8).alias('is_married')
    ])

    print("  ✓ is_married created")
    print(f"    Married proportion: {df['is_married'].mean():.2%}")

# 3. has_children
if 'nchild' in df.columns:
    df = df.with_columns([
        (pl.col('nchild') > 0).cast(pl.Int8).alias('has_children')
    ])
    print("\n[3] ✓ has_children created")

# ==========================================
# E. METROPOLITAN STATUS
# ==========================================
# print("\n" + "=" * 50)
# print("E. METROPOLITAN STATUS")
# print("=" * 50)
#
# # Check for metro variables
# metro_vars = ['metro', 'metarea', 'migmet131']
# found_metro = [v for v in metro_vars if v in df.columns]
#
# if found_metro:
#     metro_col = found_metro[0]
#     print(f"\n[1] Creating in_metro from {metro_col}...")
#
#     # Most IPUMS metro codes: 0 = Not in metro, >0 = In metro area
#     df = df.with_columns([
#         (pl.col(metro_col) > 0).cast(pl.Int8).alias('in_metro')
#     ])
#
#     print("  ✓ in_metro created")
#     print(f"    Metro proportion: {df['in_metro'].mean():.2%}")
# else:
#     print("\n[1] No metro variable found, skipping...")

# ==========================================
# F. INTERACTION FEATURES (Key for Research Questions)
# ==========================================
print("\n" + "=" * 50)
print("F. INTERACTION FEATURES")
print("=" * 50)

# These help answer your research questions about differential returns

# 1. immigrant × education
if all(col in df.columns for col in ['is_immigrant', 'educ']):
    print("\n[1] Creating immigrant_x_education...")
    df = df.with_columns([
        (pl.col('is_immigrant') * pl.col('educ')).alias('immigrant_x_education')
    ])
    print("  ✓ immigrant_x_education created")

# 2. immigrant × STEM
if all(col in df.columns for col in ['is_immigrant', 'is_stem']):
    print("\n[2] Creating immigrant_x_stem...")
    df = df.with_columns([
        (pl.col('is_immigrant') * pl.col('is_stem')).alias('immigrant_x_stem')
    ])
    print("  ✓ immigrant_x_stem created")

# 3. immigrant × proficient_english
if all(col in df.columns for col in ['is_immigrant', 'proficient_english']):
    print("\n[3] Creating immigrant_x_english...")
    df = df.with_columns([
        (pl.col('is_immigrant') * pl.col('proficient_english')).alias('immigrant_x_english')
    ])
    print("  ✓ immigrant_x_english created")

# 4. years_in_us × education (returns to tenure)
if all(col in df.columns for col in ['years_in_us', 'educ', 'is_immigrant']):
    print("\n[4] Creating tenure_x_education...")
    df = df.with_columns([
        pl.when(pl.col('is_immigrant') == 1)
        .then(pl.col('years_in_us') * pl.col('educ'))
        .otherwise(pl.lit(0))
        .alias('tenure_x_education')
    ])
    print("  ✓ tenure_x_education created")

# ==========================================
# G. ONE-HOT ENCODING (for XGBoost)
# ==========================================
print("\n" + "=" * 50)
print("G. ONE-HOT ENCODING CATEGORICAL VARIABLES")
print("=" * 50)

# For XGBoost, we can either:
# Option A: Convert to Categorical dtype (memory efficient, works with enable_categorical=True)
# Option B: One-hot encode (more columns but explicit)

# We'll use Option B for critical variables like race, region
# And Option A (Categorical dtype) for high-cardinality like degree_category, origin_region

print("\n[Option A] Converting high-cardinality to Categorical dtype...")
categorical_dtype_cols = ['degree_category', 'origin_region']
for col in categorical_dtype_cols:
    if col in df.columns:
        df = df.with_columns([
            pl.col(col).cast(pl.Categorical)
        ])
        print(f"  ✓ {col} → Categorical dtype")

print("\n[Option B] One-hot encoding low-cardinality variables...")
# One-hot encode: sex, race (critical for fairness analysis)
onehot_cols = ['sex', 'race']
for col in onehot_cols:
    if col in df.columns:
        print(f"  Processing {col}...")
        df = df.to_dummies(columns=[col], separator='_')
        print(f"  ✓ {col} one-hot encoded")

# ==========================================
# H. DROP REDUNDANT/RAW COLUMNS
# ==========================================
print("\n" + "=" * 50)
print("H. DROPPING REDUNDANT COLUMNS")
print("=" * 50)

drop_cols = [
    # IDs and admin
    'serial', 'cbserial', 'pernum', 'cluster', 'strata', 'famunit',
    'ftotinc', 'incss', 'incwelfr', 'incinvst', 'incother', 'hwsei'
    # Raw versions of engineered features
    # 'degfield', 'degfieldd',  # Replaced by degree_category
    'bpl', 'bpld' # Replaced by origin_region
    'ancestr1', 'ancestr1d',  # Not using
    'language', 'languaged',  # Not using
    'speakeng',  # Replaced by proficient_english
    'citizen',  # Replaced by citizenship_status, is_naturalized
    'yrimmig',  # Replaced by years_in_us
    'yrnatur',  # Information captured in is_naturalized
    'marst',  # Replaced by is_married
    'nchild', 'nsibs',  # Replaced by family_burden

    # Detailed versions we don't need
    'raced', 'related', 'ownershpd', 'multgend', 'gradeattd', 'educd', 'empstatd', 'classwkrd',

    # Migration variables we're not using
    'migrate1', 'migrate1d', 'migplac1', 'migcounty1', 'migmet131', 'movedin',

    # Geographic detail we don't need (keeping stateicp, region)
    'countyicp', 'city', 'cityerr', 'citypop',

    # Household composition (keeping hhtype)
    'nfams', 'lingisol',

    # # School variables (not in labor force analysis)
    # 'school', 'gradeatt', 'schltype',

    # Work details not needed
    'wrklstwk', 'absent', 'looking', 'availble', 'wrkrecal', 'workedyr',
    'pwstate2', 'carpool', 'riders',

    # Child age variables
    'eldch', 'yngch',
    'gradeatt', 'school',
     'migmet131', 'migmet131_missing'
]

# Only drop columns that exist
existing_drop = [c for c in drop_cols if c in df.columns]
df_final = df.drop(existing_drop)

print(f"  Dropped {len(existing_drop)} columns")
print(f"  Remaining columns: {len(df_final.columns)}")

# ==========================================
# I. FINAL CHECKS & SAVE
# ==========================================
print("\n" + "=" * 50)
print("FINAL CHECKS & SUMMARY")
print("=" * 50)

print(f"\nFinal shape: {df_final.shape}")
print(f"Final memory: {df_final.estimated_size('mb'):.1f} MB")

# Check for remaining nulls
total_nulls = sum([df_final[col].null_count() for col in df_final.columns])
print(f"\nTotal nulls remaining: {total_nulls:,}")

if total_nulls > 0:
    print("\nColumns with nulls:")
    for col in df_final.columns:
        null_count = df_final[col].null_count()
        if null_count > 0:
            print(f"  {col}: {null_count:,}")

# Show final feature list
print("\n" + "=" * 50)
print("FINAL FEATURE LIST")
print("=" * 50)

# Categorize features
target_features = ['occscore', 'classwkr']
weight_features = ['perwt', 'hhwt']
identifier_features = ['year', 'sample']
engineered_features = [
    'is_immigrant', 'years_in_us', 'citizenship_status', 'is_naturalized',
    'age_at_arrival', 'degree_category', 'is_stem', 'proficient_english',
    'origin_region', 'family_burden', 'is_married', 'has_children',
    'immigrant_x_education', 'immigrant_x_stem', 'immigrant_x_english', 'tenure_x_education'
]

print("\nTarget variables:")
for f in target_features:
    if f in df_final.columns:
        print(f"  ✓ {f}")

print("\nKey engineered features:")
for f in engineered_features:
    if f in df_final.columns:
        print(f"  ✓ {f}")

print("\nAll columns:")
print(df_final.columns)

# Save
output_file = "../ipums_data_ml_ready.parquet"
print(f"\nSaving to {output_file}...")
df_final.write_parquet(output_file, compression='snappy')
print("✓ Saved successfully!")

# Clean up memory
del df
gc.collect()

print("\n" + "=" * 50)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 50)
print(f"Output: {output_file}")
print(f"Ready for modeling with {df_final.shape[0]:,} observations")
print(f"and {df_final.shape[1]} features")