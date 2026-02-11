# IPUMS Immigration Study - Occupational Outcomes Prediction

A machine learning pipeline using XGBoost to predict occupational outcomes for immigrants and natives in the US labor force, based on IPUMS Census microdata.
<a href="/docs/migiration-occupation-model.pdf">Download paper</a>).

### Prerequisites
```bash
# Python 3.8+
pip install polars pandas numpy scikit-learn xgboost matplotlib seaborn torch
```

### GPU Setup (Optional but Recommended)
```bash
# For CUDA support (10-50x speedup on RTX 4060)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Pipeline Steps

### **Data Extraction** (`extract.py`)
Converts raw IPUMS `.dta` file to efficient Parquet format.
### **Missing Value Handling** (`missing_values.py`)
Handles structural vs. true missing data with domain knowledge.
```python
# Structural missingness: Migration variables (natives have no immigration year)
# True missingness: Random non-response (impute with median + indicator)
```

**Strategies**:
- **Migration Variables** (Structural):
  - `YRIMMIG`: Natives → 0, Immigrants → actual year
  - `YRNATUR`: Non-citizens/Natives → 0, Naturalized → actual year
  - Creates: `years_in_us` (exposure to US economy)
  
- **True Missing** (Random):
  - Numeric: Median imputation + `_missing` indicator
  - Categorical: Sentinel value (-1) + `_missing` indicator


### **Feature Engineering** (`feature_engineering.py`)
Creates research-relevant features from raw variables.
```
 - DEGFIELD → is_stem (1 if Engineering/Science/Math, 0 otherwise)
 - SPEAKENG → proficient_english (1 if speaks well/very well, 0 otherwise)
 - BPL (birthplace) → origin_region (18 categories)
 - origin_development_level (6 economic groups)
 - years_in_us: For immigrants (YEAR - YRIMMIG), For natives (AGE)
 - age_at_arrival: AGE - years_in_us (immigrants only)
 - is_naturalized: Binary flag from citizenship_status
 - family_burden: NCHILD + NSIBS
 - is_married: Binary from MARST
 - has_children: Binary from NCHILD > 0
 - immigrant_x_education: Tests differential returns to education
 - immigrant_x_stem: Tests STEM credential discounting
 - immigrant_x_english: Tests language proficiency effects
 - tenure_x_education: Tests if returns grow with US experience
```

## **Reusable Modules**

Contains reusable preprocessing and machine learning and the core components can be adapted to any tabular dataset with minimal modifications.

Complete pipeline for any dataset example:
```python
from preprocessing.preprocess_data_xgboost import preprocess_data_xgboost, create_dmatrix_from_splits
from models.xgboost_model import train_xgboost_regressor
from models.metrics import evaluate_model_comprehensive
# 1. Load YOUR data
X_train, X_test, y_train, y_test, w_train, w_test = preprocess_data_xgboost(
    data_path="your_data.csv",
    target_column='your_target',
    test_size=0.2
)

# 2. Create DMatrix
dtrain, dtest = create_dmatrix_from_splits(X_train, X_test, y_train, y_test, w_train, w_test)

# 3. Train
results = train_xgboost_regressor(dtrain, dtest, use_gpu=True, save_dir='./results')

# 4. Metrics
results = evaluate_model_comprehensive(
    y_train_true=y_train,
    y_train_pred=train_predictions,
    y_test_true=y_test,
    y_test_pred=test_predictions,
    task='regression',  # or 'classification'
    save_dir='./results'  # Auto-saves all plots
)

print(f"RMSE: {results['test_rmse']:.4f}")
print(f"R²: {results['test_r2']:.4f}")
# Done! Model trained, metrics calculated, plots saved.

```

