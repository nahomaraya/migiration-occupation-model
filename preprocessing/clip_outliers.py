import pandas as pd

def clip_outliers(df: pd.DataFrame, column: str, lower_quantile, upper_quantile) -> None:
    """
    Clip outliers using IQR method (in-place).

    Args:
        df: DataFrame to modify
        column: Column name to clip
    """
    Q1 = df[column].quantile(lower_quantile)
    Q3 = df[column].quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
