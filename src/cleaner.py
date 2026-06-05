import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class ProductionPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def sanitize_and_coerce_numeric(self, series: pd.Series) -> pd.Series:
        """Strips trailing text units (kmpl, CC, bhp, ₹, %) and forces numeric float64 type."""
        clean_series = series.astype(str).str.strip().str.lower()
        missing_indicators = ['n/a', 'na', 'unknown', 'null', 'none', '-']
        clean_series = clean_series.replace(missing_indicators, np.nan)
        
        def extract_number(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            val = val.replace(',', '')
            match = re.search(r'[-+]?\d*\.\d+|\d+', val)
            return float(match.group()) if match else np.nan

        return clean_series.apply(extract_number)

    def remove_outliers_iqr(self, df_input, columns, factor=1.5):
        clean_df = df_input.copy()
        for col in columns:
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
        return clean_df

    def run_transformations(self, form_data, numeric_cols, cat_cols):
        """Compiles custom UI configurations into a clean processed DataFrame."""
        pipeline_df = self.df.copy()

        # 1. Text-To-Numeric Unit Stripping & Coercion
        coerce_cols = form_data.getlist('coerce_cols')
        for col in coerce_cols:
            if col in pipeline_df.columns:
                pipeline_df[col] = self.sanitize_and_coerce_numeric(pipeline_df[col])
                # Shift from categorical metadata array tracking to numeric array tracking
                if col in cat_cols: cat_cols.remove(col)
                if col not in numeric_cols: numeric_cols.append(col)

        # 2. Handle Duplicates
        dup_policy = form_data.get('dup_policy')
        if dup_policy == "first":
            pipeline_df = pipeline_df.drop_duplicates(keep='first')
        elif dup_policy == "last":
            pipeline_df = pipeline_df.drop_duplicates(keep='last')

        # 3. Handle Imputations column-by-column
        for col in numeric_cols + cat_cols:
            strat = form_data.get(f"imp_{col}")
            if strat == "median":
                pipeline_df[col] = pipeline_df[col].fillna(pipeline_df[col].median())
            elif strat == "mean":
                pipeline_df[col] = pipeline_df[col].fillna(pipeline_df[col].mean())
            elif strat == "mode":
                pipeline_df[col] = pipeline_df[col].fillna(pipeline_df[col].mode()[0])
            elif strat == "zero":
                pipeline_df[col] = pipeline_df[col].fillna(0)
            elif strat == "unknown":
                pipeline_df[col] = pipeline_df[col].fillna("Unknown")
            elif strat == "drop":
                pipeline_df = pipeline_df.dropna(subset=[col])

        # 4. Handle Log Transformations
        skew_cols = form_data.getlist('skew_cols')
        for col in skew_cols:
            if col in pipeline_df.columns:
                pipeline_df[col] = np.log1p(pipeline_df[col])

        # 5. Handle Outliers
        outlier_cols = form_data.getlist('outlier_cols')
        if outlier_cols and numeric_cols:
            iqr_factor = float(form_data.get('iqr_factor', 1.5))
            # Filter only numeric columns that exist after coercion updates
            valid_outlier_cols = [c for c in outlier_cols if c in pipeline_df.columns and np.issubdtype(pipeline_df[c].dtype, np.number)]
            pipeline_df = self.remove_outliers_iqr(pipeline_df, valid_outlier_cols, factor=iqr_factor)

        # 🌟 EXTRACT PRE-ENCODED DATA REFERENCE HERE
        # This frame holds perfectly cleaned, unencoded, human-readable features
        pre_encoded_snapshot = pipeline_df.copy()

        # Isolate Target column before final array modifications
        target_variable = form_data.get('target_variable')
        target_series = None
        if target_variable and target_variable in pipeline_df.columns:
            target_series = pipeline_df[target_variable].copy()
            pipeline_df = pipeline_df.drop(columns=[target_variable])
            if target_variable in numeric_cols: numeric_cols.remove(target_variable)
            if target_variable in cat_cols: cat_cols.remove(target_variable)

        # 6. Feature Scaling
        if form_data.get('apply_scaling') and numeric_cols:
            scaler = StandardScaler()
            pipeline_df[numeric_cols] = scaler.fit_transform(pipeline_df[numeric_cols])

        # 7. One-Hot Categorical Encoding
        if form_data.get('apply_encoding') and cat_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            encoded_mat = encoder.fit_transform(pipeline_df[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)
            encoded_df = pd.DataFrame(encoded_mat, columns=encoded_cols, index=pipeline_df.index)
            pipeline_df = pipeline_df.drop(columns=cat_cols).join(encoded_df)

        # Reattach pristine target column
        if target_series is not None:
            pipeline_df[target_variable] = target_series

        return pipeline_df, pre_encoded_snapshot