import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

    def generate_descriptive_stats(self):
        """Computes comprehensive central tendencies and dispersion metrics."""
        stats_records = []
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if not series.empty:
                stats_records.append({
                    'Feature': col,
                    'Mean': round(float(series.mean()), 2),
                    'Median': round(float(series.median()), 2),
                    'Variance': round(float(series.var()), 2),
                    'Std_Deviation': round(float(series.std()), 2),
                    'Skewness': round(float(skew(series)), 2),
                    'Kurtosis': round(float(kurtosis(series)), 2)
                })
        return stats_records

    def get_chart_data(self):
        """Extracts values from features to populate JavaScript chart frameworks."""
        chart_payload = {
            "univariate_labels": [], "univariate_values": [],
            "bivariate_x": [], "bivariate_y": [],
            "has_numeric": False, "has_bivariate": False
        }
        
        # 1. Handle Univariate Sample Mapping (Default to first numeric column distribution)
        if self.numeric_cols:
            chart_payload["has_numeric"] = True
            target_col = self.numeric_cols[0]
            # Create a 10-bin frequency distribution list for the histogram chart
            counts, bins = np.histogram(self.df[target_col].dropna(), bins=10)
            chart_payload["univariate_labels"] = [f"{round(bins[i],1)}-{round(bins[i+1],1)}" for i in range(len(counts))]
            chart_payload["univariate_values"] = counts.tolist()
            chart_payload["active_univariate_name"] = target_col

            # 2. Handle Bivariate Mapping (Default to first two numeric columns)
            if len(self.numeric_cols) >= 2:
                chart_payload["has_bivariate"] = True
                col_x, col_y = self.numeric_cols[0], self.numeric_cols[1]
                # Sample the top 50 rows to keep scatter charts lightning fast and responsive
                sampled_df = self.df[[col_x, col_y]].dropna().head(50)
                chart_payload["bivariate_x"] = sampled_df[col_x].tolist()
                chart_payload["bivariate_y"] = sampled_df[col_y].tolist()
                chart_payload["active_x_name"] = col_x
                chart_payload["active_y_name"] = col_y

        return chart_payload