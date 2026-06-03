import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

    def generate_descriptive_stats(self):
        stats_records = []
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if not series.empty:
                stats_records.append({
                    'Feature': col,
                    'Mean': round(series.mean(), 2),
                    'Median': round(series.median(), 2),
                    'Variance': round(series.var(), 2),
                    'Std_Deviation': round(series.std(), 2),
                    'Skewness': round(skew(series), 2),
                    'Kurtosis': round(kurtosis(series), 2)
                })
        return stats_records