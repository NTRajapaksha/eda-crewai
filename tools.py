import os
import datetime
import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class ToolInputSchema(BaseModel):
    file_path: str = Field(..., description="The path to the dataset file, e.g., 'retail_store_sales.csv'")

class DatasetProfileTool(BaseTool):
    name: str = "Dataset Profile Tool"
    description: str = "Loads the dataset from the given file_path and returns a structured text summary: column names, dtypes as stored, number of unique values per column, percentage missing per column, and 5 sample rows formatted as a readable table."
    args_schema: type[BaseModel] = ToolInputSchema

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: Dataset not found at {file_path}. Please ensure the file is present."
    
        try:
            df = pd.read_csv(file_path)
            profile_parts = ["### Dataset Profile\n"]
            profile_parts.append(f"Total Rows: {len(df)}\nTotal Columns: {len(df.columns)}\n")
        
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique_count = df[col].nunique()
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                samples = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()
            
                profile_parts.append(
                    f"- **{col}**: Dtype: {dtype}, Unique: {unique_count}, "
                    f"Missing: {missing_pct:.2f}%, Samples: {samples}"
                )
        
            profile_parts.append("\n### 5 Sample Rows:\n")
            profile_parts.append(df.head(5).to_markdown())
        
            return "\n".join(profile_parts)
        except Exception as e:
            return f"Error reading dataset: {str(e)}"

class StatisticalSummaryTool(BaseTool):
    name: str = "Statistical Summary Tool"
    description: str = "Computes statistical summaries for the given file_path: mean, median, std, min, max, percentiles for numeric cols; top 5 value counts for categorical cols; Pearson correlation matrix; and class distribution of target 'Payment Method'."
    args_schema: type[BaseModel] = ToolInputSchema

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: Dataset not found at {file_path}."
    
        try:
            df = pd.read_csv(file_path)
            summary = ["### Statistical Summary\n"]
        
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
            if len(numeric_cols) > 0:
                summary.append("#### Numeric Columns Summaries:\n")
                stats = df[numeric_cols].describe(percentiles=[.25, .5, .75]).T
                summary.append(stats[['mean', '50%', 'std', 'min', 'max', '25%', '75%']].to_markdown())
                summary.append("\n\n#### Correlation Matrix (Numeric):\n")
                summary.append(df[numeric_cols].corr(method='pearson').to_markdown())
                summary.append("\n")
            
            if len(categorical_cols) > 0:
                summary.append("\n#### Categorical Columns Top 5 Value Counts:\n")
                for col in categorical_cols:
                    summary.append(f"**{col}**:")
                    counts = df[col].value_counts().head(5)
                    for val, count in counts.items():
                        summary.append(f"  - {val}: {count}")
                    summary.append("")
                
            if 'Payment Method' in df.columns:
                summary.append("\n#### Target Class Distribution (Payment Method):\n")
                target_counts = df['Payment Method'].value_counts(normalize=True) * 100
                for val, pct in target_counts.items():
                    summary.append(f"  - {val}: {pct:.2f}%")
                
            return "\n".join(summary)
        except Exception as e:
            return f"Error computing statistics: {str(e)}"

class OutlierDetectionTool(BaseTool):
    name: str = "Outlier Detection Tool"
    description: str = "Computes outliers from the given file_path for numeric columns (Z-score > 3, IQR > 1.5) and flags placeholder strings (ERROR, UNKNOWN, N/A, -) in all columns."
    args_schema: type[BaseModel] = ToolInputSchema

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: Dataset not found at {file_path}."
        
        try:
            df = pd.read_csv(file_path)
            anomalies = ["### Anomaly Detection Report\n"]
        
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
            for col in df.columns:
                anomalies.append(f"#### Column: {col}")
                issues_found = False
            
                # Placeholder checks for object/string columns
                if df[col].dtype == 'object':
                    placeholders = ['ERROR', 'UNKNOWN', 'N/A', '-', 'NA', 'null', '']
                    counts = df[col].astype(str).str.strip().str.upper().isin(placeholders).sum()
                    if counts > 0:
                        anomalies.append(f"- **Placeholder Strings Found**: {counts} occurrences of suspicious/missing placeholder values.")
                        issues_found = True
                    
                # Outlier checks for numeric columns
                if col in numeric_cols:
                    # Z-Score
                    mean = df[col].mean()
                    std = df[col].std()
                    if pd.notnull(mean) and pd.notnull(std) and std > 0:
                        z_scores = np.abs((df[col] - mean) / std)
                        z_outliers = (z_scores > 3).sum()
                        if z_outliers > 0:
                            anomalies.append(f"- **Z-Score Outliers (>3 std)**: {z_outliers} values found.")
                            issues_found = True
                        
                    # IQR
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if pd.notnull(q1) and pd.notnull(q3) and iqr > 0:
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        if iqr_outliers > 0:
                            anomalies.append(f"- **IQR Outliers (1.5x IQR)**: {iqr_outliers} values found.")
                            issues_found = True
                        
                    # Negative value checks (if it's a count/price/quantity usually it can't be negative)
                    negatives = (df[col] < 0).sum()
                    if negatives > 0:
                        anomalies.append(f"- **Negative Values**: {negatives} negative values found (check if this makes business sense).")
                        issues_found = True
                    
                if not issues_found:
                    anomalies.append("- No immediate obvious anomalies detected.")
                
                anomalies.append("")
            
            return "\n".join(anomalies)
        except Exception as e:
            return f"Error computing anomalies: {str(e)}"


