import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
import json
import requests
import sys
from pathlib import Path

class DataAnalyzer:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.token = os.environ.get("AIPROXY_TOKEN")
        if not self.token:
            raise ValueError("AIPROXY_TOKEN environment variable not set")
        
    def analyze(self) -> None:
        """Main analysis workflow"""
        # Get data overview
        overview = self._get_data_overview()
        
        # Initial LLM analysis to guide our approach
        analysis_plan = self._get_analysis_plan(overview)
        
        # Perform statistical analysis
        stats_results = self._analyze_statistics()
        
        # Generate visualizations
        viz_paths = self._create_visualizations(stats_results)
        
        # Create narrative
        self._generate_narrative(overview, stats_results, viz_paths)

    def _get_data_overview(self) -> Dict[str, Any]:
        """Generate dataset overview"""
        return {
            "shape": self.df.shape,
            "columns": list(self.df.dtypes.to_dict().items()),
            "sample": self.df.head(3).to_dict(),
            "missing": self.df.isnull().sum().to_dict(),
            "numeric_summary": self._safe_describe()
        }

    def _safe_describe(self) -> Dict[str, Any]:
        """Generate safe numerical summaries"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}
        return self.df[numeric_cols].describe().to_dict()

    def _get_analysis_plan(self, overview: Dict[str, Any]) -> Dict[str, Any]:
        """Get analysis suggestions from LLM"""
        prompt = {
            "messages": [{
                "role": "user",
                "content": f"Given this dataset overview: {json.dumps(overview)}, suggest 3 specific analyses that would be most insightful. Format as JSON with keys 'analyses' (list of analysis names) and 'reasons' (list of explanations)."
            }]
        }
        
        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except:
            return {"analyses": [], "reasons": []}

    def _analyze_statistics(self) -> Dict[str, Any]:
        """Perform statistical analysis"""
        results = {
            "correlation": self._safe_correlation(),
            "clusters": self._safe_clustering(),
            "outliers": self._detect_outliers()
        }
        return results

    def _safe_correlation(self) -> Dict[str, float]:
        """Calculate correlations for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        return numeric_df.corr().to_dict()

    def _safe_clustering(self) -> Dict[str, List[int]]:
        """Perform basic clustering on numeric data"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        
        from sklearn.cluster import KMeans
        n_clusters = min(3, len(self.df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Handle missing values
        filled_df = numeric_df.fillna(numeric_df.mean())
        clusters = kmeans.fit_predict(filled_df)
        return {"clusters": clusters.tolist()}

    def _detect_outliers(self) -> Dict[str, List[int]]:
        """Detect outliers using IQR method"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        
        outliers = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_indices = numeric_df[
                (numeric_df[col] < (Q1 - 1.5 * IQR)) | 
                (numeric_df[col] > (Q3 + 1.5 * IQR))
            ].index.tolist()
            if outlier_indices:
                outliers[col] = outlier_indices
        return outliers

    def _create_visualizations(self, stats_results: Dict[str, Any]) -> List[str]:
        """Generate visualizations based on analysis results"""
        viz_paths = []
        
        # Correlation heatmap
        if stats_results["correlation"]:
            corr_matrix = pd.DataFrame(stats_results["correlation"])
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig("correlation.png")
            plt.close()
            viz_paths.append("correlation.png")

        # Distribution plot
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(12, 6))
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                sns.kdeplot(data=self.df[col].dropna(), label=col)
            plt.title("Distribution of Numeric Variables")
            plt.legend()
            plt.tight_layout()
            plt.savefig("distribution.png")
            plt.close()
            viz_paths.append("distribution.png")

        # Cluster visualization if we have numeric data
        if stats_results["clusters"]:
            plt.figure(figsize=(10, 6))
            numeric_df = self.df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                plt.scatter(
                    numeric_df.iloc[:, 0],
                    numeric_df.iloc[:, 1],
                    c=stats_results["clusters"]["clusters"],
                    cmap='viridis'
                )
                plt.xlabel(numeric_df.columns[0])
                plt.ylabel(numeric_df.columns[1])
                plt.title("Cluster Analysis")
                plt.tight_layout()
                plt.savefig("clusters.png")
                plt.close()
                viz_paths.append("clusters.png")

        return viz_paths

    def _generate_narrative(self, overview: Dict[str, Any], 
                          stats_results: Dict[str, Any], 
                          viz_paths: List[str]) -> None:
        """Generate narrative README using LLM"""
        # First, analyze images
        image_insights = []
        for path in viz_paths:
            image_insights.append(self._analyze_image(path))
        
        # Create narrative prompt
        narrative_prompt = {
            "messages": [{
                "role": "user",
                "content": f"""Create a data analysis story in Markdown format with these components:
1. Overview: {json.dumps(overview)}
2. Statistical Analysis: {json.dumps(stats_results)}
3. Visualization Insights: {json.dumps(image_insights)}

Include these sections:
- Brief data description
- Analysis methodology
- Key insights
- Actionable implications

Format: Professional markdown with headers and sections."""
            }]
        }
        
        narrative = self._call_llm(narrative_prompt)
        with open("README.md", "w") as f:
            f.write(narrative)

    def _analyze_image(self, image_path: str) -> Dict[str, str]:
        """Analyze visualization using LLM vision capabilities"""
        with open(image_path, "rb") as img:
            image_data = img.read()
        
        response = requests.post(
            "https://api.aiproxy.all-it.services/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this visualization. What are the key insights?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ]
            }
        )
        return {"path": image_path, "analysis": response.json()["choices"][0]["message"]["content"]}

    def _call_llm(self, prompt: Dict[str, Any]) -> str:
        """Make API call to AI Proxy"""
        response = requests.post(
            "https://api.aiproxy.all-it.services/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            },
            json=prompt
        )
        return response.json()["choices"][0]["message"]["content"]

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: File {csv_path} not found")
        sys.exit(1)
        
    analyzer = DataAnalyzer(csv_path)
    analyzer.analyze()

if __name__ == "__main__":
    main()
