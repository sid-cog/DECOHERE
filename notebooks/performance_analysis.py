# --- Cell 5: Performance Metrics Analysis ---
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_performance_metrics(metrics_file: str, target_date: str, run_id: str, output_dir: str):
    """
    Analyze and visualize performance metrics from a JSON file.
    
    Args:
        metrics_file: Path to the metrics JSON file
        target_date: Date of the analysis
        run_id: Run identifier
        output_dir: Directory to save analysis results
    """
    # Load the performance metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Display performance metrics
    print("\nAverage Performance Metrics:")
    print("=" * 50)
    print(f"Average RMSE: {metrics['average_metrics']['rmse']:.4f}")
    print(f"Average R2 Score: {metrics['average_metrics']['r2']:.4f}")
    print(f"Average Number of Trees: {metrics['average_metrics']['n_trees']:.1f}")

    # Display cross-validation results
    print("\nCross-Validation Results:")
    print("=" * 50)
    cv_df = pd.DataFrame(metrics['cv_scores'])
    print(cv_df[['fold', 'rmse', 'r2', 'n_trees']])

    # Plot cross-validation performance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=cv_df, x='fold', y='rmse')
    plt.title('RMSE by Fold')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')

    plt.subplot(1, 2, 2)
    sns.barplot(data=cv_df, x='fold', y='r2')
    plt.title('R2 Score by Fold')
    plt.xlabel('Fold')
    plt.ylabel('R2 Score')

    plt.tight_layout()
    plt.show()

    # Display feature importance
    print("\nTop 10 Most Important Features:")
    print("=" * 50)
    importance_df = pd.DataFrame(metrics['importance_scores'])
    importance_df = importance_df.sort_values('mean_importance', ascending=False)
    print(importance_df.head(10))

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(10), x='mean_importance', y='feature')
    plt.title('Top 10 Features by Importance')
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Display SHAP values summary
    print("\nSHAP Values Summary:")
    print("=" * 50)
    shap_summary = pd.DataFrame(metrics['shap_values_summary'])
    print(shap_summary)

    # Save the analysis results
    analysis_file = os.path.join(output_dir, f"performance_analysis_{target_date}.txt")

    with open(analysis_file, 'w') as f:
        f.write(f"Performance Analysis for {target_date}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Average Performance Metrics:\n")
        f.write(f"Average RMSE: {metrics['average_metrics']['rmse']:.4f}\n")
        f.write(f"Average R2 Score: {metrics['average_metrics']['r2']:.4f}\n")
        f.write(f"Average Number of Trees: {metrics['average_metrics']['n_trees']:.1f}\n\n")
        
        f.write("Cross-Validation Results:\n")
        f.write(cv_df[['fold', 'rmse', 'r2', 'n_trees']].to_string() + "\n\n")
        
        f.write("Top 10 Most Important Features:\n")
        f.write(importance_df.head(10).to_string() + "\n\n")
        
        f.write("SHAP Values Summary:\n")
        f.write(shap_summary.to_string())

    print(f"\nAnalysis results saved to: {analysis_file}")

if __name__ == "__main__":
    # Example usage
    target_date = '2024-09-02'
    run_id = 'your_run_id'  # Replace with actual run_id
    metrics_file = os.path.join(
        'data/results/feature_selection',
        f"run_{run_id}",
        f"performance_metrics_{target_date}.json"
    )
    output_dir = os.path.join('data/results/feature_selection', f"run_{run_id}")
    
    analyze_performance_metrics(metrics_file, target_date, run_id, output_dir) 