import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional
import psutil
from IPython.display import display, HTML

class PipelineManager:
    """Manages pipeline execution and monitoring"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.results_dir = Path(project_root) / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
    def get_system_resources(self) -> Dict:
        """Get current system resource information"""
        cpu_cores = psutil.cpu_count(logical=False)
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        memory_per_worker = available_memory / (cpu_cores * 2)  # Conservative estimate
        optimal_workers = min(cpu_cores, int(available_memory / memory_per_worker))
        
        return {
            'cpu_cores': cpu_cores,
            'available_memory': available_memory,
            'memory_per_worker': memory_per_worker,
            'optimal_workers': optimal_workers
        }
    
    def run_pipeline(self, 
                    start_date: datetime, 
                    end_date: datetime, 
                    num_workers: int,
                    save_results: bool = True) -> pd.DataFrame:
        """Run the pipeline with given parameters"""
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Create command
        cmd = f"python {self.project_root}/src/pipeline/run_parallel_pipeline.py " \
              f"--start-date {start_date_str} --end-date {end_date_str} " \
              f"--num-workers {num_workers}"
        
        # Run command and capture output
        output = os.popen(cmd).read().splitlines()
        
        # Parse results
        results = []
        for line in output:
            if 'Processed' in line:
                parts = line.split()
                date = parts[1]
                duration = float(parts[-2])
                results.append({'date': date, 'duration': duration})
        
        results_df = pd.DataFrame(results)
        
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = self.results_dir / f'pipeline_results_{timestamp}.csv'
            results_df.to_csv(results_path, index=False)
        
        return results_df
    
    def get_latest_results(self) -> Optional[pd.DataFrame]:
        """Get the most recent pipeline results"""
        results_files = sorted(self.results_dir.glob('pipeline_results_*.csv'))
        if not results_files:
            return None
        
        latest_file = results_files[-1]
        return pd.read_csv(latest_file)
    
    def plot_execution_times(self, results_df: pd.DataFrame) -> None:
        """Plot pipeline execution times"""
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['date'], results_df['duration'], 'o-')
        plt.title('Pipeline Execution Time by Date')
        plt.xlabel('Date')
        plt.ylabel('Duration (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_results_summary(self, results_df: pd.DataFrame) -> Dict:
        """Get summary statistics of pipeline results"""
        return {
            'total_dates': len(results_df),
            'successful': sum(results_df['success'] & ~results_df['is_missing']),
            'failed': sum(~results_df['success']),
            'missing': sum(results_df['is_missing']),
            'avg_duration': results_df['duration'].mean()
        }
    
    def display_results_summary(self, results_df: pd.DataFrame) -> None:
        """Display formatted results summary"""
        summary = self.get_results_summary(results_df)
        display(HTML(f"""
        <h3>Pipeline Results Summary</h3>
        <ul>
            <li>Total Dates: {summary['total_dates']}</li>
            <li>Successful: {summary['successful']}</li>
            <li>Failed: {summary['failed']}</li>
            <li>Missing: {summary['missing']}</li>
            <li>Average Duration: {summary['avg_duration']:.2f} seconds</li>
        </ul>
        """)) 