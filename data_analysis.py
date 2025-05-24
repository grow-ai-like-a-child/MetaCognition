import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score, roc_curve
import json
import h5py
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MetacognitionDataManager:
    """
    Comprehensive data management for metacognition experiments.
    Handles saving, loading, and organizing experimental data.
    """
    
    def __init__(self, base_output_dir: str = "experiment_results"):
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.base_dir / "data"
        self.plots_dir = self.base_dir / "plots"
        self.reports_dir = self.base_dir / "reports"
        self.raw_dir = self.base_dir / "raw_data"
        
        for directory in [self.data_dir, self.plots_dir, self.reports_dir, self.raw_dir]:
            directory.mkdir(exist_ok=True)
    
    def save_experiment_data(self, data: List[Dict], session_id: str, 
                           formats: List[str] = ['csv', 'json', 'hdf5']) -> Dict[str, str]:
        """
        Save experiment data in multiple formats.
        
        Args:
            data: List of trial dictionaries
            session_id: Session identifier
            formats: List of formats to save ('csv', 'json', 'hdf5', 'pickle')
            
        Returns:
            Dictionary mapping format to saved file path
        """
        df = pd.DataFrame(data)
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for fmt in formats:
            if fmt == 'csv':
                filepath = self.data_dir / f"{session_id}_{timestamp}.csv"
                df.to_csv(filepath, index=False)
                
            elif fmt == 'json':
                filepath = self.data_dir / f"{session_id}_{timestamp}.json"
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
            elif fmt == 'hdf5':
                filepath = self.data_dir / f"{session_id}_{timestamp}.h5"
                with h5py.File(filepath, 'w') as f:
                    # Save main dataframe
                    df_numeric = df.select_dtypes(include=[np.number])
                    df_string = df.select_dtypes(include=['object'])
                    
                    if not df_numeric.empty:
                        f.create_dataset('numeric_data', data=df_numeric.values)
                        f.attrs['numeric_columns'] = [col.encode() for col in df_numeric.columns]
                    
                    if not df_string.empty:
                        dt = h5py.special_dtype(vlen=str)
                        for col in df_string.columns:
                            f.create_dataset(f'string_{col}', data=df_string[col].astype(str), dtype=dt)
                    
                    # Save metadata
                    f.attrs['session_id'] = session_id.encode()
                    f.attrs['timestamp'] = timestamp.encode()
                    f.attrs['n_trials'] = len(data)
                    
            elif fmt == 'pickle':
                filepath = self.data_dir / f"{session_id}_{timestamp}.pkl"
                df.to_pickle(filepath)
            
            saved_files[fmt] = str(filepath)
        
        return saved_files
    
    def load_experiment_data(self, session_id: str = None, 
                           file_path: str = None, format: str = 'csv') -> pd.DataFrame:
        """
        Load experiment data from saved files.
        
        Args:
            session_id: Session ID to load (loads most recent if None)
            file_path: Direct file path to load
            format: File format to load
            
        Returns:
            DataFrame with experiment data
        """
        if file_path:
            filepath = Path(file_path)
        else:
            # Find most recent file for session
            if session_id:
                pattern = f"{session_id}_*.{format}"
            else:
                pattern = f"*.{format}"
            
            files = list(self.data_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No {format} files found for session {session_id}")
            
            filepath = max(files, key=lambda x: x.stat().st_mtime)
        
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif format == 'pickle':
            return pd.read_pickle(filepath)
        elif format == 'hdf5':
            # Reconstruct DataFrame from HDF5
            with h5py.File(filepath, 'r') as f:
                df_data = {}
                
                if 'numeric_data' in f:
                    numeric_cols = [col.decode() for col in f.attrs['numeric_columns']]
                    numeric_data = f['numeric_data'][:]
                    for i, col in enumerate(numeric_cols):
                        df_data[col] = numeric_data[:, i]
                
                # Load string columns
                for key in f.keys():
                    if key.startswith('string_'):
                        col_name = key[7:]  # Remove 'string_' prefix
                        df_data[col_name] = [s.decode() if isinstance(s, bytes) else s for s in f[key][:]]
                
                return pd.DataFrame(df_data)
    
    def combine_sessions(self, session_ids: List[str] = None) -> pd.DataFrame:
        """
        Combine data from multiple sessions.
        
        Args:
            session_ids: List of session IDs to combine (all if None)
            
        Returns:
            Combined DataFrame
        """
        if session_ids is None:
            # Get all CSV files
            files = list(self.data_dir.glob("*.csv"))
        else:
            files = []
            for session_id in session_ids:
                session_files = list(self.data_dir.glob(f"{session_id}_*.csv"))
                files.extend(session_files)
        
        if not files:
            raise FileNotFoundError("No data files found to combine")
        
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            # Extract session info from filename
            session_info = file.stem.split('_')[0]
            df['file_session'] = session_info
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

class MetacognitionAnalyzer:
    """
    Comprehensive analysis of metacognitive performance.
    Calculates key metrics and performs statistical comparisons.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with experiment data.
        
        Args:
            data: DataFrame with columns: model, trial, correct, confidence, response_time, etc.
        """
        self.data = data.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and validate data for analysis."""
        required_columns = ['model', 'correct', 'confidence']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert data types
        self.data['correct'] = self.data['correct'].astype(bool)
        self.data['confidence'] = pd.to_numeric(self.data['confidence'], errors='coerce')
        
        # Remove invalid confidence ratings
        self.data = self.data[self.data['confidence'].between(1, 6)]
        
        # Add derived columns
        self.data['accuracy_numeric'] = self.data['correct'].astype(int)
        
        print(f"Data prepared: {len(self.data)} trials, {self.data['model'].nunique()} models")
    
    def calculate_basic_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic performance metrics for each model."""
        metrics = {}
        
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            
            metrics[model] = {
                'n_trials': len(model_data),
                'accuracy': model_data['correct'].mean(),
                'mean_confidence': model_data['confidence'].mean(),
                'std_confidence': model_data['confidence'].std(),
                'mean_response_time': model_data.get('response_time', pd.Series([np.nan])).mean(),
                'accuracy_se': np.sqrt(model_data['correct'].mean() * (1 - model_data['correct'].mean()) / len(model_data))
            }
        
        return metrics
    
    def calculate_metacognitive_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metacognitive sensitivity (confidence difference between correct/incorrect).
        """
        sensitivity_metrics = {}
        
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            
            correct_conf = model_data[model_data['correct']]['confidence']
            incorrect_conf = model_data[~model_data['correct']]['confidence']
            
            if len(correct_conf) > 0 and len(incorrect_conf) > 0:
                # Type 1 sensitivity (traditional metacognitive sensitivity)
                meta_sens = correct_conf.mean() - incorrect_conf.mean()
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(correct_conf) - 1) * correct_conf.var() + 
                                    (len(incorrect_conf) - 1) * incorrect_conf.var()) / 
                                   (len(correct_conf) + len(incorrect_conf) - 2))
                cohens_d = meta_sens / pooled_std if pooled_std > 0 else 0
                
                # Statistical test
                t_stat, p_value = stats.ttest_ind(correct_conf, incorrect_conf)
                
                sensitivity_metrics[model] = {
                    'metacognitive_sensitivity': meta_sens,
                    'correct_confidence_mean': correct_conf.mean(),
                    'incorrect_confidence_mean': incorrect_conf.mean(),
                    'correct_confidence_std': correct_conf.std(),
                    'incorrect_confidence_std': incorrect_conf.std(),
                    'cohens_d': cohens_d,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'n_correct': len(correct_conf),
                    'n_incorrect': len(incorrect_conf)
                }
            else:
                sensitivity_metrics[model] = {
                    'metacognitive_sensitivity': np.nan,
                    'correct_confidence_mean': correct_conf.mean() if len(correct_conf) > 0 else np.nan,
                    'incorrect_confidence_mean': incorrect_conf.mean() if len(incorrect_conf) > 0 else np.nan,
                    'correct_confidence_std': correct_conf.std() if len(correct_conf) > 0 else np.nan,
                    'incorrect_confidence_std': incorrect_conf.std() if len(incorrect_conf) > 0 else np.nan,
                    'cohens_d': np.nan,
                    't_statistic': np.nan,
                    'p_value': np.nan,
                    'n_correct': len(correct_conf),
                    'n_incorrect': len(incorrect_conf)
                }
        
        return sensitivity_metrics
    
    def calculate_calibration_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate confidence calibration metrics."""
        calibration_metrics = {}
        
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            
            # Group by confidence level
            conf_groups = model_data.groupby('confidence')['correct'].agg(['mean', 'count']).reset_index()
            conf_groups = conf_groups[conf_groups['count'] >= 3]  # Minimum trials per confidence level
            
            if len(conf_groups) > 1:
                confidences = conf_groups['confidence'].values
                accuracies = conf_groups['mean'].values
                
                # Calibration slope (regression of accuracy on confidence)
                if len(confidences) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(confidences, accuracies)
                    calibration_slope = slope
                    calibration_r2 = r_value ** 2
                else:
                    calibration_slope = np.nan
                    calibration_r2 = np.nan
                
                # Overconfidence/underconfidence
                mean_conf_normalized = (model_data['confidence'].mean() - 1) / 5  # Normalize to 0-1
                mean_accuracy = model_data['correct'].mean()
                overconfidence = mean_conf_normalized - mean_accuracy
                
                # Brier score (lower is better)
                brier_score = np.mean((model_data['accuracy_numeric'] - 
                                     (model_data['confidence'] - 1) / 5) ** 2)
                
                calibration_metrics[model] = {
                    'calibration_slope': calibration_slope,
                    'calibration_r2': calibration_r2,
                    'overconfidence': overconfidence,
                    'brier_score': brier_score,
                    'mean_confidence_normalized': mean_conf_normalized,
                    'mean_accuracy': mean_accuracy,
                    'n_confidence_levels': len(conf_groups)
                }
            else:
                calibration_metrics[model] = {
                    'calibration_slope': np.nan,
                    'calibration_r2': np.nan,
                    'overconfidence': np.nan,
                    'brier_score': np.nan,
                    'mean_confidence_normalized': np.nan,
                    'mean_accuracy': model_data['correct'].mean(),
                    'n_confidence_levels': len(conf_groups)
                }
        
        return calibration_metrics
    
    def calculate_type2_roc_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate Type-2 ROC analysis for metacognitive efficiency.
        """
        type2_metrics = {}
        
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            
            if len(model_data) > 10:  # Minimum trials for ROC analysis
                # Type-2 ROC: confidence ratings predict correctness
                try:
                    auc = roc_auc_score(model_data['correct'], model_data['confidence'])
                    
                    # Get ROC curve
                    fpr, tpr, thresholds = roc_curve(model_data['correct'], model_data['confidence'])
                    
                    type2_metrics[model] = {
                        'type2_auc': auc,
                        'type2_auc_normalized': (auc - 0.5) * 2,  # Normalized to -1 to 1
                        'n_trials': len(model_data),
                        'roc_curve': {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'thresholds': thresholds.tolist()
                        }
                    }
                except ValueError:
                    # No variation in correct/incorrect or confidence
                    type2_metrics[model] = {
                        'type2_auc': np.nan,
                        'type2_auc_normalized': np.nan,
                        'n_trials': len(model_data),
                        'roc_curve': None
                    }
            else:
                type2_metrics[model] = {
                    'type2_auc': np.nan,
                    'type2_auc_normalized': np.nan,
                    'n_trials': len(model_data),
                    'roc_curve': None
                }
        
        return type2_metrics
    
    def compare_models(self) -> Dict[str, Any]:
        """Statistical comparison between models."""
        models = self.data['model'].unique()
        
        if len(models) < 2:
            return {"error": "Need at least 2 models for comparison"}
        
        comparisons = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                data1 = self.data[self.data['model'] == model1]
                data2 = self.data[self.data['model'] == model2]
                
                # Accuracy comparison
                acc1 = data1['correct'].mean()
                acc2 = data2['correct'].mean()
                
                # Confidence comparison
                conf1 = data1['confidence'].mean()
                conf2 = data2['confidence'].mean()
                
                # Statistical tests
                acc_chi2, acc_p = stats.chi2_contingency([
                    [data1['correct'].sum(), len(data1) - data1['correct'].sum()],
                    [data2['correct'].sum(), len(data2) - data2['correct'].sum()]
                ])[:2]
                
                conf_t, conf_p = stats.ttest_ind(data1['confidence'], data2['confidence'])
                
                comparisons[comparison_key] = {
                    'accuracy_diff': acc1 - acc2,
                    'accuracy_chi2': acc_chi2,
                    'accuracy_p_value': acc_p,
                    'confidence_diff': conf1 - conf2,
                    'confidence_t_stat': conf_t,
                    'confidence_p_value': conf_p,
                    'n_trials_1': len(data1),
                    'n_trials_2': len(data2)
                }
        
        return comparisons
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        basic_metrics = self.calculate_basic_metrics()
        sensitivity_metrics = self.calculate_metacognitive_sensitivity()
        calibration_metrics = self.calculate_calibration_metrics()
        type2_metrics = self.calculate_type2_roc_analysis()
        model_comparisons = self.compare_models()
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE METACOGNITION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total trials: {len(self.data)}")
        report.append(f"Models analyzed: {', '.join(self.data['model'].unique())}")
        report.append("")
        
        # Basic Metrics
        report.append("1. BASIC PERFORMANCE METRICS")
        report.append("-" * 40)
        for model, metrics in basic_metrics.items():
            report.append(f"\n{model.upper()}:")
            report.append(f"  Trials: {metrics['n_trials']}")
            report.append(f"  Accuracy: {metrics['accuracy']:.3f} ± {metrics['accuracy_se']:.3f}")
            report.append(f"  Mean Confidence: {metrics['mean_confidence']:.2f} ± {metrics['std_confidence']:.2f}")
            if not np.isnan(metrics['mean_response_time']):
                report.append(f"  Mean Response Time: {metrics['mean_response_time']:.2f}s")
        
        # Metacognitive Sensitivity
        report.append("\n\n2. METACOGNITIVE SENSITIVITY")
        report.append("-" * 40)
        for model, metrics in sensitivity_metrics.items():
            report.append(f"\n{model.upper()}:")
            if not np.isnan(metrics['metacognitive_sensitivity']):
                report.append(f"  Metacognitive Sensitivity: {metrics['metacognitive_sensitivity']:.3f}")
                report.append(f"  Correct Confidence: {metrics['correct_confidence_mean']:.2f} ± {metrics['correct_confidence_std']:.2f}")
                report.append(f"  Incorrect Confidence: {metrics['incorrect_confidence_mean']:.2f} ± {metrics['incorrect_confidence_std']:.2f}")
                report.append(f"  Effect Size (Cohen's d): {metrics['cohens_d']:.3f}")
                report.append(f"  Statistical Test: t={metrics['t_statistic']:.3f}, p={metrics['p_value']:.4f}")
            else:
                report.append("  Insufficient data for sensitivity analysis")
        
        # Calibration
        report.append("\n\n3. CONFIDENCE CALIBRATION")
        report.append("-" * 40)
        for model, metrics in calibration_metrics.items():
            report.append(f"\n{model.upper()}:")
            if not np.isnan(metrics['calibration_slope']):
                report.append(f"  Calibration Slope: {metrics['calibration_slope']:.3f}")
                report.append(f"  Calibration R²: {metrics['calibration_r2']:.3f}")
                report.append(f"  Overconfidence: {metrics['overconfidence']:.3f}")
                report.append(f"  Brier Score: {metrics['brier_score']:.3f}")
            else:
                report.append("  Insufficient data for calibration analysis")
        
        # Type-2 ROC
        report.append("\n\n4. TYPE-2 ROC ANALYSIS")
        report.append("-" * 40)
        for model, metrics in type2_metrics.items():
            report.append(f"\n{model.upper()}:")
            if not np.isnan(metrics['type2_auc']):
                report.append(f"  Type-2 AUC: {metrics['type2_auc']:.3f}")
                report.append(f"  Normalized AUC: {metrics['type2_auc_normalized']:.3f}")
            else:
                report.append("  Insufficient data for ROC analysis")
        
        # Model Comparisons
        if 'error' not in model_comparisons:
            report.append("\n\n5. MODEL COMPARISONS")
            report.append("-" * 40)
            for comparison, metrics in model_comparisons.items():
                report.append(f"\n{comparison.replace('_', ' ').upper()}:")
                report.append(f"  Accuracy Difference: {metrics['accuracy_diff']:.3f} (p={metrics['accuracy_p_value']:.4f})")
                report.append(f"  Confidence Difference: {metrics['confidence_diff']:.3f} (p={metrics['confidence_p_value']:.4f})")
        
        return "\n".join(report)

class MetacognitionVisualizer:
    """
    Create publication-ready visualizations for metacognition analysis.
    """
    
    def __init__(self, data: pd.DataFrame, output_dir: str = "experiment_results/plots"):
        self.data = data.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set consistent style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    def plot_basic_performance(self, save: bool = True) -> plt.Figure:
        """Plot basic performance metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Basic Performance Metrics by Model', fontsize=16, fontweight='bold')
        
        models = self.data['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        # Accuracy
        ax = axes[0, 0]
        accuracy_data = self.data.groupby('model')['correct'].agg(['mean', 'sem']).reset_index()
        bars = ax.bar(accuracy_data['model'], accuracy_data['mean'], 
                     yerr=accuracy_data['sem'], capsize=5, color=colors)
        ax.set_title('Accuracy by Model')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, accuracy_data['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom')
        
        # Mean Confidence
        ax = axes[0, 1]
        conf_data = self.data.groupby('model')['confidence'].agg(['mean', 'sem']).reset_index()
        bars = ax.bar(conf_data['model'], conf_data['mean'], 
                     yerr=conf_data['sem'], capsize=5, color=colors)
        ax.set_title('Mean Confidence by Model')
        ax.set_ylabel('Mean Confidence Rating')
        ax.set_ylim(1, 6)
        
        for bar, mean_val in zip(bars, conf_data['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{mean_val:.2f}', ha='center', va='bottom')
        
        # Confidence Distribution
        ax = axes[1, 0]
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model]
            conf_counts = model_data['confidence'].value_counts().sort_index()
            conf_props = conf_counts / len(model_data)
            
            x_pos = np.arange(1, 7) + i * 0.3 - len(models) * 0.15
            ax.bar(x_pos, conf_props.reindex(range(1, 7), fill_value=0), 
                  width=0.3, label=model, color=colors[i], alpha=0.7)
        
        ax.set_title('Confidence Rating Distribution')
        ax.set_xlabel('Confidence Rating')
        ax.set_ylabel('Proportion of Trials')
        ax.set_xticks(range(1, 7))
        ax.legend()
        
        # Response Time (if available)
        ax = axes[1, 1]
        if 'response_time' in self.data.columns and not self.data['response_time'].isna().all():
            rt_data = self.data.groupby('model')['response_time'].agg(['mean', 'sem']).reset_index()
            bars = ax.bar(rt_data['model'], rt_data['mean'], 
                         yerr=rt_data['sem'], capsize=5, color=colors)
            ax.set_title('Mean Response Time by Model')
            ax.set_ylabel('Response Time (seconds)')
            
            for bar, mean_val in zip(bars, rt_data['mean']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{mean_val:.2f}s', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Response Time\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Response Time by Model')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'basic_performance.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'basic_performance.pdf', bbox_inches='tight')
        
        return fig
    
    def plot_metacognitive_sensitivity(self, save: bool = True) -> plt.Figure:
        """Plot metacognitive sensitivity analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Metacognitive Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        models = self.data['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        # Confidence by correctness
        ax = axes[0]
        x_pos = np.arange(len(models))
        width = 0.35
        
        correct_conf = []
        incorrect_conf = []
        correct_sem = []
        incorrect_sem = []
        
        for model in models:
            model_data = self.data[self.data['model'] == model]
            correct_data = model_data[model_data['correct']]['confidence']
            incorrect_data = model_data[~model_data['correct']]['confidence']
            
            correct_conf.append(correct_data.mean())
            incorrect_conf.append(incorrect_data.mean())
            correct_sem.append(correct_data.sem())
            incorrect_sem.append(incorrect_data.sem())
        
        bars1 = ax.bar(x_pos - width/2, correct_conf, width, label='Correct', 
                      yerr=correct_sem, capsize=5, color='lightgreen', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, incorrect_conf, width, label='Incorrect', 
                      yerr=incorrect_sem, capsize=5, color='lightcoral', alpha=0.8)
        
        ax.set_title('Confidence by Response Correctness')
        ax.set_ylabel('Mean Confidence Rating')
        ax.set_xlabel('Model')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(1, 6)
        
        # Add value labels
        for bars, values in [(bars1, correct_conf), (bars2, incorrect_conf)]:
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Sensitivity scores
        ax = axes[1]
        sensitivity_scores = []
        sensitivity_sem = []
        
        for model in models:
            model_data = self.data[self.data['model'] == model]
            correct_data = model_data[model_data['correct']]['confidence']
            incorrect_data = model_data[~model_data['correct']]['confidence']
            
            if len(correct_data) > 0 and len(incorrect_data) > 0:
                sens = correct_data.mean() - incorrect_data.mean()
                # Bootstrap SEM for sensitivity
                n_bootstrap = 1000
                bootstrap_sens = []
                for _ in range(n_bootstrap):
                    correct_sample = np.random.choice(correct_data, len(correct_data), replace=True)
                    incorrect_sample = np.random.choice(incorrect_data, len(incorrect_data), replace=True)
                    bootstrap_sens.append(correct_sample.mean() - incorrect_sample.mean())
                
                sensitivity_scores.append(sens)
                sensitivity_sem.append(np.std(bootstrap_sens))
            else:
                sensitivity_scores.append(0)
                sensitivity_sem.append(0)
        
        bars = ax.bar(models, sensitivity_scores, yerr=sensitivity_sem, 
                     capsize=5, color=colors, alpha=0.7)
        ax.set_title('Metacognitive Sensitivity\n(Correct - Incorrect Confidence)')
        ax.set_ylabel('Sensitivity Score')
        ax.set_xlabel('Model')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, sensitivity_scores):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.05,
                   f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
        
        # Confidence distributions by correctness
        ax = axes[2]
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model]
            
            # Create violin plots
            correct_conf = model_data[model_data['correct']]['confidence']
            incorrect_conf = model_data[~model_data['correct']]['confidence']
            
            positions = [i*2, i*2+0.5]
            parts = ax.violinplot([correct_conf, incorrect_conf], positions=positions, 
                                 widths=0.4, showmeans=True)
            
            parts['bodies'][0].set_facecolor('lightgreen')
            parts['bodies'][1].set_facecolor('lightcoral')
            parts['bodies'][0].set_alpha(0.7)
            parts['bodies'][1].set_alpha(0.7)
        
        ax.set_title('Confidence Distributions\nby Correctness')
        ax.set_ylabel('Confidence Rating')
        ax.set_xlabel('Model')
        
        # Set x-axis labels
        tick_positions = [i*2 + 0.25 for i in range(len(models))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(models)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightgreen', alpha=0.7, label='Correct'),
                          Patch(facecolor='lightcoral', alpha=0.7, label='Incorrect')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'metacognitive_sensitivity.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'metacognitive_sensitivity.pdf', bbox_inches='tight')
        
        return fig
    
    def plot_confidence_calibration(self, save: bool = True) -> plt.Figure:
        """Plot confidence calibration analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Confidence Calibration Analysis', fontsize=16, fontweight='bold')
        
        models = self.data['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        # Calibration curves
        ax = axes[0, 0]
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model]
            
            # Group by confidence level
            conf_groups = model_data.groupby('confidence').agg({
                'correct': ['mean', 'sem', 'count']
            }).reset_index()
            conf_groups.columns = ['confidence', 'accuracy', 'accuracy_sem', 'count']
            conf_groups = conf_groups[conf_groups['count'] >= 3]  # Minimum trials
            
            if len(conf_groups) > 1:
                ax.errorbar(conf_groups['confidence'], conf_groups['accuracy'], 
                           yerr=conf_groups['accuracy_sem'], 
                           marker='o', label=model, color=colors[i], capsize=5)
        
        # Perfect calibration line
        ax.plot([1, 6], [1/6, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        ax.set_title('Confidence Calibration Curves')
        ax.set_xlabel('Confidence Rating')
        ax.set_ylabel('Accuracy')
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overconfidence analysis
        ax = axes[0, 1]
        overconf_data = []
        for model in models:
            model_data = self.data[self.data['model'] == model]
            mean_conf_norm = (model_data['confidence'].mean() - 1) / 5
            mean_accuracy = model_data['correct'].mean()
            overconf = mean_conf_norm - mean_accuracy
            overconf_data.append(overconf)
        
        bars = ax.bar(models, overconf_data, color=colors, alpha=0.7)
        ax.set_title('Overconfidence\n(Normalized Confidence - Accuracy)')
        ax.set_ylabel('Overconfidence')
        ax.set_xlabel('Model')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, overconf_data):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.01,
                   f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
        
        # Calibration scatter plot
        ax = axes[1, 0]
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model]
            
            # Bin confidence ratings
            conf_bins = np.linspace(1, 6, 11)
            bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
            
            bin_accuracy = []
            bin_confidence = []
            
            for j in range(len(conf_bins) - 1):
                mask = (model_data['confidence'] >= conf_bins[j]) & (model_data['confidence'] < conf_bins[j+1])
                if mask.sum() > 0:
                    bin_accuracy.append(model_data[mask]['correct'].mean())
                    bin_confidence.append(bin_centers[j])
            
            if bin_accuracy:
                ax.scatter(bin_confidence, bin_accuracy, label=model, 
                          color=colors[i], s=50, alpha=0.7)
        
        ax.plot([1, 6], [1/6, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.set_title('Confidence vs Accuracy Scatter')
        ax.set_xlabel('Mean Confidence Rating')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Brier Score
        ax = axes[1, 1]
        brier_scores = []
        for model in models:
            model_data = self.data[self.data['model'] == model]
            conf_normalized = (model_data['confidence'] - 1) / 5
            brier = np.mean((model_data['correct'].astype(int) - conf_normalized) ** 2)
            brier_scores.append(brier)
        
        bars = ax.bar(models, brier_scores, color=colors, alpha=0.7)
        ax.set_title('Brier Score\n(Lower = Better Calibration)')
        ax.set_ylabel('Brier Score')
        ax.set_xlabel('Model')
        
        for bar, val in zip(bars, brier_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'confidence_calibration.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'confidence_calibration.pdf', bbox_inches='tight')
        
        return fig
    
    def plot_type2_roc_analysis(self, save: bool = True) -> plt.Figure:
        """Plot Type-2 ROC analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Type-2 ROC Analysis (Metacognitive Efficiency)', fontsize=16, fontweight='bold')
        
        models = self.data['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        # ROC Curves
        ax = axes[0]
        auc_scores = []
        
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model]
            
            if len(model_data) > 10:
                try:
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(model_data['correct'], model_data['confidence'])
                    auc = roc_auc_score(model_data['correct'], model_data['confidence'])
                    
                    ax.plot(fpr, tpr, label=f'{model} (AUC = {auc:.3f})', 
                           color=colors[i], linewidth=2)
                    auc_scores.append(auc)
                except ValueError:
                    auc_scores.append(np.nan)
            else:
                auc_scores.append(np.nan)
        
        # Chance line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.5)')
        
        ax.set_title('Type-2 ROC Curves')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # AUC Comparison
        ax = axes[1]
        valid_aucs = [auc for auc in auc_scores if not np.isnan(auc)]
        valid_models = [model for model, auc in zip(models, auc_scores) if not np.isnan(auc)]
        valid_colors = [colors[i] for i, auc in enumerate(auc_scores) if not np.isnan(auc)]
        
        if valid_aucs:
            bars = ax.bar(valid_models, valid_aucs, color=valid_colors, alpha=0.7)
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Chance Level')
            
            ax.set_title('Type-2 AUC Scores')
            ax.set_ylabel('AUC Score')
            ax.set_xlabel('Model')
            ax.set_ylim(0, 1)
            
            for bar, val in zip(bars, valid_aucs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Insufficient Data\nfor ROC Analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'type2_roc_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'type2_roc_analysis.pdf', bbox_inches='tight')
        
        return fig
    
    def plot_trial_progression(self, save: bool = True) -> plt.Figure:
        """Plot performance and confidence over trial progression."""
        if 'trial' not in self.data.columns:
            # Create trial numbers if not available
            self.data['trial'] = self.data.groupby('model').cumcount() + 1
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance and Confidence Over Trial Progression', fontsize=16, fontweight='bold')
        
        models = self.data['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        # Rolling accuracy
        ax = axes[0, 0]
        window_size = min(10, len(self.data) // 5)  # Adaptive window size
        
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model].sort_values('trial')
            if len(model_data) >= window_size:
                rolling_acc = model_data['correct'].rolling(window=window_size, center=True).mean()
                ax.plot(model_data['trial'], rolling_acc, label=model, color=colors[i], linewidth=2)
        
        ax.set_title(f'Rolling Accuracy (Window = {window_size})')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rolling confidence
        ax = axes[0, 1]
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model].sort_values('trial')
            if len(model_data) >= window_size:
                rolling_conf = model_data['confidence'].rolling(window=window_size, center=True).mean()
                ax.plot(model_data['trial'], rolling_conf, label=model, color=colors[i], linewidth=2)
        
        ax.set_title(f'Rolling Confidence (Window = {window_size})')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Mean Confidence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cumulative accuracy
        ax = axes[1, 0]
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model].sort_values('trial')
            cumulative_acc = model_data['correct'].cumsum() / (model_data['trial'])
            ax.plot(model_data['trial'], cumulative_acc, label=model, color=colors[i], linewidth=2)
        
        ax.set_title('Cumulative Accuracy')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Cumulative Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning curve (if contrast data available)
        ax = axes[1, 1]
        if 'contrast' in self.data.columns:
            for i, model in enumerate(models):
                model_data = self.data[self.data['model'] == model].sort_values('trial')
                if len(model_data) >= window_size:
                    rolling_contrast = model_data['contrast'].rolling(window=window_size, center=True).mean()
                    ax.plot(model_data['trial'], rolling_contrast, label=model, color=colors[i], linewidth=2)
            
            ax.set_title(f'Adaptive Contrast Level (Window = {window_size})')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Contrast Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Contrast Data\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Adaptive Contrast Level')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'trial_progression.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'trial_progression.pdf', bbox_inches='tight')
        
        return fig
    
    def create_summary_dashboard(self, save: bool = True) -> plt.Figure:
        """Create a comprehensive summary dashboard."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Metacognition Investigation - Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        models = self.data['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        accuracy_data = self.data.groupby('model')['correct'].agg(['mean', 'sem'])
        bars = ax1.bar(accuracy_data.index, accuracy_data['mean'], 
                      yerr=accuracy_data['sem'], capsize=5, color=colors, alpha=0.7)
        ax1.set_title('Accuracy by Model', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, val in zip(bars, accuracy_data['mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Confidence comparison
        ax2 = fig.add_subplot(gs[0, 1])
        conf_data = self.data.groupby('model')['confidence'].agg(['mean', 'sem'])
        bars = ax2.bar(conf_data.index, conf_data['mean'], 
                      yerr=conf_data['sem'], capsize=5, color=colors, alpha=0.7)
        ax2.set_title('Mean Confidence', fontweight='bold')
        ax2.set_ylabel('Confidence Rating')
        ax2.set_ylim(1, 6)
        
        for bar, val in zip(bars, conf_data['mean']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Metacognitive sensitivity
        ax3 = fig.add_subplot(gs[0, 2])
        sensitivity_scores = []
        for model in models:
            model_data = self.data[self.data['model'] == model]
            correct_conf = model_data[model_data['correct']]['confidence'].mean()
            incorrect_conf = model_data[~model_data['correct']]['confidence'].mean()
            sensitivity_scores.append(correct_conf - incorrect_conf)
        
        bars = ax3.bar(models, sensitivity_scores, color=colors, alpha=0.7)
        ax3.set_title('Metacognitive Sensitivity', fontweight='bold')
        ax3.set_ylabel('Sensitivity Score')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, sensitivity_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.05,
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
        
        # 4. Sample size
        ax4 = fig.add_subplot(gs[0, 3])
        sample_sizes = self.data['model'].value_counts()
        bars = ax4.bar(sample_sizes.index, sample_sizes.values, color=colors, alpha=0.7)
        ax4.set_title('Sample Size', fontweight='bold')
        ax4.set_ylabel('Number of Trials')
        
        for bar, val in zip(bars, sample_sizes.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}', ha='center', va='bottom', fontsize=10)
        
        # 5. Confidence by correctness
        ax5 = fig.add_subplot(gs[1, :2])
        width = 0.35
        x_pos = np.arange(len(models))
        
        correct_conf = []
        incorrect_conf = []
        for model in models:
            model_data = self.data[self.data['model'] == model]
            correct_conf.append(model_data[model_data['correct']]['confidence'].mean())
            incorrect_conf.append(model_data[~model_data['correct']]['confidence'].mean())
        
        ax5.bar(x_pos - width/2, correct_conf, width, label='Correct', 
               color='lightgreen', alpha=0.8)
        ax5.bar(x_pos + width/2, incorrect_conf, width, label='Incorrect', 
               color='lightcoral', alpha=0.8)
        
        ax5.set_title('Confidence by Response Correctness', fontweight='bold')
        ax5.set_ylabel('Mean Confidence')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(models)
        ax5.legend()
        
        # 6. Calibration curves
        ax6 = fig.add_subplot(gs[1, 2:])
        for i, model in enumerate(models):
            model_data = self.data[self.data['model'] == model]
            conf_groups = model_data.groupby('confidence')['correct'].agg(['mean', 'count'])
            conf_groups = conf_groups[conf_groups['count'] >= 3]
            
            if len(conf_groups) > 1:
                ax6.plot(conf_groups.index, conf_groups['mean'], 
                        marker='o', label=model, color=colors[i], linewidth=2, markersize=6)
        
        ax6.plot([1, 6], [1/6, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax6.set_title('Confidence Calibration Curves', fontweight='bold')
        ax6.set_xlabel('Confidence Rating')
        ax6.set_ylabel('Accuracy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Confidence distribution heatmap
        ax7 = fig.add_subplot(gs[2, :2])
        conf_matrix = []
        for model in models:
            model_data = self.data[self.data['model'] == model]
            conf_dist = model_data['confidence'].value_counts().sort_index()
            conf_props = conf_dist / len(model_data)
            conf_matrix.append([conf_props.get(i, 0) for i in range(1, 7)])
        
        im = ax7.imshow(conf_matrix, cmap='Blues', aspect='auto')
        ax7.set_title('Confidence Distribution Heatmap', fontweight='bold')
        ax7.set_xlabel('Confidence Rating')
        ax7.set_ylabel('Model')
        ax7.set_xticks(range(6))
        ax7.set_xticklabels(range(1, 7))
        ax7.set_yticks(range(len(models)))
        ax7.set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(6):
                text = ax7.text(j, i, f'{conf_matrix[i][j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax7, label='Proportion of Trials')
        
        # 8. Performance over trials
        ax8 = fig.add_subplot(gs[2, 2:])
        if 'trial' in self.data.columns:
            window_size = min(10, len(self.data) // 10)
            for i, model in enumerate(models):
                model_data = self.data[self.data['model'] == model].sort_values('trial')
                if len(model_data) >= window_size:
                    rolling_acc = model_data['correct'].rolling(window=window_size, center=True).mean()
                    ax8.plot(model_data['trial'], rolling_acc, label=model, color=colors[i], linewidth=2)
            
            ax8.set_title(f'Rolling Accuracy Over Trials (Window={window_size})', fontweight='bold')
            ax8.set_xlabel('Trial Number')
            ax8.set_ylabel('Accuracy')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Trial Data\nNot Available', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Performance Over Trials', fontweight='bold')
        
        # 9. Summary statistics table
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('tight')
        ax9.axis('off')
        
        # Calculate summary stats
        summary_data = []
        for model in models:
            model_data = self.data[self.data['model'] == model]
            
            # Basic metrics
            accuracy = model_data['correct'].mean()
            mean_conf = model_data['confidence'].mean()
            
            # Metacognitive sensitivity
            correct_conf = model_data[model_data['correct']]['confidence'].mean()
            incorrect_conf = model_data[~model_data['correct']]['confidence'].mean()
            meta_sens = correct_conf - incorrect_conf
            
            # Overconfidence
            conf_norm = (mean_conf - 1) / 5
            overconf = conf_norm - accuracy
            
            # Type-2 AUC
            try:
                auc = roc_auc_score(model_data['correct'], model_data['confidence'])
            except:
                auc = np.nan
            
            summary_data.append([
                model, 
                f"{accuracy:.3f}",
                f"{mean_conf:.2f}",
                f"{meta_sens:.3f}",
                f"{overconf:.3f}",
                f"{auc:.3f}" if not np.isnan(auc) else "N/A",
                len(model_data)
            ])
        
        columns = ['Model', 'Accuracy', 'Mean Conf.', 'Meta Sens.', 'Overconf.', 'Type-2 AUC', 'N Trials']
        table = ax9.table(cellText=summary_data, colLabels=columns, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax9.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        if save:
            plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'summary_dashboard.pdf', bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(self, save: bool = True) -> Dict[str, plt.Figure]:
        """Generate all visualization plots."""
        plots = {}
        
        print("Generating basic performance plots...")
        plots['basic_performance'] = self.plot_basic_performance(save=save)
        
        print("Generating metacognitive sensitivity plots...")
        plots['metacognitive_sensitivity'] = self.plot_metacognitive_sensitivity(save=save)
        
        print("Generating confidence calibration plots...")
        plots['confidence_calibration'] = self.plot_confidence_calibration(save=save)
        
        print("Generating Type-2 ROC analysis plots...")
        plots['type2_roc'] = self.plot_type2_roc_analysis(save=save)
        
        print("Generating trial progression plots...")
        plots['trial_progression'] = self.plot_trial_progression(save=save)
        
        print("Generating summary dashboard...")
        plots['summary_dashboard'] = self.create_summary_dashboard(save=save)
        
        if save:
            print(f"All plots saved to: {self.output_dir}")
        
        return plots

# Main analysis function
def analyze_experiment_data(data_file: str, output_dir: str = "experiment_results"):
    """
    Complete analysis pipeline for metacognition experiment data.
    
    Args:
        data_file: Path to CSV file with experiment data
        output_dir: Directory for saving results
    """
    print("Starting comprehensive metacognition analysis...")
    
    # Initialize components
    data_manager = MetacognitionDataManager(output_dir)
    
    # Load data
    print(f"Loading data from: {data_file}")
    data = pd.read_csv(data_file)
    
    # Initialize analyzer and visualizer
    analyzer = MetacognitionAnalyzer(data)
    visualizer = MetacognitionVisualizer(data, output_dir + "/plots")
    
    # Generate comprehensive report
    print("Generating analysis report...")
    report = analyzer.generate_comprehensive_report()
    
    # Save report
    report_file = Path(output_dir) / "reports" / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Analysis report saved to: {report_file}")
    
    # Generate all visualizations
    print("Generating visualizations...")
    plots = visualizer.generate_all_plots(save=True)
    
    # Save detailed metrics as JSON
    basic_metrics = analyzer.calculate_basic_metrics()
    sensitivity_metrics = analyzer.calculate_metacognitive_sensitivity()
    calibration_metrics = analyzer.calculate_calibration_metrics()
    type2_metrics = analyzer.calculate_type2_roc_analysis()
    model_comparisons = analyzer.compare_models()
    
    detailed_metrics = {
        'basic_metrics': basic_metrics,
        'sensitivity_metrics': sensitivity_metrics,
        'calibration_metrics': calibration_metrics,
        'type2_metrics': type2_metrics,
        'model_comparisons': model_comparisons,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    metrics_file = Path(output_dir) / "reports" / f"detailed_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(detailed_metrics, f, indent=2, default=str)
    
    print(f"Detailed metrics saved to: {metrics_file}")
    print("\nAnalysis complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(report.split('\n\n')[1])  # Print basic metrics section
    
    return {
        'data': data,
        'analyzer': analyzer,
        'visualizer': visualizer,
        'plots': plots,
        'report': report,
        'metrics': detailed_metrics
    }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "experiment_results"
        
        results = analyze_experiment_data(data_file, output_dir)
    else:
        print("Usage: python data_analysis.py <data_file.csv> [output_directory]")
        print("Example: python data_analysis.py experiment_results/session_20241201_143022.csv") 