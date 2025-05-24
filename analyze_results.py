#!/usr/bin/env python3
"""
Quick Analysis Script for Metacognition Investigation Platform

This script provides simplified functions to analyze experimental data.
It can be used from command line or imported as a module.

Usage:
    python analyze_results.py --data path/to/data.csv --output results_folder
    
Or in Python:
    from analyze_results import quick_analysis
    results = quick_analysis("data.csv")
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Import our analysis modules
from data_analysis import (
    MetacognitionDataManager, 
    MetacognitionAnalyzer, 
    MetacognitionVisualizer,
    analyze_experiment_data
)

def quick_analysis(data_path: str, output_dir: str = None, show_plots: bool = True):
    """
    Perform quick analysis of experiment data.
    
    Args:
        data_path: Path to CSV file with experiment data
        output_dir: Output directory (auto-generated if None)
        show_plots: Whether to display plots
    
    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        # Auto-generate output directory name
        data_file = Path(data_path)
        timestamp = data_file.stem.split('_')[-1] if '_' in data_file.stem else "analysis"
        output_dir = f"analysis_results_{timestamp}"
    
    print(f"🔍 Analyzing data from: {data_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Run full analysis
    results = analyze_experiment_data(data_path, output_dir)
    
    # Show plots if requested
    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()
    
    return results

def compare_sessions(session_files: list, output_dir: str = "comparison_results"):
    """
    Compare results across multiple experimental sessions.
    
    Args:
        session_files: List of CSV files to compare
        output_dir: Output directory for comparison results
    
    Returns:
        Combined analysis results
    """
    print(f"📊 Comparing {len(session_files)} sessions...")
    
    # Load and combine data
    all_data = []
    for i, file_path in enumerate(session_files):
        data = pd.read_csv(file_path)
        # Add session identifier
        session_name = Path(file_path).stem
        data['session'] = session_name
        data['session_number'] = i + 1
        all_data.append(data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Save combined data
    os.makedirs(output_dir, exist_ok=True)
    combined_file = Path(output_dir) / "combined_sessions.csv"
    combined_data.to_csv(combined_file, index=False)
    
    print(f"💾 Combined data saved to: {combined_file}")
    
    # Run analysis on combined data
    results = analyze_experiment_data(str(combined_file), output_dir)
    
    return results

def generate_example_data(n_trials: int = 200, models: list = None, save_path: str = "example_data.csv"):
    """
    Generate example experimental data for testing the analysis pipeline.
    
    Args:
        n_trials: Number of trials per model
        models: List of model names (default: ['Human', 'AI_Model_1', 'AI_Model_2'])
        save_path: Where to save the example data
    
    Returns:
        Path to saved example data
    """
    import numpy as np
    
    if models is None:
        models = ['Human', 'AI_Model_1', 'AI_Model_2']
    
    print(f"🎲 Generating example data with {n_trials} trials per model...")
    
    all_data = []
    
    for model in models:
        # Different performance characteristics for each model
        if model == 'Human':
            base_accuracy = 0.75
            confidence_bias = 0.2  # Slightly overconfident
            sensitivity = 0.8  # Good metacognitive sensitivity
        elif model == 'AI_Model_1':
            base_accuracy = 0.85
            confidence_bias = -0.1  # Slightly underconfident
            sensitivity = 0.6  # Moderate metacognitive sensitivity
        else:  # AI_Model_2
            base_accuracy = 0.70
            confidence_bias = 0.4  # Overconfident
            sensitivity = 0.4  # Poor metacognitive sensitivity
        
        for trial in range(n_trials):
            # Generate trial difficulty
            contrast = np.random.beta(2, 5)  # Biased toward easier trials
            
            # Performance depends on contrast and model ability
            trial_difficulty = 1 - contrast
            accuracy_prob = base_accuracy - (trial_difficulty * 0.3)
            correct = np.random.random() < accuracy_prob
            
            # Confidence depends on correctness and model characteristics
            if correct:
                base_conf = 4.5 + (sensitivity * 1.0)
            else:
                base_conf = 3.5 - (sensitivity * 0.8)
            
            # Add confidence bias
            base_conf += confidence_bias
            
            # Add noise and constrain to 1-6 range
            confidence = np.clip(base_conf + np.random.normal(0, 0.8), 1, 6)
            confidence = int(np.round(confidence))
            
            # Response time (somewhat realistic)
            response_time = np.random.lognormal(mean=0.5, sigma=0.3)
            
            trial_data = {
                'model': model,
                'trial': trial + 1,
                'contrast': contrast,
                'correct': correct,
                'confidence': confidence,
                'response_time': response_time,
                'session': 'example_session',
                'timestamp': f"2024-12-01T14:30:{trial:02d}"
            }
            
            all_data.append(trial_data)
    
    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(save_path, index=False)
    
    print(f"💾 Example data saved to: {save_path}")
    print(f"📈 Generated {len(df)} trials across {len(models)} models")
    
    return save_path

def summary_stats(data_path: str):
    """
    Print quick summary statistics for experiment data.
    
    Args:
        data_path: Path to CSV file with experiment data
    """
    print(f"📊 Summary Statistics for: {data_path}")
    print("=" * 50)
    
    data = pd.read_csv(data_path)
    
    print(f"Total trials: {len(data)}")
    print(f"Models: {', '.join(data['model'].unique())}")
    print(f"Columns: {', '.join(data.columns)}")
    
    print("\nPer-Model Statistics:")
    print("-" * 30)
    
    for model in data['model'].unique():
        model_data = data[data['model'] == model]
        accuracy = model_data['correct'].mean()
        mean_conf = model_data['confidence'].mean()
        
        print(f"{model}:")
        print(f"  Trials: {len(model_data)}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Mean Confidence: {mean_conf:.2f}")
        
        if len(model_data[model_data['correct']]) > 0 and len(model_data[~model_data['correct']]) > 0:
            correct_conf = model_data[model_data['correct']]['confidence'].mean()
            incorrect_conf = model_data[~model_data['correct']]['confidence'].mean()
            sensitivity = correct_conf - incorrect_conf
            print(f"  Metacognitive Sensitivity: {sensitivity:.3f}")
        print()

def main():
    """Command line interface for the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze Metacognition Experiment Data')
    
    parser.add_argument('--data', '-d', type=str, help='Path to experiment data CSV file')
    parser.add_argument('--output', '-o', type=str, help='Output directory for results')
    parser.add_argument('--compare', '-c', nargs='+', help='Multiple CSV files to compare')
    parser.add_argument('--generate-example', '-g', action='store_true', 
                       help='Generate example data for testing')
    parser.add_argument('--trials', '-t', type=int, default=200, 
                       help='Number of trials for example data')
    parser.add_argument('--summary', '-s', action='store_true', 
                       help='Show summary statistics only')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Do not display plots')
    
    args = parser.parse_args()
    
    if args.generate_example:
        # Generate example data
        example_file = generate_example_data(n_trials=args.trials)
        print(f"\n✨ Generated example data: {example_file}")
        print("🚀 Run analysis with: python analyze_results.py --data example_data.csv")
        return
    
    if args.compare:
        # Compare multiple sessions
        if len(args.compare) < 2:
            print("❌ Need at least 2 files for comparison")
            return
        
        output_dir = args.output or "comparison_results"
        results = compare_sessions(args.compare, output_dir)
        print(f"✅ Comparison complete! Results saved to: {output_dir}")
        return
    
    if not args.data:
        print("❌ Please provide data file with --data or use --generate-example")
        parser.print_help()
        return
    
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    if args.summary:
        # Show summary statistics only
        summary_stats(args.data)
        return
    
    # Run full analysis
    try:
        results = quick_analysis(
            data_path=args.data,
            output_dir=args.output,
            show_plots=not args.no_plots
        )
        
        print("\n✅ Analysis complete!")
        print(f"📁 Results saved to: {results['analyzer'].data.name if hasattr(results['analyzer'].data, 'name') else 'analysis directory'}")
        print("📊 Check the plots and reports in the output directory")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 