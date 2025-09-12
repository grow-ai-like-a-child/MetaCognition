#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Qwen2.5-VL模型结果：计算Accuracy和Confidence distribution（修正版）
"""

import json
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import argparse

def load_answers_dict(file_path):
    """Load correct answers dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_results(file_path):
    """Load model results"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results

def analyze_results(answers_dict, model_results):
    """Analyze model results"""
    
    # Group by original_qid, merge stage1 and stage2
    grouped_results = defaultdict(dict)
    
    for result in model_results:
        original_qid = result.get('original_qid')
        stage = result.get('stage')
        
        if original_qid and stage:
            grouped_results[original_qid][f'stage{stage}'] = result
    
    # Only analyze complete results with both stage1 and stage2
    complete_results = []
    for original_qid, stages in grouped_results.items():
        if 'stage1' in stages and 'stage2' in stages:
            stage1 = stages['stage1']
            stage2 = stages['stage2']
            
            # Get correct answer
            correct_choice = answers_dict.get(f"{original_qid}_stage1", {}).get('correct_choice')
            if correct_choice:
                # Merge stage1 and stage2 information
                combined_result = {
                    'qid': stage1['qid'],
                    'original_qid': original_qid,
                    'task': stage1['task'],
                    'choice': stage1['choice'],
                    'confidence': stage2['confidence'],
                    'latency_ms': stage1['latency_ms'],
                    'probabilities': stage1['probabilities'],
                    'correct_choice': correct_choice,
                    'is_correct': stage1['choice'] == correct_choice
                }
                complete_results.append(combined_result)
    
    print(f"Total of {len(complete_results)} complete results (including stage1 and stage2)")
    
    # Group by task type
    task_results = defaultdict(list)
    for result in complete_results:
        task = result['task']
        task_results[task].append(result)
    
    # Calculate statistics for each task type
    task_stats = {}
    
    for task, results in task_results.items():
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / total if total > 0 else 0
        
        # Confidence distribution
        confidences = [r['confidence'] for r in results]
        conf_counter = Counter(confidences)
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Average latency
        avg_latency = sum(r['latency_ms'] for r in results) / len(results) if results else 0
        
        task_stats[task] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'confidence_distribution': dict(conf_counter),
            'avg_confidence': avg_confidence,
            'avg_latency_ms': avg_latency
        }
    
    return task_stats, complete_results

def print_analysis_results(task_stats):
    """Print analysis results"""
    print("\n" + "="*80)
    print("Qwen2.5-VL 模型结果分析（修正版）")
    print("="*80)
    
    total_correct = 0
    total_questions = 0
    
    for task, stats in task_stats.items():
        print(f"\n📊 {task} task:")
        print(f"  Total questions: {stats['total']}")
        print(f"  Correct answers: {stats['correct']}")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
        print(f"  Average confidence: {stats['avg_confidence']:.2f}")
        print(f"  Average latency: {stats['avg_latency_ms']:.1f}ms")
        
        print(f"  Confidence distribution:")
        for conf_level in sorted(stats['confidence_distribution'].keys()):
            count = stats['confidence_distribution'][conf_level]
            percentage = count / stats['total'] * 100
            print(f"    {conf_level}: {count} ({percentage:.1f}%)")
        
        total_correct += stats['correct']
        total_questions += stats['total']
    
    # Overall statistics
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"\n🎯 Overall statistics:")
    print(f"  Total questions: {total_questions}")
    print(f"  总Correct answers: {total_correct}")
    print(f"  总体Accuracy: {overall_accuracy:.2%}")

def create_confidence_plots(task_stats, output_dir="analysis_plots"):
    """Create confidence distribution plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Task accuracy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    tasks = list(task_stats.keys())
    accuracies = [task_stats[task]['accuracy'] for task in tasks]
    
    bars = ax1.bar(tasks, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Task Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Display percentages on bars
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confidence distribution
    conf_levels = [1, 2, 3, 4, 5]
    conf_matrix = []
    
    for task in tasks:
        row = []
        for conf in conf_levels:
            count = task_stats[task]['confidence_distribution'].get(conf, 0)
            total = task_stats[task]['total']
            percentage = count / total * 100 if total > 0 else 0
            row.append(percentage)
        conf_matrix.append(row)
    
    im = ax2.imshow(conf_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(conf_levels)))
    ax2.set_yticks(range(len(tasks)))
    ax2.set_xticklabels([f'Conf {i}' for i in conf_levels])
    ax2.set_yticklabels(tasks)
    
    # Add numerical annotations
    for i in range(len(tasks)):
        for j in range(len(conf_levels)):
            text = ax2.text(j, i, f'{conf_matrix[i][j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    ax2.set_title('Confidence Distribution Heatmap (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Confidence Level')
    ax2.set_ylabel('Task Type')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/analysis_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Chart saved to: {output_dir}/analysis_plot.png")

def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL metacognitive reasoning accuracy analysis")
    parser.add_argument("--answers", type=str, default="../../raw/ground_truth.json", help="Correct answers file path")
    parser.add_argument("--results", type=str, required=True, help="Model results file path")
    parser.add_argument("--output", type=str, default="analysis_results.csv", help="Output CSV file path")
    parser.add_argument("--plot", type=str, default="analysis_plot.png", help="Output chart file path")
    
    args = parser.parse_args()
    
    # Loading data
    print("正在Loading data...")
    answers_dict = load_answers_dict(args.answers)
    model_results = load_model_results(args.results)
    
    print(f"Loaded {len(answers_dict)} correct answers")
    print(f"Loaded {len(model_results)} model results")
    
    # Analyzing results
    print("\n正在Analyzing results...")
    task_stats, complete_results = analyze_results(answers_dict, model_results)
    
    # Printing results
    print_analysis_results(task_stats)
    
    # Creating charts
    try:
        create_confidence_plots(task_stats, output_dir="analysis_plots")
    except Exception as e:
        print(f"Creating charts时出错: {e}")
        print("Skipping chart generation...")
    
    # Save detailed results to CSV
    detailed_results = []
    for result in complete_results:
        detailed_results.append({
            'qid': result['qid'],
            'original_qid': result['original_qid'],
            'task': result['task'],
            'model_choice': result['choice'],
            'correct_choice': result['correct_choice'],
            'is_correct': result['is_correct'],
            'confidence': result['confidence'],
            'latency_ms': result['latency_ms'],
            'prob_a': result['probabilities']['A'],
            'prob_b': result['probabilities']['B']
        })
    
    df = pd.DataFrame(detailed_results)
    df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n📄 Detailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
