#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析InternVL3-38B模型性能：正确率和置信度统计
"""

import json
import re
from collections import defaultdict
from pathlib import Path

def load_ground_truth():
    """
    加载真实标签数据
    """
    ground_truth = {}
    
    # 加载问题文件获取真实标签
    questions_file = Path("../../data/prompt/questions_two_stage_concise.jsonl")
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            question = json.loads(line.strip())
            if question.get('stage') == 1:  # 只处理stage1问题
                qid = question['qid']
                original_qid = question['original_qid']
                
                # 从context中提取真实答案
                context = question.get('context', {})
                derived = context.get('derived', {})
                gt = derived.get('gt', {})
                
                if 'more_symbol' in gt:
                    more_symbol = gt['more_symbol']
                    if more_symbol == 'symA':
                        ground_truth[original_qid] = 'A'
                    elif more_symbol == 'symB':
                        ground_truth[original_qid] = 'B'
    
    return ground_truth

def analyze_model_performance():
    """
    分析InternVL3模型性能
    """
    # 加载真实标签
    ground_truth = load_ground_truth()
    print(f"加载了 {len(ground_truth)} 个真实标签")
    
    # 加载模型结果
    results_file = Path("internvl3_complete_results.jsonl")
    results = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    print(f"加载了 {len(results)} 个模型结果")
    
    # 按original_qid分组，只分析stage1结果
    stage1_results = {}
    stage2_results = {}
    
    for result in results:
        original_qid = result['original_qid']
        stage = result['stage']
        
        if stage == 1:
            stage1_results[original_qid] = result
        elif stage == 2:
            stage2_results[original_qid] = result
    
    print(f"找到 {len(stage1_results)} 个Stage 1结果")
    print(f"找到 {len(stage2_results)} 个Stage 2结果")
    
    # 分析Stage 1正确率
    print("\n" + "=" * 60)
    print("InternVL3-38B Stage 1 性能分析")
    print("=" * 60)
    
    correct_count = 0
    total_count = 0
    confidence_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for original_qid, result in stage1_results.items():
        if original_qid not in ground_truth:
            continue
            
        total_count += 1
        predicted = result['choice']
        true_answer = ground_truth[original_qid]
        
        # 检查是否正确
        is_correct = (predicted == true_answer)
        if is_correct:
            correct_count += 1
        
        # 获取对应的Stage 2置信度
        if original_qid in stage2_results:
            confidence = stage2_results[original_qid]['confidence']
            confidence_stats[confidence]['total'] += 1
            if is_correct:
                confidence_stats[confidence]['correct'] += 1
    
    # 计算总体准确率
    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n📊 Stage 1 总体性能:")
    print(f"   总题目数: {total_count}")
    print(f"   正确答案数: {correct_count}")
    print(f"   整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # 按置信度统计
    print(f"\n📈 按置信度分组统计:")
    print("-" * 50)
    
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        percentage = stats['total'] / total_count * 100
        
        print(f"   置信度 {conf}: {stats['correct']:4d}/{stats['total']:4d} = {accuracy:.4f} ({accuracy*100:5.2f}%) | 占比: {percentage:5.2f}%")
    
    # 按任务类型统计
    print(f"\n📋 按任务类型统计:")
    print("-" * 50)
    
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for original_qid, result in stage1_results.items():
        if original_qid not in ground_truth:
            continue
            
        task = result.get('task', 'Unknown')
        predicted = result['choice']
        true_answer = ground_truth[original_qid]
        is_correct = (predicted == true_answer)
        
        task_stats[task]['total'] += 1
        if is_correct:
            task_stats[task]['correct'] += 1
    
    for task in sorted(task_stats.keys()):
        stats = task_stats[task]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        percentage = stats['total'] / total_count * 100
        
        print(f"   {task:12s}: {stats['correct']:4d}/{stats['total']:4d} = {accuracy:.4f} ({accuracy*100:5.2f}%) | 占比: {percentage:5.2f}%")
    
    # 置信度校准分析
    print(f"\n🎯 置信度校准分析:")
    print("-" * 50)
    
    # 计算每个置信度水平的平均准确率
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            calibration_error = abs(accuracy - (conf / 5.0))  # 理想情况下，置信度/5应该等于准确率
            print(f"   置信度 {conf}: 准确率={accuracy:.3f}, 理想值={conf/5.0:.3f}, 校准误差={calibration_error:.3f}")
    
    # 保存详细比较结果
    comparison_data = []
    for original_qid, result in stage1_results.items():
        if original_qid not in ground_truth:
            continue
            
        predicted = result['choice']
        true_answer = ground_truth[original_qid]
        is_correct = (predicted == true_answer)
        
        confidence = 0
        if original_qid in stage2_results:
            confidence = stage2_results[original_qid]['confidence']
        
        comparison_data.append({
            'original_qid': original_qid,
            'task': result.get('task', 'Unknown'),
            'predicted': predicted,
            'true_answer': true_answer,
            'is_correct': is_correct,
            'confidence': confidence,
            'probabilities': result.get('probabilities', {}),
            'latency_ms': result.get('latency_ms', 0)
        })
    
    # 保存比较结果
    with open('internvl3_comparison_with_ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细比较结果已保存到: internvl3_comparison_with_ground_truth.json")
    print(f"   包含 {len(comparison_data)} 个题目的详细分析")

if __name__ == "__main__":
    analyze_model_performance()
