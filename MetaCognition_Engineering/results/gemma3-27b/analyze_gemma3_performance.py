#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Gemma3-27B模型性能：正确率和置信度统计
"""

import json
import re
from collections import defaultdict
from pathlib import Path

def load_ground_truth():
    """
    加载真实标签数据 - 从ground_truth.json文件
    """
    ground_truth = {}
    
    # 加载ground_truth.json文件
    ground_truth_file = Path("../../data/raw/ground_truth.json")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # 提取stage1问题的答案，并转换为original_qid格式
    for qid, answer_data in gt_data.items():
        if '_stage1' in qid:
            # 将qid转换为original_qid (去掉_stage1后缀)
            original_qid = qid.replace('_stage1', '')
            correct_choice = answer_data.get('correct_choice')
            if correct_choice in ['A', 'B']:
                ground_truth[original_qid] = correct_choice
    
    return ground_truth

def analyze_model_performance():
    """
    分析Gemma3模型性能
    """
    # 加载真实标签
    ground_truth = load_ground_truth()
    print(f"加载了 {len(ground_truth)} 个真实标签")
    
    # 加载模型结果
    results_file = Path("gemma3_27b_concise_full_results.jsonl")
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
    print("Gemma3-27B Stage 1 性能分析")
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
    
    # 概率分析（Gemma3特有的真实logits概率）
    print(f"\n🎲 概率分析 (基于真实logits):")
    print("-" * 50)
    
    probability_stats = {
        'high_confidence': {'total': 0, 'correct': 0},  # A或B概率 > 0.8
        'medium_confidence': {'total': 0, 'correct': 0},  # 0.6 <= 概率 <= 0.8
        'low_confidence': {'total': 0, 'correct': 0},   # 概率 < 0.6
    }
    
    for original_qid, result in stage1_results.items():
        if original_qid not in ground_truth:
            continue
            
        predicted = result['choice']
        true_answer = ground_truth[original_qid]
        is_correct = (predicted == true_answer)
        
        # 获取概率
        probs = result.get('probabilities', {})
        if predicted in probs:
            prob = probs[predicted]
            
            if prob > 0.8:
                category = 'high_confidence'
            elif prob >= 0.6:
                category = 'medium_confidence'
            else:
                category = 'low_confidence'
            
            probability_stats[category]['total'] += 1
            if is_correct:
                probability_stats[category]['correct'] += 1
    
    for category, stats in probability_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            percentage = stats['total'] / total_count * 100
            print(f"   {category:15s}: {stats['correct']:4d}/{stats['total']:4d} = {accuracy:.4f} ({accuracy*100:5.2f}%) | 占比: {percentage:5.2f}%")
    
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
    
    # 延迟分析
    print(f"\n⏱️ 推理延迟分析:")
    print("-" * 50)
    
    stage1_latencies = [result.get('latency_ms', 0) for result in stage1_results.values()]
    stage2_latencies = [result.get('latency_ms', 0) for result in stage2_results.values()]
    
    if stage1_latencies:
        avg_stage1 = sum(stage1_latencies) / len(stage1_latencies)
        print(f"   Stage 1平均延迟: {avg_stage1:.1f}ms")
    
    if stage2_latencies:
        avg_stage2 = sum(stage2_latencies) / len(stage2_latencies)
        print(f"   Stage 2平均延迟: {avg_stage2:.1f}ms")
    
    if stage1_latencies and stage2_latencies:
        total_avg = (sum(stage1_latencies) + sum(stage2_latencies)) / (len(stage1_latencies) + len(stage2_latencies))
        print(f"   总平均延迟: {total_avg:.1f}ms")
    
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
            'latency_ms': result.get('latency_ms', 0),
            'stage1_latency': result.get('latency_ms', 0),
            'stage2_latency': stage2_results.get(original_qid, {}).get('latency_ms', 0) if original_qid in stage2_results else 0
        })
    
    # 保存比较结果
    with open('gemma3_comparison_with_ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细比较结果已保存到: gemma3_comparison_with_ground_truth.json")
    print(f"   包含 {len(comparison_data)} 个题目的详细分析")
    
    # 打印汇总统计
    print(f"\n📈 汇总统计:")
    print("-" * 50)
    print(f"   ✅ 总体准确率: {overall_accuracy*100:.2f}%")
    print(f"   ⚡ 平均延迟: {total_avg:.1f}ms" if 'total_avg' in locals() else "   ⚡ 延迟信息不完整")
    print(f"   🎯 高置信度(5)准确率: {confidence_stats[5]['correct']/confidence_stats[5]['total']*100:.2f}%" if confidence_stats[5]['total'] > 0 else "   🎯 无置信度5的数据")
    print(f"   📊 高概率(>0.8)准确率: {probability_stats['high_confidence']['correct']/probability_stats['high_confidence']['total']*100:.2f}%" if probability_stats['high_confidence']['total'] > 0 else "   📊 无高概率数据")

if __name__ == "__main__":
    analyze_model_performance()
