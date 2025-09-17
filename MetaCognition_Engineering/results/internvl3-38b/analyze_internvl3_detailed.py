#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析InternVL3-38B模型性能
"""

import json
from collections import defaultdict
from pathlib import Path

def analyze_detailed_performance():
    """
    详细分析InternVL3模型性能
    """
    # 加载正确比较结果
    comparison_file = Path("internvl3_correct_comparison.json")
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("InternVL3-38B 详细性能分析")
    print("=" * 60)
    
    # 基本统计
    total_questions = len(data)
    correct_answers = sum(1 for item in data if item['is_correct'])
    overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    print(f"\n📊 总体性能:")
    print(f"   总题目数: {total_questions}")
    print(f"   正确答案数: {correct_answers}")
    print(f"   整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # 按置信度统计
    print(f"\n📈 按置信度分组统计:")
    print("-" * 50)
    
    confidence_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for item in data:
        conf = item['confidence']
        confidence_stats[conf]['total'] += 1
        if item['is_correct']:
            confidence_stats[conf]['correct'] += 1
    
    # 按置信度排序
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        percentage = stats['total'] / total_questions * 100
        
        print(f"   置信度 {conf}: {stats['correct']:4d}/{stats['total']:4d} = {accuracy:.4f} ({accuracy*100:5.2f}%) | 占比: {percentage:5.2f}%")
    
    # 按任务类型统计
    print(f"\n📋 按任务类型统计:")
    print("-" * 50)
    
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for item in data:
        task = item.get('task', 'Unknown')
        task_stats[task]['total'] += 1
        if item['is_correct']:
            task_stats[task]['correct'] += 1
    
    for task in sorted(task_stats.keys()):
        stats = task_stats[task]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        percentage = stats['total'] / total_questions * 100
        
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
    
    # 预测分布分析
    print(f"\n🔍 预测分布分析:")
    print("-" * 50)
    
    predicted_dist = defaultdict(int)
    correct_dist = defaultdict(lambda: {'A': 0, 'B': 0})
    
    for item in data:
        predicted = item['predicted_choice']
        correct = item['correct_choice']
        predicted_dist[predicted] += 1
        
        if item['is_correct']:
            correct_dist[correct][predicted] += 1
    
    print(f"   预测A: {predicted_dist['A']} 次 ({predicted_dist['A']/total_questions*100:.1f}%)")
    print(f"   预测B: {predicted_dist['B']} 次 ({predicted_dist['B']/total_questions*100:.1f}%)")
    
    print(f"\n   正确答案分布:")
    for correct_choice in ['A', 'B']:
        total_correct = sum(correct_dist[correct_choice].values())
        if total_correct > 0:
            print(f"     正确答案{correct_choice}: {total_correct} 题")
            for predicted_choice in ['A', 'B']:
                count = correct_dist[correct_choice][predicted_choice]
                percentage = count / total_correct * 100
                print(f"       预测{predicted_choice}: {count} 题 ({percentage:.1f}%)")
    
    # 置信度与准确率的关系
    print(f"\n📊 置信度与准确率关系:")
    print("-" * 50)
    
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            print(f"   置信度 {conf}: {accuracy:.3f} 准确率 (基于 {stats['total']} 个样本)")
    
    # 错误案例分析
    print(f"\n❌ 错误案例分析:")
    print("-" * 50)
    
    wrong_cases = [item for item in data if not item['is_correct']]
    print(f"   总错误数: {len(wrong_cases)}")
    
    # 按置信度分组的错误
    wrong_by_confidence = defaultdict(int)
    for item in wrong_cases:
        wrong_by_confidence[item['confidence']] += 1
    
    print(f"   按置信度分组的错误:")
    for conf in sorted(wrong_by_confidence.keys()):
        count = wrong_by_confidence[conf]
        percentage = count / len(wrong_cases) * 100
        print(f"     置信度 {conf}: {count} 个错误 ({percentage:.1f}%)")
    
    # 高置信度但错误的案例
    high_conf_wrong = [item for item in wrong_cases if item['confidence'] >= 4]
    print(f"   高置信度(≥4)但错误: {len(high_conf_wrong)} 个")
    
    if len(high_conf_wrong) > 0:
        print(f"   示例高置信度错误案例:")
        for i, item in enumerate(high_conf_wrong[:3]):  # 显示前3个
            print(f"     {i+1}. {item['qid']}: 预测={item['predicted_choice']}, 正确={item['correct_choice']}, 置信度={item['confidence']}")

if __name__ == "__main__":
    analyze_detailed_performance()
