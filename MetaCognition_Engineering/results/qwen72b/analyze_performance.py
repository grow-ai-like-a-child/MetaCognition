#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Qwen2.5-VL-72B模型性能：正确率和置信度统计
"""

import json
from collections import defaultdict

def analyze_model_performance():
    """
    分析模型性能
    """
    # 加载数据
    with open('/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/results/qwen72b/qwen72b_simple_comparison.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("Qwen2.5-VL-72B 模型性能分析")
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
    print(f"\n🎯 按任务类型统计:")
    print("-" * 50)
    
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for item in data:
        # 从qid中提取任务类型
        task_type = item['qid'].split('-')[0]
        task_stats[task_type]['total'] += 1
        if item['is_correct']:
            task_stats[task_type]['correct'] += 1
    
    for task in sorted(task_stats.keys()):
        stats = task_stats[task]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        percentage = stats['total'] / total_questions * 100
        
        print(f"   {task:8s}: {stats['correct']:4d}/{stats['total']:4d} = {accuracy:.4f} ({accuracy*100:5.2f}%) | 占比: {percentage:5.2f}%")
    
    # 置信度校准分析
    print(f"\n🎯 置信度校准分析:")
    print("-" * 50)
    
    # 高置信度(4-5) vs 低置信度(1-3)的准确率对比
    high_conf_correct = sum(1 for item in data if item['confidence'] >= 4 and item['is_correct'])
    high_conf_total = sum(1 for item in data if item['confidence'] >= 4)
    low_conf_correct = sum(1 for item in data if item['confidence'] < 4 and item['is_correct'])
    low_conf_total = sum(1 for item in data if item['confidence'] < 4)
    
    high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
    low_conf_accuracy = low_conf_correct / low_conf_total if low_conf_total > 0 else 0
    
    print(f"   高置信度(4-5): {high_conf_correct:4d}/{high_conf_total:4d} = {high_conf_accuracy:.4f} ({high_conf_accuracy*100:5.2f}%)")
    print(f"   低置信度(1-3): {low_conf_correct:4d}/{low_conf_total:4d} = {low_conf_accuracy:.4f} ({low_conf_accuracy*100:5.2f}%)")
    
    # 置信度分布
    print(f"\n📊 置信度分布:")
    print("-" * 30)
    for conf in sorted(confidence_stats.keys()):
        count = confidence_stats[conf]['total']
        percentage = count / total_questions * 100
        bar = "█" * int(percentage / 2)  # 每2%一个方块
        print(f"   {conf}: {count:4d} ({percentage:5.2f}%) {bar}")
    
    # 元认知能力分析
    print(f"\n🧠 元认知能力分析:")
    print("-" * 50)
    
    # 计算置信度与准确率的相关性
    conf_acc_correlation = []
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            conf_acc_correlation.append((conf, accuracy, stats['total']))
    
    print("   置信度 vs 准确率:")
    for conf, acc, count in conf_acc_correlation:
        print(f"   {conf}: {acc:.4f} (n={count})")
    
    # 判断模型是否过度自信
    if len(conf_acc_correlation) >= 2:
        high_conf_acc = conf_acc_correlation[-1][1]  # 最高置信度的准确率
        low_conf_acc = conf_acc_correlation[0][1]    # 最低置信度的准确率
        
        if high_conf_acc < low_conf_acc:
            print(f"\n   ⚠️  模型可能过度自信：高置信度准确率({high_conf_acc:.4f}) < 低置信度准确率({low_conf_acc:.4f})")
        else:
            print(f"\n   ✅ 模型置信度校准良好：高置信度准确率({high_conf_acc:.4f}) > 低置信度准确率({low_conf_acc:.4f})")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_model_performance()
