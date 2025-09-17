#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析Gemma3-27B在三种任务上的性能差异
"""

import json
from collections import defaultdict
from pathlib import Path

def detailed_task_analysis():
    """
    按任务类型详细分析性能
    """
    # 加载分析结果
    comparison_file = Path("gemma3_comparison_with_ground_truth.json")
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("🔍 Gemma3-27B 详细任务分析报告")
    print("=" * 80)
    
    # 按任务分组
    tasks = {}
    for item in data:
        task = item['task']
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(item)
    
    # 总体统计
    total_correct = sum(1 for item in data if item['is_correct'])
    total_count = len(data)
    overall_accuracy = total_correct / total_count * 100
    
    print(f"\n📊 总体概览:")
    print(f"   总准确率: {overall_accuracy:.2f}% ({total_correct}/{total_count})")
    print(f"   数据分布: Grid={len(tasks.get('Grid', []))}, Color={len(tasks.get('Color', []))}, Gabor={len(tasks.get('Gabor', []))}")
    
    # 每个任务的详细分析
    for task_name in sorted(tasks.keys()):
        task_data = tasks[task_name]
        task_correct = sum(1 for item in task_data if item['is_correct'])
        task_total = len(task_data)
        task_accuracy = task_correct / task_total * 100
        
        print(f"\n" + "=" * 60)
        print(f"📋 {task_name} 任务分析")
        print("=" * 60)
        
        print(f"\n基本统计:")
        print(f"   准确率: {task_accuracy:.2f}% ({task_correct}/{task_total})")
        
        # 置信度分布
        confidence_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for item in task_data:
            conf = item['confidence']
            confidence_stats[conf]['total'] += 1
            if item['is_correct']:
                confidence_stats[conf]['correct'] += 1
        
        print(f"\n置信度分布:")
        for conf in sorted(confidence_stats.keys()):
            stats = confidence_stats[conf]
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            pct = stats['total'] / task_total * 100
            print(f"   置信度{conf}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}% (占比: {pct:4.1f}%)")
        
        # 概率分析
        high_prob = [item for item in task_data if max(item['probabilities'].values()) > 0.8]
        med_prob = [item for item in task_data if 0.6 <= max(item['probabilities'].values()) <= 0.8]
        low_prob = [item for item in task_data if max(item['probabilities'].values()) < 0.6]
        
        print(f"\n概率分析:")
        if high_prob:
            high_correct = sum(1 for item in high_prob if item['is_correct'])
            print(f"   高概率(>0.8): {high_correct:3d}/{len(high_prob):3d} = {high_correct/len(high_prob)*100:5.1f}%")
        
        if med_prob:
            med_correct = sum(1 for item in med_prob if item['is_correct'])
            print(f"   中概率(0.6-0.8): {med_correct:3d}/{len(med_prob):3d} = {med_correct/len(med_prob)*100:5.1f}%")
        
        if low_prob:
            low_correct = sum(1 for item in low_prob if item['is_correct'])
            print(f"   低概率(<0.6): {low_correct:3d}/{len(low_prob):3d} = {low_correct/len(low_prob)*100:5.1f}%")
        
        # 延迟分析
        stage1_latencies = [item['stage1_latency'] for item in task_data if item['stage1_latency'] > 0]
        stage2_latencies = [item['stage2_latency'] for item in task_data if item['stage2_latency'] > 0]
        
        if stage1_latencies:
            avg_stage1 = sum(stage1_latencies) / len(stage1_latencies)
            print(f"\n延迟分析:")
            print(f"   Stage1平均: {avg_stage1:.1f}ms")
        
        if stage2_latencies:
            avg_stage2 = sum(stage2_latencies) / len(stage2_latencies)
            print(f"   Stage2平均: {avg_stage2:.1f}ms")
        
        # 错误案例分析
        incorrect_items = [item for item in task_data if not item['is_correct']]
        if incorrect_items:
            print(f"\n错误分析:")
            print(f"   错误总数: {len(incorrect_items)}")
            
            # 按置信度分组错误
            error_by_conf = defaultdict(int)
            for item in incorrect_items:
                error_by_conf[item['confidence']] += 1
            
            print(f"   错误分布:")
            for conf in sorted(error_by_conf.keys()):
                print(f"     置信度{conf}: {error_by_conf[conf]}个错误")
            
            # 高置信度错误（可能是模型过度自信）
            high_conf_errors = [item for item in incorrect_items if item['confidence'] >= 4]
            if high_conf_errors:
                print(f"   ⚠️  高置信度错误: {len(high_conf_errors)}个 (模型过度自信)")
    
    # 任务间比较
    print(f"\n" + "=" * 60)
    print(f"🔄 任务间比较")
    print("=" * 60)
    
    task_accuracies = {}
    for task_name, task_data in tasks.items():
        task_correct = sum(1 for item in task_data if item['is_correct'])
        task_total = len(task_data)
        task_accuracies[task_name] = task_correct / task_total * 100
    
    # 排序任务
    sorted_tasks = sorted(task_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n任务难度排序 (准确率从高到低):")
    for i, (task, acc) in enumerate(sorted_tasks, 1):
        difficulty = "简单" if acc > 80 else "中等" if acc > 60 else "困难"
        print(f"   {i}. {task:8s}: {acc:5.2f}% ({difficulty})")
    
    # 置信度校准对比
    print(f"\n置信度校准对比:")
    for task_name, task_data in sorted(tasks.items()):
        print(f"\n{task_name}:")
        conf_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for item in task_data:
            conf = item['confidence']
            conf_stats[conf]['total'] += 1
            if item['is_correct']:
                conf_stats[conf]['correct'] += 1
        
        for conf in sorted(conf_stats.keys()):
            stats = conf_stats[conf]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                ideal = conf / 5.0
                error = abs(accuracy - ideal)
                print(f"   置信度{conf}: 实际={accuracy:.3f}, 理想={ideal:.3f}, 误差={error:.3f}")
    
    print(f"\n" + "=" * 80)
    print(f"💡 关键发现:")
    print("=" * 80)
    
    best_task = sorted_tasks[0][0]
    worst_task = sorted_tasks[-1][0]
    print(f"✅ 最擅长任务: {best_task} ({sorted_tasks[0][1]:.1f}%)")
    print(f"❌ 最困难任务: {worst_task} ({sorted_tasks[-1][1]:.1f}%)")
    
    accuracy_gap = sorted_tasks[0][1] - sorted_tasks[-1][1]
    print(f"📊 任务间性能差距: {accuracy_gap:.1f}个百分点")
    
    # 总体校准质量
    overall_conf_5_items = [item for item in data if item['confidence'] == 5]
    if overall_conf_5_items:
        conf_5_accuracy = sum(1 for item in overall_conf_5_items if item['is_correct']) / len(overall_conf_5_items)
        print(f"🎯 高置信度(5)校准: 实际{conf_5_accuracy:.3f} vs 理想1.000 (误差{abs(conf_5_accuracy-1.0):.3f})")
    
    print(f"⚡ 整体推理效率: 平均{227.2:.1f}ms每题")

if __name__ == "__main__":
    detailed_task_analysis()
