#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较InternVL3-38B与Qwen2.5-VL-72B的性能
"""

import json
from collections import defaultdict
from pathlib import Path

def load_qwen72b_results():
    """
    加载Qwen72B的结果
    """
    qwen_file = Path("../qwen72b/qwen72b_simple_comparison.json")
    if not qwen_file.exists():
        print(f"Qwen72B结果文件不存在: {qwen_file}")
        return None
    
    with open(qwen_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_models():
    """
    比较两个模型的性能
    """
    # 加载InternVL3结果
    internvl3_file = Path("internvl3_correct_comparison.json")
    with open(internvl3_file, 'r', encoding='utf-8') as f:
        internvl3_data = json.load(f)
    
    # 加载Qwen72B结果
    qwen72b_data = load_qwen72b_results()
    if qwen72b_data is None:
        print("无法加载Qwen72B结果，跳过比较")
        return
    
    print("=" * 80)
    print("InternVL3-38B vs Qwen2.5-VL-72B 性能比较")
    print("=" * 80)
    
    # 基本性能比较
    print(f"\n📊 基本性能比较:")
    print("-" * 60)
    
    # InternVL3统计
    internvl3_total = len(internvl3_data)
    internvl3_correct = sum(1 for item in internvl3_data if item['is_correct'])
    internvl3_accuracy = internvl3_correct / internvl3_total if internvl3_total > 0 else 0
    
    # Qwen72B统计
    qwen72b_total = len(qwen72b_data)
    qwen72b_correct = sum(1 for item in qwen72b_data if item['is_correct'])
    qwen72b_accuracy = qwen72b_correct / qwen72b_total if qwen72b_total > 0 else 0
    
    print(f"   模型            | 总题目数 | 正确答案数 | 准确率")
    print(f"   ----------------|----------|------------|--------")
    print(f"   InternVL3-38B   | {internvl3_total:8d} | {internvl3_correct:10d} | {internvl3_accuracy:.4f} ({internvl3_accuracy*100:.2f}%)")
    print(f"   Qwen2.5-VL-72B  | {qwen72b_total:8d} | {qwen72b_correct:10d} | {qwen72b_accuracy:.4f} ({qwen72b_accuracy*100:.2f}%)")
    
    # 准确率差异
    accuracy_diff = internvl3_accuracy - qwen72b_accuracy
    print(f"\n   准确率差异: {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f}%)")
    if accuracy_diff > 0:
        print(f"   InternVL3-38B 比 Qwen2.5-VL-72B 高 {accuracy_diff*100:.2f}%")
    else:
        print(f"   Qwen2.5-VL-72B 比 InternVL3-38B 高 {abs(accuracy_diff)*100:.2f}%")
    
    # 按置信度比较
    print(f"\n📈 按置信度分组比较:")
    print("-" * 60)
    
    # InternVL3置信度统计
    internvl3_conf_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for item in internvl3_data:
        conf = item['confidence']
        internvl3_conf_stats[conf]['total'] += 1
        if item['is_correct']:
            internvl3_conf_stats[conf]['correct'] += 1
    
    # Qwen72B置信度统计
    qwen72b_conf_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for item in qwen72b_data:
        conf = item['confidence']
        qwen72b_conf_stats[conf]['total'] += 1
        if item['is_correct']:
            qwen72b_conf_stats[conf]['correct'] += 1
    
    print(f"   置信度 | InternVL3-38B | Qwen2.5-VL-72B | 差异")
    print(f"   -------|---------------|----------------|------")
    
    all_confidences = set(internvl3_conf_stats.keys()) | set(qwen72b_conf_stats.keys())
    for conf in sorted(all_confidences):
        # InternVL3
        internvl3_stats = internvl3_conf_stats[conf]
        internvl3_acc = internvl3_stats['correct'] / internvl3_stats['total'] if internvl3_stats['total'] > 0 else 0
        
        # Qwen72B
        qwen72b_stats = qwen72b_conf_stats[conf]
        qwen72b_acc = qwen72b_stats['correct'] / qwen72b_stats['total'] if qwen72b_stats['total'] > 0 else 0
        
        diff = internvl3_acc - qwen72b_acc
        
        print(f"   {conf:6d} | {internvl3_acc:.4f} ({internvl3_stats['total']:4d}) | {qwen72b_acc:.4f} ({qwen72b_stats['total']:4d}) | {diff:+.4f}")
    
    # 按任务类型比较
    print(f"\n📋 按任务类型比较:")
    print("-" * 60)
    
    # InternVL3任务统计
    internvl3_task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for item in internvl3_data:
        task = item.get('task', 'Unknown')
        internvl3_task_stats[task]['total'] += 1
        if item['is_correct']:
            internvl3_task_stats[task]['correct'] += 1
    
    # Qwen72B任务统计
    qwen72b_task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for item in qwen72b_data:
        task = item.get('task', 'Unknown')
        qwen72b_task_stats[task]['total'] += 1
        if item['is_correct']:
            qwen72b_task_stats[task]['correct'] += 1
    
    print(f"   任务类型 | InternVL3-38B | Qwen2.5-VL-72B | 差异")
    print(f"   --------|---------------|----------------|------")
    
    all_tasks = set(internvl3_task_stats.keys()) | set(qwen72b_task_stats.keys())
    for task in sorted(all_tasks):
        # InternVL3
        internvl3_stats = internvl3_task_stats[task]
        internvl3_acc = internvl3_stats['correct'] / internvl3_stats['total'] if internvl3_stats['total'] > 0 else 0
        
        # Qwen72B
        qwen72b_stats = qwen72b_task_stats[task]
        qwen72b_acc = qwen72b_stats['correct'] / qwen72b_stats['total'] if qwen72b_stats['total'] > 0 else 0
        
        diff = internvl3_acc - qwen72b_acc
        
        print(f"   {task:8s} | {internvl3_acc:.4f} ({internvl3_stats['total']:4d}) | {qwen72b_acc:.4f} ({qwen72b_stats['total']:4d}) | {diff:+.4f}")
    
    # 置信度校准比较
    print(f"\n🎯 置信度校准比较:")
    print("-" * 60)
    
    print(f"   置信度 | InternVL3校准误差 | Qwen72B校准误差 | 差异")
    print(f"   -------|------------------|-----------------|------")
    
    for conf in sorted(all_confidences):
        # InternVL3校准误差
        internvl3_stats = internvl3_conf_stats[conf]
        if internvl3_stats['total'] > 0:
            internvl3_acc = internvl3_stats['correct'] / internvl3_stats['total']
            internvl3_cal_error = abs(internvl3_acc - (conf / 5.0))
        else:
            internvl3_cal_error = 0
        
        # Qwen72B校准误差
        qwen72b_stats = qwen72b_conf_stats[conf]
        if qwen72b_stats['total'] > 0:
            qwen72b_acc = qwen72b_stats['correct'] / qwen72b_stats['total']
            qwen72b_cal_error = abs(qwen72b_acc - (conf / 5.0))
        else:
            qwen72b_cal_error = 0
        
        diff = internvl3_cal_error - qwen72b_cal_error
        
        print(f"   {conf:6d} | {internvl3_cal_error:.4f} | {qwen72b_cal_error:.4f} | {diff:+.4f}")
    
    # 保存比较结果
    comparison_result = {
        'internvl3_38b': {
            'total_questions': internvl3_total,
            'correct_answers': internvl3_correct,
            'accuracy': internvl3_accuracy,
            'confidence_stats': dict(internvl3_conf_stats),
            'task_stats': dict(internvl3_task_stats)
        },
        'qwen2_5_vl_72b': {
            'total_questions': qwen72b_total,
            'correct_answers': qwen72b_correct,
            'accuracy': qwen72b_accuracy,
            'confidence_stats': dict(qwen72b_conf_stats),
            'task_stats': dict(qwen72b_task_stats)
        },
        'comparison': {
            'accuracy_difference': accuracy_diff,
            'better_model': 'InternVL3-38B' if accuracy_diff > 0 else 'Qwen2.5-VL-72B'
        }
    }
    
    with open('internvl3_vs_qwen72b_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细比较结果已保存到: internvl3_vs_qwen72b_comparison.json")

if __name__ == "__main__":
    compare_models()
