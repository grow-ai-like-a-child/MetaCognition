#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Qwen2.5-VL-72B模型在三种任务类型中的置信度分布
"""

import json
from collections import defaultdict

def analyze_confidence_by_task():
    """
    分析不同任务类型的置信度分布
    """
    # 加载数据
    with open('/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/results/qwen72b/qwen72b_simple_comparison.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("Qwen2.5-VL-72B 模型三种任务类型的置信度分布分析")
    print("=" * 80)
    
    # 按任务类型分组
    task_data = defaultdict(list)
    for item in data:
        task_type = item['qid'].split('-')[0]
        task_data[task_type].append(item)
    
    # 分析每种任务类型
    for task_type in ['GRID', 'COL', 'GAB']:
        if task_type not in task_data:
            continue
            
        task_items = task_data[task_type]
        total = len(task_items)
        correct = sum(1 for item in task_items if item['is_correct'])
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n🎯 {task_type} 任务分析:")
        print("-" * 60)
        print(f"   总题目数: {total}")
        print(f"   正确答案数: {correct}")
        print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 置信度分布
        print(f"\n📊 {task_type} 置信度分布:")
        print("-" * 40)
        
        confidence_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for item in task_items:
            conf = item['confidence']
            confidence_stats[conf]['total'] += 1
            if item['is_correct']:
                confidence_stats[conf]['correct'] += 1
        
        # 按置信度排序显示
        for conf in sorted(confidence_stats.keys()):
            stats = confidence_stats[conf]
            accuracy_at_conf = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            percentage = stats['total'] / total * 100
            
            # 创建可视化条形图
            bar_length = int(percentage / 2)  # 每2%一个方块
            bar = "█" * bar_length
            
            print(f"   置信度 {conf}: {stats['correct']:3d}/{stats['total']:3d} = {accuracy_at_conf:.4f} ({accuracy_at_conf*100:5.2f}%) | 占比: {percentage:5.2f}% {bar}")
        
        # 置信度校准分析
        print(f"\n🧠 {task_type} 置信度校准:")
        print("-" * 30)
        
        # 计算不同置信度水平的准确率
        conf_acc_pairs = []
        for conf in sorted(confidence_stats.keys()):
            stats = confidence_stats[conf]
            if stats['total'] > 0:
                accuracy_at_conf = stats['correct'] / stats['total']
                conf_acc_pairs.append((conf, accuracy_at_conf, stats['total']))
        
        if len(conf_acc_pairs) >= 2:
            # 检查置信度校准
            high_conf_acc = conf_acc_pairs[-1][1]  # 最高置信度的准确率
            low_conf_acc = conf_acc_pairs[0][1]    # 最低置信度的准确率
            
            print(f"   最高置信度准确率: {high_conf_acc:.4f}")
            print(f"   最低置信度准确率: {low_conf_acc:.4f}")
            
            if high_conf_acc > low_conf_acc:
                print(f"   ✅ 置信度校准良好")
            elif high_conf_acc < low_conf_acc:
                print(f"   ⚠️  置信度校准不佳（过度自信）")
            else:
                print(f"   ➖ 置信度校准中性")
        
        # 置信度分布统计
        print(f"\n📈 {task_type} 置信度统计:")
        print("-" * 30)
        
        conf_values = [item['confidence'] for item in task_items]
        if conf_values:
            avg_confidence = sum(conf_values) / len(conf_values)
            min_confidence = min(conf_values)
            max_confidence = max(conf_values)
            
            print(f"   平均置信度: {avg_confidence:.2f}")
            print(f"   最低置信度: {min_confidence}")
            print(f"   最高置信度: {max_confidence}")
            
            # 置信度分布百分比
            conf_distribution = defaultdict(int)
            for conf in conf_values:
                conf_distribution[conf] += 1
            
            print(f"   置信度分布:")
            for conf in sorted(conf_distribution.keys()):
                count = conf_distribution[conf]
                percentage = count / total * 100
                print(f"     {conf}: {count:3d} ({percentage:5.2f}%)")
    
    # 跨任务类型比较
    print(f"\n🔄 跨任务类型比较:")
    print("-" * 60)
    
    comparison_data = []
    for task_type in ['GRID', 'COL', 'GAB']:
        if task_type in task_data:
            task_items = task_data[task_type]
            total = len(task_items)
            correct = sum(1 for item in task_items if item['is_correct'])
            accuracy = correct / total if total > 0 else 0
            
            # 计算平均置信度
            conf_values = [item['confidence'] for item in task_items]
            avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0
            
            # 计算置信度5的占比
            conf_5_count = sum(1 for conf in conf_values if conf == 5)
            conf_5_percentage = conf_5_count / total * 100 if total > 0 else 0
            
            comparison_data.append({
                'task': task_type,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'conf_5_percentage': conf_5_percentage,
                'total': total
            })
    
    # 显示比较表格
    print(f"{'任务类型':<8} {'准确率':<10} {'平均置信度':<12} {'置信度5占比':<12} {'题目数':<8}")
    print("-" * 60)
    for data in comparison_data:
        print(f"{data['task']:<8} {data['accuracy']:<10.4f} {data['avg_confidence']:<12.2f} {data['conf_5_percentage']:<12.2f} {data['total']:<8}")
    
    # 分析置信度与准确率的关系
    print(f"\n🔍 置信度与准确率关系分析:")
    print("-" * 50)
    
    for data in comparison_data:
        task = data['task']
        accuracy = data['accuracy']
        avg_conf = data['avg_confidence']
        conf_5_pct = data['conf_5_percentage']
        
        print(f"{task}:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   平均置信度: {avg_conf:.2f}")
        print(f"   置信度5占比: {conf_5_pct:.2f}%")
        
        # 判断是否过度自信
        if conf_5_pct > 90 and accuracy < 0.6:
            print(f"   ⚠️  可能过度自信：高置信度占比({conf_5_pct:.1f}%)但准确率较低({accuracy:.4f})")
        elif conf_5_pct < 50 and accuracy > 0.8:
            print(f"   ✅ 置信度适中：低置信度占比({conf_5_pct:.1f}%)且准确率高({accuracy:.4f})")
        else:
            print(f"   ➖ 置信度与准确率匹配度一般")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_confidence_by_task()
