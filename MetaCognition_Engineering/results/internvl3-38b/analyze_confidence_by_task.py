#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析InternVL3-38B每类任务的置信度分布
"""

import json
from collections import defaultdict
from pathlib import Path

def analyze_confidence_by_task():
    """
    分析每类任务的置信度分布
    """
    # 加载正确比较结果
    comparison_file = Path("internvl3_correct_comparison.json")
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("InternVL3-38B 每类任务置信度分布分析")
    print("=" * 80)
    
    # 按任务类型分组
    task_groups = defaultdict(list)
    for item in data:
        task = item.get('task', 'Unknown')
        task_groups[task].append(item)
    
    print(f"\n📊 任务类型概览:")
    print("-" * 60)
    for task in sorted(task_groups.keys()):
        count = len(task_groups[task])
        percentage = count / len(data) * 100
        print(f"   {task:12s}: {count:4d} 题 ({percentage:5.1f}%)")
    
    # 分析每个任务的置信度分布
    for task in sorted(task_groups.keys()):
        task_data = task_groups[task]
        
        print(f"\n🔍 {task} 任务详细分析:")
        print("=" * 60)
        
        # 基本统计
        total = len(task_data)
        correct = sum(1 for item in task_data if item['is_correct'])
        accuracy = correct / total if total > 0 else 0
        
        print(f"   总题目数: {total}")
        print(f"   正确答案数: {correct}")
        print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 置信度分布
        print(f"\n   置信度分布:")
        print("   " + "-" * 50)
        
        confidence_dist = defaultdict(int)
        confidence_correct = defaultdict(int)
        
        for item in task_data:
            conf = item['confidence']
            confidence_dist[conf] += 1
            if item['is_correct']:
                confidence_correct[conf] += 1
        
        for conf in sorted(confidence_dist.keys()):
            count = confidence_dist[conf]
            correct_count = confidence_correct[conf]
            percentage = count / total * 100
            accuracy_at_conf = correct_count / count if count > 0 else 0
            
            print(f"     置信度 {conf}: {count:4d} 题 ({percentage:5.1f}%) | 正确: {correct_count:4d} | 准确率: {accuracy_at_conf:.4f}")
        
        # 置信度与准确率关系
        print(f"\n   置信度与准确率关系:")
        print("   " + "-" * 50)
        
        for conf in sorted(confidence_dist.keys()):
            count = confidence_dist[conf]
            correct_count = confidence_correct[conf]
            accuracy_at_conf = correct_count / count if count > 0 else 0
            calibration_error = abs(accuracy_at_conf - (conf / 5.0))
            
            print(f"     置信度 {conf}: 准确率={accuracy_at_conf:.3f}, 理想值={conf/5.0:.3f}, 校准误差={calibration_error:.3f}")
        
        # 预测分布
        print(f"\n   预测分布:")
        print("   " + "-" * 30)
        
        predicted_dist = defaultdict(int)
        for item in task_data:
            predicted = item['predicted_choice']
            predicted_dist[predicted] += 1
        
        for choice in ['A', 'B']:
            count = predicted_dist[choice]
            percentage = count / total * 100
            print(f"     预测{choice}: {count:4d} 题 ({percentage:5.1f}%)")
        
        # 错误案例分析
        wrong_cases = [item for item in task_data if not item['is_correct']]
        if wrong_cases:
            print(f"\n   错误案例分析 (共{len(wrong_cases)}个错误):")
            print("   " + "-" * 50)
            
            # 按置信度分组的错误
            wrong_by_confidence = defaultdict(int)
            for item in wrong_cases:
                wrong_by_confidence[item['confidence']] += 1
            
            for conf in sorted(wrong_by_confidence.keys()):
                count = wrong_by_confidence[conf]
                percentage = count / len(wrong_cases) * 100
                print(f"     置信度 {conf}: {count:4d} 个错误 ({percentage:5.1f}%)")
            
            # 高置信度但错误的案例
            high_conf_wrong = [item for item in wrong_cases if item['confidence'] >= 4]
            if high_conf_wrong:
                print(f"\n     高置信度(≥4)但错误: {len(high_conf_wrong)} 个")
                
                # 显示一些示例
                print(f"     示例高置信度错误案例:")
                for i, item in enumerate(high_conf_wrong[:3]):
                    print(f"       {i+1}. {item['qid']}: 预测={item['predicted_choice']}, 正确={item['correct_choice']}, 置信度={item['confidence']}")
    
    # 跨任务置信度对比
    print(f"\n📈 跨任务置信度对比:")
    print("=" * 80)
    
    print(f"   任务类型 | 置信度3 | 置信度4 | 置信度5 | 平均置信度")
    print("   " + "-" * 60)
    
    for task in sorted(task_groups.keys()):
        task_data = task_groups[task]
        
        # 计算置信度分布
        confidence_dist = defaultdict(int)
        for item in task_data:
            conf = item['confidence']
            confidence_dist[conf] += 1
        
        total = len(task_data)
        conf3_pct = confidence_dist[3] / total * 100 if total > 0 else 0
        conf4_pct = confidence_dist[4] / total * 100 if total > 0 else 0
        conf5_pct = confidence_dist[5] / total * 100 if total > 0 else 0
        
        # 计算平均置信度
        total_conf = sum(item['confidence'] for item in task_data)
        avg_conf = total_conf / total if total > 0 else 0
        
        print(f"   {task:8s} | {conf3_pct:6.1f}% | {conf4_pct:6.1f}% | {conf5_pct:6.1f}% | {avg_conf:8.2f}")
    
    # 保存详细分析结果
    analysis_result = {}
    
    for task in sorted(task_groups.keys()):
        task_data = task_groups[task]
        
        # 基本统计
        total = len(task_data)
        correct = sum(1 for item in task_data if item['is_correct'])
        accuracy = correct / total if total > 0 else 0
        
        # 置信度分布
        confidence_dist = defaultdict(int)
        confidence_correct = defaultdict(int)
        
        for item in task_data:
            conf = item['confidence']
            confidence_dist[conf] += 1
            if item['is_correct']:
                confidence_correct[conf] += 1
        
        # 计算平均置信度
        total_conf = sum(item['confidence'] for item in task_data)
        avg_conf = total_conf / total if total > 0 else 0
        
        analysis_result[task] = {
            'total_questions': total,
            'correct_answers': correct,
            'accuracy': accuracy,
            'average_confidence': avg_conf,
            'confidence_distribution': dict(confidence_dist),
            'confidence_accuracy': {
                conf: correct_count / confidence_dist[conf] if confidence_dist[conf] > 0 else 0
                for conf, correct_count in confidence_correct.items()
            }
        }
    
    # 保存结果
    with open('internvl3_confidence_by_task_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细分析结果已保存到: internvl3_confidence_by_task_analysis.json")

if __name__ == "__main__":
    analyze_confidence_by_task()
