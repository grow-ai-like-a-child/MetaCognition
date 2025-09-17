#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据ground_truth中的correct_choice与InternVL3结果进行正确匹配
"""

import json
from pathlib import Path

def create_correct_comparison():
    """
    创建正确的比较分析，基于ground_truth中的correct_choice
    """
    # 加载ground_truth
    ground_truth_file = Path("../../data/raw/ground_truth.json")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"加载了 {len(ground_truth)} 个ground_truth条目")
    
    # 加载InternVL3结果
    results_file = Path("internvl3_complete_results.jsonl")
    results = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    print(f"加载了 {len(results)} 个InternVL3结果")
    
    # 按qid分组结果
    stage1_results = {}
    stage2_results = {}
    
    for result in results:
        qid = result['qid']
        stage = result['stage']
        
        if stage == 1:
            stage1_results[qid] = result
        elif stage == 2:
            stage2_results[qid] = result
    
    print(f"找到 {len(stage1_results)} 个Stage 1结果")
    print(f"找到 {len(stage2_results)} 个Stage 2结果")
    
    # 创建匹配结果
    matched_results = []
    
    for qid, stage1_result in stage1_results.items():
        if qid not in ground_truth:
            print(f"警告: {qid} 在ground_truth中未找到")
            continue
        
        # 获取ground_truth中的正确选项
        correct_choice = ground_truth[qid]['correct_choice']
        
        # 获取模型预测的选项
        predicted_choice = stage1_result['choice']
        
        # 判断是否正确
        is_correct = (predicted_choice == correct_choice)
        
        # 获取对应的Stage 2置信度
        stage2_qid = qid.replace('_stage1', '_stage2')
        confidence = 0
        if stage2_qid in stage2_results:
            confidence = stage2_results[stage2_qid]['confidence']
        
        # 创建匹配结果条目
        matched_result = {
            'qid': qid,
            'original_qid': stage1_result['original_qid'],
            'task': stage1_result.get('task', 'Unknown'),
            'predicted_choice': predicted_choice,
            'correct_choice': correct_choice,
            'is_correct': is_correct,
            'confidence': confidence,
            'probabilities': stage1_result.get('probabilities', {}),
            'latency_ms': stage1_result.get('latency_ms', 0),
            'raw_text': stage1_result.get('raw_text', ''),
            'image_path': stage1_result.get('image_path', '')
        }
        
        matched_results.append(matched_result)
    
    # 保存匹配结果
    output_file = 'internvl3_correct_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matched_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 正确匹配结果已保存到: {output_file}")
    print(f"   包含 {len(matched_results)} 个题目的分析")
    
    # 基本统计
    total = len(matched_results)
    correct = sum(1 for item in matched_results if item['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 基本统计:")
    print(f"   总题目数: {total}")
    print(f"   正确答案数: {correct}")
    print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 按置信度统计
    from collections import defaultdict
    confidence_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for item in matched_results:
        conf = item['confidence']
        confidence_stats[conf]['total'] += 1
        if item['is_correct']:
            confidence_stats[conf]['correct'] += 1
    
    print(f"\n📈 按置信度分组统计:")
    print("-" * 50)
    
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        percentage = stats['total'] / total * 100
        
        print(f"   置信度 {conf}: {stats['correct']:4d}/{stats['total']:4d} = {accuracy:.4f} ({accuracy*100:5.2f}%) | 占比: {percentage:5.2f}%")
    
    # 显示一些示例
    print(f"\n🔍 示例结果 (前5个):")
    print("-" * 80)
    for i, item in enumerate(matched_results[:5]):
        status = "✓" if item['is_correct'] else "✗"
        print(f"   {i+1}. {item['qid']}: 预测={item['predicted_choice']}, 正确={item['correct_choice']}, 置信度={item['confidence']} {status}")

if __name__ == "__main__":
    create_correct_comparison()
