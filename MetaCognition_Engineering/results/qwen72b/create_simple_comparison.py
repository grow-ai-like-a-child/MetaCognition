#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建简化的对比JSON文件，只包含模型选项、正确答案和置信度
"""

import json
from typing import Dict, Any, List

def create_simple_comparison():
    """
    创建简化的对比数据
    """
    # 文件路径
    ground_truth_file = '/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/raw/ground_truth.json'
    model_results_file = '/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/results/qwen72b/qwen2.5-vl-72b_full_results.jsonl'
    output_file = '/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/results/qwen72b/qwen72b_simple_comparison.json'
    
    print("正在加载ground truth...")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    # 转换为qid到correct_choice的映射
    ground_truth_map = {}
    for qid, data in ground_truth_data.items():
        if 'correct_choice' in data:
            original_qid = qid.replace('_stage1', '')
            ground_truth_map[original_qid] = data['correct_choice']
    
    print("正在加载模型结果...")
    # 加载stage1结果
    stage1_results = {}
    with open(model_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('stage') == 1:
                original_qid = data.get('original_qid')
                if original_qid:
                    stage1_results[original_qid] = data
    
    # 加载stage2结果（置信度）
    confidence_map = {}
    with open(model_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('stage') == 2:
                original_qid = data.get('original_qid')
                confidence = data.get('confidence', 0)
                if original_qid:
                    confidence_map[original_qid] = confidence
    
    print("正在创建简化对比数据...")
    simple_results = []
    
    for qid in sorted(ground_truth_map.keys()):
        if qid in stage1_results and qid in confidence_map:
            stage1_data = stage1_results[qid]
            correct_choice = ground_truth_map[qid]
            model_choice = stage1_data.get('choice', '')
            confidence = confidence_map[qid]
            
            # 判断是否正确
            is_correct = model_choice == correct_choice
            
            simple_item = {
                'qid': qid,
                'model_choice': model_choice,
                'correct_choice': correct_choice,
                'confidence': confidence,
                'is_correct': is_correct
            }
            
            simple_results.append(simple_item)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simple_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n简化结果已保存到: {output_file}")
    print(f"总题目数: {len(simple_results)}")
    
    # 计算统计信息
    correct_count = sum(1 for item in simple_results if item['model_choice'] == item['correct_choice'])
    accuracy = correct_count / len(simple_results) if simple_results else 0
    print(f"正确答案数: {correct_count}")
    print(f"准确率: {accuracy:.4f}")
    
    # 按置信度统计
    confidence_stats = {}
    for item in simple_results:
        conf = item['confidence']
        if conf not in confidence_stats:
            confidence_stats[conf] = {'total': 0, 'correct': 0}
        confidence_stats[conf]['total'] += 1
        if item['model_choice'] == item['correct_choice']:
            confidence_stats[conf]['correct'] += 1
    
    print("\n按置信度分组的统计:")
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"置信度 {conf}: {stats['correct']}/{stats['total']} = {accuracy:.4f}")

if __name__ == "__main__":
    create_simple_comparison()
