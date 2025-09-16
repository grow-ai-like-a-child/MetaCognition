#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较Qwen2.5-VL-72B结果与ground truth，生成包含模型选项、正确答案和置信度的JSON文件
"""

import json
import re
from typing import Dict, Any, List

def load_ground_truth(ground_truth_file: str) -> Dict[str, str]:
    """
    加载ground truth文件，返回qid到correct_choice的映射
    """
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    # 转换为qid到correct_choice的映射
    ground_truth_map = {}
    for qid, data in ground_truth_data.items():
        if 'correct_choice' in data:
            # 将GRID-0001_stage1转换为GRID-0001
            original_qid = qid.replace('_stage1', '')
            ground_truth_map[original_qid] = data['correct_choice']
    
    return ground_truth_map

def load_model_results(results_file: str) -> List[Dict[str, Any]]:
    """
    加载模型结果文件，只保留stage1的结果
    """
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('stage') == 1:  # 只保留stage1的结果
                results.append(data)
    return results

def extract_confidence_from_stage2(results_file: str) -> Dict[str, int]:
    """
    从stage2结果中提取置信度信息
    """
    confidence_map = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('stage') == 2:  # stage2包含置信度
                original_qid = data.get('original_qid')
                confidence = data.get('confidence', 0)
                if original_qid:
                    confidence_map[original_qid] = confidence
    return confidence_map

def create_comparison_data(model_results: List[Dict[str, Any]], 
                          ground_truth_map: Dict[str, str], 
                          confidence_map: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    创建对比数据
    """
    comparison_data = []
    
    for result in model_results:
        original_qid = result.get('original_qid')
        if not original_qid:
            continue
            
        # 获取模型的选择
        model_choice = result.get('choice', '')
        
        # 获取正确答案
        correct_choice = ground_truth_map.get(original_qid, '')
        
        # 获取置信度
        confidence = confidence_map.get(original_qid, 0)
        
        # 判断是否正确
        is_correct = model_choice == correct_choice
        
        comparison_item = {
            'qid': original_qid,
            'task': result.get('task', ''),
            'model_choice': model_choice,
            'correct_choice': correct_choice,
            'confidence': confidence,
            'is_correct': is_correct,
            'latency_ms': result.get('latency_ms', 0),
            'probabilities': result.get('probabilities', {})
        }
        
        comparison_data.append(comparison_item)
    
    return comparison_data

def main():
    # 文件路径
    ground_truth_file = '/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/raw/ground_truth.json'
    model_results_file = '/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/results/qwen72b/qwen2.5-vl-72b_full_results.jsonl'
    output_file = '/home/ubuntu/MetaCognition/MetaCognition_Engineering/data/results/qwen72b/qwen72b_comparison_with_ground_truth.json'
    
    print("正在加载ground truth...")
    ground_truth_map = load_ground_truth(ground_truth_file)
    print(f"加载了 {len(ground_truth_map)} 个ground truth条目")
    
    print("正在加载模型结果...")
    model_results = load_model_results(model_results_file)
    print(f"加载了 {len(model_results)} 个stage1结果")
    
    print("正在提取置信度信息...")
    confidence_map = extract_confidence_from_stage2(model_results_file)
    print(f"提取了 {len(confidence_map)} 个置信度值")
    
    print("正在创建对比数据...")
    comparison_data = create_comparison_data(model_results, ground_truth_map, confidence_map)
    
    # 计算统计信息
    total_questions = len(comparison_data)
    correct_answers = sum(1 for item in comparison_data if item['is_correct'])
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # 按置信度分组统计
    confidence_stats = {}
    for item in comparison_data:
        conf = item['confidence']
        if conf not in confidence_stats:
            confidence_stats[conf] = {'total': 0, 'correct': 0}
        confidence_stats[conf]['total'] += 1
        if item['is_correct']:
            confidence_stats[conf]['correct'] += 1
    
    # 计算每个置信度级别的准确率
    for conf in confidence_stats:
        total = confidence_stats[conf]['total']
        correct = confidence_stats[conf]['correct']
        confidence_stats[conf]['accuracy'] = correct / total if total > 0 else 0
    
    # 创建最终结果
    final_result = {
        'summary': {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'confidence_statistics': confidence_stats
        },
        'detailed_results': comparison_data
    }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"总题目数: {total_questions}")
    print(f"正确答案数: {correct_answers}")
    print(f"准确率: {accuracy:.4f}")
    
    print("\n按置信度分组的统计:")
    for conf in sorted(confidence_stats.keys()):
        stats = confidence_stats[conf]
        print(f"置信度 {conf}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

if __name__ == "__main__":
    main()
