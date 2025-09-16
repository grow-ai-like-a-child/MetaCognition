#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细对比分析：answers_dict与qwen2.5-vl-7b结果的逐题对比（修正版）
"""

import json
from collections import defaultdict
import re

def load_answers_dict(file_path):
    """Load correct answers dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_results(file_path):
    """Load model results"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results

def clean_model_choice(choice):
    """清理模型选择，移除多余的标点符号"""
    if isinstance(choice, str):
        # 提取A或B，忽略其他字符
        match = re.search(r'[AB]', choice)
        if match:
            return match.group()
    return choice

def create_detailed_comparison(answers_dict, model_results):
    """Create detailed question-by-question comparison analysis"""
    
    # Group by original_qid, merge stage1 and stage2
    grouped_results = defaultdict(dict)
    
    for result in model_results:
        original_qid = result.get('original_qid')
        stage = result.get('stage')
        
        if original_qid and stage:
            grouped_results[original_qid][f'stage{stage}'] = result
    
    # 创建详细对比数据
    detailed_comparison = []
    
    for original_qid, stages in grouped_results.items():
        if 'stage1' in stages and 'stage2' in stages:
            stage1 = stages['stage1']
            stage2 = stages['stage2']
            
            # Get correct answer
            correct_choice = answers_dict.get(f"{original_qid}_stage1", {}).get('correct_choice')
            
            if correct_choice:
                # 清理模型选择
                cleaned_choice = clean_model_choice(stage1['choice'])
                
                # 创建对比记录
                comparison_record = {
                    "question_id": original_qid,
                    "task_type": stage1['task'],
                    "model_choice_raw": stage1['choice'],  # 原始选择
                    "model_choice": cleaned_choice,  # 清理后的选择
                    "correct_answer": correct_choice,
                    "is_correct": cleaned_choice == correct_choice,
                    "confidence": stage2['confidence'],
                    "model_probabilities": {
                        "A": stage1['probabilities']['A'],
                        "B": stage1['probabilities']['B']
                    },
                    "latency_ms": stage1['latency_ms'],
                    "stage1_qid": stage1['qid'],
                    "stage2_qid": stage2['qid']
                }
                
                detailed_comparison.append(comparison_record)
    
    return detailed_comparison

def generate_summary_statistics(detailed_comparison):
    """Generate summary statistics"""
    
    # Group by task type统计
    task_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'confidence_sum': 0,
        'confidence_distribution': defaultdict(int)
    })
    
    total_correct = 0
    total_questions = 0
    total_confidence = 0
    
    for record in detailed_comparison:
        task = record['task_type']
        is_correct = record['is_correct']
        confidence = record['confidence']
        
        task_stats[task]['total'] += 1
        task_stats[task]['confidence_sum'] += confidence
        task_stats[task]['confidence_distribution'][confidence] += 1
        
        if is_correct:
            task_stats[task]['correct'] += 1
            total_correct += 1
        
        total_questions += 1
        total_confidence += confidence
    
    # 计算准确率和Average confidence
    for task in task_stats:
        stats = task_stats[task]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        stats['avg_confidence'] = stats['confidence_sum'] / stats['total'] if stats['total'] > 0 else 0
        stats['confidence_distribution'] = dict(stats['confidence_distribution'])
    
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    overall_avg_confidence = total_confidence / total_questions if total_questions > 0 else 0
    
    summary = {
        "overall_statistics": {
            "total_questions": total_questions,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "overall_avg_confidence": overall_avg_confidence
        },
        "task_statistics": dict(task_stats)
    }
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL metacognitive reasoning comparison analysis")
    parser.add_argument("--answers", type=str, default="../../raw/ground_truth.json", help="Correct answers file path")
    parser.add_argument("--results", type=str, required=True, help="Model results file path")
    parser.add_argument("--output", type=str, default="comparison_results.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    print("正在Loading data...")
    
    # Loading data
    answers_dict = load_answers_dict(args.answers)
    model_results = load_model_results(args.results)
    
    print(f"Loaded {len(answers_dict)} correct answers")
    print(f"Loaded {len(model_results)} model results")
    
    # 创建详细对比
    print("正在创建详细对比分析...")
    detailed_comparison = create_detailed_comparison(answers_dict, model_results)
    
    print(f"成功匹配 {len(detailed_comparison)} 道题目")
    
    # 生成汇总统计
    summary_stats = generate_summary_statistics(detailed_comparison)
    
    # 创建完整的Analyzing results
    analysis_result = {
        "summary": summary_stats,
        "detailed_comparison": detailed_comparison
    }
    
    # 保存到JSON文件
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    print(f"
📊 详细对比分析完成！")
    print(f"📄 结果已保存到: {args.output}")
    
    # 打印汇总信息
    print(f"
🎯 汇总统计:")
    print(f"  Total questions: {summary_stats['overall_statistics']['total_questions']}")
    print(f"  总Correct answers: {summary_stats['overall_statistics']['total_correct']}")
    print(f"  总体Accuracy: {summary_stats['overall_statistics']['overall_accuracy']:.2%}")
    print(f"  总体Average confidence: {summary_stats['overall_statistics']['overall_avg_confidence']:.2f}")
    
    print(f"
📈 各task类型统计:")
    for task, stats in summary_stats['task_statistics'].items():
        print(f"  {task}:")
        print(f"    题目数: {stats['total']}")
        print(f"    Correct answers: {stats['correct']}")
        print(f"    Accuracy: {stats['accuracy']:.2%}")
        print(f"    Average confidence: {stats['avg_confidence']:.2f}")
        print(f"    Confidence distribution: {stats['confidence_distribution']}")
if __name__ == "__main__":
    main()
