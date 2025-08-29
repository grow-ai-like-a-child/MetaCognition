#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证两阶段runner的输出结果
"""

import json
import csv
from pathlib import Path

def validate_questions_file(file_path: str):
    """验证questions_two_stage.jsonl文件"""
    print(f"\n=== 验证 {file_path} ===")
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                questions.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析错误: {e}")
                continue
    
    print(f"✅ 成功读取 {len(questions)} 个问题")
    
    # 检查阶段分布
    stage1_count = len([q for q in questions if q.get('stage') == 1])
    stage2_count = len([q for q in questions if q.get('stage') == 2])
    
    print(f"📊 阶段分布:")
    print(f"  第一阶段: {stage1_count}")
    print(f"  第二阶段: {stage2_count}")
    
    # 检查任务类型分布
    task_counts = {}
    for q in questions:
        task = q.get('task', 'Unknown')
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print(f"📊 任务类型分布:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")
    
    # 检查示例问题
    print(f"\n📋 示例问题:")
    for i, q in enumerate(questions[:3]):
        print(f"  问题 {i+1}:")
        print(f"    qid: {q.get('qid', 'N/A')}")
        print(f"    task: {q.get('task', 'N/A')}")
        print(f"    stage: {q.get('stage', 'N/A')}")
        print(f"    prompt: {q.get('prompt', 'N/A')[:100]}...")
        print()

def validate_responses_file(file_path: str):
    """验证responses_two_stage.jsonl文件"""
    print(f"\n=== 验证 {file_path} ===")
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    responses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                responses.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析错误: {e}")
                continue
    
    print(f"✅ 成功读取 {len(responses)} 个响应")
    
    # 检查阶段分布
    stage1_count = len([r for r in responses if r.get('stage') == 1])
    stage2_count = len([r for r in responses if r.get('stage') == 2])
    
    print(f"📊 阶段分布:")
    print(f"  第一阶段: {stage1_count}")
    print(f"  第二阶段: {stage2_count}")
    
    # 检查置信度分布
    confidences = [r.get('confidence', 0) for r in responses]
    conf_dist = {}
    for conf in confidences:
        conf_dist[conf] = conf_dist.get(conf, 0) + 1
    
    print(f"📊 置信度分布:")
    for conf in sorted(conf_dist.keys()):
        print(f"  {conf}: {conf_dist[conf]}")
    
    # 检查示例响应
    print(f"\n📋 示例响应:")
    for i, r in enumerate(responses[:3]):
        print(f"  响应 {i+1}:")
        print(f"    qid: {r.get('qid', 'N/A')}")
        print(f"    choice: {r.get('choice', 'N/A')}")
        print(f"    confidence: {r.get('confidence', 'N/A')}")
        print(f"    stage: {r.get('stage', 'N/A')}")
        print()

def validate_eval_file(file_path: str):
    """验证eval_two_stage.csv文件"""
    print(f"\n=== 验证 {file_path} ===")
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    rows = []
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"✅ 成功读取 {len(rows)} 行评估结果")
    
    if not rows:
        print("❌ 没有评估结果")
        return
    
    # 检查列名
    print(f"📊 CSV列名:")
    for col in rows[0].keys():
        print(f"  {col}")
    
    # 检查任务类型分布
    task_counts = {}
    for row in rows:
        task = row.get('task', 'Unknown')
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print(f"\n📊 任务类型分布:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")
    
    # 检查正确性分布
    correct_count = len([row for row in rows if row.get('is_correct', '').lower() == 'true'])
    total_count = len(rows)
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    
    print(f"\n📊 正确性分布:")
    print(f"  正确: {correct_count}")
    print(f"  错误: {total_count - correct_count}")
    print(f"  准确率: {accuracy:.1f}%")
    
    # 检查示例行
    print(f"\n📋 示例评估行:")
    for i, row in enumerate(rows[:3]):
        print(f"  行 {i+1}:")
        print(f"    qid: {row.get('qid', 'N/A')}")
        print(f"    task: {row.get('task', 'N/A')}")
        print(f"    stage1_choice: {row.get('stage1_choice', 'N/A')}")
        print(f"    stage2_confidence: {row.get('stage2_confidence', 'N/A')}")
        print(f"    is_correct: {row.get('is_correct', 'N/A')}")
        print()

def validate_file_consistency():
    """验证文件之间的一致性"""
    print(f"\n=== 验证文件一致性 ===")
    
    # 检查文件是否存在
    files = {
        'questions': 'questions_two_stage.jsonl',
        'responses': 'responses_two_stage.jsonl', 
        'eval': 'eval_two_stage.csv'
    }
    
    existing_files = {}
    for name, path in files.items():
        if Path(path).exists():
            existing_files[name] = path
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (不存在)")
    
    if len(existing_files) < 2:
        print("⚠️  需要至少两个文件来进行一致性检查")
        return
    
    # 检查问题数量和响应数量是否一致
    if 'questions' in existing_files and 'responses' in existing_files:
        with open(existing_files['questions'], 'r', encoding='utf-8') as f:
            q_count = sum(1 for line in f if line.strip())
        
        with open(existing_files['responses'], 'r', encoding='utf-8') as f:
            r_count = sum(1 for line in f if line.strip())
        
        if q_count == r_count:
            print(f"✅ 问题数量 ({q_count}) 和响应数量 ({r_count}) 一致")
        else:
            print(f"❌ 问题数量 ({q_count}) 和响应数量 ({r_count}) 不一致")
    
    # 检查评估结果数量是否合理
    if 'eval' in existing_files:
        with open(existing_files['eval'], 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            eval_count = sum(1 for row in reader)
        
        if 'questions' in existing_files:
            with open(existing_files['questions'], 'r', encoding='utf-8') as f:
                q_count = sum(1 for line in f if line.strip())
            
            expected_eval_count = q_count // 2  # 每个原始题目对应一行评估
            if eval_count == expected_eval_count:
                print(f"✅ 评估结果数量 ({eval_count}) 符合预期 ({expected_eval_count})")
            else:
                print(f"❌ 评估结果数量 ({eval_count}) 不符合预期 ({expected_eval_count})")

def main():
    """主函数"""
    print("🔍 两阶段Runner输出验证")
    print("=" * 50)
    
    # 验证各个文件
    validate_questions_file('questions_two_stage.jsonl')
    validate_responses_file('responses_two_stage.jsonl')
    validate_eval_file('eval_two_stage.csv')
    
    # 验证文件一致性
    validate_file_consistency()
    
    print("\n✅ 验证完成!")

if __name__ == "__main__":
    main()
