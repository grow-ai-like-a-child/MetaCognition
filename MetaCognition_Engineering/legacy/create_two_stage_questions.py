#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建两阶段提问的问题文件
第一阶段：实验任务
第二阶段：置信度评估
"""

import json

def create_two_stage_questions(input_file, output_file):
    """
    创建两阶段提问的问题文件
    
    Args:
        input_file: 原始questions.jsonl文件
        output_file: 输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    print(f"读取了 {len(questions)} 个问题")
    
    # 创建两阶段问题
    two_stage_questions = []
    
    for i, q in enumerate(questions):
        # 第一阶段：实验任务（不包含置信度要求）
        stage1_prompt = create_stage1_prompt(q)
        
        # 第二阶段：置信度评估
        stage2_prompt = create_stage2_prompt(q)
        
        # 创建两个阶段的问题
        stage1_question = {
            "qid": f"{q['qid']}_stage1",
            "task": q['task'],
            "image_path": q['image_path'],
            "prompt": stage1_prompt,
            "prompt_hash": q.get('prompt_hash', ''),
            "context": q['context'],
            "stage": 1,
            "original_qid": q['qid']
        }
        
        stage2_question = {
            "qid": f"{q['qid']}_stage2",
            "task": q['task'],
            "image_path": q['image_path'],
            "prompt": stage2_prompt,
            "prompt_hash": q.get('prompt_hash', ''),
            "context": q['context'],
            "stage": 2,
            "original_qid": q['qid']
        }
        
        two_stage_questions.append(stage1_question)
        two_stage_questions.append(stage2_question)
        
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个问题...")
    
    print(f"创建了 {len(two_stage_questions)} 个两阶段问题")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in two_stage_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    print(f"已保存到: {output_file}")
    
    # 统计信息
    stage1_count = len([q for q in two_stage_questions if q['stage'] == 1])
    stage2_count = len([q for q in two_stage_questions if q['stage'] == 2])
    
    print(f"\n统计信息:")
    print(f"第一阶段问题: {stage1_count}")
    print(f"第二阶段问题: {stage2_count}")
    print(f"总计: {len(two_stage_questions)}")

def create_stage1_prompt(q):
    """创建第一阶段的提示（实验任务）"""
    task = q['task']
    
    if task == "Grid":
        # 从原始prompt中提取符号信息
        prompt = q['prompt']
        # 移除置信度要求，只保留实验任务
        if "Answer with A or B, then on a new line write 'confidence: N' (1-5)." in prompt:
            prompt = prompt.replace("Answer with A or B, then on a new line write 'confidence: N' (1-5).", "Answer with A or B.")
        return prompt
    
    elif task == "Color":
        # 从原始prompt中提取布局信息
        prompt = q['prompt']
        # 移除置信度要求
        if "Answer with A or B, then on a new line write 'confidence: N' (1-5)." in prompt:
            prompt = prompt.replace("Answer with A or B, then on a new line write 'confidence: N' (1-5).", "Answer with A or B.")
        return prompt
    
    elif task == "Gabor":
        # 从原始prompt中提取方向信息
        prompt = q['prompt']
        # 移除置信度要求
        if "Answer with A or B, then on a new line write 'confidence: N' (1-5)." in prompt:
            prompt = prompt.replace("Answer with A or B, then on a new line write 'confidence: N' (1-5).", "Answer with A or B.")
        return prompt
    
    else:
        return q['prompt']

def create_stage2_prompt(q):
    """创建第二阶段的提示（置信度评估）"""
    task = q['task']
    
    if task == "Grid":
        return "Based on your previous answer, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
    
    elif task == "Color":
        return "Based on your previous answer about which side is brighter, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
    
    elif task == "Gabor":
        return "Based on your previous answer about the stripe orientation, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
    
    else:
        return "How confident are you in your previous answer? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."

if __name__ == "__main__":
    input_file = "questions.jsonl"
    output_file = "questions_two_stage.jsonl"
    
    create_two_stage_questions(input_file, output_file)
