#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建简洁版本的数据集，修改prompt格式以强制模型给出更简洁的回答
"""

import json
from pathlib import Path

def create_concise_dataset(input_file: str, output_file: str):
    """
    创建简洁版本的数据集
    """
    print(f"读取原始数据集: {input_file}")
    
    # 读取原始数据
    questions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    print(f"原始数据集包含 {len(questions)} 个题目条目")
    
    # 修改prompt
    modified_questions = []
    for question in questions:
        # 创建新的题目对象
        new_question = question.copy()
        
        # 修改Stage 1的prompt
        if question["stage"] == 1:
            old_prompt = question["prompt"]
            # 替换 "Answer with A or B." 为 "Answer with A or B only without any reasoning."
            new_prompt = old_prompt.replace(
                "Answer with A or B.",
                "Answer with A or B only without any reasoning."
            )
            new_question["prompt"] = new_prompt
            
        # 修改Stage 2的prompt
        elif question["stage"] == 2:
            old_prompt = question["prompt"]
            # 替换 "Answer with 'confidence: N' where N is your confidence level." 
            # 为 "Answer with 'confidence: N' only, where N is your confidence level."
            new_prompt = old_prompt.replace(
                "Answer with 'confidence: N' where N is your confidence level.",
                "Answer with 'confidence: N' only, where N is your confidence level."
            )
            new_question["prompt"] = new_prompt
        
        modified_questions.append(new_question)
    
    # 保存修改后的数据集
    print(f"保存修改后的数据集: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in modified_questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"完成！修改后的数据集包含 {len(modified_questions)} 个题目条目")
    
    # 显示一些示例
    print("\n修改示例:")
    print("Stage 1 原prompt:", questions[0]["prompt"])
    print("Stage 1 新prompt:", modified_questions[0]["prompt"])
    print("\nStage 2 原prompt:", questions[1]["prompt"])
    print("Stage 2 新prompt:", modified_questions[1]["prompt"])

if __name__ == "__main__":
    input_file = "questions_two_stage.jsonl"
    output_file = "questions_two_stage_concise.jsonl"
    
    create_concise_dataset(input_file, output_file)
