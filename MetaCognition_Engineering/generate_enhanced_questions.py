#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成增强版两阶段题目文件
修改stage1和stage2的prompt，要求模型提供解释
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

def prompt_hash(s: str) -> str:
    """生成prompt的哈希值"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def enhance_stage1_prompt(original_prompt: str) -> str:
    """
    增强stage1的prompt，添加解释要求
    """
    # 将"Answer with A or B"替换为"Please answer with A or B and explain why you choose this answer."
    enhanced = original_prompt.replace(
        "Answer with A or B.",
        "Please answer with A or B and explain why you choose this answer."
    )
    return enhanced

def enhance_stage2_prompt(original_prompt: str) -> str:
    """
    增强stage2的prompt，添加解释要求
    """
    # 将原有的prompt替换为新的版本
    enhanced = "How confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Please answer with 'confidence: N' where N is your confidence level, and also explain why you rate your confidence as this level."
    return enhanced

def process_questions(input_file: Path, output_file: Path):
    """
    处理题目文件，生成增强版本
    """
    enhanced_questions = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            question = json.loads(line)
            
            # 创建增强版题目
            enhanced_question = question.copy()
            
            if question.get("stage") == 1:
                # 增强stage1的prompt
                enhanced_question["prompt"] = enhance_stage1_prompt(question["prompt"])
                enhanced_question["prompt_hash"] = prompt_hash(enhanced_question["prompt"])
                
            elif question.get("stage") == 2:
                # 增强stage2的prompt
                enhanced_question["prompt"] = enhance_stage2_prompt(question["prompt"])
                enhanced_question["prompt_hash"] = prompt_hash(enhanced_question["prompt"])
            
            enhanced_questions.append(enhanced_question)
    
    # 保存增强版题目
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in enhanced_questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"处理完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"处理题目数量: {len(enhanced_questions)}")

def main():
    input_file = Path("questions_two_stage.jsonl")
    output_file = Path("questions_two_stage_enhanced.jsonl")
    
    if not input_file.exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    process_questions(input_file, output_file)

if __name__ == "__main__":
    main()
