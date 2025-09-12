#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试两阶段问题的prompt内容
"""

import json

def test_prompts():
    """测试几个具体问题的prompt"""
    
    with open('questions_two_stage.jsonl', 'r', encoding='utf-8') as f:
        questions = []
        for line in f:
            try:
                questions.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                continue
    
    print(f"成功读取 {len(questions)} 个问题")
    
    # 测试COL-0538
    col_538_stage1 = None
    col_538_stage2 = None
    
    for q in questions:
        if q.get('qid') == 'COL-0538_stage1':
            col_538_stage1 = q
        elif q.get('qid') == 'COL-0538_stage2':
            col_538_stage2 = q
    
    if col_538_stage1:
        print("\n=== COL-0538 Stage 1 ===")
        print(f"Task: {col_538_stage1.get('task', 'N/A')}")
        print(f"Prompt: {col_538_stage1.get('prompt', 'N/A')}")
        print(f"Stage: {col_538_stage1.get('stage', 'N/A')}")
    
    if col_538_stage2:
        print("\n=== COL-0538 Stage 2 ===")
        print(f"Task: {col_538_stage2.get('task', 'N/A')}")
        print(f"Prompt: {col_538_stage2.get('prompt', 'N/A')}")
        print(f"Stage: {col_538_stage2.get('stage', 'N/A')}")
    
    # 测试Gridtask
    grid_1_stage1 = None
    grid_1_stage2 = None
    
    for q in questions:
        if q.get('qid') == 'GRID-0001_stage1':
            grid_1_stage1 = q
        elif q.get('qid') == 'GRID-0001_stage2':
            grid_1_stage2 = q
    
    if grid_1_stage1:
        print("\n=== GRID-0001 Stage 1 ===")
        print(f"Task: {grid_1_stage1.get('task', 'N/A')}")
        print(f"Prompt: {grid_1_stage1.get('prompt', 'N/A')}")
        print(f"Stage: {grid_1_stage1.get('stage', 'N/A')}")
    
    if grid_1_stage2:
        print("\n=== GRID-0001 Stage 2 ===")
        print(f"Task: {grid_1_stage2.get('task', 'N/A')}")
        print(f"Prompt: {grid_1_stage2.get('prompt', 'N/A')}")
        print(f"Stage: {grid_1_stage2.get('stage', 'N/A')}")

if __name__ == "__main__":
    test_prompts()
