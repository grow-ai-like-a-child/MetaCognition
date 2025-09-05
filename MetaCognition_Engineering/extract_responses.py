#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取模型回答并格式化为CSV表格
"""

import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any

def extract_choice_from_stage1(raw_text: str) -> str:
    """从Stage 1的raw_text中提取选择（A或B）"""
    # 匹配 A. 或 B. 开头的选择
    match = re.match(r'^([AB])\.', raw_text.strip())
    if match:
        return match.group(1)
    
    # 如果没有找到标准格式，尝试其他模式
    if 'A' in raw_text and 'B' in raw_text:
        if raw_text.find('A') < raw_text.find('B'):
            return 'A'
        else:
            return 'B'
    
    return 'Unknown'

def extract_confidence_from_stage2(raw_text: str) -> str:
    """从Stage 2的raw_text中提取置信度"""
    match = re.search(r'confidence:\s*(\d+)', raw_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return 'Unknown'

def process_responses(jsonl_file: Path, output_csv: Path):
    """处理响应文件并生成CSV"""
    
    # 按original_qid分组
    responses_by_original = {}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            response = json.loads(line)
            original_qid = response['original_qid']
            stage = response['stage']
            
            if original_qid not in responses_by_original:
                responses_by_original[original_qid] = {}
            
            responses_by_original[original_qid][f'stage{stage}'] = response
    
    # 准备CSV数据
    csv_data = []
    
    for original_qid, stages in responses_by_original.items():
        stage1 = stages.get('stage1', {})
        stage2 = stages.get('stage2', {})
        
        # 提取Stage 1信息
        stage1_choice = extract_choice_from_stage1(stage1.get('raw_text', ''))
        stage1_raw = stage1.get('raw_text', '')
        stage1_latency = stage1.get('latency_ms', 0)
        
        # 提取Stage 2信息
        stage2_confidence = extract_confidence_from_stage2(stage2.get('raw_text', ''))
        stage2_raw = stage2.get('raw_text', '')
        stage2_latency = stage2.get('latency_ms', 0)
        
        # 构建CSV行
        csv_row = {
            'Question_ID': original_qid,
            'Task': stage1.get('task', ''),
            'Stage1_Choice': stage1_choice,
            'Stage1_Raw': stage1_raw,
            'Stage1_Latency_ms': f"{stage1_latency:.1f}",
            'Stage2_Confidence': stage2_confidence,
            'Stage2_Raw': stage2_raw,
            'Stage2_Latency_ms': f"{stage2_latency:.1f}"
        }
        
        csv_data.append(csv_row)
    
    # 写入CSV文件
    if csv_data:
        fieldnames = [
            'Question_ID', 'Task', 
            'Stage1_Choice', 'Stage1_Raw', 'Stage1_Latency_ms',
            'Stage2_Confidence', 'Stage2_Raw', 'Stage2_Latency_ms'
        ]
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"CSV文件已生成: {output_csv}")
        print(f"处理了 {len(csv_data)} 道题目")
    else:
        print("没有找到有效的响应数据")

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("用法: python extract_responses.py <input.jsonl> <output.csv>")
        print("示例: python extract_responses.py metacognition_10.jsonl responses_10.csv")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    if not input_file.exists():
        print(f"输入文件不存在: {input_file}")
        sys.exit(1)
    
    process_responses(input_file, output_file)

if __name__ == "__main__":
    main()
