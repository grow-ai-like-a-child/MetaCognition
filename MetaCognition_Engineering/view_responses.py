#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看模型响应的脚本
"""

import json
import sys
from pathlib import Path

def view_responses(jsonl_file: Path):
    """查看响应文件"""
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            response = json.loads(line)
            
            print(f"{'='*60}")
            print(f"题目 {i}: {response['qid']} ({response['task']})")
            print(f"阶段: Stage {response['stage']}")
            print(f"原始题目ID: {response['original_qid']}")
            print(f"延迟: {response['latency_ms']:.1f}ms")
            print(f"置信度: {response['confidence']}")
            print(f"{'='*60}")
            print("模型回答:")
            print(response['raw_text'])
            print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python view_responses.py <response_file.jsonl>")
        sys.exit(1)
    
    response_file = Path(sys.argv[1])
    if not response_file.exists():
        print(f"文件不存在: {response_file}")
        sys.exit(1)
    
    view_responses(response_file)
