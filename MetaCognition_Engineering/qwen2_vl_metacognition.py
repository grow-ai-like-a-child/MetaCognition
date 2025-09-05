#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Qwen2-VL-7B-Instruct模型进行元认知推理的脚本
每道题目的Stage 1和Stage 2在同一个聊天会话中进行
"""

import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2VLMetacognitionInference:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct", device: str = "auto"):
        """
        初始化Qwen2-VL模型
        """
        print(f"正在加载模型: {model_name}")
        print(f"设备: {device}")
        
        # 加载模型和处理器
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print("模型加载完成！")
    
    def process_metacognition_question(self, stage1_question: Dict[str, Any], stage2_question: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        处理一道题目的元认知流程
        Stage 1和Stage 2在同一个聊天会话中进行
        """
        print(f"处理题目: {stage1_question['original_qid']}")
        
        # 构建完整的对话历史
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{stage1_question['image_path']}",
                    },
                    {"type": "text", "text": stage1_question["prompt"]},
                ],
            }
        ]
        
        # 第一阶段：处理Stage 1
        print(f"  - Stage 1: {stage1_question['qid']}")
        start_time = time.time()
        
        # 准备Stage 1输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # 生成Stage 1回答
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1280,
                do_sample=False,
                temperature=0.0
            )
        
        # 解码Stage 1回答
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        stage1_response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        stage1_latency = (time.time() - start_time) * 1000
        
        # 将Stage 1的回答添加到对话历史
        messages.append({
            "role": "assistant",
            "content": stage1_response
        })
        
        # 第二阶段：处理Stage 2（基于Stage 1的回答）
        print(f"  - Stage 2: {stage2_question['qid']}")
        start_time = time.time()
        
        # 添加Stage 2的问题到对话历史
        messages.append({
            "role": "user",
            "content": stage2_question["prompt"]
        })
        
        # 准备Stage 2输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # 生成Stage 2回答
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1280,
                do_sample=False,
                temperature=0.0
            )
        
        # 解码Stage 2回答
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        stage2_response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        stage2_latency = (time.time() - start_time) * 1000
        
        # 提取置信度
        stage1_confidence = 0
        stage2_confidence = 0
        
        # 从stage2响应中提取置信度
        match = re.search(r'confidence:\s*(\d+)', stage2_response, re.IGNORECASE)
        if match:
            stage2_confidence = int(match.group(1))
        
        # 构建结果
        stage1_result = {
            "qid": stage1_question["qid"],
            "task": stage1_question["task"],
            "choice": stage1_response,
            "confidence": stage1_confidence,
            "raw_text": stage1_response,
            "latency_ms": stage1_latency,
            "stage": 1,
            "original_qid": stage1_question["original_qid"]
        }
        
        stage2_result = {
            "qid": stage2_question["qid"],
            "task": stage2_question["task"],
            "choice": stage2_response,
            "confidence": stage2_confidence,
            "raw_text": stage2_response,
            "latency_ms": stage2_latency,
            "stage": 2,
            "original_qid": stage2_question["original_qid"]
        }
        
        return stage1_result, stage2_result
    
    def process_questions_metacognition(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按元认知流程处理题目
        每道题目的Stage 1和Stage 2在同一个聊天会话中
        不同题目之间是独立的聊天会话
        """
        # 按original_qid分组
        questions_by_original = {}
        for question in questions:
            original_qid = question["original_qid"]
            if original_qid not in questions_by_original:
                questions_by_original[original_qid] = {}
            stage = question["stage"]
            questions_by_original[original_qid][f"stage{stage}"] = question
        
        results = []
        total_questions = len(questions_by_original)
        
        for i, (original_qid, stage_questions) in enumerate(questions_by_original.items(), 1):
            print(f"\n处理题目 {i}/{total_questions}: {original_qid}")
            
            # 获取stage1和stage2的题目
            stage1_question = stage_questions.get("stage1")
            stage2_question = stage_questions.get("stage2")
            
            if not stage1_question or not stage2_question:
                print(f"  警告: 题目 {original_qid} 缺少stage1或stage2")
                continue
            
            try:
                stage1_result, stage2_result = self.process_metacognition_question(stage1_question, stage2_question)
                results.extend([stage1_result, stage2_result])
                
            except Exception as e:
                print(f"  错误: 处理题目 {original_qid} 时出错: {e}")
                # 添加错误结果
                error_result = {
                    "qid": f"{original_qid}_error",
                    "task": stage1_question.get("task", "Unknown"),
                    "choice": "",
                    "confidence": 0,
                    "raw_text": f"ERROR: {str(e)}",
                    "latency_ms": 0,
                    "stage": 0,
                    "original_qid": original_qid
                }
                results.append(error_result)
        
        return results

def load_questions(jsonl_path: Path) -> List[Dict[str, Any]]:
    """加载题目文件"""
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def save_results(results: List[Dict[str, Any]], output_path: Path):
    """保存结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="使用Qwen2-VL-7B-Instruct进行元认知推理")
    parser.add_argument("--questions", type=Path, required=True, help="题目文件路径")
    parser.add_argument("--output", type=Path, required=True, help="输出结果文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="模型名称")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--max_questions", type=int, help="最大处理题目数量（用于测试）")
    
    args = parser.parse_args()
    
    # 加载题目
    print(f"加载题目文件: {args.questions}")
    questions = load_questions(args.questions)
    
    if args.max_questions:
        # 限制题目数量，按original_qid限制
        # 先按original_qid分组，然后取前N个完整的题目
        questions_by_original = {}
        for question in questions:
            original_qid = question["original_qid"]
            if original_qid not in questions_by_original:
                questions_by_original[original_qid] = []
            questions_by_original[original_qid].append(question)
        
        # 取前N个完整的题目（包含stage1和stage2）
        limited_questions = []
        count = 0
        for original_qid, q_list in questions_by_original.items():
            if count >= args.max_questions:
                break
            # 检查是否有完整的stage1和stage2
            has_stage1 = any(q["stage"] == 1 for q in q_list)
            has_stage2 = any(q["stage"] == 2 for q in q_list)
            if has_stage1 and has_stage2:
                limited_questions.extend(q_list)
                count += 1
        
        questions = limited_questions
        print(f"限制处理题目数量: {count} 道题目")
    
    print(f"总共 {len(questions)} 个题目条目")
    
    # 初始化模型
    inference = Qwen2VLMetacognitionInference(args.model, args.device)
    
    # 处理题目
    print("开始元认知推理...")
    start_time = time.time()
    
    results = inference.process_questions_metacognition(questions)
    
    total_time = time.time() - start_time
    print(f"\n处理完成！总耗时: {total_time:.2f}秒")
    print(f"平均每题耗时: {total_time/len(set(q['original_qid'] for q in questions)):.2f}秒")
    
    # 保存结果
    print(f"保存结果到: {args.output}")
    save_results(results, args.output)
    
    print("完成！")

if __name__ == "__main__":
    main()
