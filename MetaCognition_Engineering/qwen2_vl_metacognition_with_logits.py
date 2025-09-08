#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Qwen2-VL-32B-Instruct模型进行元认知推理的脚本（带logits计算）
每道题目的Stage 1和Stage 2在同一个聊天会话中进行
"""

import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
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
        
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
        
        print("模型加载完成！")
    
    def prepare_inputs(self, text: str, image_inputs: List, video_inputs: List):
        """
        准备模型输入并返回input_ids和offset
        """
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # 获取input_ids和offset
        input_ids = inputs.input_ids[0]
        offset = len(input_ids)  # 这是生成开始的位置
        
        return inputs, input_ids, offset
    
    def get_reduction_fn(self, reduction: str):
        """获取概率归约函数"""
        if reduction == "mean":
            return lambda x: sum(x) / len(x)
        elif reduction == "sum":
            return sum
        elif reduction == "min":
            return min
        elif reduction == "max":
            return max
        else:
            return lambda x: x  # 返回原始列表
    
    def compute_choice_probabilities(self, text: str, image_inputs: List, video_inputs: List):
        """
        计算选项A和B的概率（通过softmax）
        """
        inputs, input_ids, offset = self.prepare_inputs(text, image_inputs, video_inputs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 获取最后一个token位置的logits（用于预测下一个token）
        last_token_logits = logits[0, -1, :]  # 形状: [vocab_size]
        
        # 获取A和B token的ID
        tokenizer = self.processor.tokenizer
        
        # 尝试不同的A和B token表示
        a_tokens = ['A', 'A.', 'A)', 'A)']
        b_tokens = ['B', 'B.', 'B)', 'B)']
        
        a_logits = []
        b_logits = []
        
        for a_token in a_tokens:
            a_id = tokenizer.convert_tokens_to_ids(a_token)
            if a_id != tokenizer.unk_token_id:
                a_logits.append(last_token_logits[a_id].item())
        
        for b_token in b_tokens:
            b_id = tokenizer.convert_tokens_to_ids(b_token)
            if b_id != tokenizer.unk_token_id:
                b_logits.append(last_token_logits[b_id].item())
        
        # 计算A和B的平均logits
        a_logit = sum(a_logits) / len(a_logits) if a_logits else -1000.0
        b_logit = sum(b_logits) / len(b_logits) if b_logits else -1000.0
        
        # 将logits转换为概率（使用softmax）
        logits_tensor = torch.tensor([a_logit, b_logit])
        probabilities = F.softmax(logits_tensor, dim=0)
        
        a_prob = probabilities[0].item()
        b_prob = probabilities[1].item()
        
        return {"A": a_prob, "B": b_prob}
    
    def process_metacognition_question(self, stage1_question: Dict[str, Any], stage2_question: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        处理一道题目的元认知流程（带logits计算）
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
        
        # 计算Stage 1的A和B选项概率
        stage1_choice_probs = self.compute_choice_probabilities(text, image_inputs, video_inputs)
        
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
        
        # Stage 2不需要计算logits
        
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
        
        # 构建结果（Stage 1包含A和B的概率，Stage 2不包含概率）
        stage1_result = {
            "qid": stage1_question["qid"],
            "task": stage1_question["task"],
            "choice": stage1_response,
            "confidence": stage1_confidence,
            "raw_text": stage1_response,
            "latency_ms": stage1_latency,
            "stage": 1,
            "original_qid": stage1_question["original_qid"],
            "probabilities": stage1_choice_probs  # 包含A和B的概率
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
            # Stage 2不包含logits
        }
        
        return stage1_result, stage2_result
    
    def process_questions_metacognition(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按元认知流程处理题目（带logits计算）
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
        
        for i, (original_qid, stages) in enumerate(questions_by_original.items()):
            print(f"处理进度: {i+1}/{total_questions}")
            
            try:
                stage1_question = stages.get("stage1")
                stage2_question = stages.get("stage2")
                
                if not stage1_question or not stage2_question:
                    print(f"跳过题目 {original_qid}: 缺少stage1或stage2")
                    continue
                
                stage1_result, stage2_result = self.process_metacognition_question(
                    stage1_question, stage2_question
                )
                
                results.extend([stage1_result, stage2_result])
                
            except Exception as e:
                print(f"处理题目 {original_qid} 时出错: {e}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                continue
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL元认知推理")
    parser.add_argument("--questions", type=str, required=True, help="题目JSONL文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSONL文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="模型名称")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    
    args = parser.parse_args()
    
    # 加载题目
    questions = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    print(f"加载了 {len(questions)} 道题目")
    
    # 初始化模型
    inference = Qwen2VLMetacognitionInference(args.model, args.device)
    
    # 处理题目
    results = inference.process_questions_metacognition(questions)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"处理完成，结果保存到: {args.output}")
    print(f"总共处理了 {len(results)} 个结果")

if __name__ == "__main__":
    main()
