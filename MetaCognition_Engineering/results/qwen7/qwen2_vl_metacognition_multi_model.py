#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用不同型号的Qwen2.5-VL模型进行元认知推理的脚本（带logits计算，保存prompt）
支持7B、14B、32B、72B等不同规模的模型
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
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "auto"):
        """
        初始化Qwen2-VL模型
        支持不同规模的模型：7B、14B、32B、72B
        """
        print(f"正在加载模型: {model_name}")
        print(f"设备: {device}")
        
        # 根据模型大小调整torch_dtype和device_map
        model_size = self._get_model_size(model_name)
        torch_dtype = self._get_optimal_dtype(model_size)
        device_map = self._get_optimal_device_map(model_size, device)
        
        print(f"模型大小: {model_size}")
        print(f"使用数据类型: {torch_dtype}")
        print(f"设备映射: {device_map}")
        
        # 加载模型和处理器
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
        
        print("模型加载完成！")
        print(f"模型参数量: {self._count_parameters():,}")
    
    def _get_model_size(self, model_name: str) -> str:
        """从模型名称中提取模型大小"""
        if "7B" in model_name:
            return "7B"
        elif "14B" in model_name:
            return "14B"
        elif "32B" in model_name:
            return "32B"
        elif "72B" in model_name:
            return "72B"
        else:
            return "unknown"
    
    def _get_optimal_dtype(self, model_size: str) -> torch.dtype:
        """根据模型大小选择最优的数据类型"""
        if model_size in ["7B", "14B"]:
            return torch.bfloat16  # 较小模型可以使用bfloat16
        elif model_size in ["32B", "72B"]:
            return torch.bfloat16  # 大模型也使用bfloat16以节省内存
        else:
            return torch.bfloat16
    
    def _get_optimal_device_map(self, model_size: str, device: str) -> str:
        """根据模型大小选择最优的设备映射"""
        if device == "auto":
            if model_size in ["7B", "14B"]:
                return "auto"  # 小模型可以自动分配
            elif model_size in ["32B", "72B"]:
                return "auto"  # 大模型也使用auto，让transformers自动处理
            else:
                return "auto"
        else:
            return device
    
    def _count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.model.parameters())
    
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
        
        input_ids = inputs.input_ids[0]
        offset = len(input_ids)  # 这是生成开始的位置
        
        return inputs, input_ids, offset
    
    def compute_choice_probabilities(self, text: str, image_inputs: List, video_inputs: List):
        """
        计算选项A和B的概率（通过softmax）
        """
        inputs, input_ids, offset = self.prepare_inputs(text, image_inputs, video_inputs)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 获取最后一个token位置的logits
        last_token_logits = logits[0, -1, :]
        
        tokenizer = self.processor.tokenizer
        a_token = 'A'
        b_token = 'B'
        a_id = tokenizer.convert_tokens_to_ids(a_token)
        b_id = tokenizer.convert_tokens_to_ids(b_token)

        probs = F.softmax(last_token_logits, dim=-1)

        a_prob = probs[a_id].item() if a_id != tokenizer.unk_token_id else 0.0
        b_prob = probs[b_id].item() if b_id != tokenizer.unk_token_id else 0.0
        
        return {"A": a_prob, "B": b_prob}
    
    def process_metacognition_question(self, stage1_question: Dict[str, Any], stage2_question: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        处理一道题目的元认知流程（带logits计算）
        Stage 1和Stage 2在同一个聊天会话中进行
        """
        print(f"处理题目: {stage1_question['original_qid']}")
        
        # 构建对话历史
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{stage1_question['image_path']}"},
                    {"type": "text", "text": stage1_question["prompt"]},
                ],
            }
        ]
        
        # Stage 1
        print(f"  - Stage 1: {stage1_question['qid']}")
        start_time = time.time()
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stage1_prompt = text
        image_inputs, video_inputs = process_vision_info(messages)
        
        stage1_choice_probs = self.compute_choice_probabilities(text, image_inputs, video_inputs)
        
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1280,
                do_sample=False,
                temperature=0.0
            )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        stage1_response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        stage1_latency = (time.time() - start_time) * 1000
        
        messages.append({"role": "assistant", "content": stage1_response})
        
        # Stage 2
        print(f"  - Stage 2: {stage2_question['qid']}")
        start_time = time.time()
        
        messages.append({"role": "user", "content": stage2_question["prompt"]})
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stage2_prompt = text
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1280,
                do_sample=False,
                temperature=0.0
            )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        stage2_response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        stage2_latency = (time.time() - start_time) * 1000
        
        stage1_confidence = 0
        stage2_confidence = 0
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
            "original_qid": stage1_question["original_qid"],
            "probabilities": stage1_choice_probs,
            "prompt": stage1_prompt
        }
        
        stage2_result = {
            "qid": stage2_question["qid"],
            "task": stage2_question["task"],
            "choice": stage2_response,
            "confidence": stage2_confidence,
            "raw_text": stage2_response,
            "latency_ms": stage2_latency,
            "stage": 2,
            "original_qid": stage2_question["original_qid"],
            "prompt": stage2_prompt
        }
        
        return stage1_result, stage2_result
    
    def process_questions_metacognition(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按元认知流程处理题目（带logits计算）
        每道题目的Stage 1和Stage 2在同一个聊天会话中
        """
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
                
                stage1_result, stage2_result = self.process_metacognition_question(stage1_question, stage2_question)
                results.extend([stage1_result, stage2_result])
                
            except Exception as e:
                print(f"处理题目 {original_qid} 时出错: {e}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                continue
        
        return results

def get_model_recommendations():
    """获取不同GPU配置的模型推荐"""
    recommendations = {
        "A100-40GB": {
            "recommended": ["Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-14B-Instruct"],
            "possible": ["Qwen/Qwen2.5-VL-72B-Instruct"],
            "not_recommended": []
        },
        "A100-80GB": {
            "recommended": ["Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct"],
            "possible": ["Qwen/Qwen2.5-VL-14B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"],
            "not_recommended": []
        },
        "RTX-4090": {
            "recommended": ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-14B-Instruct"],
            "possible": ["Qwen/Qwen2.5-VL-32B-Instruct"],
            "not_recommended": ["Qwen/Qwen2.5-VL-72B-Instruct"]
        },
        "RTX-3090": {
            "recommended": ["Qwen/Qwen2.5-VL-7B-Instruct"],
            "possible": ["Qwen/Qwen2.5-VL-14B-Instruct"],
            "not_recommended": ["Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]
        }
    }
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL元认知推理 - 多模型支持")
    parser.add_argument("--questions", type=str, required=True, help="题目JSONL文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSONL文件路径")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                       help="模型名称 (支持: 7B, 14B, 32B, 72B)")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--show-recommendations", action="store_true", 
                       help="显示不同GPU配置的模型推荐")
    
    args = parser.parse_args()
    
    if args.show_recommendations:
        print("=== Qwen2.5-VL模型推荐 ===")
        recommendations = get_model_recommendations()
        for gpu, models in recommendations.items():
            print(f"\n{gpu}:")
            print(f"  推荐: {', '.join(models['recommended'])}")
            print(f"  可能: {', '.join(models['possible'])}")
            if models['not_recommended']:
                print(f"  不推荐: {', '.join(models['not_recommended'])}")
        return
    
    questions = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    print(f"Loaded {len(questions)} 道题目")
    print(f"使用模型: {args.model}")
    
    inference = Qwen2VLMetacognitionInference(args.model, args.device)
    
    # 分批处理并保存结果
    batch_size = 100
    total_batches = (len(questions) + batch_size - 1) // batch_size
    
    print(f"将分 {total_batches} 批处理，每批 {batch_size} 个题目")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]
            
            print(f"处理第 {batch_idx + 1}/{total_batches} 批 (题目 {start_idx + 1}-{end_idx})")
            
            try:
                batch_results = inference.process_questions_metacognition(batch_questions)
                
                # 立即保存这批结果
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()  # 强制写入磁盘
                
                print(f"第 {batch_idx + 1} 批完成，处理了 {len(batch_results)} 个结果")
                
            except Exception as e:
                print(f"第 {batch_idx + 1} 批处理失败: {e}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                continue
    
    print(f"处理完成，结果保存到: {args.output}")
    
    # 统计最终结果
    with open(args.output, 'r', encoding='utf-8') as f:
        total_results = sum(1 for line in f if line.strip())
    print(f"总共处理了 {total_results} 个结果")

if __name__ == "__main__":
    main()
