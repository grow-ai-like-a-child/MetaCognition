#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Ovis2-34B模型进行元认知推理的脚本（带logits计算，保存prompt，实时写入）
每道题目的Stage 1和Stage 2在同一个聊天会话中进行
实时写入结果，避免数据丢失
"""

import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from PIL import Image

class Ovis2MetacognitionInference:
    def __init__(self, model_name: str = "AIDC-AI/Ovis2-34B", device: str = "auto"):
        """
        初始化Ovis2-34B模型
        """
        print(f"正在加载模型: {model_name}")
        print(f"设备: {device}")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True
        ).cuda()
        
        # 获取tokenizer
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        
        print("模型加载完成！")
    
    def prepare_inputs(self, text: str, image_path: str):
        """
        准备模型输入并返回input_ids和offset
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 构建查询
        query = f'<image>\n{text}'
        
        # 预处理输入
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [image], max_partition=9
        )
        
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype, 
                device=self.visual_tokenizer.device
            )
        pixel_values = [pixel_values]
        
        offset = len(input_ids[0])  # 这是生成开始的位置
        
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask
        }, input_ids, offset, query, prompt
    
    def compute_choice_probabilities(self, text: str, image_path: str):
        """
        计算选项A和B的概率（通过softmax）
        """
        inputs, input_ids, offset, query, prompt = self.prepare_inputs(text, image_path)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs['attention_mask'],
                labels=None,
                return_dict=True
            )
            logits = outputs.logits

        # 获取最后一个token位置的logits
        last_token_logits = logits[0, -1, :]
        
        a_token = 'A'
        b_token = 'B'
        a_tokens = self.text_tokenizer.encode(a_token, add_special_tokens=False)
        b_tokens = self.text_tokenizer.encode(b_token, add_special_tokens=False)
        
        a_id = a_tokens[0] if a_tokens else self.text_tokenizer.unk_token_id
        b_id = b_tokens[0] if b_tokens else self.text_tokenizer.unk_token_id

        probs = F.softmax(last_token_logits, dim=-1)

        a_prob = probs[a_id].item() if a_id != self.text_tokenizer.unk_token_id else 0.0
        b_prob = probs[b_id].item() if b_id != self.text_tokenizer.unk_token_id else 0.0
        
        return {"A": a_prob, "B": b_prob}
    
    def generate_response(self, text: str, image_path: str, max_new_tokens: int = 1280):
        """
        生成模型响应
        """
        inputs, input_ids, offset, query, prompt = self.prepare_inputs(text, image_path)
        
        # 生成响应
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            
            output_ids = self.model.generate(
                inputs['input_ids'], 
                pixel_values=inputs['pixel_values'], 
                attention_mask=inputs['attention_mask'], 
                **gen_kwargs
            )[0]
            
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # 提取响应部分（去除输入prompt）
            if query in output:
                response = output.split(query)[-1].strip()
            else:
                response = output.strip()
                
            return response, prompt
    
    def process_metacognition_question(self, stage1_question: Dict[str, Any], stage2_question: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        处理一道题目的元认知流程（带logits计算）
        Stage 1和Stage 2在同一个聊天会话中进行
        """
        print(f"处理题目: {stage1_question['original_qid']}")
        
        # Stage 1
        print(f"  - Stage 1: {stage1_question['qid']}")
        start_time = time.time()
        
        # 获取Stage 1的选择概率
        stage1_choice_probs = self.compute_choice_probabilities(
            stage1_question["prompt"], 
            stage1_question["image_path"]
        )
        
        # 生成Stage 1响应
        stage1_response, stage1_prompt = self.generate_response(
            stage1_question["prompt"], 
            stage1_question["image_path"]
        )
        stage1_latency = (time.time() - start_time) * 1000
        
        # Stage 2
        print(f"  - Stage 2: {stage2_question['qid']}")
        start_time = time.time()
        
        # 构建Stage 2的上下文（包含Stage 1的对话历史）
        stage2_context = f"Previous question: {stage1_question['prompt']}\nPrevious answer: {stage1_response}\n\n{stage2_question['prompt']}"
        
        # 生成Stage 2响应
        stage2_response, stage2_prompt = self.generate_response(
            stage2_context,
            stage2_question["image_path"]
        )
        stage2_latency = (time.time() - start_time) * 1000
        
        # 提取置信度
        stage1_confidence = 0
        stage2_confidence = 0
        
        # 从Stage 2响应中提取置信度分数
        confidence_patterns = [
            r'confidence:\s*(\d+)',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*10',
            r'(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)/10',
            r'(\d+(?:\.\d+)?)\s*分',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, stage2_response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # 规范化到0-10范围
                    if score > 10:
                        score = score / 10  # 百分比转换
                    stage2_confidence = min(max(score, 0), 10)
                    break
                except ValueError:
                    continue
        
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
    
    def process_questions_metacognition(self, questions: List[Dict[str, Any]], output_file: str) -> List[Dict[str, Any]]:
        """
        按元认知流程处理题目（带logits计算）
        每道题目的Stage 1和Stage 2在同一个聊天会话中
        实时写入结果到文件
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
        
        # 打开输出文件进行实时写入
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (original_qid, stages) in enumerate(questions_by_original.items()):
                print(f"处理进度: {i+1}/{total_questions}")
                
                try:
                    stage1_question = stages.get("stage1")
                    stage2_question = stages.get("stage2")
                    
                    if not stage1_question or not stage2_question:
                        print(f"跳过题目 {original_qid}: 缺少stage1或stage2")
                        continue
                    
                    stage1_result, stage2_result = self.process_metacognition_question(stage1_question, stage2_question)
                    
                    # 立即写入结果到文件
                    f.write(json.dumps(stage1_result, ensure_ascii=False) + '\n')
                    f.write(json.dumps(stage2_result, ensure_ascii=False) + '\n')
                    f.flush()  # 强制刷新缓冲区
                    
                    results.extend([stage1_result, stage2_result])
                    
                    print(f"已保存题目 {original_qid} 的结果")
                    
                except Exception as e:
                    print(f"处理题目 {original_qid} 时出错: {e}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    continue
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Ovis2-34B元认知推理（实时写入版本）")
    parser.add_argument("--questions", type=str, required=True, help="题目JSONL文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSONL文件路径")
    parser.add_argument("--model", type=str, default="AIDC-AI/Ovis2-34B", help="模型名称")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    
    args = parser.parse_args()
    
    questions = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    print(f"Loaded {len(questions)} 道题目")
    
    inference = Ovis2MetacognitionInference(args.model, args.device)
    results = inference.process_questions_metacognition(questions, args.output)
    
    print(f"处理完成，结果保存到: {args.output}")
    print(f"总共处理了 {len(results)} 个结果")

if __name__ == "__main__":
    main()
