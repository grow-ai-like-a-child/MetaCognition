#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 3 27B IT 元认知推理实验
基于 https://huggingface.co/google/gemma-3-27b-it
使用本地GPU运行
"""

import json
import argparse
import time
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import numpy as np

class Gemma3MetacognitionInference:
    def __init__(self, model_name="google/gemma-3-27b-it", device="auto"):
        """
        初始化Gemma 3模型 - 使用本地GPU
        """
        print(f"正在加载模型: {model_name}")
        
        # 设置设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 检查GPU内存
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU内存: {gpu_memory:.1f} GB")
            if gpu_memory < 20:
                print("⚠️  警告: GPU内存可能不足，建议至少20GB")
        
        # 加载模型和处理器
        try:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            print("✅ 模型加载完成！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请确保:")
            print("1. 已登录Hugging Face: huggingface-cli login")
            print("2. 有访问google/gemma-3-27b-it的权限")
            print("3. GPU内存充足（建议20GB+）")
            raise
    
    def prepare_image(self, image_path):
        """
        准备图像输入
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None
    
    def generate_response(self, text, image_path):
        """
        生成模型响应 - 使用本地GPU
        """
        try:
            # 准备图像
            image = self.prepare_image(image_path)
            if image is None:
                return "Error: 无法加载图像"
            
            # 构建消息格式
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant that answers questions about images. Answer with A or B only, without any reasoning."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ]
            
            # 应用聊天模板
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # 生成响应
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=512, 
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                generation = generation[0][input_len:]
            
            # 解码响应
            response = self.processor.decode(generation, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return "Error: 生成失败"
    
    def process_vision_info(self, messages):
        """
        处理视觉信息（模仿qwen2的process_vision_info）
        """
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if message["role"] == "user" and "content" in message:
                for content in message["content"]:
                    if content["type"] == "image":
                        image_path = content["image"]
                        if image_path.startswith("file://"):
                            image_path = image_path[7:]  # 移除 "file://" 前缀
                        image = self.prepare_image(image_path)
                        image_inputs.append(image)
                    elif content["type"] == "video":
                        # 处理视频（如果需要）
                        pass
        
        return image_inputs, video_inputs
    
    def prepare_inputs(self, text, image_inputs=None, video_inputs=None):
        """
        准备模型输入并返回input_ids和offset（严格按照qwen2的方法）
        """
        if image_inputs:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs if video_inputs else [],
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
            )
        
        inputs = inputs.to(self.device)
        
        input_ids = inputs.input_ids[0]
        offset = len(input_ids)  # 这是生成开始的位置
        
        return inputs, input_ids, offset
    
    def compute_choice_probabilities(self, text, image_inputs=None, video_inputs=None):
        """
        计算选项A和B的概率（通过softmax，严格按照qwen2的方法）
        """
        try:
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
                    
        except Exception as e:
            print(f"基于logits计算概率时出错: {e}")
            return {"A": 0.5, "B": 0.5}
    
    def process_metacognition_question(self, stage1_question, stage2_question):
        """
        处理元认知问题（两阶段）
        """
        print(f"处理题目: {stage1_question['original_qid']}")
        
        # 构建消息格式（模仿qwen2）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{stage1_question['image_path']}"},
                    {"type": "text", "text": stage1_question["prompt"]},
                ],
            }
        ]
        
        # Stage 1: 回答问题
        print("  Stage 1: 回答问题...")
        start_time = time.time()
        
        stage1_response = self.generate_response(
            stage1_question['prompt'], 
            stage1_question['image_path']
        )
        
        stage1_latency = (time.time() - start_time) * 1000
        
        # 提取选择
        choice_match = re.search(r'\b([AB])\b', stage1_response.upper())
        choice = choice_match.group(1) if choice_match else "A"
        
        # 计算概率（使用qwen2的方法）
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        stage1_probs = self.compute_choice_probabilities(text, image_inputs, video_inputs)
        
        # Stage 2: 评估置信度
        print("  Stage 2: 评估置信度...")
        start_time = time.time()
        
        # 构建置信度评估提示
        confidence_prompt = f"Based on your previous answer '{choice}', how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' only, where N is your confidence level."
        
        stage2_response = self.generate_response(
            confidence_prompt,
            stage1_question['image_path']  # 使用相同的图像
        )
        
        stage2_latency = (time.time() - start_time) * 1000
        
        # 提取置信度
        confidence_match = re.search(r'confidence:\s*(\d+)', stage2_response.lower())
        confidence = int(confidence_match.group(1)) if confidence_match else 3
        
        # Stage 2不需要计算A/B概率，因为它是置信度问题
        stage2_probs = {"A": 0.5, "B": 0.5}  # 置信度问题没有A/B选择
        
        # 构建结果
        stage1_result = {
            "qid": stage1_question['qid'],
            "task": stage1_question.get('task', 'Unknown'),
            "choice": choice,
            "confidence": 0,  # Stage 1没有置信度
            "raw_text": stage1_response,
            "latency_ms": stage1_latency,
            "stage": 1,
            "original_qid": stage1_question['original_qid'],
            "probabilities": stage1_probs,
            "prompt": stage1_question['prompt'],
            "image_path": stage1_question['image_path'],
            "timestamp": time.time()
        }
        
        stage2_result = {
            "qid": stage2_question['qid'],
            "task": stage2_question.get('task', 'Unknown'),
            "choice": f"confidence: {confidence}",
            "confidence": confidence,
            "raw_text": stage2_response,
            "latency_ms": stage2_latency,
            "stage": 2,
            "original_qid": stage2_question['original_qid'],
            "probabilities": stage2_probs,
            "prompt": confidence_prompt,
            "image_path": stage1_question['image_path'],  # 使用相同的图像
            "timestamp": time.time()
        }
        
        return stage1_result, stage2_result
    
    def run_experiment(self, questions_file, output_file, max_questions=None):
        """
        运行元认知实验
        """
        # 加载问题
        questions = []
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line.strip()))
        
        print(f"加载了 {len(questions)} 个问题")
        
        # 按original_qid分组
        questions_by_original = {}
        for q in questions:
            original_qid = q['original_qid']
            stage = q['stage']
            
            if original_qid not in questions_by_original:
                questions_by_original[original_qid] = {}
            
            if stage == 1:
                questions_by_original[original_qid]['stage1'] = q
            elif stage == 2:
                questions_by_original[original_qid]['stage2'] = q
        
        # 只处理stage1问题，然后找到对应的stage2
        stage1_questions = [q for q in questions if q['stage'] == 1]
        
        if max_questions:
            stage1_questions = stage1_questions[:max_questions]
        
        print(f"将处理 {len(stage1_questions)} 个题目")
        
        results = []
        total_questions = len(stage1_questions)
        
        # 打开输出文件进行实时写入
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, stage1_question in enumerate(stage1_questions):
                print(f"处理进度: {i+1}/{total_questions}")
                
                try:
                    # 找到对应的stage2问题
                    original_qid = stage1_question['original_qid']
                    stage2_qid = original_qid + "_stage2"
                    stage2_question = next((q for q in questions if q['qid'] == stage2_qid), None)
                    
                    if not stage2_question:
                        print(f"跳过题目 {original_qid}: 找不到对应的stage2问题")
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
    parser = argparse.ArgumentParser(description="Gemma 3 27B IT 元认知推理实验")
    parser.add_argument("--questions", type=str, required=True, help="题目JSONL文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSONL文件路径")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it", help="模型名称")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--max_questions", type=int, default=None, help="最大处理题目数（用于测试）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    inference = Gemma3MetacognitionInference(args.model, args.device)
    
    # 运行实验
    print("开始运行元认知实验...")
    results = inference.run_experiment(args.questions, args.output, args.max_questions)
    
    print(f"实验完成！结果已保存到: {args.output}")
    print(f"总共处理了 {len(results)} 个结果")

if __name__ == "__main__":
    main()
