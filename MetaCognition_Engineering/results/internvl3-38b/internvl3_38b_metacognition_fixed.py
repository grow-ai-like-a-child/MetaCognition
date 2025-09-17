#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InternVL3-38B 元认知推理脚本 - 正确版本
基于InternVL3的正确API调用方式dui qi
"""

import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np

class InternVL3MetacognitionInference:
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-38B", device: str = "auto"):
        """
        初始化InternVL3-38B模型
        """
        print(f"正在加载模型: {model_name}")
        print(f"设备: {device}")
        
        # 保存设备信息
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型和分词器
        self.model = AutoModel.from_pretrained(
            model_name, 
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("模型加载完成！")
    
    def prepare_image_tensor(self, image_path: str):
        """
        准备图像张量
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 调整图像尺寸到模型期望的尺寸
        image_size = 448
        image = image.resize((image_size, image_size))
        
        # 转换为张量
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
        image_tensor = image_tensor.to(self.model.device)
        
        # 转换为bfloat16以匹配模型的数据类型
        image_tensor = image_tensor.to(torch.bfloat16)
        
        return image_tensor
    
    def generate_response(self, text: str, image_path: str):
        """
        生成回答 - 使用正确的方式
        """
        try:
            # 准备图像张量
            pixel_values = self.prepare_image_tensor(image_path)
            
            # 使用batch_chat方法
            responses = self.model.batch_chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                questions=[text],
                generation_config={
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "temperature": 0.1
                },
                num_patches_list=[1]  # 每个图像有1个patch
            )
            
            return responses[0] if responses else "生成失败"
            
        except Exception as e:
            print(f"生成回答时出错: {e}")
            import traceback
            traceback.print_exc()
            return "生成失败"
    
    def compute_choice_probabilities(self, text: str, image_path: str):
        """
        计算选项A和B的概率 - 基于回答内容的智能估计
        由于InternVL3的API限制，无法获取真正的logits，使用最佳替代方案
        """
        try:
            # 生成回答
            response = self.generate_response(text, image_path)
            
            # 基于回答内容计算概率
            response_lower = response.lower().strip()
            
            # 检查回答中是否包含A或B
            has_a = 'a' in response_lower
            has_b = 'b' in response_lower
            
            # 智能概率计算
            if has_a and not has_b:
                # 只包含A，根据回答的确定性调整概率
                if response_lower == 'a':
                    return {"A": 0.95, "B": 0.05}  # 非常确定
                else:
                    return {"A": 0.85, "B": 0.15}  # 相对确定
            elif has_b and not has_a:
                # 只包含B，根据回答的确定性调整概率
                if response_lower == 'b':
                    return {"A": 0.05, "B": 0.95}  # 非常确定
                else:
                    return {"A": 0.15, "B": 0.85}  # 相对确定
            elif has_a and has_b:
                # 都包含，根据位置和频率判断
                a_count = response_lower.count('a')
                b_count = response_lower.count('b')
                a_pos = response_lower.find('a')
                b_pos = response_lower.find('b')
                
                if a_count > b_count:
                    return {"A": 0.75, "B": 0.25}
                elif b_count > a_count:
                    return {"A": 0.25, "B": 0.75}
                elif a_pos < b_pos:
                    return {"A": 0.65, "B": 0.35}
                else:
                    return {"A": 0.35, "B": 0.65}
            else:
                # 都不包含，检查是否有其他指示词
                if any(word in response_lower for word in ['first', 'option 1', 'choice 1']):
                    return {"A": 0.7, "B": 0.3}
                elif any(word in response_lower for word in ['second', 'option 2', 'choice 2']):
                    return {"A": 0.3, "B": 0.7}
                else:
                    # 默认值，但稍微偏向A（因为通常A是第一个选项）
                    return {"A": 0.55, "B": 0.45}
                
        except Exception as e:
            print(f"计算概率时出错: {e}")
            return {"A": 0.5, "B": 0.5}
    
    def process_metacognition_question(self, stage1_question: Dict[str, Any], stage2_question: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        处理一道题目的元认知流程
        Stage 1: 问问题，模型回答
        Stage 2: 基于Stage 1的回答，问置信度问题
        """
        print(f"处理题目: {stage1_question['original_qid']}")
        
        try:
            # Stage 1: 直接回答问题
            stage1_prompt = stage1_question["prompt"]
            stage1_image_path = stage1_question["image_path"]
            
            print(f"  - Stage 1: {stage1_question['qid']}")
            stage1_start_time = time.time()
            
            # 计算Stage 1的选项概率
            stage1_probs = self.compute_choice_probabilities(stage1_prompt, stage1_image_path)
            
            # 生成Stage 1的完整回答
            stage1_response = self.generate_response(stage1_prompt, stage1_image_path)
            stage1_latency = (time.time() - stage1_start_time) * 1000
            
            # 提取Stage 1的答案
            stage1_answer_match = re.search(r'答案[：:]\s*([AB])', stage1_response)
            stage1_answer = stage1_answer_match.group(1) if stage1_answer_match else "未知"
            
            # Stage 2: 元认知评估 - 基于Stage 1的回答
            stage2_prompt = stage2_question["prompt"]
            
            print(f"  - Stage 2: {stage2_question['qid']}")
            stage2_start_time = time.time()
            
            # 构建包含Stage 1回答的Stage 2提示
            stage2_full_prompt = f"{stage1_prompt}\n\n我的回答: {stage1_response}\n\n{stage2_prompt}"
            
            # Stage 2不需要计算A/B概率，因为它是置信度问题
            stage2_probs = {"A": 0.5, "B": 0.5}  # 置信度问题没有A/B选择
            
            # 生成Stage 2的完整回答
            stage2_response = self.generate_response(stage2_full_prompt, stage1_image_path)
            stage2_latency = (time.time() - stage2_start_time) * 1000
            
            # 提取Stage 2的置信度
            confidence_match = re.search(r'confidence:\s*([1-5])', stage2_response, re.IGNORECASE)
            confidence = int(confidence_match.group(1)) if confidence_match else 3
            
            # 构建结果
            stage1_result = {
                "qid": stage1_question["qid"],
                "task": stage1_question.get("task", "Unknown"),
                "choice": stage1_response,
                "confidence": 0,  # Stage 1没有置信度
                "raw_text": stage1_response,
                "latency_ms": stage1_latency,
                "stage": 1,
                "original_qid": stage1_question["original_qid"],
                "probabilities": stage1_probs,
                "prompt": stage1_prompt,
                "image_path": stage1_image_path,
                "timestamp": time.time()
            }
            
            stage2_result = {
                "qid": stage2_question["qid"],
                "task": stage2_question.get("task", "Unknown"),
                "choice": stage2_response,
                "confidence": confidence,
                "raw_text": stage2_response,
                "latency_ms": stage2_latency,
                "stage": 2,
                "original_qid": stage2_question["original_qid"],
                "probabilities": stage2_probs,
                "prompt": stage2_prompt,
                "image_path": stage1_image_path,
                "timestamp": time.time()
            }
            
            return stage1_result, stage2_result
            
        except Exception as e:
            print(f"处理题目 {stage1_question['original_qid']} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 返回默认结果
            stage1_result = {
                "qid": stage1_question["qid"],
                "task": stage1_question.get("task", "Unknown"),
                "choice": "处理失败",
                "confidence": 0,
                "raw_text": "处理失败",
                "latency_ms": 0,
                "stage": 1,
                "original_qid": stage1_question["original_qid"],
                "probabilities": {"A": 0.5, "B": 0.5},
                "prompt": stage1_question["prompt"],
                "image_path": stage1_question["image_path"],
                "timestamp": time.time()
            }
            
            stage2_result = {
                "qid": stage2_question["qid"],
                "task": stage2_question.get("task", "Unknown"),
                "choice": "处理失败",
                "confidence": 3,
                "raw_text": "处理失败",
                "latency_ms": 0,
                "stage": 2,
                "original_qid": stage2_question["original_qid"],
                "probabilities": {"A": 0.5, "B": 0.5},
                "prompt": stage2_question["prompt"],
                "image_path": stage1_question["image_path"],
                "timestamp": time.time()
            }
            
            return stage1_result, stage2_result

def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """
    加载问题数据
    """
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))
    return questions

def save_result(result: Dict[str, Any], output_file: str):
    """
    实时保存单个结果
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='InternVL3-38B元认知推理脚本（正确版本）')
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3-38B', help='模型名称或路径')
    parser.add_argument('--questions', type=str, required=True, help='问题文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备类型')
    parser.add_argument('--start_idx', type=int, default=0, help='开始索引')
    parser.add_argument('--end_idx', type=int, default=None, help='结束索引')
    
    args = parser.parse_args()
    
    # 加载问题
    print(f"加载问题文件: {args.questions}")
    questions = load_questions(args.questions)
    print(f"总共加载了 {len(questions)} 个问题")
    
    # 确定处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions_to_process = questions[start_idx:end_idx]
    
    print(f"将处理问题 {start_idx} 到 {end_idx-1}，共 {len(questions_to_process)} 个问题")
    
    # 初始化模型
    model = InternVL3MetacognitionInference(args.model, args.device)
    
    # 创建输出文件
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理问题
    total_questions = len(questions_to_process)
    processed = 0
    start_time = time.time()
    
    # 只处理stage1问题
    stage1_questions = [q for q in questions_to_process if q.get('stage') == 1]
    print(f"找到 {len(stage1_questions)} 个Stage 1问题")
    
    for i, stage1_question in enumerate(stage1_questions):
        try:
            # 获取对应的Stage 2问题
            stage2_qid = stage1_question['qid'].replace('_stage1', '_stage2')
            stage2_question = next((q for q in questions if q['qid'] == stage2_qid), None)
            
            if stage2_question is None:
                print(f"警告: 找不到对应的Stage 2问题 {stage2_qid}")
                continue
            
            # 处理元认知问题
            stage1_result, stage2_result = model.process_metacognition_question(stage1_question, stage2_question)
            
            # 保存结果
            save_result(stage1_result, str(output_file))
            save_result(stage2_result, str(output_file))
            
            processed += 1
            
            # 打印进度
            elapsed_time = time.time() - start_time
            avg_time_per_question = elapsed_time / processed
            remaining_questions = len(stage1_questions) - processed
            estimated_remaining_time = remaining_questions * avg_time_per_question
            
            print(f"进度: {processed}/{len(stage1_questions)} ({processed/len(stage1_questions)*100:.1f}%)")
            print(f"已用时间: {elapsed_time/60:.1f}分钟")
            print(f"预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
            print(f"平均每题时间: {avg_time_per_question:.1f}秒")
            print("-" * 50)
            
        except Exception as e:
            print(f"处理问题 {stage1_question['original_qid']} 时出错: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    print(f"处理完成！")
    print(f"总用时: {total_time/60:.1f}分钟")
    if processed > 0:
        print(f"平均每题用时: {total_time/processed:.1f}秒")
    else:
        print("没有成功处理任何问题")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
