#!/usr/bin/env python3
"""
Ovis2-34B 元认知实验代码
基于 AIDC-AI/Ovis2-34B 模型实现两阶段元认知任务

Ovis2-34B 特点:
- ViT: aimv2-1B-patch14-448  
- LLM: Qwen2.5-32B-Instruct
- 增强的推理能力和多模态处理
- 支持视频和多图像输入
"""

import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ovis2_34b_experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class Ovis2MetacognitionProcessor:
    """Ovis2-34B 元认知处理器"""
    
    def __init__(self, model_path: str = "AIDC-AI/Ovis2-34B", device: str = "auto"):
        """
        初始化 Ovis2-34B 模型
        
        Args:
            model_path: 模型路径
            device: 设备配置
        """
        self.model_path = model_path
        self.device = device
        
        logging.info(f"正在加载 Ovis2-34B 模型: {model_path}")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=32768,
            trust_remote_code=True
        ).cuda()
        
        # 获取tokenizer
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        
        logging.info("Ovis2-34B 模型加载完成!")
        
        # 选项映射
        self.choice_options = ['A', 'B']
        
    def generate_response(self, text: str, image_path: str, max_new_tokens: int = 512) -> str:
        """
        生成模型响应
        
        Args:
            text: 输入文本
            image_path: 图像路径
            max_new_tokens: 最大新token数
            
        Returns:
            生成的响应文本
        """
        try:
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
            
            # 生成响应
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                
                output_ids = self.model.generate(
                    input_ids, 
                    pixel_values=pixel_values, 
                    attention_mask=attention_mask, 
                    **gen_kwargs
                )[0]
                
                output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
                
                # 提取响应部分（去除输入prompt）
                if query in output:
                    response = output.split(query)[-1].strip()
                else:
                    response = output.strip()
                    
                return response
                
        except Exception as e:
            logging.error(f"生成响应时出错: {e}")
            return f"Error: {str(e)}"
    
    def compute_choice_probabilities(self, text: str, image_path: str) -> Dict[str, float]:
        """
        计算选择概率 (基于logits的精确计算)
        
        Args:
            text: 输入文本
            image_path: 图像路径
            
        Returns:
            选择概率字典 {'A': prob_a, 'B': prob_b}
        """
        try:
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
            
            # 获取选择token的ID
            choice_tokens = {}
            for choice in self.choice_options:
                tokens = self.text_tokenizer.encode(choice, add_special_tokens=False)
                if len(tokens) == 1:
                    choice_tokens[choice] = tokens[0]
                else:
                    logging.warning(f"选择 {choice} 对应多个token: {tokens}")
                    choice_tokens[choice] = tokens[0]  # 使用第一个token
            
            with torch.no_grad():
                # 前向传播获取logits
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=None,
                    return_dict=True
                )
                
                # 获取最后一个位置的logits
                last_token_logits = outputs.logits[0, -1, :]
                
                # 计算选择概率
                choice_logits = torch.tensor([
                    last_token_logits[choice_tokens[choice]].item()
                    for choice in self.choice_options
                ])
                
                choice_probs = torch.softmax(choice_logits, dim=-1)
                
                return {
                    choice: prob.item() 
                    for choice, prob in zip(self.choice_options, choice_probs)
                }
                
        except Exception as e:
            logging.error(f"计算选择概率时出错: {e}")
            # 返回均匀分布作为fallback
            return {choice: 1.0/len(self.choice_options) for choice in self.choice_options}
    
    def extract_choice_from_response(self, response: str) -> str:
        """从响应中提取选择"""
        response = response.strip().upper()
        
        # 直接匹配 A 或 B
        if response in ['A', 'B']:
            return response
            
        # 匹配包含选择的模式
        patterns = [
            r'答案是\s*([AB])',
            r'选择\s*([AB])',
            r'答案：\s*([AB])',
            r'选择：\s*([AB])',
            r'\b([AB])\b',
            r'([AB])\s*选项',
            r'([AB])\s*是',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # 如果找不到明确的选择，返回第一个出现的A或B
        for char in response:
            if char in ['A', 'B']:
                return char
                
        return 'A'  # 默认返回A
    
    def extract_confidence_from_response(self, response: str) -> float:
        """从响应中提取置信度分数"""
        # 匹配不同的置信度表达模式
        patterns = [
            r'置信度[：:]\s*(\d+(?:\.\d+)?)',
            r'confidence[：:]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)/10',
            r'(\d+(?:\.\d+)?)\s*分',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*10',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    score = float(matches[-1])  # 取最后一个匹配
                    
                    # 规范化到0-1范围
                    if score > 10:
                        score = score / 100  # 百分比转换
                    elif score > 1:
                        score = score / 10   # 10分制转换
                        
                    return min(max(score, 0.0), 1.0)
                    
                except ValueError:
                    continue
        
        return 0.5  # 默认中等置信度
    
    def process_metacognition_question(self, qid: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个元认知问题
        
        Args:
            qid: 问题ID
            question_data: 问题数据
            
        Returns:
            处理结果
        """
        try:
            stage = question_data.get('stage', 1)
            
            if stage == 1:
                # Stage 1: 回答原始问题
                logging.info(f"  Stage 1: 回答问题...")
                
                prompt = question_data['prompt']
                image_path = question_data['image_path']
                
                # 生成回答
                stage1_response = self.generate_response(prompt, image_path)
                
                # 提取选择
                stage1_choice = self.extract_choice_from_response(stage1_response)
                
                # 计算选择概率
                stage1_probs = self.compute_choice_probabilities(prompt, image_path)
                
                return {
                    'qid': qid,
                    'stage': 1,
                    'response': stage1_response,
                    'choice': stage1_choice,
                    'choice_probabilities': stage1_probs,
                    'task': question_data.get('task', ''),
                    'original_qid': question_data.get('original_qid', qid),
                    'timestamp': question_data.get('timestamp', ''),
                    'model': 'ovis2-34b'
                }
                
            else:  # stage == 2
                # Stage 2: 评估置信度
                logging.info(f"  Stage 2: 评估置信度...")
                
                prompt = question_data['prompt']
                image_path = question_data['image_path']
                
                # 生成置信度评估
                stage2_response = self.generate_response(prompt, image_path)
                
                # 提取置信度分数
                confidence_score = self.extract_confidence_from_response(stage2_response)
                
                return {
                    'qid': qid,
                    'stage': 2,
                    'response': stage2_response,
                    'confidence_score': confidence_score,
                    'task': question_data.get('task', ''),
                    'original_qid': question_data.get('original_qid', qid),
                    'timestamp': question_data.get('timestamp', ''),
                    'model': 'ovis2-34b'
                }
                
        except Exception as e:
            logging.error(f"处理问题 {qid} 时出错: {e}")
            return {
                'qid': qid,
                'stage': question_data.get('stage', 1),
                'error': str(e),
                'model': 'ovis2-34b'
            }

def main():
    parser = argparse.ArgumentParser(description='Ovis2-34B 元认知实验')
    parser.add_argument('--questions', type=str, required=True, help='问题文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--model', type=str, default='AIDC-AI/Ovis2-34B', help='模型路径')
    parser.add_argument('--device', type=str, default='auto', help='设备配置')
    
    args = parser.parse_args()
    
    # 初始化处理器
    processor = Ovis2MetacognitionProcessor(
        model_path=args.model,
        device=args.device
    )
    
    # 加载问题
    questions = []
    with open(args.questions, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    logging.info(f"加载了 {len(questions)} 个问题")
    
    # 处理问题
    results = []
    
    for i, question in enumerate(tqdm(questions, desc="处理问题")):
        qid = question.get('qid', f'question_{i}')
        
        print(f"处理进度: {i+1}/{len(questions)}")
        print(f"处理题目: {qid}")
        
        result = processor.process_metacognition_question(qid, question)
        results.append(result)
        
        print(f"已保存题目 {qid} 的结果")
        
        # 定期保存结果
        if (i + 1) % 100 == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            logging.info(f"已保存前 {i+1} 个结果")
    
    # 保存最终结果
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logging.info(f"实验完成！结果已保存到: {args.output}")
    logging.info(f"总共处理了 {len(results)} 个结果")

if __name__ == "__main__":
    main()
