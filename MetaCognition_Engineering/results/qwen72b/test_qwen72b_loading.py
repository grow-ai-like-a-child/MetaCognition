#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Qwen2.5-VL-72B模型加载和基本功能
"""

import torch
import time
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("测试Qwen2.5-VL-72B模型加载")
    print("=" * 50)
    
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    device = "auto"
    
    print(f"模型名称: {model_name}")
    print(f"设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    print("\n开始加载模型...")
    start_time = time.time()
    
    try:
        # 加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        # 加载处理器
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
        
        load_time = time.time() - start_time
        print(f"模型加载成功！耗时: {load_time:.2f}秒")
        
        # 测试基本推理
        print("\n测试基本推理...")
        test_inference(model, processor)
        
        return True
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return False

def test_inference(model, processor):
    """测试基本推理功能"""
    try:
        # 创建测试消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, how are you?"}
                ],
            }
        ]
        
        # 处理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
        inputs = inputs.to(model.device)
        
        print(f"输入形状: {inputs.input_ids.shape}")
        print(f"设备: {inputs.input_ids.device}")
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0
            )
        
        # 解码输出
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        print(f"模型回复: {response}")
        print("基本推理测试成功！")
        
    except Exception as e:
        print(f"推理测试失败: {e}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")

def main():
    """主函数"""
    print("开始测试Qwen2.5-VL-72B模型...")
    
    success = test_model_loading()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！模型可以正常使用")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ 测试失败！请检查模型和依赖")
        print("=" * 50)

if __name__ == "__main__":
    main()
