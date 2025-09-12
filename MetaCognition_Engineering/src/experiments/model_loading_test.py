#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同Qwen2.5-VL模型的加载情况
"""

import torch
import time
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

def test_model_loading(model_name: str, max_retries: int = 3):
    """测试模型加载"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    for attempt in range(max_retries):
        try:
            print(f"尝试 {attempt + 1}/{max_retries}...")
            
            # 检查GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            start_time = time.time()
            
            # 加载模型
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
            
            load_time = time.time() - start_time
            
            # 计算参数量
            param_count = sum(p.numel() for p in model.parameters())
            
            # 检查GPU内存使用
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_cache = torch.cuda.memory_reserved() / 1024**3
            else:
                gpu_memory = 0
                gpu_cache = 0
            
            print(f"✅ 模型加载成功!")
            print(f"   参数量: {param_count:,}")
            print(f"   加载时间: {load_time:.2f} 秒")
            print(f"   GPU内存使用: {gpu_memory:.2f} GB")
            print(f"   GPU内存缓存: {gpu_cache:.2f} GB")
            
            # 简单测试推理
            print("   测试推理...")
            test_text = "Hello, how are you?"
            inputs = processor(text=[test_text], return_tensors="pt")
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.0
                )
            
            print("   ✅ 推理测试成功!")
            
            # 清理内存
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            if attempt < max_retries - 1:
                print("   等待5秒后重试...")
                time.sleep(5)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"   经过{max_retries}次尝试后仍然失败")
                return False
    
    return False

def main():
    print("Qwen2.5-VL模型加载测试")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试不同大小的模型
    models_to_test = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-14B-Instruct", 
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        success = test_model_loading(model_name)
        results[model_name] = success
        
        # 在测试之间等待一下
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
    
    # 总结结果
    print(f"\n{'='*60}")
    print("测试结果总结")
    print(f"{'='*60}")
    
    for model_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{model_name}: {status}")
    
    # 推荐
    print(f"\n{'='*60}")
    print("推荐")
    print(f"{'='*60}")
    
    successful_models = [name for name, success in results.items() if success]
    if successful_models:
        print("可以成功加载的模型:")
        for model in successful_models:
            print(f"  - {model}")
        
        # 根据GPU内存推荐
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 70:
                print("\n推荐使用: Qwen/Qwen2.5-VL-72B-Instruct (最佳性能)")
            elif gpu_memory >= 30:
                print("\n推荐使用: Qwen/Qwen2.5-VL-32B-Instruct (平衡性能)")
            elif gpu_memory >= 15:
                print("\n推荐使用: Qwen/Qwen2.5-VL-14B-Instruct (良好性能)")
            else:
                print("\n推荐使用: Qwen/Qwen2.5-VL-7B-Instruct (基础性能)")
    else:
        print("❌ 没有模型可以成功加载，请检查环境配置")

if __name__ == "__main__":
    main()
