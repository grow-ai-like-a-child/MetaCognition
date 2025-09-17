#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理InternVL3文件夹，保留重要文件，删除冗余文件
"""

import os
from pathlib import Path

def analyze_files():
    """
    分析文件重要性并建议保留/删除
    """
    print("=" * 80)
    print("InternVL3文件夹文件分析")
    print("=" * 80)
    
    files = [
        # 核心实验文件
        ("internvl3_38b_metacognition_fixed.py", "核心实验脚本", "KEEP", "主要的元认知实验代码"),
        ("internvl3_complete_results.jsonl", "完整实验结果", "KEEP", "最重要的实验结果数据"),
        ("internvl3_experiment.log", "实验日志", "KEEP", "实验运行日志，用于调试"),
        
        # 分析脚本
        ("analyze_confidence_by_task.py", "置信度分析脚本", "KEEP", "每类任务置信度分布分析"),
        ("analyze_internvl3_detailed.py", "详细性能分析脚本", "KEEP", "详细的性能分析"),
        ("compare_with_qwen72b.py", "模型对比脚本", "KEEP", "与Qwen72B的对比分析"),
        ("create_correct_comparison.py", "正确匹配脚本", "KEEP", "基于ground_truth的正确匹配"),
        ("visualize_confidence_distribution.py", "可视化脚本", "KEEP", "置信度分布可视化"),
        
        # 结果数据文件
        ("internvl3_correct_comparison.json", "正确匹配结果", "KEEP", "最重要的分析结果"),
        ("internvl3_confidence_by_task_analysis.json", "置信度分析结果", "KEEP", "置信度分析数据"),
        ("internvl3_vs_qwen72b_comparison.json", "模型对比结果", "KEEP", "模型对比数据"),
        
        # 可视化文件
        ("internvl3_confidence_distribution.png", "可视化图表PNG", "KEEP", "置信度分布图表"),
        ("internvl3_confidence_distribution.pdf", "可视化图表PDF", "KEEP", "置信度分布图表PDF版本"),
        
        # 冗余文件
        ("analyze_internvl3_performance.py", "性能分析脚本", "DELETE", "被analyze_internvl3_detailed.py替代"),
        ("analyze_performance.py", "旧性能分析脚本", "DELETE", "被analyze_internvl3_detailed.py替代"),
        ("compare_with_ground_truth.py", "旧对比脚本", "DELETE", "被create_correct_comparison.py替代"),
        ("create_simple_comparison.py", "旧简化比较脚本", "DELETE", "被create_correct_comparison.py替代"),
        ("create_internvl3_simple_comparison.py", "简化比较脚本", "DELETE", "被create_correct_comparison.py替代"),
        
        # 冗余数据文件
        ("internvl3_comparison_with_ground_truth.json", "旧对比结果", "DELETE", "被internvl3_correct_comparison.json替代"),
        ("internvl3_simple_comparison.json", "简化比较结果", "DELETE", "被internvl3_correct_comparison.json替代"),
    ]
    
    print(f"\n📋 文件分析结果:")
    print("-" * 80)
    print(f"{'文件名':<40} {'类型':<15} {'建议':<8} {'说明'}")
    print("-" * 80)
    
    keep_count = 0
    delete_count = 0
    
    for filename, file_type, recommendation, description in files:
        status = "✅ 保留" if recommendation == "KEEP" else "❌ 删除"
        print(f"{filename:<40} {file_type:<15} {status:<8} {description}")
        
        if recommendation == "KEEP":
            keep_count += 1
        else:
            delete_count += 1
    
    print("-" * 80)
    print(f"总计: {len(files)} 个文件")
    print(f"建议保留: {keep_count} 个文件")
    print(f"建议删除: {delete_count} 个文件")
    
    return files

def cleanup_files():
    """
    执行文件清理
    """
    files_to_delete = [
        "analyze_internvl3_performance.py",
        "analyze_performance.py", 
        "compare_with_ground_truth.py",
        "create_simple_comparison.py",
        "create_internvl3_simple_comparison.py",
        "internvl3_comparison_with_ground_truth.json",
        "internvl3_simple_comparison.json"
    ]
    
    print(f"\n🧹 开始清理文件...")
    print("-" * 50)
    
    deleted_count = 0
    for filename in files_to_delete:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"✅ 已删除: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败: {filename} - {e}")
        else:
            print(f"⚠️  文件不存在: {filename}")
    
    print(f"\n清理完成: 删除了 {deleted_count} 个文件")

def show_remaining_files():
    """
    显示保留的文件
    """
    print(f"\n📁 保留的核心文件:")
    print("-" * 50)
    
    core_files = [
        ("internvl3_38b_metacognition_fixed.py", "核心实验脚本"),
        ("internvl3_complete_results.jsonl", "完整实验结果"),
        ("internvl3_correct_comparison.json", "正确匹配结果"),
        ("analyze_confidence_by_task.py", "置信度分析脚本"),
        ("analyze_internvl3_detailed.py", "详细性能分析脚本"),
        ("compare_with_qwen72b.py", "模型对比脚本"),
        ("create_correct_comparison.py", "正确匹配脚本"),
        ("visualize_confidence_distribution.py", "可视化脚本"),
        ("internvl3_confidence_by_task_analysis.json", "置信度分析结果"),
        ("internvl3_vs_qwen72b_comparison.json", "模型对比结果"),
        ("internvl3_confidence_distribution.png", "可视化图表PNG"),
        ("internvl3_confidence_distribution.pdf", "可视化图表PDF"),
        ("internvl3_experiment.log", "实验日志")
    ]
    
    for filename, description in core_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_str = format_file_size(size)
            print(f"✅ {filename:<40} {size_str:>10} - {description}")
        else:
            print(f"❌ {filename:<40} {'缺失':>10} - {description}")

def format_file_size(size_bytes):
    """
    格式化文件大小
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

if __name__ == "__main__":
    # 分析文件
    files = analyze_files()
    
    # 询问是否执行清理
    print(f"\n❓ 是否执行文件清理？(y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes', '是']:
        cleanup_files()
        show_remaining_files()
    else:
        print("取消清理操作")
        show_remaining_files()
