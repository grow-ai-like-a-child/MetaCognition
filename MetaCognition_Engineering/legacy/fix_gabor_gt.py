#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正Gabor任务中的错误标注
根据theta_deg角度计算正确的gt标签

角度定义：
- 0° = 12点钟方向 = 竖直（Vertical）
- 90° = 6点钟方向 = 水平（Horizontal）
"""

import json
import math

def calculate_gabor_gt(theta_deg):
    """
    根据角度计算Gabor任务的正确答案
    
    Args:
        theta_deg: 条纹角度（0-360度）
    
    Returns:
        "vertical" 或 "horizontal"
    """
    # 标准化到0-180度范围
    theta_normalized = theta_deg % 180
    
    # 计算距离竖直和水平的最小距离
    # 0° = 竖直，90° = 水平
    dist_to_vertical = min(theta_normalized, 180 - theta_normalized)  # 距离0°
    dist_to_horizontal = abs(theta_normalized - 90)                   # 距离90°
    
    if dist_to_vertical < dist_to_horizontal:
        return "vertical"    # 更接近竖直
    else:
        return "horizontal"  # 更接近水平

def fix_gabor_annotations(input_file, output_file):
    """
    修正JSON文件中的Gabor标注
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共有 {len(data)} 个题目")
    
    # 统计信息
    total_gabor = 0
    corrected_count = 0
    errors = []
    
    for i, item in enumerate(data):
        if item.get("task") == "Gabor":
            total_gabor += 1
            
            # 获取当前角度和标注
            theta_deg = item["derived"]["theta_deg"]
            current_gt = item["derived"]["gt"]
            
            # 计算正确的标注
            correct_gt = calculate_gabor_gt(theta_deg)
            
            # 检查是否需要修正
            if current_gt != correct_gt:
                corrected_count += 1
                errors.append({
                    "qid": item["qid"],
                    "theta_deg": theta_deg,
                    "old_gt": current_gt,
                    "new_gt": correct_gt,
                    "index": i
                })
                
                # 修正标注
                item["derived"]["gt"] = correct_gt
                
                print(f"修正 {item['qid']}: {theta_deg}° {current_gt} -> {correct_gt}")
    
    print(f"\n统计结果:")
    print(f"总Gabor题目数: {total_gabor}")
    print(f"需要修正的题目数: {corrected_count}")
    print(f"修正比例: {corrected_count/total_gabor*100:.2f}%")
    
    if corrected_count > 0:
        print(f"\n前10个修正的例子:")
        for i, error in enumerate(errors[:10]):
            print(f"  {error['qid']}: {error['theta_deg']}° {error['old_gt']} -> {error['new_gt']}")
        
        # 保存修正后的文件
        print(f"\n正在保存修正后的文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("修正完成！")
    else:
        print("所有Gabor标注都是正确的！")

def analyze_angle_distribution(input_file):
    """
    分析角度分布和标注情况
    """
    print(f"\n正在分析角度分布...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    angle_stats = {}
    
    for item in data:
        if item.get("task") == "Gabor":
            theta_deg = item["derived"]["theta_deg"]
            gt = item["derived"]["gt"]
            
            # 标准化到0-180度
            theta_normalized = theta_deg % 180
            
            if theta_normalized not in angle_stats:
                angle_stats[theta_normalized] = {"vertical": 0, "horizontal": 0}
            
            angle_stats[theta_normalized][gt] += 1
    
    print("角度分布和标注统计:")
    for angle in sorted(angle_stats.keys()):
        stats = angle_stats[angle]
        correct_gt = calculate_gabor_gt(angle)
        print(f"  {angle:3.0f}°: V={stats['vertical']:3d}, H={stats['horizontal']:3d} (正确应该是: {correct_gt})")

if __name__ == "__main__":
    input_file = "catalog_2688_newspec.json"
    output_file = "catalog_2688_newspec_correctly_fixed.json"
    
    # 先分析角度分布
    analyze_angle_distribution(input_file)
    
    # 修正标注
    fix_gabor_annotations(input_file, output_file)
