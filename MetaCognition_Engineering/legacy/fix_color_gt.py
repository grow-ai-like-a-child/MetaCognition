#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正Colortask中的错误标注
根据layout和brighter_side计算正确的gt标签

映射规则：
- LR布局: L->left, R->right
- UD布局: L->top, R->bottom  
- DRUL布局: L->bottom-left, R->top-right
- DLUR布局: L->top-left, R->bottom-right
"""

import json

def calculate_color_gt(layout, brighter_side):
    """
    根据布局和更亮的一边计算正确答案
    
    Args:
        layout: 布局类型 ("LR", "UD", "DRUL", "DLUR")
        brighter_side: 更亮的一边 ("L" 或 "R")
    
    Returns:
        正确答案字符串
    """
    if layout == "LR":
        return "left" if brighter_side == "L" else "right"
    elif layout == "UD":
        return "top" if brighter_side == "L" else "bottom"
    elif layout == "DRUL":
        return "bottom-left" if brighter_side == "L" else "top-right"
    elif layout == "DLUR":
        return "top-left" if brighter_side == "L" else "bottom-right"
    else:
        raise ValueError(f"Unknown layout: {layout}")

def fix_color_annotations(input_file, output_file):
    """
    修正JSON文件中的Color标注
    
    Args:
        input_file: 输入JSON文件路径
        output_file: Output JSON file path
    """
    print(f"正在读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total of {len(data)} 个题目")
    
    # 统计信息
    total_color = 0
    corrected_count = 0
    errors = []
    
    for i, item in enumerate(data):
        if item.get("task") == "Color":
            total_color += 1
            
            # 获取当前参数
            layout = item["params"]["layout"]
            brighter_side = item["params"]["brighter_side"]
            current_gt = item["derived"]["gt"]
            
            # 计算正确的标注
            correct_gt = calculate_color_gt(layout, brighter_side)
            
            # 检查是否需要修正
            if current_gt != correct_gt:
                corrected_count += 1
                errors.append({
                    "qid": item["qid"],
                    "layout": layout,
                    "brighter_side": brighter_side,
                    "old_gt": current_gt,
                    "new_gt": correct_gt,
                    "index": i
                })
                
                # 修正标注
                item["derived"]["gt"] = correct_gt
                
                print(f"修正 {item['qid']}: {layout} {brighter_side} {current_gt} -> {correct_gt}")
    
    print(f"\n统计结果:")
    print(f"总Color题目数: {total_color}")
    print(f"需要修正的题目数: {corrected_count}")
    print(f"修正比例: {corrected_count/total_color*100:.2f}%")
    
    if corrected_count > 0:
        print(f"\n前10个修正的例子:")
        for i, error in enumerate(errors[:10]):
            print(f"  {error['qid']}: {error['layout']} {error['brighter_side']} {error['old_gt']} -> {error['new_gt']}")
        
        # 保存修正后的文件
        print(f"\n正在保存修正后的文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("修正完成！")
    else:
        print("所有Color标注都是正确的！")

def analyze_color_distribution(input_file):
    """
    分析Colortask的布局和标注分布
    """
    print(f"\n正在分析Colortask分布...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    layout_stats = {}
    
    for item in data:
        if item.get("task") == "Color":
            layout = item["params"]["layout"]
            brighter_side = item["params"]["brighter_side"]
            gt = item["derived"]["gt"]
            
            if layout not in layout_stats:
                layout_stats[layout] = {"L": 0, "R": 0}
            
            layout_stats[layout][brighter_side] += 1
    
    print("布局分布统计:")
    for layout in sorted(layout_stats.keys()):
        stats = layout_stats[layout]
        print(f"  {layout}: L={stats['L']:3d}, R={stats['R']:3d}")

if __name__ == "__main__":
    input_file = "catalog_2688_newspec.json"
    output_file = "catalog_2688_newspec_color_fixed.json"
    
    # 先分析分布
    analyze_color_distribution(input_file)
    
    # 修正标注
    fix_color_annotations(input_file, output_file)
