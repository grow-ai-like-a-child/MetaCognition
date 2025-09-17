#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化InternVL3-38B每类任务的置信度分布
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

def create_confidence_visualization():
    """
    创建置信度分布可视化
    """
    # 加载分析结果
    analysis_file = Path("internvl3_confidence_by_task_analysis.json")
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('InternVL3-38B 每类任务置信度分布分析', fontsize=16, fontweight='bold')
    
    # 1. 置信度分布饼图
    ax1 = axes[0, 0]
    tasks = list(analysis_data.keys())
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # 为每个任务创建置信度分布
    conf3_data = [analysis_data[task]['confidence_distribution'].get('3', 0) for task in tasks]
    conf4_data = [analysis_data[task]['confidence_distribution'].get('4', 0) for task in tasks]
    conf5_data = [analysis_data[task]['confidence_distribution'].get('5', 0) for task in tasks]
    
    x = np.arange(len(tasks))
    width = 0.25
    
    ax1.bar(x - width, conf3_data, width, label='置信度 3', color='#ff9999', alpha=0.8)
    ax1.bar(x, conf4_data, width, label='置信度 4', color='#66b3ff', alpha=0.8)
    ax1.bar(x + width, conf5_data, width, label='置信度 5', color='#99ff99', alpha=0.8)
    
    ax1.set_xlabel('任务类型')
    ax1.set_ylabel('题目数量')
    ax1.set_title('各任务置信度分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 置信度百分比分布
    ax2 = axes[0, 1]
    
    # 计算百分比
    conf3_pct = [conf3_data[i] / analysis_data[task]['total_questions'] * 100 for i, task in enumerate(tasks)]
    conf4_pct = [conf4_data[i] / analysis_data[task]['total_questions'] * 100 for i, task in enumerate(tasks)]
    conf5_pct = [conf5_data[i] / analysis_data[task]['total_questions'] * 100 for i, task in enumerate(tasks)]
    
    ax2.bar(x - width, conf3_pct, width, label='置信度 3', color='#ff9999', alpha=0.8)
    ax2.bar(x, conf4_pct, width, label='置信度 4', color='#66b3ff', alpha=0.8)
    ax2.bar(x + width, conf5_pct, width, label='置信度 5', color='#99ff99', alpha=0.8)
    
    ax2.set_xlabel('任务类型')
    ax2.set_ylabel('百分比 (%)')
    ax2.set_title('各任务置信度百分比分布')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 置信度与准确率关系
    ax3 = axes[1, 0]
    
    conf_levels = [3, 4, 5]
    for i, task in enumerate(tasks):
        accuracies = []
        for conf in conf_levels:
            if conf in analysis_data[task]['confidence_accuracy']:
                accuracies.append(analysis_data[task]['confidence_accuracy'][conf])
            else:
                accuracies.append(0)
        
        ax3.plot(conf_levels, accuracies, marker='o', linewidth=2, label=task, color=colors[i])
    
    # 添加理想校准线
    ideal_line = [conf/5.0 for conf in conf_levels]
    ax3.plot(conf_levels, ideal_line, 'k--', alpha=0.7, label='理想校准线')
    
    ax3.set_xlabel('置信度')
    ax3.set_ylabel('准确率')
    ax3.set_title('置信度与准确率关系')
    ax3.set_xticks(conf_levels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. 任务性能对比
    ax4 = axes[1, 1]
    
    task_names = tasks
    accuracies = [analysis_data[task]['accuracy'] * 100 for task in tasks]
    avg_confidences = [analysis_data[task]['average_confidence'] for task in tasks]
    
    # 创建双轴
    ax4_twin = ax4.twinx()
    
    bars = ax4.bar(task_names, accuracies, alpha=0.7, color=colors, label='准确率 (%)')
    line = ax4_twin.plot(task_names, avg_confidences, 'ro-', linewidth=2, markersize=8, label='平均置信度')
    
    ax4.set_xlabel('任务类型')
    ax4.set_ylabel('准确率 (%)', color='blue')
    ax4_twin.set_ylabel('平均置信度', color='red')
    ax4.set_title('任务性能与平均置信度对比')
    
    # 添加数值标签
    for i, (acc, conf) in enumerate(zip(accuracies, avg_confidences)):
        ax4.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom')
        ax4_twin.text(i, conf + 0.05, f'{conf:.2f}', ha='center', va='bottom', color='red')
    
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('internvl3_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('internvl3_confidence_distribution.pdf', bbox_inches='tight')
    
    print("📊 置信度分布可视化图表已保存:")
    print("   - internvl3_confidence_distribution.png")
    print("   - internvl3_confidence_distribution.pdf")
    
    # 显示图表
    plt.show()

def create_detailed_confidence_table():
    """
    创建详细的置信度分布表格
    """
    # 加载分析结果
    analysis_file = Path("internvl3_confidence_by_task_analysis.json")
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    print("\n" + "=" * 100)
    print("InternVL3-38B 每类任务置信度分布详细表格")
    print("=" * 100)
    
    for task in ['Color', 'Gabor', 'Grid']:
        if task not in analysis_data:
            continue
            
        data = analysis_data[task]
        print(f"\n📋 {task} 任务:")
        print("-" * 80)
        print(f"{'置信度':<8} {'题目数':<8} {'百分比':<8} {'正确数':<8} {'准确率':<10} {'校准误差':<10}")
        print("-" * 80)
        
        total = data['total_questions']
        for conf in [3, 4, 5]:
            count = data['confidence_distribution'].get(str(conf), 0)
            percentage = count / total * 100 if total > 0 else 0
            correct = int(count * data['confidence_accuracy'].get(str(conf), 0))
            accuracy = data['confidence_accuracy'].get(str(conf), 0)
            calibration_error = abs(accuracy - (conf / 5.0))
            
            print(f"{conf:<8} {count:<8} {percentage:<8.1f}% {correct:<8} {accuracy:<10.4f} {calibration_error:<10.4f}")
        
        print(f"\n   总题目数: {total}")
        print(f"   平均置信度: {data['average_confidence']:.2f}")
        print(f"   整体准确率: {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")

if __name__ == "__main__":
    try:
        create_confidence_visualization()
    except ImportError:
        print("matplotlib未安装，跳过可视化图表生成")
        print("请运行: pip install matplotlib")
    
    create_detailed_confidence_table()
