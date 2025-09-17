#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比Gemma3-27B和InternVL3-38B的性能
"""

import json
from pathlib import Path

def main():
    """
    对比两个模型的性能数据
    """
    print("=" * 80)
    print("🔍 Gemma3-27B vs InternVL3-38B 性能对比")
    print("=" * 80)
    
    # Gemma3性能数据
    gemma3_file = Path("gemma3_comparison_with_ground_truth.json")
    if gemma3_file.exists():
        with open(gemma3_file, 'r', encoding='utf-8') as f:
            gemma3_data = json.load(f)
        
        gemma3_total = len(gemma3_data)
        gemma3_correct = sum(1 for item in gemma3_data if item['is_correct'])
        gemma3_accuracy = gemma3_correct / gemma3_total if gemma3_total > 0 else 0
        
        # 计算平均延迟
        stage1_latencies = [item['stage1_latency'] for item in gemma3_data if item['stage1_latency'] > 0]
        stage2_latencies = [item['stage2_latency'] for item in gemma3_data if item['stage2_latency'] > 0]
        gemma3_avg_latency = (sum(stage1_latencies) + sum(stage2_latencies)) / (len(stage1_latencies) + len(stage2_latencies)) if stage1_latencies and stage2_latencies else 0
        
        # 高置信度(5)统计
        high_conf_items = [item for item in gemma3_data if item['confidence'] == 5]
        gemma3_high_conf_correct = sum(1 for item in high_conf_items if item['is_correct'])
        gemma3_high_conf_accuracy = gemma3_high_conf_correct / len(high_conf_items) if high_conf_items else 0
        
        print(f"\n📊 Gemma3-27B 性能:")
        print(f"   ✅ 总体准确率: {gemma3_accuracy*100:.2f}% ({gemma3_correct}/{gemma3_total})")
        print(f"   ⚡ 平均延迟: {gemma3_avg_latency:.1f}ms")
        print(f"   🎯 高置信度(5)准确率: {gemma3_high_conf_accuracy*100:.2f}% ({gemma3_high_conf_correct}/{len(high_conf_items)})")
        print(f"   📈 数据规模: {gemma3_total} 个题目")
    
    # InternVL3性能数据
    internvl3_file = Path("../internvl3-38b/internvl3_comparison_with_ground_truth.json")
    if internvl3_file.exists():
        with open(internvl3_file, 'r', encoding='utf-8') as f:
            internvl3_data = json.load(f)
        
        internvl3_total = len(internvl3_data)
        internvl3_correct = sum(1 for item in internvl3_data if item['is_correct'])
        internvl3_accuracy = internvl3_correct / internvl3_total if internvl3_total > 0 else 0
        
        # 计算平均延迟
        internvl3_latencies = [item['latency_ms'] for item in internvl3_data if item['latency_ms'] > 0]
        internvl3_avg_latency = sum(internvl3_latencies) / len(internvl3_latencies) if internvl3_latencies else 0
        
        # 高置信度(5)统计
        high_conf_items = [item for item in internvl3_data if item['confidence'] == 5]
        internvl3_high_conf_correct = sum(1 for item in high_conf_items if item['is_correct'])
        internvl3_high_conf_accuracy = internvl3_high_conf_correct / len(high_conf_items) if high_conf_items else 0
        
        print(f"\n📊 InternVL3-38B 性能:")
        print(f"   ✅ 总体准确率: {internvl3_accuracy*100:.2f}% ({internvl3_correct}/{internvl3_total})")
        print(f"   ⚡ 平均延迟: {internvl3_avg_latency:.1f}ms")
        print(f"   🎯 高置信度(5)准确率: {internvl3_high_conf_accuracy*100:.2f}% ({internvl3_high_conf_correct}/{len(high_conf_items)})")
        print(f"   📈 数据规模: {internvl3_total} 个题目")
        
        # 性能对比
        if gemma3_file.exists():
            print(f"\n🔥 直接对比:")
            print("-" * 50)
            
            accuracy_diff = gemma3_accuracy - internvl3_accuracy
            latency_diff = gemma3_avg_latency - internvl3_avg_latency
            
            print(f"   准确率差异: {accuracy_diff*100:+.2f}% (Gemma3 {'更好' if accuracy_diff > 0 else '更差' if accuracy_diff < 0 else '相同'})")
            print(f"   延迟差异: {latency_diff:+.1f}ms (Gemma3 {'更慢' if latency_diff > 0 else '更快' if latency_diff < 0 else '相同'})")
            
            # 效率比较
            if latency_diff < 0:
                speedup = internvl3_avg_latency / gemma3_avg_latency if gemma3_avg_latency > 0 else 1
                print(f"   速度提升: {speedup:.1f}x 更快")
            
            print(f"\n🏆 优势总结:")
            print("-" * 50)
            
            if accuracy_diff > 0.01:  # 1%以上差异认为显著
                print(f"   ✅ Gemma3准确率明显更高 (+{accuracy_diff*100:.2f}%)")
            elif accuracy_diff < -0.01:
                print(f"   ❌ InternVL3准确率明显更高 (+{-accuracy_diff*100:.2f}%)")
            else:
                print(f"   ⚖️ 两模型准确率相近")
            
            if latency_diff < -50:  # 50ms以上差异认为显著
                print(f"   ⚡ Gemma3速度明显更快 ({-latency_diff:.0f}ms)")
            elif latency_diff > 50:
                print(f"   🐌 InternVL3速度明显更快 ({latency_diff:.0f}ms)")
            else:
                print(f"   ⚖️ 两模型速度相近")
            
            # 数据质量对比
            print(f"\n📊 数据质量对比:")
            print("-" * 50)
            print(f"   Gemma3数据规模: {gemma3_total} 题")
            print(f"   InternVL3数据规模: {internvl3_total} 题")
            
            if gemma3_total > internvl3_total:
                print(f"   📈 Gemma3数据更完整 (+{gemma3_total - internvl3_total} 题)")
            elif internvl3_total > gemma3_total:
                print(f"   📈 InternVL3数据更完整 (+{internvl3_total - gemma3_total} 题)")
            else:
                print(f"   ⚖️ 数据规模相同")
    
    else:
        print(f"\n❌ 未找到InternVL3性能数据文件: {internvl3_file}")
    
    if not gemma3_file.exists():
        print(f"\n❌ 未找到Gemma3性能数据文件: {gemma3_file}")
        print("请先运行 analyze_gemma3_performance.py 生成分析数据")

if __name__ == "__main__":
    main()
