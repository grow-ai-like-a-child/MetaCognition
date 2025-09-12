# Qwen2.5-VL-7B 元认知推理实验

## 文件说明

### 核心脚本
- `qwen2_vl_metacognition_multi_model.py` - 多模型推理脚本，支持不同Qwen模型
- `qwen2_vl_metacognition_with_logits.py` - 原始单模型推理脚本

### 数据文件
- `questions_two_stage_concise.jsonl` - 两阶段问题数据集（5376题）
- `answers_dict.json` - 正确答案字典
- `catalog_2688_newspec_correctly_fixed.json` - 题目目录文件
- `catalog_runner_two_stage.py` - 数据生成脚本

### 实验结果
- `qwen2.5-vl-7b_full_results.jsonl` - Qwen 7B模型完整推理结果
- `qwen7b_detailed_analysis.csv` - 详细分析CSV文件
- `qwen7b_detailed_analysis_corrected.csv` - 修正版详细分析CSV文件

### 对比分析
- `qwen7b_detailed_comparison.json` - 详细对比分析JSON
- `qwen7b_detailed_comparison_fixed.json` - 修正版详细对比分析JSON
- `qwen7b_simplified_comparison.json` - 简化版对比分析JSON（推荐使用）

### 分析脚本
- `analyze_qwen_results.py` - 结果分析脚本
- `analyze_qwen_results_corrected.py` - 修正版结果分析脚本
- `detailed_comparison_analysis.py` - 详细对比分析脚本
- `detailed_comparison_analysis_fixed.py` - 修正版详细对比分析脚本

## 实验结果摘要

### 总体表现
- **总题目数**: 2,688题
- **总正确数**: 1,534题
- **总体正确率**: 57.07%

### 各任务类型表现
- **Grid任务**: 94.31% (845/896)
- **Gabor任务**: 37.95% (340/896)
- **Color任务**: 38.95% (349/896)

### 置信度分布
- 主要给出置信度3，很少给出高置信度(4-5)
- 平均置信度: 3.13

## 使用方法

1. 运行推理：
```bash
python3 qwen2_vl_metacognition_multi_model.py --model Qwen/Qwen2.5-VL-7B-Instruct --questions questions_two_stage_concise.jsonl --output results.jsonl
```

2. 分析结果：
```bash
python3 analyze_qwen_results_corrected.py
```

3. 详细对比分析：
```bash
python3 detailed_comparison_analysis_fixed.py
```
