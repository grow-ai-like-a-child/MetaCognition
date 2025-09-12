# Qwen2.5-VL-32B 元认知推理实验

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
- `qwen2.5-vl-32b_full_results.jsonl` - Qwen 32B模型完整推理结果（待生成）
- `qwen32b_detailed_analysis.csv` - 详细分析CSV文件（待生成）
- `qwen32b_detailed_analysis_corrected.csv` - 修正版详细分析CSV文件（待生成）

### 对比分析
- `qwen32b_detailed_comparison.json` - 详细对比分析JSON（待生成）
- `qwen32b_detailed_comparison_fixed.json` - 修正版详细对比分析JSON（待生成）
- `qwen32b_simplified_comparison.json` - 简化版对比分析JSON（待生成）

### 分析脚本
- `analyze_qwen_results.py` - 结果分析脚本（待复制）
- `analyze_qwen_results_corrected.py` - 修正版结果分析脚本（待复制）
- `detailed_comparison_analysis.py` - 详细对比分析脚本（待复制）
- `detailed_comparison_analysis_fixed.py` - 修正版详细对比分析脚本（待复制）

## 实验结果摘要

### 总体表现
- **总题目数**: 2,688题
- **总正确数**: 待运行
- **总体正确率**: 待运行

### 各任务类型表现
- **Grid任务**: 待运行
- **Gabor任务**: 待运行
- **Color任务**: 待运行

### 置信度分布
- 平均置信度: 待运行
- 置信度分布: 待运行

## 使用方法

1. 运行32B模型推理：
```bash
python3 qwen2_vl_metacognition_multi_model.py --model Qwen/Qwen2.5-VL-32B-Instruct --questions questions_two_stage_concise.jsonl --output qwen2.5-vl-32b_full_results.jsonl
```

2. 分析结果（运行后复制分析脚本）：
```bash
python3 analyze_qwen_results_corrected.py
```

3. 详细对比分析：
```bash
python3 detailed_comparison_analysis_fixed.py
```

## 注意事项

- 32B模型需要更多GPU内存，请确保有足够的显存
- 建议使用批处理模式运行，避免内存溢出
- 运行时间会比7B模型长，请耐心等待
