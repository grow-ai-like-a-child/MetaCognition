# Qwen2.5-VL-72B 元认知推理实验

## 文件说明

### 核心脚本
- `qwen2_vl_metacognition_72b.py` - Qwen2.5-VL-72B模型推理脚本（原始版本，批量写入）
- `qwen2_vl_metacognition_72b_realtime.py` - 实时写入版本，防止数据丢失
- `compare_with_ground_truth.py` - 与标准答案对比分析脚本
- `create_simple_comparison.py` - 生成简化对比结果脚本
- `analyze_performance.py` - 性能分析脚本
- `analyze_confidence_by_task.py` - 按任务类型分析置信度分布脚本

### 数据文件
- `questions_two_stage_concise.jsonl` - 两阶段问题数据集（5376题）
- `questions_two_stage_enhanced.jsonl` - 增强版两阶段问题数据集
- `catalog_2688_newspec_correctly_fixed.json` - 题目目录文件
- `ground_truth.json` - 正确答案字典

### 实验结果
- `qwen2.5-vl-72b_full_results.jsonl` - Qwen 72B模型完整推理结果（2,688题）
- `qwen2.5-vl-72b_realtime_results.jsonl` - 实时写入版本结果（部分完成）
- `qwen72b_comparison_with_ground_truth.json` - 详细对比分析结果
- `qwen72b_simple_comparison.json` - 简化对比结果（包含正确性标记）

### 配置文件
- `gpu_config.txt` - GPU配置信息
- `cpu_config.txt` - CPU配置信息

## 实验结果摘要

### 总体表现
- **总题目数**: 2,688题
- **总正确数**: 1,526题
- **总体正确率**: 56.77%

### 各任务类型表现
- **GRID任务**: 94.98% (1,512/1,592) ✅ 表现极佳
- **COL任务**: 39.40% (344/872) ⚠️ 表现较差
- **GAB任务**: 35.94% (320/891) ⚠️ 表现最差

### 置信度分布
- **置信度5**: 98.62% (2,651题) - 模型过度自信
- **置信度4**: 0.78% (21题)
- **置信度3**: 0.60% (16题)
- **平均置信度**: 4.99

### 元认知能力分析
- ✅ **置信度校准良好**: 高置信度(4-5)准确率(57.04%) > 低置信度(1-3)准确率(12.50%)
- ⚠️ **过度自信问题**: 98.62%的题目都给出最高置信度(5)
- 📊 **任务差异明显**: GRID任务表现优秀，COL和GAB任务表现较差

## 使用方法

### 1. 运行模型推理

**原始版本（批量写入）：**
```bash
python3 qwen2_vl_metacognition_72b.py --questions ../../prompt/questions_two_stage_concise.jsonl --output qwen2.5-vl-72b_full_results.jsonl --model Qwen/Qwen2.5-VL-72B-Instruct --device auto
```

**实时写入版本（推荐）：**
```bash
python3 qwen2_vl_metacognition_72b_realtime.py --questions ../../prompt/questions_two_stage_concise.jsonl --output qwen2.5-vl-72b_realtime_results.jsonl --model Qwen/Qwen2.5-VL-72B-Instruct --device auto
```

**使用运行脚本：**
```bash
# 原始版本
./run_qwen72b_experiment.sh

# 实时版本
./run_qwen72b_realtime.sh
```

### 2. 结果分析

**生成详细对比结果：**
```bash
python3 compare_with_ground_truth.py
```

**生成简化对比结果：**
```bash
python3 create_simple_comparison.py
```

**性能分析：**
```bash
python3 analyze_performance.py
```

**按任务类型分析置信度：**
```bash
python3 analyze_confidence_by_task.py
```

### 3. 自定义参数

**使用特定设备：**
```bash
python3 qwen2_vl_metacognition_72b_realtime.py --questions ../../prompt/questions_two_stage_concise.jsonl --output results.jsonl --device cuda:0
```

**使用自定义模型路径：**
```bash
python3 qwen2_vl_metacognition_72b_realtime.py --questions ../../prompt/questions_two_stage_concise.jsonl --output results.jsonl --model /path/to/local/model
```

## 关键发现

### 1. 任务表现差异
- **GRID任务**: 94.98%准确率，模型在网格计数任务上表现优秀
- **COL任务**: 39.40%准确率，颜色判断任务表现较差
- **GAB任务**: 35.94%准确率，Gabor模式识别任务表现最差

### 2. 元认知能力
- **置信度校准**: 模型能够正确识别自己的不确定性
- **过度自信**: 98.62%的题目都给出最高置信度(5)
- **校准质量**: 高置信度题目的准确率确实更高

### 3. 技术改进
- **实时写入**: 解决了批量写入可能导致的数据丢失问题
- **结果分析**: 提供了多种分析工具和可视化结果
- **错误处理**: 改进了QID匹配和数据格式问题

## 注意事项

- 72B模型需要大量GPU内存，建议使用多GPU或高显存GPU
- **推荐使用实时写入版本**，避免长时间运行后数据丢失
- 运行时间较长，建议使用后台运行和进度监控
- 确保有足够的磁盘空间存储模型和结果文件
- 建议在运行前检查GPU内存使用情况

## 硬件要求

- **GPU**: 建议使用A100 80GB或H100等高端GPU
- **内存**: 建议至少64GB系统内存
- **存储**: 建议至少200GB可用空间
- **网络**: 需要稳定的网络连接下载模型
