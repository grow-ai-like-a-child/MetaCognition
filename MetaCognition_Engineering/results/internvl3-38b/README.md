# InternVL3-38B 元认知实验结果

本文件夹包含InternVL3-38B模型在元认知实验中的完整结果和分析。实验采用两阶段元认知推理：第一阶段回答选择题，第二阶段评估置信度。

## 📁 文件结构

### 🔬 核心实验文件
- **`internvl3_38b_metacognition_fixed.py`** (15KB) - 主要的元认知实验脚本
  - 支持InternVL3-38B模型的元认知推理
  - 实现两阶段推理流程（答题+置信度评估）
  - 包含智能概率计算和错误处理
- **`internvl3_complete_results.jsonl`** (2.6MB) - 完整的实验结果数据
  - 包含5376条记录（2688个stage1 + 2688个stage2）
  - 每行包含：qid, task, choice, confidence, probabilities, latency等
- **`internvl3_experiment.log`** (1.2MB) - 实验运行日志
  - 详细的运行过程记录
  - 错误信息和调试信息

### 📊 分析脚本
- **`analyze_confidence_by_task.py`** (7.4KB) - 每类任务置信度分布分析
  - 按Color、Gabor、Grid任务分别分析置信度分布
  - 计算置信度校准误差
  - 生成详细的置信度统计表格
- **`analyze_internvl3_detailed.py`** (5.8KB) - 详细性能分析脚本
  - 总体性能统计
  - 按置信度分组分析
  - 按任务类型分析
  - 置信度校准分析
- **`compare_with_qwen72b.py`** (7.6KB) - 与Qwen2.5-VL-72B的对比分析
  - 准确率对比
  - 置信度分布对比
  - 任务类型性能对比
  - 置信度校准对比
- **`create_correct_comparison.py`** (4.6KB) - 基于ground_truth的正确匹配脚本
  - 将模型结果与真实标签匹配
  - 生成标准化的比较数据
- **`visualize_confidence_distribution.py`** (6.8KB) - 置信度分布可视化脚本
  - 生成多维度可视化图表
  - 支持PNG和PDF格式输出

### 📈 结果数据文件
- **`internvl3_correct_comparison.json`** (981KB) - 正确的匹配结果
  - 包含每题的模型选择、正确选项、是否正确、置信度
  - 基于ground_truth.json的准确匹配
  - 用于所有后续分析的基础数据
- **`internvl3_confidence_by_task_analysis.json`** (1.1KB) - 每类任务的置信度分析数据
  - 按任务类型统计的置信度分布
  - 置信度与准确率关系
  - 平均置信度等指标
- **`internvl3_vs_qwen72b_comparison.json`** (1.4KB) - 与Qwen72B的对比数据
  - 两个模型的详细对比结果
  - 性能差异分析

### 🎨 可视化文件
- **`internvl3_confidence_distribution.png`** (271KB) - 置信度分布可视化图表（PNG格式）
- **`internvl3_confidence_distribution.pdf`** (22KB) - 置信度分布可视化图表（PDF格式）
  - 包含4个子图：置信度分布、百分比分布、置信度-准确率关系、任务性能对比

## 🎯 主要发现

### 📊 总体性能
- **总题目数**: 2,688题（每类任务896题）
- **正确答案数**: 1,266题
- **整体准确率**: 47.10%
- **平均置信度**: 4.41（1-5分制）

### 🔄 与Qwen2.5-VL-72B对比
- **Qwen72B准确率**: 56.77%
- **InternVL3准确率**: 47.10%
- **性能差异**: InternVL3比Qwen72B低9.67%
- **置信度分布差异**: InternVL3在低置信度(3)表现更好，高置信度(5)表现略差

### 🎯 按任务类型分析
1. **Grid任务**: 90.62%准确率（最容易）
   - 平均置信度: 4.91
   - 置信度分布: 92%为置信度5
   - 校准误差: 0.078（校准最好）

2. **Color任务**: 29.13%准确率（中等难度）
   - 平均置信度: 4.41
   - 置信度分布: 70.5%为置信度5
   - 校准误差: 0.668（校准最差）

3. **Gabor任务**: 21.54%准确率（最难）
   - 平均置信度: 3.98
   - 置信度分布: 45.2%为置信度3，42.7%为置信度5
   - 校准误差: 0.843（校准最差）

### 📈 置信度分布与校准
- **置信度3**: 25.89%准确率（676题，25.15%占比）
  - 校准误差: 0.341（过度自信）
- **置信度4**: 35.26%准确率（173题，6.44%占比）
  - 校准误差: 0.447（过度自信）
- **置信度5**: 56.01%准确率（1,839题，68.42%占比）
  - 校准误差: 0.440（严重过度自信）

### 🔍 关键洞察
1. **任务难度差异巨大**: Grid任务表现最好，Gabor任务最困难
2. **置信度校准问题**: 模型存在严重过度自信问题，置信度5的准确率只有56%
3. **预测分布**: Color任务偏向预测A（61.9%），其他任务相对平衡
4. **置信度策略**: Gabor任务相对谨慎，Grid任务高度自信
5. **模型对比**: InternVL3在整体性能上低于Qwen72B，但在某些置信度水平上表现更好

## 🚀 使用方法

### 📊 运行分析脚本
```bash
# 1. 详细性能分析（推荐首先运行）
python analyze_internvl3_detailed.py
# 输出：总体性能、置信度分布、任务类型分析、校准分析

# 2. 每类任务置信度分析
python analyze_confidence_by_task.py
# 输出：Color、Gabor、Grid任务的详细置信度分布统计

# 3. 与Qwen72B对比分析
python compare_with_qwen72b.py
# 输出：两个模型的详细对比结果

# 4. 生成可视化图表
python visualize_confidence_distribution.py
# 输出：internvl3_confidence_distribution.png/pdf

# 5. 重新生成正确匹配结果
python create_correct_comparison.py
# 输出：internvl3_correct_comparison.json
```

### 🔬 重新运行实验
```bash
# 运行完整实验（需要GPU）
python internvl3_38b_metacognition_fixed.py \
    --questions ../../data/prompt/questions_two_stage_concise.jsonl \
    --output new_results.jsonl

# 运行小规模测试（3题）
python internvl3_38b_metacognition_fixed.py \
    --questions ../../data/prompt/questions_two_stage_concise.jsonl \
    --output test_results.jsonl \
    --max_questions 3
```

### 📋 依赖要求
```bash
# 安装必要的Python包
pip install torch transformers pillow numpy matplotlib

# 确保有足够的GPU内存（推荐16GB+）
# 模型大小约38B参数
```

## 📋 数据格式

### 🔬 实验结果格式 (internvl3_complete_results.jsonl)
每行一个JSON对象，包含完整的实验记录：
```json
{
  "qid": "GRID-0001_stage1",           // 问题ID（包含stage信息）
  "task": "Grid",                      // 任务类型（Grid/Color/Gabor）
  "choice": "A",                       // 模型选择的答案
  "confidence": 4,                     // 置信度（1-5分制）
  "raw_text": "A",                     // 模型原始回答文本
  "latency_ms": 1976.5,               // 推理延迟（毫秒）
  "stage": 1,                          // 阶段（1=答题，2=置信度评估）
  "original_qid": "GRID-0001",        // 原始问题ID
  "probabilities": {"A": 0.95, "B": 0.05}, // 选项概率分布
  "prompt": "Which one has more: O or X?...", // 输入提示
  "image_path": "../../cache/images/Grid/GRID-0001.png", // 图像路径
  "timestamp": 1758000400.758947       // 时间戳
}
```

### 📊 正确匹配结果格式 (internvl3_correct_comparison.json)
基于ground_truth的标准化比较数据：
```json
{
  "qid": "GRID-0001_stage1",           // 问题ID
  "original_qid": "GRID-0001",        // 原始问题ID
  "task": "Grid",                      // 任务类型
  "predicted_choice": "A",             // 模型预测的答案
  "correct_choice": "A",               // 正确答案（来自ground_truth）
  "is_correct": true,                  // 是否正确
  "confidence": 4,                     // 置信度
  "probabilities": {"A": 0.95, "B": 0.05}, // 概率分布
  "latency_ms": 1976.5,               // 推理延迟
  "raw_text": "A",                     // 原始回答
  "image_path": "../../cache/images/Grid/GRID-0001.png" // 图像路径
}
```

### 📈 置信度分析结果格式 (internvl3_confidence_by_task_analysis.json)
按任务类型统计的置信度分析：
```json
{
  "Color": {
    "total_questions": 896,
    "correct_answers": 261,
    "accuracy": 0.2913,
    "average_confidence": 4.41,
    "confidence_distribution": {"3": 263, "4": 1, "5": 632},
    "confidence_accuracy": {"3": 0.1901, "4": 1.0, "5": 0.3323}
  },
  "Gabor": { ... },
  "Grid": { ... }
}
```

### 🔄 模型对比结果格式 (internvl3_vs_qwen72b_comparison.json)
两个模型的详细对比数据：
```json
{
  "internvl3_38b": {
    "total_questions": 2688,
    "correct_answers": 1266,
    "accuracy": 0.4710,
    "confidence_stats": {...},
    "task_stats": {...}
  },
  "qwen2_5_vl_72b": { ... },
  "comparison": {
    "accuracy_difference": -0.0967,
    "better_model": "Qwen2.5-VL-72B"
  }
}
```

## 🔍 关键洞察

1. **任务难度差异巨大**: Grid任务表现最好，Gabor任务最困难
2. **置信度校准问题**: 模型存在过度自信问题，置信度5的准确率只有56%
3. **预测分布**: Color任务偏向预测A，其他任务相对平衡
4. **模型对比**: InternVL3在整体性能上低于Qwen72B

## 📝 注意事项

### 🔧 技术注意事项
- 所有分析脚本都基于`internvl3_correct_comparison.json`文件
- 可视化需要matplotlib库：`pip install matplotlib`
- 实验日志包含详细的运行信息和错误记录
- 结果文件采用UTF-8编码，支持中文显示

### 💾 数据完整性
- `internvl3_complete_results.jsonl`包含所有原始实验数据
- `internvl3_correct_comparison.json`是分析的基础数据文件
- 所有JSON文件都经过验证，确保数据格式正确

### 🚀 性能要求
- 运行完整实验需要16GB+ GPU内存
- 模型推理时间约1.5-2秒/题
- 完整实验运行时间约2-3小时

### 📊 分析建议
1. 首先运行`analyze_internvl3_detailed.py`了解总体性能
2. 然后运行`analyze_confidence_by_task.py`了解任务差异
3. 使用`compare_with_qwen72b.py`进行模型对比
4. 最后运行`visualize_confidence_distribution.py`生成图表

### 🔄 更新记录
- **2024-09-16**: 完成InternVL3-38B元认知实验
- **2024-09-16**: 完成与Qwen72B的对比分析
- **2024-09-16**: 完成每类任务置信度分布分析
- **2024-09-16**: 生成可视化图表和完整文档
