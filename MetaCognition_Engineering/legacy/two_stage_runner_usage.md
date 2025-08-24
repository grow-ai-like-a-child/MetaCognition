# 两阶段提问的Catalog Runner使用说明

## 🎯 概述

`catalog_runner_two_stage.py` 是支持两阶段提问的Catalog Runner版本，可以：

1. **生成图片**：从catalog生成实验图片
2. **创建两阶段问题**：生成questions_two_stage.jsonl
3. **调用模型**：支持mock和OpenAI API
4. **收集结果**：生成评估CSV文件

## 📋 主要特性

### **两阶段提问设计**
- **第一阶段**：纯实验任务，不包含置信度要求
- **第二阶段**：基于第一阶段的回答，评估置信度

### **支持的任务类型**
- **Grid**：网格符号数量判断
- **Gabor**：条纹方向判断  
- **Color**：亮度差异判断

## 🚀 使用方法

### **1. 生成图片**
```bash
python catalog_runner_two_stage.py gen --catalog catalog_2688_newspec.json --imgroot cache/images
```

**参数说明**：
- `--catalog`：题目表文件路径
- `--imgroot`：图片输出根目录
- `--only-task`：（可选）只生成特定任务类型

### **2. 创建两阶段问题**
```bash
python catalog_runner_two_stage.py pack --catalog catalog_2688_newspec.json --imgroot cache/images --out questions_two_stage.jsonl
```

**输出文件**：`questions_two_stage.jsonl`
- 包含5,376个问题（2,688个原始题目 × 2个阶段）
- 每个问题都有`stage`字段标识阶段
- 通过`original_qid`关联两个阶段

### **3. 调用模型**
```bash
# Mock模式（离线测试）
python catalog_runner_two_stage.py feed --pack questions_two_stage.jsonl --out responses_two_stage.jsonl --engine mock --workers 8

# OpenAI模式（需要API密钥）
python catalog_runner_two_stage.py feed --pack questions_two_stage.jsonl --out responses_two_stage.jsonl --engine openai --workers 4
```

**参数说明**：
- `--pack`：问题文件路径
- `--out`：响应输出文件路径
- `--engine`：模型引擎（mock/openai）
- `--workers`：并发数

### **4. 收集评估结果**
```bash
python catalog_runner_two_stage.py collect --resp responses_two_stage.jsonl --catalog catalog_2688_newspec.json --out eval_two_stage.csv
```

**输出文件**：`eval_two_stage.csv`
- 包含每个原始题目的完整评估结果
- 两个阶段的答案、置信度、延迟时间
- 任务特定的参数信息

## 📊 输出文件格式

### **questions_two_stage.jsonl**
```json
{
  "qid": "GRID-0001_stage1",
  "task": "Grid",
  "image_path": "cache/images/Grid/GRID-0001.png",
  "prompt": "Which one has more: O or X?\nChoose one:\nA. O\nB. X\nAnswer with A or B.",
  "stage": 1,
  "original_qid": "GRID-0001"
}
```

### **responses_two_stage.jsonl**
```json
{
  "qid": "GRID-0001_stage1",
  "task": "Grid",
  "choice": "A",
  "confidence": 0,
  "raw_text": "A",
  "latency_ms": 450,
  "stage": 1,
  "original_qid": "GRID-0001"
}
```

### **eval_two_stage.csv**
| qid | task | stage1_choice | stage1_confidence | stage2_confidence | is_correct | grid_level | ... |
|-----|------|---------------|-------------------|-------------------|------------|------------|-----|
| GRID-0001 | Grid | A | 0 | 4 | true | 1 | ... |

## 🔧 关键改进

### **1. 提示生成**
- `prompt_for_stage1()`：生成纯实验任务提示
- `prompt_for_stage2()`：生成置信度评估提示

### **2. Mock答案**
- `mock_answer_stage1()`：生成实验任务答案
- `mock_answer_stage2()`：生成置信度评估

### **3. 结果收集**
- 按`original_qid`分组两个阶段的响应
- 生成包含两个阶段信息的评估CSV

## 📈 实验流程

### **完整流程示例**
```bash
# 1. 生成图片（如果还没有）
python catalog_runner_two_stage.py gen --catalog catalog_2688_newspec.json --imgroot cache/images

# 2. 创建两阶段问题
python catalog_runner_two_stage.py pack --catalog catalog_2688_newspec.json --imgroot cache/images --out questions_two_stage.jsonl

# 3. 调用模型（Mock模式）
python catalog_runner_two_stage.py feed --pack questions_two_stage.jsonl --out responses_two_stage.jsonl --engine mock --workers 8

# 4. 收集评估结果
python catalog_runner_two_stage.py collect --resp responses_two_stage.jsonl --catalog catalog_2688_newspec.json --out eval_two_stage.csv
```

## ⚠️ 注意事项

1. **图片路径**：确保`imgroot`目录下有对应的任务子目录
2. **OpenAI API**：使用OpenAI引擎需要设置`OPENAI_API_KEY`环境变量
3. **文件关联**：两个阶段的问题通过`original_qid`字段关联
4. **评估逻辑**：正确性判断基于第一阶段的答案

## 🎯 优势

- ✅ **更自然的对话流程**：先完成任务，再评估信心
- ✅ **避免任务混淆**：模型不会被置信度要求干扰
- ✅ **更准确的置信度**：基于已完成的任务来评估
- ✅ **便于分析**：可以分别分析任务准确性和置信度准确性
- ✅ **符合人类认知**：先思考答案，再评估确定性
