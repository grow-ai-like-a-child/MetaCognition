# 两阶段提问示例

## 🎯 设计理念

将原来的单阶段提问（实验任务+置信度）拆分为两个独立的阶段：

1. **第一阶段**：纯实验任务，不涉及置信度
2. **第二阶段**：基于第一阶段的回答，评估置信度

## 📋 具体示例

### Grid任务示例

#### 第一阶段：实验任务
```
qid: "GRID-0001_stage1"
task: "Grid"
image_path: "cache/images/Grid/GRID-0001.png"
prompt: "Which one has more: O or X?
Choose one:
A. O
B. X
Answer with A or B."
stage: 1
```

**模型回答示例**：
```
B
```

#### 第二阶段：置信度评估
```
qid: "GRID-0001_stage2"
task: "Grid"
image_path: "cache/images/Grid/GRID-0001.png"
prompt: "Based on your previous answer, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
stage: 2
```

**模型回答示例**：
```
confidence: 4
```

### Color任务示例

#### 第一阶段：实验任务
```
qid: "COL-0001_stage1"
task: "Color"
image_path: "cache/images/Color/COL-0001.png"
prompt: "Which side is brighter?"
stage: 1
```

**模型回答示例**：
```
The right side appears brighter.
```

#### 第二阶段：置信度评估
```
qid: "COL-0001_stage2"
task: "Color"
image_path: "cache/images/Color/COL-0001.png"
prompt: "Based on your previous answer about which side is brighter, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
stage: 2
```

**模型回答示例**：
```
confidence: 3
```

### Gabor任务示例

#### 第一阶段：实验任务
```
qid: "GAB-0001_stage1"
task: "Gabor"
image_path: "cache/images/Gabor/GAB-0001.png"
prompt: "Are the stripes more vertical or horizontal?"
stage: 1
```

**模型回答示例**：
```
The stripes appear more vertical.
```

#### 第二阶段：置信度评估
```
qid: "GAB-0001_stage2"
task: "Gabor"
image_path: "cache/images/Gabor/GAB-0001.png"
prompt: "Based on your previous answer about the stripe orientation, how confident are you in your choice? Rate your confidence from 1 to 5, where 1 means very uncertain and 5 means very certain. Answer with 'confidence: N' where N is your confidence level."
stage: 2
```

**模型回答示例**：
```
confidence: 2
```

## 🔄 实验流程

### 传统单阶段流程
```
用户 → 模型 → 回答（任务答案 + 置信度）
```

### 新的两阶段流程
```
第一阶段：用户 → 模型 → 任务答案
第二阶段：用户 → 模型 → 置信度评估
```

## ✅ 优势

1. **更自然的对话流程**：先完成任务，再评估信心
2. **避免任务混淆**：模型不会在回答任务时被置信度要求干扰
3. **更准确的置信度**：基于已完成的任务回答来评估信心
4. **便于分析**：可以分别分析任务准确性和置信度准确性
5. **符合人类认知**：先思考答案，再评估确定性

## 📊 数据统计

- **原始问题数**：2,688
- **两阶段问题数**：5,376
- **第一阶段**：2,688（实验任务）
- **第二阶段**：2,688（置信度评估）

## 🎯 使用方法

1. 使用 `questions_two_stage.jsonl` 进行实验
2. 先运行所有 `stage: 1` 的问题
3. 再运行所有 `stage: 2` 的问题
4. 通过 `original_qid` 字段关联两个阶段的回答
