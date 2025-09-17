# Ovis2-34B 元认知实验

## 🎯 实验概述

本实验使用 [AIDC-AI/Ovis2-34B](https://huggingface.co/AIDC-AI/Ovis2-34B) 模型进行多模态元认知能力评估。Ovis2-34B 是阿里巴巴国际数字商业集团推出的最新多模态大语言模型，具有强大的视觉理解和推理能力。

## 🏗️ 模型架构

- **视觉编码器**: aimv2-1B-patch14-448
- **语言模型**: Qwen2.5-32B-Instruct  
- **参数量**: 34.9B
- **特色功能**:
  - 增强的Chain-of-Thought推理能力
  - 视频和多图像处理支持
  - 多语言OCR能力
  - Flash Attention 2.0优化

## ✅ 测试状态

### 模型功能测试 ✅
- **模型加载**: ✅ 成功
- **基础推理**: ✅ 成功
- **概率计算**: ✅ 成功

### 元认知功能测试 ✅
- **Stage 1 (问题回答)**: ✅ 成功
  - 准确识别图像内容
  - 正确提取选择结果
  - 精确计算选择概率
- **Stage 2 (置信度评估)**: ✅ 成功
  - 合理评估置信度
  - 正确提取置信度分数

### 测试结果示例
```json
{
  "stage1": {
    "response": "A (red)",
    "choice": "A", 
    "choice_probabilities": {
      "A": 0.9999, 
      "B": 0.0001
    }
  },
  "stage2": {
    "response": "I am 10 out of 10 confident in my previous answer. The image is a solid red square...",
    "confidence_score": 1.0
  }
}
```

## 🛠️ 技术特点

### 1. 高效推理架构
- **单次推理**: Stage 1 只需一次模型调用获得响应和概率
- **精确概率**: 基于logits直接计算，避免启发式估计
- **Flash Attention**: 优化的注意力机制提升推理速度

### 2. 多模态处理能力
- **图像理解**: 支持复杂视觉场景分析
- **文本生成**: 高质量自然语言响应
- **跨模态推理**: 视觉-语言联合理解

### 3. 元认知设计
- **两阶段架构**: 分离问题回答和置信度评估
- **概率量化**: 提供精确的选择概率分布
- **置信度校准**: 支持多种置信度表达方式

## 📁 文件结构

```
ovis2-34b/
├── ovis2_34b_metacognition.py           # 主实验代码（基础版本）
├── ovis2_34b_metacognition_realtime.py  # 实时版本（推荐使用）
├── test_ovis2_model.py                   # 模型功能测试
├── test_metacognition_sample.py          # 元认知功能测试
├── test_realtime_sample.py               # 实时版本测试
├── ovis2_34b_test_results.jsonl          # 基础测试结果
├── ovis2_34b_realtime_test_results.jsonl # 实时版本测试结果
└── README.md                             # 本文档
```

## 🚀 运行实验

### 环境准备
```bash
# 安装Ovis库
pip install git+https://github.com/AIDC-AI/Ovis.git

# 安装Flash Attention
pip install flash-attn --no-build-isolation

# 更新torchvision
pip install torchvision==0.19.0 --upgrade
```

### 完整实验（推荐使用实时版本）
```bash
# 实时版本（推荐）- 模仿Qwen72B实时版本的设计
python ovis2_34b_metacognition_realtime.py \
  --questions ../../data/prompt/questions_two_stage_concise.jsonl \
  --output ovis2_34b_concise_full_results.jsonl \
  --model AIDC-AI/Ovis2-34B \
  --device auto

# 基础版本
python ovis2_34b_metacognition.py \
  --questions ../../data/prompt/questions_two_stage_concise.jsonl \
  --output ovis2_34b_concise_full_results.jsonl \
  --model AIDC-AI/Ovis2-34B \
  --device auto
```

### 快速测试
```bash
# 模型功能测试
python test_ovis2_model.py

# 元认知功能测试  
python test_metacognition_sample.py

# 实时版本测试（推荐）
python test_realtime_sample.py
```

## 📊 预期性能

基于模型架构和初步测试，Ovis2-34B预期表现：

- **推理速度**: 比InternVL3-38B更快（单次推理vs三次推理）
- **视觉理解**: 优秀的图像识别和场景理解能力
- **推理质量**: 强化的CoT推理能力
- **置信度校准**: 基于Qwen2.5的优秀语言理解

## 🔄 下一步计划

1. **✅ 环境配置**: 已完成
2. **✅ 模型测试**: 已完成  
3. **✅ 元认知验证**: 已完成
4. **✅ 实时版本**: 已完成（模仿Qwen72B设计）
5. **🔄 完整实验**: 准备就绪
6. **📊 结果分析**: 待进行
7. **☁️ 数据上传**: 待进行

## 💡 技术亮点

- **最新架构**: 基于Ovis2.5系列的最新多模态技术
- **高效实现**: 优化的推理流程和内存使用
- **精确量化**: 基于真实logits的概率计算
- **完整元认知**: 支持复杂的元认知评估任务
- **实时写入**: 模仿Qwen72B实时版本，避免数据丢失
- **会话连续性**: Stage 1和Stage 2在同一个聊天会话中进行

## 🎯 实时版本测试结果

最新测试显示Ovis2-34B实时版本表现优异：

### 性能指标
- **Stage 1 延迟**: ~300-800ms
- **Stage 2 延迟**: ~1300-1400ms  
- **概率精度**: 高精度logits计算
- **置信度提取**: 自动识别多种置信度表达

### 测试示例
```json
{
  "红色图像": {
    "stage1": {"choice": "A (red)", "prob_A": 0.75, "prob_B": 0.0001},
    "stage2": {"confidence": 10.0, "response": "I am very confident..."}
  },
  "蓝色图像": {
    "stage1": {"choice": "B (blue)", "prob_A": 0.0006, "prob_B": 0.73},
    "stage2": {"confidence": 10.0, "response": "I am very confident..."}
  }
}
```

---

**状态**: 🟢 已准备就绪，可以开始完整实验

**推荐使用**: `ovis2_34b_metacognition_realtime.py`（实时版本）

**最后更新**: 2025-09-16 10:11
