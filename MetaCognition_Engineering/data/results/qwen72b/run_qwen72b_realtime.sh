#!/bin/bash

# Qwen2.5-VL-72B 实时写入版本运行脚本
# 确保每处理完一道题目就立即保存结果，避免数据丢失

echo "开始运行 Qwen2.5-VL-72B 实时写入版本实验..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 创建日志目录
mkdir -p logs

# 记录开始时间
echo "实验开始时间: $(date)" > logs/qwen72b_realtime_start.log

# 运行实验（实时写入版本）
python3 -u qwen2_vl_metacognition_72b_realtime.py \
    --questions ../../prompt/questions_two_stage_concise.jsonl \
    --output qwen2.5-vl-72b_realtime_results.jsonl \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --device auto \
    2>&1 | tee logs/qwen72b_realtime_run.log

# 记录结束时间
echo "实验结束时间: $(date)" >> logs/qwen72b_realtime_start.log

# 检查结果文件
if [ -f "qwen2.5-vl-72b_realtime_results.jsonl" ]; then
    echo "结果文件已生成:"
    ls -lh qwen2.5-vl-72b_realtime_results.jsonl
    echo "文件行数: $(wc -l < qwen2.5-vl-72b_realtime_results.jsonl)"
else
    echo "错误: 结果文件未生成"
    exit 1
fi

echo "实验完成！"
