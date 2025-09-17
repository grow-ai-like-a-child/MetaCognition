#!/bin/bash
# Qwen2.5-VL-72B 元认知推理实验运行脚本

echo "=========================================="
echo "Qwen2.5-VL-72B 元认知推理实验"
echo "=========================================="

# 设置基本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUESTIONS_FILE="$PROJECT_ROOT/data/processed/questions_two_stage.jsonl"
OUTPUT_FILE="$SCRIPT_DIR/qwen2.5-vl-72b_full_results.jsonl"

# 检查问题文件是否存在
if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "错误: 问题文件不存在: $QUESTIONS_FILE"
    echo "请先运行 full_pipeline.py 生成问题文件"
    exit 1
fi

echo "问题文件: $QUESTIONS_FILE"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "开始运行Qwen2.5-VL-72B推理..."
echo "预计运行时间: 6-12小时"
echo ""

# 运行推理脚本
cd "$SCRIPT_DIR"
python3 qwen2_vl_metacognition_72b.py \
    --questions "$QUESTIONS_FILE" \
    --output "$OUTPUT_FILE" \
    --model "Qwen/Qwen2.5-VL-72B-Instruct" \
    --device "auto"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "实验完成！"
    echo "结果文件: $OUTPUT_FILE"
    echo "=========================================="
    
    # 显示结果文件信息
    if [ -f "$OUTPUT_FILE" ]; then
        echo "结果文件大小: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "结果行数: $(wc -l < "$OUTPUT_FILE")"
    fi
else
    echo ""
    echo "=========================================="
    echo "实验失败！请检查错误信息"
    echo "=========================================="
    exit 1
fi
