#!/bin/bash
# Qwen2.5-VL元认知实验运行脚本

# 设置默认参数
QUESTIONS_FILE="questions_two_stage_concise.jsonl"
OUTPUT_DIR="results"
MODEL_SIZE="7B"  # 默认使用7B模型
S3_SYNC=false
S3_BUCKET="s3://perceptualmetacognition/"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --questions)
            QUESTIONS_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test-loading)
            TEST_LOADING=true
            shift
            ;;
        --show-recommendations)
            SHOW_RECOMMENDATIONS=true
            shift
            ;;
        --s3-sync)
            S3_SYNC=true
            shift
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model SIZE           模型大小 (7B, 14B, 32B, 72B) [默认: 7B]"
            echo "  --questions FILE       题目文件路径 [默认: questions_two_stage_concise.jsonl]"
            echo "  --output-dir DIR       输出目录 [默认: results]"
            echo "  --test-loading         只测试模型加载，不运行完整实验"
            echo "  --show-recommendations 显示模型推荐"
            echo "  --s3-sync             启用S3数据同步"
            echo "  --s3-bucket BUCKET    S3存储桶地址 [默认: s3://perceptualmetacognition/]"
            echo "  -h, --help            显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --model 7B"
            echo "  $0 --model 32B --questions my_questions.jsonl"
            echo "  $0 --test-loading"
            echo "  $0 --show-recommendations"
            echo "  $0 --model 7B --s3-sync"
            echo "  $0 --model 72B --s3-sync --s3-bucket s3://my-bucket/"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查题目文件是否存在
if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "错误: 题目文件 '$QUESTIONS_FILE' 不存在"
    exit 1
fi

# 显示系统信息
echo "=== 系统信息 ==="
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""
echo "内存信息:"
free -h
echo ""

# 显示模型推荐
if [ "$SHOW_RECOMMENDATIONS" = true ]; then
    echo "=== 模型推荐 ==="
    python3 qwen2_vl_metacognition_multi_model.py --show-recommendations
    exit 0
fi

# 测试模型加载
if [ "$TEST_LOADING" = true ]; then
    echo "=== 测试模型加载 ==="
    python3 test_model_loading.py
    exit 0
fi

# 设置模型名称
case $MODEL_SIZE in
    7B)
        MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
        ;;
    14B)
        MODEL_NAME="Qwen/Qwen2.5-VL-14B-Instruct"
        ;;
    32B)
        MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
        ;;
    72B)
        MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct"
        ;;
    *)
        echo "错误: 不支持的模型大小 '$MODEL_SIZE'"
        echo "支持的模型大小: 7B, 14B, 32B, 72B"
        exit 1
        ;;
esac

# 生成输出文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="$OUTPUT_DIR/qwen2.5-vl-${MODEL_SIZE,,}_metacognition_${TIMESTAMP}.jsonl"

echo "=== 开始元认知实验 ==="
echo "模型: $MODEL_NAME"
echo "题目文件: $QUESTIONS_FILE"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 运行实验
python3 qwen2_vl_metacognition_multi_model.py \
    --model "$MODEL_NAME" \
    --questions "$QUESTIONS_FILE" \
    --output "$OUTPUT_FILE"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 实验完成 ==="
    echo "结果文件: $OUTPUT_FILE"
    
    # 统计结果
    if [ -f "$OUTPUT_FILE" ]; then
        TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
        echo "总结果数: $TOTAL_LINES"
        
        # 显示文件大小
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "文件大小: $FILE_SIZE"
    fi
    
    # S3同步
    if [ "$S3_SYNC" = true ]; then
        echo ""
        echo "=== 开始S3同步 ==="
        
        # 检查S3同步脚本是否存在
        S3_SCRIPT="$(dirname "$0")/s3_sync.sh"
        if [ -f "$S3_SCRIPT" ]; then
            echo "使用S3同步脚本: $S3_SCRIPT"
            
            # 同步当前模型的结果
            MODEL_DIR="$OUTPUT_DIR/qwen${MODEL_SIZE,,}"
            if [ -d "$MODEL_DIR" ]; then
                echo "同步 $MODEL_DIR 到 S3..."
                bash "$S3_SCRIPT" upload
            else
                echo "警告: 模型目录 $MODEL_DIR 不存在"
            fi
        else
            echo "错误: S3同步脚本不存在: $S3_SCRIPT"
            echo "请确保 s3_sync.sh 在 scripts 目录中"
        fi
    fi
else
    echo "实验失败，请检查错误信息"
    exit 1
fi
