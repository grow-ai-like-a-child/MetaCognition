#!/bin/bash
# 监控Qwen 7B处理进度的脚本

echo "=== Qwen 7B 处理进度监控 ==="
echo "开始时间: $(date)"
echo ""

while true; do
    # 检查进程是否还在运行
    if ! ps -p 58256 > /dev/null 2>&1; then
        echo "程序已结束！"
        break
    fi
    
    # 检查输出文件
    if [ -f "qwen2.5-vl-7b_full_results.jsonl" ]; then
        lines=$(wc -l < qwen2.5-vl-7b_full_results.jsonl)
        size=$(du -h qwen2.5-vl-7b_full_results.jsonl | cut -f1)
        
        # 计算进度
        total_expected=10754  # 5377 * 2 (每个题目2个阶段)
        progress=$((lines * 100 / total_expected))
        
        echo "[$(date '+%H:%M:%S')] 进度: $lines/$total_expected ($progress%) - 文件大小: $size"
        
        # 显示最后处理的题目
        last_qid=$(tail -1 qwen2.5-vl-7b_full_results.jsonl | grep -o '"qid":"[^"]*"' | cut -d'"' -f4)
        echo "  当前处理: $last_qid"
        
    else
        echo "[$(date '+%H:%M:%S')] 等待输出文件生成..."
    fi
    
    echo ""
    sleep 60  # 每分钟检查一次
done

echo "监控结束: $(date)"
