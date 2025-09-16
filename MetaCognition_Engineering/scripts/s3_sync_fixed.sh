#!/bin/bash
# AWS S3 数据同步脚本
# 用于管理元认知实验数据的同步

# 配置变量
S3_BUCKET="s3://perceptualmetacognition/"
LOCAL_RESULTS_DIR="results"
LOCAL_DATA_DIR="data"
LOG_FILE="s3_sync.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $1" | tee -a "$LOG_FILE"
}

# 检查AWS CLI是否安装
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI 未安装。正在安装..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            sudo apt-get install -y awscli
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install awscli
            else
                log_error "请先安装 Homebrew 或手动安装 AWS CLI"
                exit 1
            fi
        else
            log_error "不支持的操作系统。请手动安装 AWS CLI"
            exit 1
        fi
    else
        log_success "AWS CLI 已安装"
    fi
}

# 配置AWS凭据
configure_aws() {
    log "配置 AWS 凭据..."
    aws configure set aws_access_key_id YOUR_AWS_ACCESS_KEY_ID
    aws configure set aws_secret_access_key YOUR_AWS_SECRET_ACCESS_KEY
    aws configure set default.region us-east-1
    aws configure set default.output json
    
    # 验证配置
    if aws sts get-caller-identity &> /dev/null; then
        log_success "AWS 配置验证成功"
    else
        log_error "AWS 配置验证失败"
        exit 1
    fi
}

# 从S3下载数据
download_from_s3() {
    local s3_path="$1"
    local local_path="$2"
    
    log "从 S3 下载数据: $s3_path -> $local_path"
    
    if aws s3 cp "$s3_path" "$local_path" --recursive; then
        log_success "下载完成: $s3_path"
    else
        log_error "下载失败: $s3_path"
        return 1
    fi
}

# 上传数据到S3
upload_to_s3() {
    local local_path="$1"
    local s3_path="$2"
    
    log "上传数据到 S3: $local_path -> $s3_path"
    
    if aws s3 sync "$local_path" "$s3_path" --delete; then
        log_success "上传完成: $s3_path"
    else
        log_error "上传失败: $s3_path"
        return 1
    fi
}

# 同步结果数据
sync_results() {
    log "开始同步结果数据..."
    
    # 同步各个模型的结果
    for model_dir in qwen7 qwen32 qwen72b; do
        if [ -d "$LOCAL_RESULTS_DIR/$model_dir" ]; then
            log "同步 $model_dir 结果..."
            upload_to_s3 "$LOCAL_RESULTS_DIR/$model_dir" "$S3_BUCKET$model_dir/"
        else
            log_warning "目录不存在: $LOCAL_RESULTS_DIR/$model_dir"
        fi
    done
}

# 同步原始数据
sync_data() {
    log "开始同步原始数据..."
    
    if [ -d "$LOCAL_DATA_DIR" ]; then
        upload_to_s3 "$LOCAL_DATA_DIR" "${S3_BUCKET}data/"
    else
        log_warning "数据目录不存在: $LOCAL_DATA_DIR"
    fi
}

# 从S3恢复数据
restore_from_s3() {
    log "从 S3 恢复数据..."
    
    # 恢复结果数据
    for model_dir in qwen7 qwen32 qwen72b; do
        log "恢复 $model_dir 结果..."
        download_from_s3 "$S3_BUCKET$model_dir/" "$LOCAL_RESULTS_DIR/$model_dir/"
    done
    
    # 恢复原始数据
    log "恢复原始数据..."
    download_from_s3 "${S3_BUCKET}data/" "$LOCAL_DATA_DIR/"
}

# 显示S3存储状态
show_s3_status() {
    log "S3 存储状态:"
    aws s3 ls "$S3_BUCKET" --recursive --human-readable --summarize
}

# 清理本地缓存
cleanup_local() {
    log "清理本地缓存..."
    
    # 清理Python缓存
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # 清理日志文件
    find . -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    log_success "本地缓存清理完成"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  sync     同步数据到S3 [默认]"
    echo "  upload   上传数据到S3"
    echo "  download 从S3下载数据"
    echo "  status   显示S3存储状态"
    echo "  cleanup  清理本地缓存"
    echo "  help     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 sync      # 同步所有数据到S3"
    echo "  $0 download  # 从S3恢复数据"
    echo "  $0 status    # 查看S3状态"
}

# 主函数
main() {
    echo "=== AWS S3 数据同步工具 ==="
    echo ""
    
    # 检查参数
    case "${1:-sync}" in
        "download"|"down")
            check_aws_cli
            configure_aws
            restore_from_s3
            ;;
        "upload"|"up")
            check_aws_cli
            configure_aws
            sync_results
            sync_data
            ;;
        "sync")
            check_aws_cli
            configure_aws
            sync_results
            sync_data
            ;;
        "status")
            check_aws_cli
            configure_aws
            show_s3_status
            ;;
        "cleanup")
            cleanup_local
            ;;
        "help"|"-h"|"--help")
            show_help
            exit 0
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
    
    log_success "操作完成"
}

# 运行主函数
main "$@"
