#!/bin/bash
# 运行所有模型并同步到S3的完整管道脚本

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QUESTIONS_FILE="data/prompt/questions_two_stage_concise.jsonl"
OUTPUT_DIR="results"
S3_BUCKET="s3://perceptualmetacognition/"

# 支持的模型列表
MODELS=("7B" "32B" "72B")

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

log_model() {
    echo -e "${PURPLE}[$(date '+%Y-%m-%d %H:%M:%S')] 🤖${NC} $1"
}

# 检查依赖
check_dependencies() {
    log "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI 未安装。请先运行: ./scripts/s3_sync.sh help"
        exit 1
    fi
    
    # 检查实验脚本
    if [ ! -f "$SCRIPT_DIR/run_metacognition_experiments.sh" ]; then
        log_error "实验脚本不存在: $SCRIPT_DIR/run_metacognition_experiments.sh"
        exit 1
    fi
    
    # 检查S3同步脚本
    if [ ! -f "$SCRIPT_DIR/s3_sync.sh" ]; then
        log_error "S3同步脚本不存在: $SCRIPT_DIR/s3_sync.sh"
        exit 1
    fi
    
    log_success "所有依赖检查通过"
}

# 配置AWS
configure_aws() {
    log "配置AWS..."
    bash "$SCRIPT_DIR/s3_sync.sh" download
    if [ $? -eq 0 ]; then
        log_success "AWS配置完成"
    else
        log_error "AWS配置失败"
        exit 1
    fi
}

# 运行单个模型实验
run_model_experiment() {
    local model_size="$1"
    
    log_model "开始运行 $model_size 模型实验..."
    
    # 运行实验
    bash "$SCRIPT_DIR/run_metacognition_experiments.sh" \
        --model "$model_size" \
        --questions "$QUESTIONS_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --s3-sync \
        --s3-bucket "$S3_BUCKET"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "$model_size 模型实验完成"
        return 0
    else
        log_error "$model_size 模型实验失败 (退出码: $exit_code)"
        return $exit_code
    fi
}

# 运行所有模型
run_all_models() {
    log "开始运行所有模型实验..."
    
    local failed_models=()
    local successful_models=()
    
    for model in "${MODELS[@]}"; do
        log_model "准备运行 $model 模型..."
        
        if run_model_experiment "$model"; then
            successful_models+=("$model")
        else
            failed_models+=("$model")
        fi
        
        # 模型间休息，避免资源冲突
        if [ "$model" != "${MODELS[-1]}" ]; then
            log "等待30秒后继续下一个模型..."
            sleep 30
        fi
    done
    
    # 报告结果
    echo ""
    log "=== 实验完成报告 ==="
    
    if [ ${#successful_models[@]} -gt 0 ]; then
        log_success "成功完成的模型: ${successful_models[*]}"
    fi
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        log_error "失败的模型: ${failed_models[*]}"
        return 1
    fi
    
    return 0
}

# 同步所有数据到S3
sync_all_to_s3() {
    log "同步所有数据到S3..."
    
    bash "$SCRIPT_DIR/s3_sync.sh" upload
    
    if [ $? -eq 0 ]; then
        log_success "所有数据已同步到S3"
    else
        log_error "S3同步失败"
        return 1
    fi
}

# 显示S3状态
show_s3_status() {
    log "显示S3存储状态..."
    bash "$SCRIPT_DIR/s3_sync.sh" status
}

# 清理本地数据
cleanup_local() {
    log "清理本地数据..."
    bash "$SCRIPT_DIR/s3_sync.sh" cleanup
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --models LIST       指定要运行的模型列表 (用逗号分隔) [默认: 7B,32B,72B]"
    echo "  --questions FILE    题目文件路径 [默认: data/prompt/questions_two_stage_concise.jsonl]"
    echo "  --output-dir DIR    输出目录 [默认: results]"
    echo "  --s3-bucket BUCKET  S3存储桶地址 [默认: s3://perceptualmetacognition/]"
    echo "  --skip-download     跳过从S3下载数据"
    echo "  --skip-upload       跳过上传到S3"
    echo "  --status-only       只显示S3状态"
    echo "  --cleanup-only      只清理本地数据"
    echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 运行所有模型并同步到S3"
    echo "  $0 --models 7B,32B                   # 只运行7B和32B模型"
    echo "  $0 --skip-download                   # 跳过下载，直接运行实验"
    echo "  $0 --status-only                     # 只查看S3状态"
    echo "  $0 --cleanup-only                    # 只清理本地数据"
}

# 解析命令行参数
parse_arguments() {
    SKIP_DOWNLOAD=false
    SKIP_UPLOAD=false
    STATUS_ONLY=false
    CLEANUP_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --models)
                IFS=',' read -ra MODELS <<< "$2"
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
            --s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --skip-upload)
                SKIP_UPLOAD=true
                shift
                ;;
            --status-only)
                STATUS_ONLY=true
                shift
                ;;
            --cleanup-only)
                CLEANUP_ONLY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    echo "=== MetaCognition 全模型实验管道 ==="
    echo "S3存储桶: $S3_BUCKET"
    echo "输出目录: $OUTPUT_DIR"
    echo "题目文件: $QUESTIONS_FILE"
    echo "模型列表: ${MODELS[*]}"
    echo ""
    
    # 解析参数
    parse_arguments "$@"
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT" || exit 1
    
    # 状态检查
    if [ "$STATUS_ONLY" = true ]; then
        check_dependencies
        show_s3_status
        exit 0
    fi
    
    # 清理模式
    if [ "$CLEANUP_ONLY" = true ]; then
        cleanup_local
        exit 0
    fi
    
    # 检查依赖
    check_dependencies
    
    # 下载数据（如果需要）
    if [ "$SKIP_DOWNLOAD" = false ]; then
        configure_aws
    else
        log_warning "跳过从S3下载数据"
    fi
    
    # 运行所有模型
    if run_all_models; then
        log_success "所有模型实验完成"
        
        # 上传到S3（如果需要）
        if [ "$SKIP_UPLOAD" = false ]; then
            sync_all_to_s3
        else
            log_warning "跳过上传到S3"
        fi
        
        # 显示最终状态
        show_s3_status
        
        log_success "管道执行完成！"
    else
        log_error "部分模型实验失败"
        exit 1
    fi
}

# 运行主函数
main "$@"
