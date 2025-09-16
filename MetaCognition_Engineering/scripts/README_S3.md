# AWS S3 数据同步管道

这个目录包含了用于管理MetaCognition实验数据的AWS S3同步管道。

## 文件说明

### 核心脚本

1. **`s3_sync.sh`** - S3同步核心脚本
   - 处理数据的上传、下载和同步
   - 支持结果数据和原始数据的同步
   - 包含AWS CLI配置和验证

2. **`run_metacognition_experiments.sh`** - 增强的实验运行脚本
   - 添加了S3同步选项
   - 支持在实验完成后自动同步结果

3. **`run_all_models_with_s3.sh`** - 完整管道脚本
   - 运行所有模型 (7B, 32B, 72B)
   - 自动处理S3同步
   - 支持批量操作和错误处理

4. **`quick_start_s3.sh`** - 快速启动脚本
   - 一键安装和配置AWS CLI
   - 启动完整实验管道

## 使用方法

### 1. 快速开始

```bash
# 一键启动（推荐）
./scripts/quick_start_s3.sh
```

### 2. 手动配置

```bash
# 1. 配置AWS CLI
aws configure set aws_access_key_id YOUR_AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key YOUR_AWS_SECRET_ACCESS_KEY

# 2. 运行单个模型实验并同步
./scripts/run_metacognition_experiments.sh --model 7B --s3-sync

# 3. 运行所有模型并同步
./scripts/run_all_models_with_s3.sh
```

### 3. S3同步操作

```bash
# 上传数据到S3
./scripts/s3_sync.sh upload

# 从S3下载数据
./scripts/s3_sync.sh download

# 查看S3状态
./scripts/s3_sync.sh status

# 清理本地缓存
./scripts/s3_sync.sh cleanup
```

## 配置说明

### S3存储桶结构

```
s3://perceptualmetacognition/
├── qwen7/                    # 7B模型结果
│   ├── qwen2.5-vl-7b_full_results.jsonl
│   ├── qwen7b_results_detailes.json
│   └── ...
├── qwen32/                   # 32B模型结果
│   ├── qwen2.5-vl-32b_full_results.jsonl
│   ├── qwen32b_results_detailes.json
│   └── ...
├── qwen72b/                  # 72B模型结果
│   ├── qwen2.5-vl-72b_full_results.jsonl
│   ├── qwen72b_comparison_with_ground_truth.json
│   └── ...
└── data/                     # 原始数据
    ├── raw/
    └── prompt/
```

### 环境变量

可以通过以下环境变量自定义配置：

```bash
export S3_BUCKET="s3://your-bucket/"
export OUTPUT_DIR="your_results"
export QUESTIONS_FILE="your_questions.jsonl"
```

## 高级用法

### 1. 选择性运行模型

```bash
# 只运行特定模型
./scripts/run_all_models_with_s3.sh --models 7B,32B
```

### 2. 跳过某些步骤

```bash
# 跳过从S3下载
./scripts/run_all_models_with_s3.sh --skip-download

# 跳过上传到S3
./scripts/run_all_models_with_s3.sh --skip-upload
```

### 3. 状态检查

```bash
# 只查看S3状态
./scripts/run_all_models_with_s3.sh --status-only

# 只清理本地数据
./scripts/run_all_models_with_s3.sh --cleanup-only
```

## 故障排除

### 1. AWS CLI问题

```bash
# 检查AWS配置
aws sts get-caller-identity

# 重新配置
aws configure
```

### 2. 权限问题

确保AWS凭据有足够的权限：
- `s3:GetObject`
- `s3:PutObject`
- `s3:DeleteObject`
- `s3:ListBucket`

### 3. 网络问题

```bash
# 检查网络连接
aws s3 ls s3://perceptualmetacognition/

# 使用代理（如果需要）
export AWS_PROXY=http://proxy:port
```

## 监控和日志

### 日志文件

- `s3_sync.log` - S3同步操作日志
- `qwen72b_full_run.log` - 72B模型运行日志
- 其他模型特定的日志文件

### 进度监控

```bash
# 监控S3同步进度
tail -f s3_sync.log

# 监控实验进度
./scripts/monitor_progress.sh
```

## 最佳实践

1. **定期同步**: 建议在每次实验后立即同步到S3
2. **备份重要数据**: 重要结果建议本地和S3双重备份
3. **监控存储使用**: 定期检查S3存储使用情况
4. **版本控制**: 使用时间戳标记不同版本的结果
5. **错误处理**: 注意监控同步过程中的错误

## 支持

如果遇到问题，请检查：

1. AWS CLI是否正确安装和配置
2. 网络连接是否正常
3. S3存储桶权限是否正确
4. 本地磁盘空间是否充足

更多帮助信息：

```bash
./scripts/s3_sync.sh help
./scripts/run_all_models_with_s3.sh --help
./scripts/run_metacognition_experiments.sh --help
```
