# MetaCognition S3 同步管道使用指南

## 🚀 快速开始

### 方法1: 一键启动（推荐）
```bash
./scripts/quick_start_s3.sh
```

### 方法2: 手动步骤
```bash
# 1. 配置AWS（已自动完成）
aws configure set aws_access_key_id YOUR_AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key YOUR_AWS_SECRET_ACCESS_KEY

# 2. 运行所有模型实验并同步到S3
./scripts/run_all_models_with_s3.sh
```

## 📋 可用脚本

### 1. 核心S3同步脚本
```bash
./scripts/s3_sync.sh [命令]
```
- `sync` - 同步所有数据到S3（默认）
- `upload` - 上传数据到S3
- `download` - 从S3下载数据
- `status` - 显示S3存储状态
- `cleanup` - 清理本地缓存
- `help` - 显示帮助信息

### 2. 单模型实验脚本
```bash
./scripts/run_metacognition_experiments.sh [选项]
```
- `--model SIZE` - 模型大小 (7B, 32B, 72B)
- `--s3-sync` - 启用S3同步
- `--s3-bucket BUCKET` - 指定S3存储桶

### 3. 全模型管道脚本
```bash
./scripts/run_all_models_with_s3.sh [选项]
```
- `--models LIST` - 指定模型列表 (用逗号分隔)
- `--skip-download` - 跳过从S3下载
- `--skip-upload` - 跳过上传到S3
- `--status-only` - 只显示S3状态

## 🎯 使用场景

### 场景1: 运行所有模型实验
```bash
# 运行所有模型 (7B, 32B, 72B) 并同步到S3
./scripts/run_all_models_with_s3.sh
```

### 场景2: 运行特定模型
```bash
# 只运行7B和32B模型
./scripts/run_all_models_with_s3.sh --models 7B,32B

# 运行单个模型并同步
./scripts/run_metacognition_experiments.sh --model 72B --s3-sync
```

### 场景3: 数据管理
```bash
# 查看S3状态
./scripts/s3_sync.sh status

# 从S3恢复数据
./scripts/s3_sync.sh download

# 清理本地缓存
./scripts/s3_sync.sh cleanup
```

### 场景4: 状态检查
```bash
# 只检查S3状态，不运行实验
./scripts/run_all_models_with_s3.sh --status-only
```

## 📁 S3存储结构

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

## ⚙️ 配置选项

### 环境变量
```bash
export S3_BUCKET="s3://your-bucket/"
export OUTPUT_DIR="your_results"
export QUESTIONS_FILE="your_questions.jsonl"
```

### 自定义配置
```bash
# 使用自定义S3存储桶
./scripts/run_all_models_with_s3.sh --s3-bucket s3://my-bucket/

# 使用自定义输出目录
./scripts/run_metacognition_experiments.sh --output-dir my_results --s3-sync
```

## 🔧 故障排除

### 1. AWS CLI问题
```bash
# 检查AWS配置
aws sts get-caller-identity

# 重新配置
aws configure
```

### 2. 权限问题
确保AWS凭据有足够权限：
- `s3:GetObject`
- `s3:PutObject`
- `s3:DeleteObject`
- `s3:ListBucket`

### 3. 网络问题
```bash
# 检查S3连接
aws s3 ls s3://perceptualmetacognition/
```

## 📊 监控和日志

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

## 🎉 完成！

现在您可以使用以下命令开始：

```bash
# 一键启动所有实验
./scripts/quick_start_s3.sh
```

或者

```bash
# 手动运行所有模型
./scripts/run_all_models_with_s3.sh
```

所有结果将自动同步到S3存储桶 `s3://perceptualmetacognition/`！
