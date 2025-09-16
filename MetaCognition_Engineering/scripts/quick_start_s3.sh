#!/bin/bash
# 快速启动S3同步管道

echo "=== MetaCognition S3 快速启动 ==="
echo ""

# 检查AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装"
    echo "正在安装 AWS CLI..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y awscli
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install awscli
        else
            echo "请先安装 Homebrew 或手动安装 AWS CLI"
            exit 1
        fi
    else
        echo "不支持的操作系统。请手动安装 AWS CLI"
        exit 1
    fi
fi

# 配置AWS
echo "🔧 配置 AWS 凭据..."
aws configure set aws_access_key_id YOUR_AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key YOUR_AWS_SECRET_ACCESS_KEY
aws configure set default.region us-east-1
aws configure set default.output json

# 验证配置
if aws sts get-caller-identity &> /dev/null; then
    echo "✅ AWS 配置成功"
else
    echo "❌ AWS 配置失败"
    exit 1
fi

echo ""
echo "🚀 启动完整实验管道..."
echo "这将运行所有模型 (7B, 32B, 72B) 并同步到 S3"
echo ""

# 运行完整管道
bash "$(dirname "$0")/run_all_models_with_s3.sh"

echo ""
echo "🎉 完成！"
