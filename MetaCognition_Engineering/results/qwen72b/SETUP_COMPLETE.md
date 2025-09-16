# Qwen2.5-VL-72B 实验准备完成

## 📁 文件结构

```
qwen72b/
├── qwen2_vl_metacognition_72b.py    # 主推理脚本
├── test_qwen72b_loading.py          # 模型加载测试脚本
├── run_qwen72b_experiment.sh        # 实验运行脚本
├── README.md                         # 详细说明文档
├── gpu_config.txt                    # GPU配置信息
├── cpu_config.txt                    # CPU配置信息
└── SETUP_COMPLETE.md                 # 本文件
```

## 🚀 快速开始

### 1. 测试模型加载
```bash
cd data/results/qwen72b
python3 test_qwen72b_loading.py
```

### 2. 运行完整实验
```bash
cd data/results/qwen72b
./run_qwen72b_experiment.sh
```

### 3. 手动运行推理
```bash
cd data/results/qwen72b
python3 qwen2_vl_metacognition_72b.py \
    --questions ../../processed/questions_two_stage.jsonl \
    --output qwen2.5-vl-72b_full_results.jsonl
```

## ⚙️ 系统配置

### GPU配置
- **型号**: 2x NVIDIA H100 80GB HBM3
- **总显存**: 163GB
- **推荐设置**: device="auto" (自动使用双GPU)

### CPU配置
- **型号**: Intel Xeon Platinum 8480+
- **核心数**: 52核 (26核 x 2线程)
- **内存**: 建议128GB+

## 📊 预期性能

- **推理速度**: 4-8 tokens/秒 (双GPU)
- **总运行时间**: 6-12小时
- **内存使用**: ~80-90% 单GPU显存
- **输出文件**: ~500MB-1GB

## 🔧 故障排除

### 常见问题
1. **内存不足**: 使用单GPU模式 `--device cuda:0`
2. **模型下载失败**: 检查网络连接和Hugging Face访问
3. **依赖缺失**: 安装 `transformers`, `torch`, `qwen-vl-utils`

### 监控命令
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控系统资源
htop

# 检查进程
ps aux | grep python
```

## 📝 注意事项

1. **长时间运行**: 建议使用 `screen` 或 `tmux` 保持会话
2. **磁盘空间**: 确保有足够空间存储模型和结果
3. **网络稳定**: 模型首次下载需要稳定网络
4. **定期检查**: 长时间运行建议定期检查进度

## 🎯 下一步

1. 运行测试脚本验证环境
2. 开始完整实验
3. 监控运行状态
4. 分析实验结果

---
**准备完成时间**: $(date)
**系统**: Ubuntu 22.04 LTS
**Python**: 3.10+
**PyTorch**: 2.0+
