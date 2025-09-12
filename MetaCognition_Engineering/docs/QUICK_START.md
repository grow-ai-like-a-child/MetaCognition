# Quick Start Guide

## 🚀 5-Minute Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Check GPU status
nvidia-smi
```

### 2. Run 7B Model Experiment
```bash
# Navigate to 7B results directory
cd data/results/qwen7

# Run experiment (if no result files exist yet)
python qwen2_vl_metacognition_multi_model.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --questions ../../processed/questions_two_stage_concise.jsonl \
  --output qwen2.5-vl-7b_results.jsonl

# Analyze results
python ../../../src/analysis/accuracy_analysis.py --results qwen2.5-vl-7b_results.jsonl
```

### 3. View Results
```bash
# View simplified comparison results
cat qwen7b_simplified_comparison.json | head -10

# View detailed analysis CSV
head qwen7b_detailed_analysis_corrected.csv
```

## 📊 Expected Results

### Qwen2.5-VL-7B Performance
- **Overall Accuracy**: ~57%
- **Grid Task**: ~94% ✅
- **Gabor Task**: ~38% ⚠️
- **Color Task**: ~39% ⚠️

## 🔧 Common Issues

### Q: Out of memory error?
A: Reduce batch_size or use a smaller model

### Q: How to run 32B model?
A: Ensure 24GB+ VRAM, then:
```bash
cd data/results/qwen32
python qwen2_vl_metacognition_multi_model.py --model Qwen/Qwen2.5-VL-32B-Instruct --questions ../../processed/questions_two_stage_concise.jsonl --output qwen2.5-vl-32b_results.jsonl
```

### Q: How to generate new image data?
A: Use the full pipeline:
```bash
python src/experiments/full_pipeline.py gen
python src/experiments/full_pipeline.py pack
```

## 📈 Next Steps

1. View detailed documentation: [FILE_DESCRIPTIONS.md](FILE_DESCRIPTIONS.md)
2. Run 32B model experiments
3. Compare different model performances
4. Analyze the relationship between confidence and accuracy