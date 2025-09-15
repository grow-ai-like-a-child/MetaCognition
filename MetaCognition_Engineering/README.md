# MetaCognition Engineering

A research project for testing the performance of multimodal large language models (Qwen2.5-VL) on metacognitive reasoning tasks.

## 📋 Project Overview

This project evaluates Qwen2.5-VL models' metacognitive reasoning capabilities through three visual tasks (Grid, Gabor, Color), including:
- **Task Execution Ability**: Model's understanding and answering of visual tasks
- **Metacognitive Ability**: Model's confidence assessment of its own answers
- **Two-stage Reasoning**: Answer questions first, then assess confidence

## 🏗️ Project Structure

```
MetaCognition_Engineering/
├── src/                          # Source code
│   ├── generators/               # Image generators
│   │   ├── grid.py              # Grid counting task generator
│   │   ├── gabor.py             # Texture recognition task generator
│   │   └── color_shading.py     # Color brightness task generator
│   ├── analysis/                 # Analysis scripts
│   │   ├── accuracy_analysis.py # Accuracy analysis
│   │   └── comparison_analysis.py # Comparison analysis
│   └── experiments/              # Experiment scripts
│       ├── full_pipeline.py     # Complete experiment pipeline
│       └── model_loading_test.py # Model loading test
├── data/                         # Data files
│   ├── raw/                     # Raw data
│   │   ├── catalog_2688_newspec_correctly_fixed.json
│   │   └── ground_truth.json
│   ├── processed/               # Processed data
│   │   ├── questions_two_stage_concise.jsonl
│   │   └── questions_two_stage_enhanced.jsonl
│   └── results/                 # Experiment results
│       ├── qwen7/              # Qwen2.5-VL-7B results
│       ├── qwen32/             # Qwen2.5-VL-32B results
│       └── qwen72b/            # Qwen2.5-VL-72B results
├── scripts/                      # Run scripts
│   ├── run_metacognition_experiments.sh
│   └── monitor_progress.sh
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
└── README.md                    # Project description
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd MetaCognition_Engineering

# Install dependencies
pip install -r requirements.txt

# Ensure sufficient GPU memory
# 7B model requires ~8GB VRAM
# 32B model requires ~24GB VRAM
```

### 2. Generate Data

```bash
# Generate image data
python src/experiments/full_pipeline.py gen --catalog data/raw/catalog_2688_newspec_correctly_fixed.json --imgroot cache/images

# Generate question data
python src/experiments/full_pipeline.py pack --catalog data/raw/catalog_2688_newspec_correctly_fixed.json --imgroot cache/images --out data/processed/questions_two_stage_concise.jsonl
```

### 3. Run Experiments

```bash
# Run 7B model experiment
cd data/results/qwen7
python qwen2_vl_metacognition_multi_model.py --model Qwen/Qwen2.5-VL-7B-Instruct --questions ../../processed/questions_two_stage_concise.jsonl --output qwen2.5-vl-7b_results.jsonl

# Run 32B model experiment
cd data/results/qwen32
python qwen2_vl_metacognition_multi_model.py --model Qwen/Qwen2.5-VL-32B-Instruct --questions ../../processed/questions_two_stage_concise.jsonl --output qwen2.5-vl-32b_results.jsonl

# Run 72B model experiment
cd data/results/qwen72b
python qwen2_vl_metacognition_72b_realtime.py --model Qwen/Qwen2.5-VL-72B-Instruct --questions ../../prompt/questions_two_stage_concise.jsonl --output qwen2.5-vl-72b_results.jsonl
```

### 4. Analyze Results

```bash
# Analyze accuracy
python src/analysis/accuracy_analysis.py --results qwen2.5-vl-7b_results.jsonl --output analysis.csv

# Detailed comparison analysis
python src/analysis/comparison_analysis.py --results qwen2.5-vl-7b_results.jsonl --output comparison.json
```

## 📊 Experimental Results

### Qwen2.5-VL-7B Performance
- **Overall Accuracy**: 57.07% (1,534/2,688)
- **Grid Task**: 94.31% (845/896) ✅ Excellent
- **Gabor Task**: 37.95% (340/896) ⚠️ Fair
- **Color Task**: 38.95% (349/896) ⚠️ Fair

### Qwen2.5-VL-72B Performance
- **Overall Accuracy**: 56.77% (1,526/2,688)
- **Grid Task**: 94.98% (1,512/1,592) ✅ Excellent
- **Gabor Task**: 35.94% (320/891) ⚠️ Poor
- **Color Task**: 39.40% (344/872) ⚠️ Poor

### Confidence Distribution Comparison
- **7B Model**: Mainly confidence level 3, average 3.13
- **72B Model**: Overconfident, 98.62% at confidence level 5, average 4.99
- **Metacognitive Ability**: Both models show good calibration (higher confidence = higher accuracy)

## 🔍 Key Findings

### Model Size Impact
- **Overall Performance**: 72B model shows similar overall accuracy to 7B model (56.77% vs 57.07%)
- **Task-Specific Performance**: 72B model performs slightly better on Grid tasks but worse on Gabor tasks
- **Confidence Patterns**: 72B model is significantly more overconfident than 7B model

### Metacognitive Insights
- **Calibration Quality**: Both models demonstrate good metacognitive calibration
- **Confidence Distribution**: 7B model shows more conservative confidence, 72B model shows overconfidence
- **Task Difficulty Recognition**: Both models struggle with Gabor and Color tasks regardless of size

### Technical Improvements
- **Real-time Writing**: Implemented to prevent data loss during long experiments
- **Comprehensive Analysis**: Added multiple analysis tools for detailed performance evaluation
- **Error Handling**: Improved data format consistency and QID matching

## 🔧 Core Components

### Image Generators (src/generators/)
- **grid.py**: Generates XO grid counting task images
- **gabor.py**: Generates Gabor texture recognition task images  
- **color_shading.py**: Generates color brightness comparison task images

### Analysis Tools (src/analysis/)
- **accuracy_analysis.py**: Calculates model accuracy and confidence distribution
- **comparison_analysis.py**: Detailed comparison between model choices and correct answers

### Experiment Pipeline (src/experiments/)
- **full_pipeline.py**: Complete experiment pipeline including data generation, model inference, and result analysis
- **model_loading_test.py**: Tests loading of different models and memory usage

## 📈 Task Types

### 1. Grid Task (Grid Counting)
- **Description**: Compare the number of O's and X's in a grid
- **Difficulty**: Simple geometric counting
- **7B Performance**: 94.31% ✅
- **72B Performance**: 94.98% ✅

### 2. Gabor Task (Texture Recognition)  
- **Description**: Recognize Gabor texture orientation and frequency
- **Difficulty**: Complex texture feature recognition
- **7B Performance**: 37.95% ⚠️
- **72B Performance**: 35.94% ⚠️

### 3. Color Task (Color Brightness)
- **Description**: Compare color brightness levels
- **Difficulty**: Color perception and comparison
- **7B Performance**: 38.95% ⚠️
- **72B Performance**: 39.40% ⚠️

## 🛠️ Development Guide

### Adding New Image Generators
1. Create a new generator file in `src/generators/`
2. Implement `generate(params, out_dir)` function
3. Implement `ground_truth(params)` function
4. Register the new generator in `full_pipeline.py`

### Adding New Analysis Metrics
1. Create a new analysis script in `src/analysis/`
2. Implement corresponding analysis functions
3. Update the analysis pipeline in `full_pipeline.py`

## 📝 File Naming Conventions

- **Generators**: `{task_name}.py` (e.g., `grid.py`, `gabor.py`)
- **Analysis scripts**: `{analysis_type}_analysis.py` (e.g., `accuracy_analysis.py`)
- **Experiment scripts**: `{purpose}.py` (e.g., `full_pipeline.py`)
- **Data files**: `{description}.{format}` (e.g., `questions_two_stage_concise.jsonl`)
- **Result files**: `{model_name}_{experiment_type}.{format}` (e.g., `qwen2.5-vl-7b_results.jsonl`)

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📞 Contact

For questions or suggestions, please contact:
- Create an Issue
- Send email to [your-email@example.com]

## 🙏 Acknowledgments

- Thanks to the Qwen team for providing excellent multimodal models
- Thanks to all contributors for their efforts