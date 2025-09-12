# File Descriptions

## 📁 Directory Structure

### src/ Source Code Directory

#### src/generators/ Image Generators
- **grid.py** - Grid counting task image generator
  - Function: Generate XO grid images for counting tasks
  - Input: Shape, size, density parameters
  - Output: Grid images and correct answers

- **gabor.py** - Gabor texture task image generator  
  - Function: Generate Gabor texture images for texture recognition tasks
  - Input: Orientation, frequency, contrast parameters
  - Output: Texture images and correct answers

- **color_shading.py** - Color brightness task image generator
  - Function: Generate color brightness comparison images
  - Input: Color, brightness, contrast parameters
  - Output: Color images and correct answers

#### src/analysis/ Analysis Scripts
- **accuracy_analysis.py** - Accuracy analysis script
  - Function: Calculate model accuracy, confidence distribution
  - Input: Model result JSONL files
  - Output: Analysis reports and CSV files

- **comparison_analysis.py** - Comparison analysis script
  - Function: Detailed comparison between model choices and correct answers
  - Input: Model results and correct answers
  - Output: Question-by-question comparison JSON files

#### src/experiments/ Experiment Scripts
- **full_pipeline.py** - Complete experiment pipeline
  - Function: End-to-end experiment process management
  - Support: Data generation, model inference, result analysis
  - Commands: `gen`, `pack`, `feed`, `collect`

- **model_loading_test.py** - Model loading test
  - Function: Test loading of different models and memory usage
  - Purpose: Verify environment configuration and model compatibility

### data/ Data Directory

#### data/raw/ Raw Data
- **catalog_2688_newspec_correctly_fixed.json** - Question catalog file
  - Contains: Complete definitions for 2688 questions
  - Format: JSON, includes Grid, Gabor, Color task types

- **ground_truth.json** - Correct answers file
  - Contains: Correct answers for all questions
  - Format: JSON, key-value pair mapping

#### data/processed/ Processed Data
- **questions_two_stage_concise.jsonl** - Two-stage question dataset
  - Contains: 5376 two-stage questions (2 stages per question)
  - Format: JSONL, one JSON object per line

- **questions_two_stage_enhanced.jsonl** - Enhanced question dataset
  - Contains: Question data with additional information
  - Format: JSONL

#### data/results/ Experiment Results
- **qwen7/** - Qwen2.5-VL-7B experiment results
  - **qwen2.5-vl-7b_full_results.jsonl** - 7B model complete inference results
  - **qwen7b_simplified_comparison.json** - Simplified comparison analysis
  - **qwen7b_detailed_analysis_corrected.csv** - Detailed analysis CSV

- **qwen32/** - Qwen2.5-VL-32B experiment results
  - Contains: 32B model inference scripts and preparation files
  - To run: 32B model result files

### scripts/ Run Scripts
- **run_metacognition_experiments.sh** - Experiment run script
  - Function: Automated running of different model experiments
  - Support: Parameter configuration, batch processing, progress monitoring

- **monitor_progress.sh** - Progress monitoring script
  - Function: Monitor progress of long-running experiments
  - Purpose: Track file size, processing progress

## 🔧 Core Function Modules

### 1. Image Generation Module
```python
# Usage example
from src.generators.grid import generate as generate_grid
from src.generators.gabor import generate as generate_gabor
from src.generators.color_shading import generate as generate_color

# Generate grid image
grid_image = generate_grid(params, output_dir)

# Generate Gabor texture
gabor_image = generate_gabor(params, output_dir)

# Generate color image
color_image = generate_color(params, output_dir)
```

### 2. Model Inference Module
```python
# Usage example
from qwen2_vl_metacognition_multi_model import Qwen2VLMetacognitionInference

# Initialize inference engine
inference = Qwen2VLMetacognitionInference("Qwen/Qwen2.5-VL-7B-Instruct")

# Process questions
results = inference.process_questions_metacognition(questions)
```

### 3. Analysis Module
```python
# Usage example
from src.analysis.accuracy_analysis import analyze_accuracy
from src.analysis.comparison_analysis import create_comparison

# Analyze accuracy
accuracy_stats = analyze_accuracy(model_results, ground_truth)

# Create comparison analysis
comparison_data = create_comparison(model_results, ground_truth)
```

## 📊 Data Format Specifications

### Question Data Format (JSONL)
```json
{
  "qid": "GRID-0001_stage1",
  "task": "Grid",
  "image_path": "cache/images/GRID-0001.png",
  "question": "Which one has more: O or X?",
  "choices": ["A. O", "B. X"],
  "stage": 1
}
```

### Model Result Format (JSONL)
```json
{
  "qid": "GRID-0001_stage1",
  "task": "Grid",
  "choice": "A",
  "confidence": 3,
  "probabilities": {"A": 0.621, "B": 0.379},
  "latency_ms": 1021.56
}
```

### Comparison Analysis Format (JSON)
```json
{
  "question_id": "GRID-0001",
  "task_type": "Grid",
  "model_choice": "A",
  "correct_answer": "A",
  "is_correct": true,
  "confidence": 3
}
```

## 🚀 Usage Workflow

### 1. Data Preparation
```bash
# Generate images
python src/experiments/full_pipeline.py gen

# Generate questions
python src/experiments/full_pipeline.py pack
```

### 2. Model Inference
```bash
# Run 7B model
cd data/results/qwen7
python qwen2_vl_metacognition_multi_model.py --model Qwen/Qwen2.5-VL-7B-Instruct --questions ../../processed/questions_two_stage_concise.jsonl --output results.jsonl
```

### 3. Result Analysis
```bash
# Analyze accuracy
python src/analysis/accuracy_analysis.py

# Detailed comparison
python src/analysis/comparison_analysis.py
```

## ⚠️ Important Notes

1. **Memory Requirements**: 7B model needs ~8GB VRAM, 32B model needs ~24GB VRAM
2. **File Paths**: Ensure all paths are correct, especially image paths
3. **Batch Processing**: Large datasets recommend using batch processing to avoid memory overflow
4. **Result Saving**: Long-running experiments recommend regular saving of intermediate results