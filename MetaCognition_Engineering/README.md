# Two-Stage Questioning Cognitive Experiment System

## 🎯 Project Overview

This is a cognitive experiment system that supports two-stage questioning, designed to study the performance and confidence assessment of humans and AI models in visual perception tasks. The system includes three types of cognitive tasks: grid symbol judgment, Gabor stripe orientation judgment, and color brightness difference judgment.

## ✨ Key Features

- **Two-Stage Questioning Design**: Complete experimental tasks first, then assess confidence
- **Multi-Task Support**: Grid, Gabor, and Color three cognitive tasks
- **Automated Workflow**: Complete pipeline from image generation to result evaluation
- **Flexible Configuration**: Support for both Mock and OpenAI API modes
- **Data Integrity**: Automatic generation, cache management, and result validation

## 🏗️ System Architecture

```
neweng/
├── cache/                                    # Image cache directory
│   └── images/
│       ├── Grid/                            # Grid task images
│       ├── Gabor/                           # Gabor task images
│       └── Color/                           # Color task images
├── generators/                               # Task generators
│   ├── grid.py                              # Grid task generator
│   ├── gabor.py                             # Gabor task generator
│   └── color_shading.py                     # Color task generator
├── catalog_runner_two_stage.py              # Main runner
└── catalog_2688_newspec_correctly_fixed.json # Task configuration data
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Required Python packages (see requirements.txt)
- Optional: OpenAI API key (for real model testing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Basic Usage Workflow

1. **Generate Experimental Images**
```bash
python catalog_runner_two_stage.py gen --catalog catalog_2688_newspec_correctly_fixed.json --imgroot cache/images
```

2. **Create Two-Stage Questions**
```bash
python catalog_runner_two_stage.py pack --catalog catalog_2688_newspec_correctly_fixed.json --imgroot cache/images --out questions_two_stage.jsonl
```

3. **Call Model (Mock Mode)**
```bash
python catalog_runner_two_stage.py feed --pack questions_two_stage.jsonl --out responses_two_stage.jsonl --engine mock --workers 8
```

4. **Collect Evaluation Results**
```bash
python catalog_runner_two_stage.py collect --resp responses_two_stage.jsonl --catalog catalog_2688_newspec_correctly_fixed.json --out eval_two_stage.csv
```

## 📊 Task Type Details

### Grid Task
- **Objective**: Determine which symbol appears more frequently in the grid
- **Parameters**: Grid level, shape, symbol set, color pair
- **Output**: A or B choice, confidence 1-5

### Gabor Task
- **Objective**: Determine whether stripe orientation is closer to vertical or horizontal
- **Parameters**: Gabor level, frequency, position
- **Output**: A (vertical) or B (horizontal), confidence 1-5

### Color Task
- **Objective**: Determine which side is brighter
- **Parameters**: Brightness difference, layout, font style
- **Output**: A or B choice, confidence 1-5

## 🔧 Two-Stage Questioning Design

### Stage 1: Experimental Task
- Pure cognitive task without confidence requirements
- Model focuses on the task itself, avoiding interference
- Generates A/B choice answers

### Stage 2: Confidence Assessment
- Assess confidence based on first stage answers
- 1-5 level confidence rating
- More natural cognitive process

## 📁 Output File Description

### questions_two_stage.jsonl
- Contains 5,376 questions (2,688 original questions × 2 stages)
- Each question has a stage field identifying the stage
- Connected through original_qid field

### responses_two_stage.jsonl
- Model responses to each question
- Contains choice, confidence, latency time, and other information

### eval_two_stage.csv
- Comprehensive evaluation results
- Contains complete information from both stages
- Task-specific parameters and correctness judgment

## ⚙️ Advanced Configuration

### Mock Mode
- Offline testing, no API calls required
- Configurable random seeds and latency simulation
- Suitable for development and debugging

### OpenAI Mode
- Real model API calls
- Requires OPENAI_API_KEY environment variable
- Supports concurrent processing

### Custom Parameters
- Adjustable concurrency (--workers)
- Support for specific task type filtering (--only-task)
- Flexible input/output path configuration

## 🔍 Data Validation

The system includes built-in validation mechanisms to ensure:
- Correct question count (5,376)
- Balanced stage distribution (2,688 each)
- File format consistency
- Data integrity checks

## 📈 Experimental Workflow Example

```bash
# Complete experimental workflow
cd "E:\Grow-AI\Meta cognition\Cog\neweng"

# 1. Generate images
python catalog_runner_two_stage.py gen --catalog catalog_2688_newspec_correctly_fixed.json --imgroot cache/images

# 2. Create questions
python catalog_runner_two_stage.py pack --catalog catalog_2688_newspec_correctly_fixed.json --imgroot cache/images --out questions_two_stage.jsonl

# 3. Call model
python catalog_runner_two_stage.py feed --pack questions_two_stage.jsonl --out responses_two_stage.jsonl --engine mock --workers 8

# 4. Collect results
python catalog_runner_two_stage.py collect --resp responses_two_stage.jsonl --catalog catalog_2688_newspec_correctly_fixed.json --out eval_two_stage.csv
```

## 🎯 Application Scenarios

- **Cognitive Science Research**: Study visual perception abilities of humans and AI
- **Model Evaluation**: Assess AI model performance in cognitive tasks
- **Confidence Calibration**: Study model prediction reliability
- **Educational Research**: Understand cognitive processes in learning

## 🔬 Technical Features

- **Modular Design**: Easy to extend new task types
- **Cache Mechanism**: Avoid repeated image generation
- **Error Handling**: Robust error handling and logging
- **Performance Optimization**: Support for concurrent processing and batch operations

## 📝 Important Notes

1. **Image Paths**: Ensure imgroot directory has corresponding task subdirectories
2. **API Keys**: OpenAI engine requires environment variable setup
3. **File Association**: Two-stage questions are linked through original_qid field
4. **Evaluation Logic**: Correctness judgment is based on first stage answers

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve the system:
- Report bugs or suggest new features
- Contribute new task types or generators
- Improve documentation and code quality

## 📄 License

This project uses MIT license, see LICENSE file for details.

## 📞 Contact

For questions or suggestions, please contact through:
- Submit GitHub Issue
- Send email to project maintainers

---

**Note**: This is a research tool, please ensure compliance with relevant research ethics guidelines and data protection regulations when using.
