# 🧠 Metacognition Investigation Platform

A comprehensive system for investigating metacognitive abilities in Vision Language Models (VLMs) using a two-alternative forced-choice task with Gabor patch stimuli.

## Overview

This platform implements the behavioral task described in metacognition research papers, testing whether GPT-4o and Claude Sonnet 3.5 can:

1. **Perceptual Decision Making**: Identify which temporal interval contains a higher-contrast Gabor patch
2. **Confidence Estimation**: Rate their confidence in decisions on a 1-6 scale
3. **Metacognitive Calibration**: Show appropriate confidence-accuracy relationships

## Key Features

### 🎯 **Experimental Design**
- **Two-Alternative Forced Choice (2AFC)**: Models choose between two temporal intervals
- **Gabor Patch Stimuli**: Psychophysically validated visual stimuli with adjustable contrast
- **Staircase Procedure**: Adaptively maintains ~71% performance for meaningful confidence analysis
- **Six Spatial Locations**: Patches appear at different positions around a central fixation point

### 🤖 **Vision Model Integration**
- **GPT-4o Vision**: OpenAI's latest vision-language model
- **Claude Sonnet 3.5**: Anthropic's advanced vision model
- **Concurrent Testing**: Both models tested simultaneously on identical stimuli
- **Robust Response Parsing**: Handles various response formats from models

### 📊 **Analysis & Metrics**
- **Performance Tracking**: Accuracy, response times, trial-by-trial analysis
- **Metacognitive Sensitivity**: Confidence differences between correct/incorrect trials
- **Calibration Analysis**: Confidence-accuracy relationship assessment
- **Statistical Reporting**: Comprehensive CSV/JSON data export

### 🖥️ **User Interfaces**
- **Command Line**: Full experiment runner with batch processing
- **Streamlit Web App**: Interactive interface with real-time visualization
- **Configuration System**: Flexible parameter adjustment

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (for GPT-4o)
- Anthropic API key (for Claude Sonnet 3.5)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd MetaCognition

# Install dependencies
pip install -r requirements.txt
```

### API Key Configuration
1. Create configuration files:
```bash
python experiment_runner.py setup
```

2. Edit `api_keys.json` with your actual API keys:
```json
{
  "openai_key": "your-openai-api-key-here",
  "anthropic_key": "your-anthropic-api-key-here"
}
```

## Usage

### 🚀 **Quick Start - Web Interface**
```bash
streamlit run streamlit_app.py
```
Open your browser to `http://localhost:8501` and use the interactive interface.

### 🔬 **Command Line Experiment**
```bash
# Run with default settings
python experiment_runner.py

# Customize experiment parameters in experiment_config.json
python experiment_runner.py
```

### 🧪 **Testing Individual Components**
```bash
# Test Gabor stimulus generation
python gabor_stimulus.py

# Test staircase procedure
python staircase.py

# Verify vision model interfaces
python vision_models.py
```

## Configuration

### Experiment Parameters (`experiment_config.json`)
```json
{
  "n_trials_per_session": 50,
  "target_performance": 0.71,
  "models_to_test": ["gpt-4o", "claude"],
  "staircase_params": {
    "initial_contrast": 0.5,
    "step_size_up": 0.05,
    "step_size_down": 0.02,
    "min_contrast": 0.1,
    "max_contrast": 1.0
  },
  "stimulus_params": {
    "image_size": 400,
    "patch_size": 60,
    "spatial_freq": 0.1
  }
}
```

## File Structure

```
MetaCognition/
├── gabor_stimulus.py          # Gabor patch generation
├── staircase.py               # Adaptive staircase procedure
├── vision_models.py           # VLM interfaces and analysis
├── experiment_runner.py       # Main experiment coordinator
├── streamlit_app.py          # Web interface
├── requirements.txt          # Dependencies
├── README.md                # This file
├── experiment_config.json    # Experiment parameters
├── api_keys.json            # API credentials (created by you)
└── experiment_results/      # Output directory
    ├── session_YYYYMMDD_HHMMSS/
    │   ├── trial_001.json
    │   ├── summary_report.txt
    │   ├── detailed_metrics.json
    │   └── analysis_data.csv
    └── experiment_logs/
```

## Experimental Protocol

### Trial Structure
1. **Stimulus Generation**: Create two temporal intervals with Gabor patches
2. **Contrast Assignment**: One interval has higher contrast (determined by staircase)
3. **Model Query**: Send both intervals to vision models with task instructions
4. **Response Collection**: Parse perceptual choice (1 or 2) and confidence (1-6)
5. **Staircase Update**: Adjust contrast based on model performance
6. **Data Recording**: Save all trial data and model responses

### Staircase Procedure
- **Target**: Maintain ~71% accuracy for optimal confidence analysis
- **Adaptive**: Contrast increases after errors, decreases after correct responses
- **Convergence**: Stops when performance stabilizes around target
- **Threshold Estimation**: Final contrast level represents perceptual threshold

## Results & Analysis

### Key Metrics
- **Accuracy**: Overall perceptual performance
- **Metacognitive Sensitivity**: `mean(confidence_correct) - mean(confidence_incorrect)`
- **Calibration**: Correlation between confidence and accuracy
- **Response Consistency**: Variability across trials
- **Model Comparison**: Relative performance between GPT-4o and Claude

### Output Files
- **`summary_report.txt`**: Human-readable results summary
- **`detailed_metrics.json`**: Comprehensive statistical analysis
- **`analysis_data.csv`**: Trial-by-trial data for further analysis
- **Individual trial JSON files**: Complete stimulus and response data

## Customization

### Adding New Models
1. Extend `VisionModelInterface` class in `vision_models.py`
2. Implement model-specific query methods
3. Add response parsing logic
4. Update configuration options

### Modifying Stimuli
1. Edit `GaborStimulus` class parameters
2. Adjust spatial frequency, patch size, or locations
3. Implement new stimulus types (e.g., oriented gratings)

### Custom Analysis
1. Extend `MetacognitionAnalyzer` class
2. Add new metrics or visualization functions
3. Integrate with existing reporting system

## Research Applications

### Potential Research Questions
- **Metacognitive Awareness**: Do VLMs show human-like confidence patterns?
- **Calibration Differences**: How do different models compare in confidence accuracy?
- **Task Difficulty Effects**: How does contrast manipulation affect metacognition?
- **Spatial Processing**: Are there location-specific performance differences?
- **Model Architecture**: Do different VLM architectures show distinct metacognitive profiles?

### Publication-Ready Data
- Automated statistical reporting
- Standardized metrics matching human studies
- CSV export for statistical software (R, SPSS, etc.)
- Reproducible experimental parameters

## Troubleshooting

### Common Issues
1. **API Errors**: Check API keys and rate limits
2. **Memory Issues**: Reduce trial count or image resolution
3. **Import Errors**: Verify all dependencies installed
4. **Slow Performance**: Check internet connection for API calls

### Debug Mode
Enable detailed logging by setting `logging.basicConfig(level=logging.DEBUG)` in relevant modules.

## Contributing

1. Fork the repository
2. Create feature branches
3. Add tests for new functionality
4. Submit pull requests with detailed descriptions

## Acknowledgments

Based on metacognition research methodologies from cognitive science and the specific experimental paradigm described in behavioral task studies investigating confidence judgments in perceptual decision-making.
