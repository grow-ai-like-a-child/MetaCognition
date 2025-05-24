# Metacognition Data Analysis & Visualization

This toolkit provides comprehensive analysis and visualization capabilities for the Metacognition Investigation Platform. It includes publication-ready plots, statistical analyses, and detailed reporting for metacognitive performance data.

## 🚀 Quick Start

### 1. Generate Example Data (for testing)
```bash
python analyze_results.py --generate-example --trials 200
```

### 2. Run Analysis on Your Data
```bash
python analyze_results.py --data your_experiment_data.csv --output my_results
```

### 3. Compare Multiple Sessions
```bash
python analyze_results.py --compare session1.csv session2.csv session3.csv --output comparison_results
```

## 📊 What You Get

### **Comprehensive Metrics**
- **Basic Performance**: Accuracy, confidence ratings, response times
- **Metacognitive Sensitivity**: Confidence differences between correct/incorrect trials
- **Confidence Calibration**: How well confidence matches actual performance
- **Type-2 ROC Analysis**: Metacognitive efficiency measurements
- **Statistical Comparisons**: Between models/conditions

### **Publication-Ready Visualizations**
- Basic performance comparison plots
- Metacognitive sensitivity analysis plots
- Confidence calibration curves and metrics
- Type-2 ROC curves and AUC scores
- Trial progression and learning curves
- Comprehensive summary dashboard

### **Detailed Reports**
- Statistical analysis reports (TXT format)
- Detailed metrics (JSON format)
- Multiple plot formats (PNG, PDF)
- Organized folder structure

## 📁 Data Format

Your CSV data should include these columns:

| Column | Description | Required |
|--------|-------------|----------|
| `model` | Model/condition identifier | ✅ Yes |
| `correct` | Whether response was correct (True/False) | ✅ Yes |
| `confidence` | Confidence rating (1-6 scale) | ✅ Yes |
| `trial` | Trial number | 🔄 Auto-generated if missing |
| `contrast` | Stimulus contrast/difficulty | ❌ Optional |
| `response_time` | Response time in seconds | ❌ Optional |
| `session` | Session identifier | ❌ Optional |

### Example data row:
```csv
model,trial,contrast,correct,confidence,response_time,session,timestamp
Human,1,0.8,True,5,1.23,session_001,2024-12-01T14:30:01
AI_Model_1,1,0.8,True,4,0.45,session_001,2024-12-01T14:30:02
```

## 🛠 Usage Examples

### Command Line Interface

```bash
# Basic analysis
python analyze_results.py --data experiment_data.csv

# Custom output directory
python analyze_results.py --data experiment_data.csv --output my_analysis

# Quick summary stats only
python analyze_results.py --data experiment_data.csv --summary

# Generate plots without displaying them
python analyze_results.py --data experiment_data.csv --no-plots

# Compare multiple sessions
python analyze_results.py --compare session1.csv session2.csv session3.csv
```

### Python API

```python
from analyze_results import quick_analysis, compare_sessions
from data_analysis import MetacognitionAnalyzer, MetacognitionVisualizer

# Quick analysis
results = quick_analysis("experiment_data.csv", output_dir="my_results")

# Access components
analyzer = results['analyzer']
visualizer = results['visualizer']
plots = results['plots']
report = results['report']

# Custom analysis
import pandas as pd
data = pd.read_csv("experiment_data.csv")

analyzer = MetacognitionAnalyzer(data)
basic_metrics = analyzer.calculate_basic_metrics()
sensitivity = analyzer.calculate_metacognitive_sensitivity()
calibration = analyzer.calculate_calibration_metrics()

# Custom visualization
visualizer = MetacognitionVisualizer(data, output_dir="plots")
fig = visualizer.plot_basic_performance()
dashboard = visualizer.create_summary_dashboard()
```

## 📈 Key Metrics Explained

### **Metacognitive Sensitivity**
- **Definition**: Ability to distinguish between correct and incorrect responses using confidence
- **Calculation**: Mean confidence for correct trials - Mean confidence for incorrect trials
- **Interpretation**: Higher values indicate better metacognitive awareness

### **Confidence Calibration**
- **Definition**: How well confidence ratings match actual accuracy
- **Metrics**: Calibration slope, overconfidence bias, Brier score
- **Interpretation**: Perfect calibration = confidence matches accuracy

### **Type-2 ROC Analysis**
- **Definition**: How well confidence predicts correctness (metacognitive efficiency)
- **Metric**: Area Under Curve (AUC)
- **Interpretation**: 0.5 = chance level, 1.0 = perfect metacognitive efficiency

### **Overconfidence**
- **Definition**: Tendency to be more confident than accurate
- **Calculation**: Normalized mean confidence - Mean accuracy
- **Interpretation**: Positive = overconfident, Negative = underconfident, 0 = well-calibrated

## 📊 Output Structure

```
analysis_results/
├── data/                          # Saved data files
│   ├── session_20241201_143022.csv
│   ├── session_20241201_143022.json
│   └── session_20241201_143022.h5
├── plots/                         # Visualization files
│   ├── basic_performance.png
│   ├── basic_performance.pdf
│   ├── metacognitive_sensitivity.png
│   ├── confidence_calibration.png
│   ├── type2_roc_analysis.png
│   ├── trial_progression.png
│   └── summary_dashboard.png
├── reports/                       # Analysis reports
│   ├── analysis_report_20241201_143022.txt
│   └── detailed_metrics_20241201_143022.json
└── raw_data/                      # Original data backups
```

## 🔬 Advanced Usage

### Custom Analysis Pipeline

```python
from data_analysis import MetacognitionDataManager, MetacognitionAnalyzer

# Initialize data manager
manager = MetacognitionDataManager("my_experiment_results")

# Save experiment data in multiple formats
trial_data = [
    {"model": "Human", "trial": 1, "correct": True, "confidence": 5},
    # ... more trials
]
saved_files = manager.save_experiment_data(trial_data, "session_001")

# Load and analyze
data = manager.load_experiment_data("session_001")
analyzer = MetacognitionAnalyzer(data)

# Calculate specific metrics
sensitivity = analyzer.calculate_metacognitive_sensitivity()
calibration = analyzer.calculate_calibration_metrics()
type2_roc = analyzer.calculate_type2_roc_analysis()
comparisons = analyzer.compare_models()

# Generate comprehensive report
report = analyzer.generate_comprehensive_report()
print(report)
```

### Custom Visualizations

```python
from data_analysis import MetacognitionVisualizer
import matplotlib.pyplot as plt

visualizer = MetacognitionVisualizer(data, "my_plots")

# Individual plot types
basic_fig = visualizer.plot_basic_performance()
sensitivity_fig = visualizer.plot_metacognitive_sensitivity()
calibration_fig = visualizer.plot_confidence_calibration()
roc_fig = visualizer.plot_type2_roc_analysis()
progression_fig = visualizer.plot_trial_progression()

# Comprehensive dashboard
dashboard = visualizer.create_summary_dashboard()

# Show or save
plt.show()  # Display plots
# Plots are automatically saved to output directory
```

## ⚙️ Installation Requirements

The analysis tools require these Python packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Basic plotting
- `seaborn` - Statistical visualizations  
- `plotly` - Interactive plots
- `scipy` - Statistical functions
- `scikit-learn` - ROC analysis
- `h5py` - HDF5 data format support

Install with:
```bash
pip install numpy pandas matplotlib seaborn plotly scipy scikit-learn h5py
```

## 📝 Tips for Best Results

1. **Data Quality**: Ensure your data has the required columns (`model`, `correct`, `confidence`)
2. **Sample Size**: Aim for at least 50+ trials per model for reliable statistics
3. **Confidence Scale**: Use consistent 1-6 scale for confidence ratings
4. **Model Names**: Use descriptive model names for clearer plots
5. **Multiple Sessions**: Compare results across sessions to assess reliability

## 🔍 Troubleshooting

### Common Issues

**Error: "Missing required columns"**
- Check that your CSV has `model`, `correct`, and `confidence` columns
- Ensure column names match exactly (case-sensitive)

**Empty plots or "Insufficient data" messages**
- Verify you have enough trials per model (minimum 10-20)
- Check that confidence values are between 1-6
- Ensure you have both correct and incorrect trials

**Memory issues with large datasets**
- Use HDF5 format for large datasets (`format='hdf5'`)
- Process data in chunks if needed
- Consider sampling for visualization

**Plot display issues**
- Use `--no-plots` flag if running on server without display
- Check matplotlib backend configuration
- Plots are always saved to files regardless of display

## 📚 References

This analysis toolkit implements standard metacognition research metrics:

- **Type-2 ROC**: Maniscalco & Lau (2012, 2014)
- **Confidence Calibration**: Brier (1950), Murphy (1973)
- **Metacognitive Sensitivity**: Fleming & Lau (2014)
- **Statistical Methods**: Standard psychological research practices

For more information on metacognition research methods, see:
- Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience*, 8, 443.
- Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity. *Consciousness and Cognition*, 21(1), 422-430.

---

**Happy analyzing! 🧠📊** 