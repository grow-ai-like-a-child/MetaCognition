# 📊 Data Analysis

This document explains all the metrics, statistical methods, and analysis procedures used to evaluate AI metacognitive performance in the Gabor experiment.

## 🎯 Primary Metrics

### 1. Accuracy (Perceptual Performance)
```python
def calculate_accuracy(responses):
    """Calculate overall accuracy percentage"""
    correct_responses = sum(1 for r in responses if r.correct)
    total_responses = len(responses)
    return correct_responses / total_responses if total_responses > 0 else 0

# Example: 71% accuracy (maintained by staircase)
accuracy = 0.71
```

### 2. Confidence Ratings 
```python
def analyze_confidence(responses):
    """Analyze confidence rating patterns"""
    confidences = [r.confidence for r in responses if r.confidence > 0]
    
    return {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'confidence_range': (min(confidences), max(confidences)),
        'confidence_distribution': np.bincount(confidences, minlength=7)[1:]  # 1-6 scale
    }

# Example output:
# {
#   'mean_confidence': 3.8,
#   'std_confidence': 1.2,
#   'confidence_range': (1, 6),
#   'confidence_distribution': array([2, 8, 15, 18, 6, 1])  # Counts for 1-6
# }
```

### 3. Metacognitive Sensitivity (Primary Measure)
```python
def calculate_metacognitive_sensitivity(responses):
    """
    Calculate difference in confidence between correct and incorrect trials
    This is the KEY measure of metacognitive awareness
    """
    correct_trials = [r for r in responses if r.correct and r.confidence > 0]
    incorrect_trials = [r for r in responses if not r.correct and r.confidence > 0]
    
    if not correct_trials or not incorrect_trials:
        return None
        
    mean_confidence_correct = np.mean([r.confidence for r in correct_trials])
    mean_confidence_incorrect = np.mean([r.confidence for r in incorrect_trials])
    
    sensitivity = mean_confidence_correct - mean_confidence_incorrect
    
    return {
        'sensitivity': sensitivity,
        'confidence_correct': mean_confidence_correct,
        'confidence_incorrect': mean_confidence_incorrect,
        'n_correct': len(correct_trials),
        'n_incorrect': len(incorrect_trials)
    }

# Example:
# Good metacognition: sensitivity = +1.2 (higher confidence when correct)
# Poor metacognition: sensitivity = -0.1 (lower confidence when correct)
# No metacognition: sensitivity ≈ 0 (same confidence regardless)
```

### 4. Calibration Analysis
```python
def calculate_calibration(responses):
    """
    Measure how well confidence predicts accuracy
    Perfect calibration: confidence 4/6 → 67% accuracy, confidence 6/6 → 100% accuracy
    """
    # Group by confidence level
    confidence_groups = {}
    for r in responses:
        if r.confidence > 0:
            if r.confidence not in confidence_groups:
                confidence_groups[r.confidence] = []
            confidence_groups[r.confidence].append(r.correct)
    
    calibration_data = {}
    for conf_level, correct_list in confidence_groups.items():
        actual_accuracy = np.mean(correct_list)
        expected_accuracy = (conf_level - 1) / 5  # Scale 1-6 to 0-1
        calibration_error = abs(actual_accuracy - expected_accuracy)
        
        calibration_data[conf_level] = {
            'expected_accuracy': expected_accuracy,
            'actual_accuracy': actual_accuracy,
            'calibration_error': calibration_error,
            'n_trials': len(correct_list)
        }
    
    # Overall calibration error
    overall_error = np.mean([data['calibration_error'] for data in calibration_data.values()])
    
    return {
        'by_confidence': calibration_data,
        'overall_calibration_error': overall_error
    }
```

---

## 📈 Advanced Analysis

### 5. Response Consistency
```python
def analyze_response_consistency(responses):
    """Measure how consistent the model's responses are"""
    
    # Temporal consistency (change over time)
    early_responses = responses[:len(responses)//2]
    late_responses = responses[len(responses)//2:]
    
    early_accuracy = calculate_accuracy(early_responses)
    late_accuracy = calculate_accuracy(late_responses)
    
    early_confidence = np.mean([r.confidence for r in early_responses if r.confidence > 0])
    late_confidence = np.mean([r.confidence for r in late_responses if r.confidence > 0])
    
    return {
        'accuracy_drift': late_accuracy - early_accuracy,
        'confidence_drift': late_confidence - early_confidence,
        'confidence_variability': np.std([r.confidence for r in responses if r.confidence > 0])
    }
```

### 6. Spatial Bias Analysis
```python
def analyze_spatial_bias(responses, trial_data):
    """Check for location-specific performance differences"""
    
    location_performance = {}
    
    for response, trial in zip(responses, trial_data):
        locations = (trial['first_location'], trial['second_location'])
        
        for loc in locations:
            if loc not in location_performance:
                location_performance[loc] = []
            location_performance[loc].append(response.correct)
    
    bias_analysis = {}
    for loc, performance_list in location_performance.items():
        bias_analysis[loc] = {
            'accuracy': np.mean(performance_list),
            'n_trials': len(performance_list)
        }
    
    # Test for significant differences
    accuracies = [data['accuracy'] for data in bias_analysis.values()]
    spatial_bias_magnitude = max(accuracies) - min(accuracies)
    
    return {
        'by_location': bias_analysis,
        'spatial_bias_magnitude': spatial_bias_magnitude
    }
```

### 7. Contrast Threshold Analysis
```python
def analyze_contrast_threshold(staircase_data):
    """Analyze the final contrast threshold and convergence"""
    
    # Extract threshold estimate
    final_contrast = staircase_data['current_contrast']
    
    # Analyze convergence
    performance_history = staircase_data.get('performance_history', [])
    if len(performance_history) >= 10:
        final_performance = np.mean(performance_history[-10:])
        target_deviation = abs(final_performance - 0.71)
    else:
        target_deviation = None
    
    # Reversals analysis
    reversals = staircase_data.get('reversals', [])
    
    return {
        'final_threshold': final_contrast,
        'target_deviation': target_deviation,
        'n_reversals': len(reversals),
        'converged': staircase_data.get('converged', False),
        'convergence_quality': 'good' if target_deviation and target_deviation < 0.05 else 'poor'
    }
```

---

## 🔍 Statistical Tests

### 8. Significance Testing
```python
def test_metacognitive_significance(responses):
    """Test if metacognitive sensitivity is statistically significant"""
    from scipy import stats
    
    correct_confidences = [r.confidence for r in responses if r.correct and r.confidence > 0]
    incorrect_confidences = [r.confidence for r in responses if not r.correct and r.confidence > 0]
    
    if len(correct_confidences) < 5 or len(incorrect_confidences) < 5:
        return None
    
    # t-test for difference in means
    t_stat, p_value = stats.ttest_ind(correct_confidences, incorrect_confidences)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(correct_confidences) - 1) * np.var(correct_confidences, ddof=1) + 
                         (len(incorrect_confidences) - 1) * np.var(incorrect_confidences, ddof=1)) / 
                        (len(correct_confidences) + len(incorrect_confidences) - 2))
    
    cohens_d = (np.mean(correct_confidences) - np.mean(incorrect_confidences)) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }
```

### 9. Model Comparison
```python
def compare_models(model_results):
    """Compare metacognitive performance between different models"""
    
    comparisons = {}
    model_names = list(model_results.keys())
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            
            # Compare metacognitive sensitivity
            sens1 = model_results[model1]['metacognitive_sensitivity']['sensitivity']
            sens2 = model_results[model2]['metacognitive_sensitivity']['sensitivity']
            
            # Compare calibration
            cal1 = model_results[model1]['calibration']['overall_calibration_error']
            cal2 = model_results[model2]['calibration']['overall_calibration_error']
            
            comparison_key = f"{model1}_vs_{model2}"
            comparisons[comparison_key] = {
                'sensitivity_difference': sens1 - sens2,
                'calibration_difference': cal1 - cal2,  # Lower is better
                'better_sensitivity': model1 if sens1 > sens2 else model2,
                'better_calibration': model1 if cal1 < cal2 else model2
            }
    
    return comparisons
```

---

## 📋 Analysis Report Generation

### 10. Comprehensive Analysis Function
```python
def generate_comprehensive_analysis(session_data):
    """Generate complete analysis report for a session"""
    
    models = session_data['models_tested']
    analysis_results = {}
    
    for model in models:
        model_responses = [r for r in session_data['responses'] if r['model_name'] == model]
        
        # Convert to response objects for analysis
        response_objects = [create_response_object(r) for r in model_responses]
        
        # Calculate all metrics
        analysis_results[model] = {
            'basic_performance': {
                'accuracy': calculate_accuracy(response_objects),
                'n_trials': len(response_objects),
                'response_rate': len([r for r in response_objects if r.choice in [1,2]]) / len(response_objects)
            },
            
            'confidence_analysis': analyze_confidence(response_objects),
            
            'metacognitive_sensitivity': calculate_metacognitive_sensitivity(response_objects),
            
            'calibration': calculate_calibration(response_objects),
            
            'consistency': analyze_response_consistency(response_objects),
            
            'spatial_analysis': analyze_spatial_bias(response_objects, session_data['trial_data']),
            
            'statistical_tests': test_metacognitive_significance(response_objects)
        }
    
    # Add model comparisons
    if len(models) > 1:
        analysis_results['model_comparisons'] = compare_models(analysis_results)
    
    # Add session-level analysis
    analysis_results['session_summary'] = {
        'threshold_analysis': analyze_contrast_threshold(session_data['staircase_data']),
        'total_trials': session_data['total_trials'],
        'session_duration': calculate_session_duration(session_data),
        'data_quality': assess_data_quality(session_data)
    }
    
    return analysis_results
```

---

## 📊 Visualization Support

### 11. Data for Plotting
```python
def prepare_visualization_data(analysis_results):
    """Prepare data structures for visualization"""
    
    viz_data = {}
    
    for model, results in analysis_results.items():
        if model == 'model_comparisons' or model == 'session_summary':
            continue
            
        viz_data[model] = {
            # Confidence by accuracy
            'confidence_by_accuracy': {
                'correct': [r.confidence for r in results['responses'] if r.correct],
                'incorrect': [r.confidence for r in results['responses'] if not r.correct]
            },
            
            # Calibration curve
            'calibration_curve': {
                'confidence_levels': list(range(1, 7)),
                'expected_accuracy': [(c-1)/5 for c in range(1, 7)],
                'actual_accuracy': [results['calibration']['by_confidence'].get(c, {}).get('actual_accuracy', 0) 
                                  for c in range(1, 7)]
            },
            
            # Performance over time
            'temporal_performance': {
                'trial_numbers': list(range(1, len(results['responses']) + 1)),
                'running_accuracy': calculate_running_average([r.correct for r in results['responses']]),
                'running_confidence': calculate_running_average([r.confidence for r in results['responses']])
            }
        }
    
    return viz_data

def calculate_running_average(data, window_size=10):
    """Calculate running average for temporal analysis"""
    running_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        running_avg.append(np.mean(data[start_idx:i+1]))
    return running_avg
```

---

## 🎯 Interpretation Guidelines

### Metacognitive Sensitivity Interpretation
- **> +1.0**: Excellent metacognitive awareness
- **+0.5 to +1.0**: Good metacognitive awareness
- **0 to +0.5**: Weak metacognitive awareness
- **< 0**: Paradoxical (more confident when wrong)

### Calibration Error Interpretation
- **< 0.05**: Excellent calibration
- **0.05 - 0.10**: Good calibration
- **0.10 - 0.20**: Moderate calibration
- **> 0.20**: Poor calibration

### Statistical Significance
- **p < 0.001**: Very strong evidence
- **p < 0.01**: Strong evidence
- **p < 0.05**: Moderate evidence
- **p ≥ 0.05**: No significant evidence

This comprehensive analysis framework provides **rigorous quantitative assessment** of AI metacognitive abilities! 📊🧠 