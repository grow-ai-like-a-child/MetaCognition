# 📊 Staircase Procedure

This document explains the adaptive staircase algorithm that automatically adjusts task difficulty to maintain optimal performance for measuring metacognitive sensitivity.

## 🎯 Purpose and Rationale

### Why Use a Staircase?
The staircase procedure solves a critical problem in metacognition research:

**The Problem:**
- **Too easy** (>85% accuracy): Ceiling effects, little confidence variation
- **Too hard** (<60% accuracy): Floor effects, models may give up
- **Fixed difficulty**: Different models have different capabilities

**The Solution:**
- **Adaptive difficulty**: Automatically adjusts to each model's capability
- **Target performance**: Maintains ~71% accuracy (optimal for confidence measurement)
- **Individual thresholds**: Each model gets personalized difficulty level

---

## 📈 Algorithm Overview

### Basic Principle
```
Correct Response → Make Task Harder (Decrease Contrast)
Incorrect Response → Make Task Easier (Increase Contrast)
```

### Target Performance
**71% accuracy** is maintained because:
- **Sufficient errors**: Allows comparison of confidence between correct/incorrect trials
- **Sufficient successes**: Prevents frustration and random responding
- **Psychometric optimum**: Maximally sensitive to individual differences
- **Research validated**: Standard in human metacognition studies

---

## ⚙️ Implementation Details

### Core Algorithm
```python
class AdaptiveStaircase:
    def __init__(self, target_performance=0.71):
        # Target accuracy to maintain
        self.target_performance = target_performance
        
        # Starting parameters
        self.current_contrast = 0.5  # 50% contrast
        self.step_size_up = 0.05     # Increase after errors
        self.step_size_down = 0.02   # Decrease after correct
        
        # Boundaries
        self.min_contrast = 0.1      # Floor
        self.max_contrast = 1.0      # Ceiling
        
        # History tracking
        self.history = []            # All responses (True/False)
        self.contrast_history = []   # Contrast at each trial
        self.reversals = []          # Reversal points
        
    def update(self, correct):
        """Update staircase based on trial outcome"""
        # Record response
        self.history.append(correct)
        self.contrast_history.append(self.current_contrast)
        
        # Detect reversals (direction changes)
        if len(self.history) >= 2:
            if self.history[-2] != self.history[-1]:
                self.reversals.append(len(self.history) - 1)
        
        # Adjust contrast
        if correct:
            # Make harder (decrease contrast)
            new_contrast = self.current_contrast - self.step_size_down
        else:
            # Make easier (increase contrast)  
            new_contrast = self.current_contrast + self.step_size_up
        
        # Apply boundaries
        self.current_contrast = np.clip(new_contrast, 
                                       self.min_contrast, 
                                       self.max_contrast)
```

### Asymmetric Step Sizes
```python
step_size_up = 0.05      # Larger steps when making easier
step_size_down = 0.02    # Smaller steps when making harder
```

**Rationale:**
- **Faster recovery** from making task too difficult
- **Prevents rapid oscillation** around threshold
- **Mimics human psychophysics** standards
- **Stability**: Easier to fine-tune than to recover from too-hard

---

## 📊 Convergence and Stability

### Convergence Criteria
```python
def is_converged(self):
    """Check if staircase has converged to stable performance"""
    
    # Need minimum trials
    if len(self.history) < 20:
        return False
    
    # Check recent performance
    recent_window = 10
    recent_performance = np.mean(self.history[-recent_window:])
    
    # Must be close to target
    performance_deviation = abs(recent_performance - self.target_performance)
    performance_stable = performance_deviation < 0.05  # Within 5%
    
    # Check contrast stability
    recent_contrasts = self.contrast_history[-recent_window:]
    contrast_variability = np.std(recent_contrasts)
    contrast_stable = contrast_variability < 0.02  # Low variability
    
    return performance_stable and contrast_stable
```

### Performance Tracking
```python
def get_performance_stats(self):
    """Get detailed performance statistics"""
    if not self.history:
        return None
        
    overall_performance = np.mean(self.history)
    recent_performance = np.mean(self.history[-10:]) if len(self.history) >= 10 else overall_performance
    
    return {
        'overall_accuracy': overall_performance,
        'recent_accuracy': recent_performance,
        'current_contrast': self.current_contrast,
        'n_trials': len(self.history),
        'n_reversals': len(self.reversals),
        'converged': self.is_converged(),
        'target_deviation': abs(recent_performance - self.target_performance),
        'stability': self.assess_stability()
    }
```

---

## 🔄 Trial-by-Trial Example

### Example Session Progression
```
Trial 01: Contrast=0.50, Response=Correct  → New Contrast=0.48 (↓)
Trial 02: Contrast=0.48, Response=Correct  → New Contrast=0.46 (↓)
Trial 03: Contrast=0.46, Response=Wrong    → New Contrast=0.51 (↑) [REVERSAL]
Trial 04: Contrast=0.51, Response=Correct  → New Contrast=0.49 (↓) [REVERSAL]
Trial 05: Contrast=0.49, Response=Wrong    → New Contrast=0.54 (↑) [REVERSAL]
Trial 06: Contrast=0.54, Response=Correct  → New Contrast=0.52 (↓) [REVERSAL]
...
Trial 25: Contrast=0.42, Response=Correct  → New Contrast=0.40 (↓)
Trial 26: Contrast=0.40, Response=Wrong    → New Contrast=0.45 (↑)

Performance Analysis:
- Trials 1-10:   68% accuracy (converging)
- Trials 11-20:  72% accuracy (close to target)
- Trials 21-30:  71% accuracy (converged!)
- Final threshold: 0.42 contrast
```

### Visual Representation
```
Contrast Level Over Time:

0.6 |                                    
    |     *                              
0.5 |  *     *     *                     
    |           * *   * *                
0.4 |              *     * * * * *       
    |                               * *  
0.3 |____________________________________
    0   5   10  15  20  25  30  35  40
                Trial Number

Performance Over Time:

1.0 |        * *   * * * * * *   *       
    |           *           *   *   *    
0.7 |  *   *                   *       * 
    |    *                              
0.0 |____________________________________
    0   5   10  15  20  25  30  35  40
                Trial Number

Target: 0.71 ±0.05 (shown as horizontal band)
```

---

## 🎯 Threshold Estimation

### Final Threshold Calculation
```python
def get_threshold_estimate(self):
    """Estimate the final contrast threshold"""
    
    if not self.is_converged():
        return None
    
    # Method 1: Average of last N trials
    n_final_trials = min(10, len(self.contrast_history) // 3)
    method1_threshold = np.mean(self.contrast_history[-n_final_trials:])
    
    # Method 2: Average of reversal points
    if len(self.reversals) >= 6:
        recent_reversals = self.reversals[-6:]  # Last 6 reversals
        reversal_contrasts = [self.contrast_history[r] for r in recent_reversals]
        method2_threshold = np.mean(reversal_contrasts)
    else:
        method2_threshold = method1_threshold
    
    # Combine methods (weighted average)
    final_threshold = 0.7 * method1_threshold + 0.3 * method2_threshold
    
    return {
        'threshold': final_threshold,
        'method1': method1_threshold,
        'method2': method2_threshold,
        'confidence': 'high' if self.is_converged() else 'low',
        'n_reversals': len(self.reversals)
    }
```

---

## 🔧 Parameter Tuning

### Step Size Optimization
```python
# Conservative (high precision, slow convergence)
step_size_up = 0.03
step_size_down = 0.01

# Standard (balanced)
step_size_up = 0.05  
step_size_down = 0.02

# Aggressive (fast convergence, lower precision)
step_size_up = 0.08
step_size_down = 0.03
```

### Target Performance Alternatives
```python
# Conservative (easier task, more stable)
target_performance = 0.75

# Standard (optimal for metacognition)
target_performance = 0.71

# Challenging (harder task, more errors)
target_performance = 0.67
```

### Boundary Settings
```python
# Permissive boundaries
min_contrast = 0.05    # Very low contrast
max_contrast = 1.0     # Full contrast

# Restrictive boundaries  
min_contrast = 0.15    # Easier minimum
max_contrast = 0.8     # Reduced maximum
```

---

## 🚨 Quality Control

### Problem Detection
```python
def diagnose_staircase_issues(self):
    """Identify potential problems with staircase performance"""
    
    issues = []
    
    # Check for boundary effects
    if self.current_contrast <= self.min_contrast + 0.01:
        issues.append("Hit minimum contrast boundary - task may be too hard")
    
    if self.current_contrast >= self.max_contrast - 0.01:
        issues.append("Hit maximum contrast boundary - task may be too easy")
    
    # Check performance deviation
    if len(self.history) >= 10:
        recent_performance = np.mean(self.history[-10:])
        deviation = abs(recent_performance - self.target_performance)
        
        if deviation > 0.15:
            issues.append(f"Performance far from target: {recent_performance:.2f} vs {self.target_performance:.2f}")
    
    # Check for oscillations
    if len(self.reversals) > len(self.history) * 0.5:
        issues.append("Excessive reversals - may indicate unstable convergence")
    
    # Check for monotonic trends
    if len(self.history) >= 20:
        early_performance = np.mean(self.history[:10])
        late_performance = np.mean(self.history[-10:])
        
        if abs(early_performance - late_performance) > 0.3:
            issues.append("Large performance drift - check for fatigue or learning effects")
    
    return issues
```

### Adaptive Adjustments
```python
def auto_adjust_parameters(self):
    """Automatically adjust parameters based on performance"""
    
    issues = self.diagnose_staircase_issues()
    
    for issue in issues:
        if "boundary" in issue:
            # Adjust step sizes if hitting boundaries
            self.step_size_up *= 0.8
            self.step_size_down *= 0.8
            
        elif "reversals" in issue:
            # Reduce step sizes if too oscillatory
            self.step_size_up *= 0.7
            self.step_size_down *= 0.7
            
        elif "drift" in issue:
            # Increase step sizes if not tracking properly
            self.step_size_up *= 1.2
            self.step_size_down *= 1.2
```

---

## 📊 Multi-Model Considerations

### Shared vs Individual Staircases
```python
# Option 1: Shared staircase (same difficulty for all models)
shared_staircase = AdaptiveStaircase()
for model in models:
    response = query_model(model, stimulus)
    if model == primary_model:  # Update based on one model
        shared_staircase.update(response.correct)

# Option 2: Individual staircases (personalized difficulty)
staircases = {model: AdaptiveStaircase() for model in models}
for model in models:
    response = query_model(model, stimulus)
    staircases[model].update(response.correct)
```

**Current Implementation**: Uses **shared staircase** updated by first valid response to ensure all models see identical stimuli for fair comparison.

---

## 🎯 Research Applications

### Threshold Interpretation
- **Low threshold** (e.g., 0.20): Model has excellent contrast sensitivity
- **Medium threshold** (e.g., 0.45): Model has typical contrast sensitivity  
- **High threshold** (e.g., 0.80): Model has poor contrast sensitivity

### Model Comparison
```python
def compare_thresholds(model_thresholds):
    """Compare contrast sensitivity between models"""
    
    comparisons = {}
    for model1, threshold1 in model_thresholds.items():
        for model2, threshold2 in model_thresholds.items():
            if model1 != model2:
                ratio = threshold1 / threshold2
                if ratio > 1.2:
                    comparison = f"{model1} needs {ratio:.1f}x more contrast than {model2}"
                elif ratio < 0.8:
                    comparison = f"{model1} needs {1/ratio:.1f}x less contrast than {model2}"
                else:
                    comparison = f"{model1} and {model2} have similar contrast sensitivity"
                
                comparisons[f"{model1}_vs_{model2}"] = comparison
    
    return comparisons
```

The adaptive staircase ensures **fair, personalized, and scientifically rigorous** difficulty adjustment for each AI model! 📊⚙️ 