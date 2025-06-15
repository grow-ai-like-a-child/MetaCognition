# 🔬 Experimental Design

This document explains the complete experimental methodology, trial structure, and data collection procedures for the Gabor experiment.

## 🎯 Overview

The experiment uses a **two-alternative forced choice (2AFC)** paradigm with **adaptive staircase** procedures to measure **metacognitive sensitivity** in AI vision models.

## 📋 Trial Structure

### Complete Trial Sequence

```
[PREPARATION] → [STIMULUS] → [RESPONSE] → [ANALYSIS] → [UPDATE]
```

### 1. Preparation Phase
```python
# Trial initialization
trial_number = get_next_trial()
current_contrast = staircase.get_current_contrast()
locations = randomly_select_2_of_6_positions()
target_interval = randomly_choose_1_or_2()
```

### 2. Stimulus Generation Phase
```python
# Create two temporal intervals
if target_interval == 1:
    interval_1_contrast = current_contrast
    interval_2_contrast = current_contrast * 0.7  # 30% lower
else:
    interval_1_contrast = current_contrast * 0.7  # 30% lower  
    interval_2_contrast = current_contrast

# Generate images
image_1 = create_gabor_stimulus(locations[0], interval_1_contrast)
image_2 = create_gabor_stimulus(locations[1], interval_2_contrast)
```

### 3. Model Query Phase
```python
# Send to AI models
for model in models_to_test:
    response = await query_model(
        model=model,
        first_image=image_1,
        second_image=image_2,
        prompt=create_task_prompt()
    )
    
    # Parse response
    choice = extract_choice(response)  # 1 or 2
    confidence = extract_confidence(response)  # 1-6
```

### 4. Analysis Phase
```python
# Evaluate responses
for response in responses:
    correct = (response.choice == target_interval)
    accuracy = 1 if correct else 0
    
    # Record data
    save_trial_data(trial_number, response, correct, accuracy)
```

### 5. Update Phase
```python
# Update staircase (use first valid response)
first_valid_response = get_first_valid_response(responses)
if first_valid_response:
    staircase.update(first_valid_response.correct)
```

---

## 🎨 Stimulus Design

### Gabor Patch Specifications

```python
def create_gabor_patch(contrast, spatial_freq=0.1, patch_size=60):
    """
    Create a Gabor patch with specified parameters
    
    Args:
        contrast: Contrast level (0.0-1.0)
        spatial_freq: Spatial frequency (cycles per pixel)
        patch_size: Patch diameter in pixels
    
    Returns:
        numpy array representing the Gabor patch
    """
    # Mathematical definition
    x, y = np.meshgrid(range(-patch_size//2, patch_size//2), 
                       range(-patch_size//2, patch_size//2))
    
    # Gaussian envelope
    gaussian = np.exp(-(x**2 + y**2) / (2 * (patch_size/6)**2))
    
    # Sinusoidal grating
    sine_wave = np.sin(2 * np.pi * spatial_freq * x)
    
    # Gabor patch
    gabor = contrast * gaussian * sine_wave
    
    return gabor
```

### Spatial Layout

```
Visual Field Layout (400x400 pixels):

         Location 1
          (0,-100)
             ●
Location 0        Location 2
(-100,0)            (100,0)
    ●        +        ●
             ●
         Location 3
          (0,100)

    ●                 ●
Location 5        Location 4
(-70,70)          (70,70)
```

### Contrast Manipulation

```python
# Target interval (higher contrast)
target_contrast = staircase.current_contrast

# Distractor interval (lower contrast)  
distractor_contrast = target_contrast * 0.7  # 30% reduction

# This creates a just-noticeable difference
contrast_difference = target_contrast - distractor_contrast
```

---

## 📊 Staircase Procedure

### Adaptive Algorithm

```python
class AdaptiveStaircase:
    def __init__(self, target_performance=0.71):
        self.target_performance = target_performance
        self.current_contrast = 0.5  # Start at 50%
        self.step_size_up = 0.05     # Increase after errors
        self.step_size_down = 0.02   # Decrease after correct
        
    def update(self, correct):
        if correct:
            # Make harder (decrease contrast)
            self.current_contrast -= self.step_size_down
        else:
            # Make easier (increase contrast)
            self.current_contrast += self.step_size_up
            
        # Apply bounds
        self.current_contrast = np.clip(
            self.current_contrast, 
            self.min_contrast, 
            self.max_contrast
        )
```

### Convergence Criteria

```python
def is_converged(self):
    """Check if staircase has converged"""
    if len(self.history) < 20:
        return False
        
    # Check recent performance
    recent_trials = self.history[-10:]
    recent_performance = np.mean(recent_trials)
    
    # Check if close to target
    performance_diff = abs(recent_performance - self.target_performance)
    
    return performance_diff < 0.05  # Within 5% of target
```

### Performance Tracking

```python
def get_performance_stats(self):
    """Get detailed performance statistics"""
    return {
        'overall_accuracy': np.mean(self.history),
        'recent_accuracy': np.mean(self.history[-10:]),
        'current_contrast': self.current_contrast,
        'n_trials': len(self.history),
        'n_reversals': len(self.reversals),
        'converged': self.is_converged()
    }
```

---

## 🤖 AI Model Interface

### Task Prompt

```python
def create_task_prompt():
    return """
    You are participating in a visual perception experiment. You will see two temporal intervals, 
    each containing a visual stimulus (Gabor patch). Your task is to determine which interval 
    contains the stimulus with HIGHER CONTRAST.
    
    Please respond with:
    1. CHOICE: 1 (for first interval) or 2 (for second interval)
    2. CONFIDENCE: Your confidence level from 1 (guessing) to 6 (certain)
    
    Example response:
    CHOICE: 2
    CONFIDENCE: 4
    
    Focus on the circular striped patterns and compare their visibility between the two intervals.
    """
```

### Response Parsing

```python
def parse_response(raw_response):
    """Extract choice and confidence from AI response"""
    choice = -1
    confidence = -1
    
    # Look for structured format
    if "CHOICE:" in raw_response:
        choice_match = re.search(r"CHOICE:\s*([12])", raw_response)
        if choice_match:
            choice = int(choice_match.group(1))
    
    if "CONFIDENCE:" in raw_response:
        conf_match = re.search(r"CONFIDENCE:\s*([1-6])", raw_response)
        if conf_match:
            confidence = int(conf_match.group(1))
    
    # Fallback parsing
    if choice == -1 or confidence == -1:
        numbers = re.findall(r'\b([1-6])\b', raw_response)
        if len(numbers) >= 2:
            if choice == -1 and int(numbers[0]) in [1, 2]:
                choice = int(numbers[0])
            if confidence == -1 and int(numbers[1]) in range(1, 7):
                confidence = int(numbers[1])
    
    return choice, confidence
```

---

## 📈 Data Collection

### Trial-Level Data

```python
trial_data = {
    # Trial identification
    'trial_id': f"{session_id}_trial_{trial_number:03d}",
    'trial_number': trial_number,
    'session_id': session_id,
    'timestamp': datetime.now().isoformat(),
    
    # Stimulus parameters
    'target_interval': target_interval,  # 1 or 2
    'first_location': first_location,    # 0-5
    'second_location': second_location,  # 0-5
    'first_contrast': first_contrast,    # 0.0-1.0
    'second_contrast': second_contrast,  # 0.0-1.0
    'contrast_difference': abs(first_contrast - second_contrast),
    
    # Staircase state
    'staircase_contrast': staircase.current_contrast,
    'staircase_converged': staircase.is_converged(),
    
    # Model responses
    'responses': [
        {
            'model_name': response.model_name,
            'choice': response.choice,
            'confidence': response.confidence,
            'correct': response.choice == target_interval,
            'response_time': response.response_time,
            'raw_response': response.raw_response
        }
        for response in responses
    ]
}
```

### Session-Level Data

```python
session_data = {
    'session_id': session_id,
    'start_time': session_start_time,
    'end_time': session_end_time,
    'total_trials': n_trials,
    'models_tested': list(models_tested),
    'configuration': experiment_config,
    
    # Performance summary
    'final_performance': {
        model: calculate_performance(model_responses)
        for model in models_tested
    },
    
    # Staircase summary
    'staircase_final_state': staircase.get_performance_stats(),
    'threshold_estimate': staircase.get_threshold_estimate()
}
```

---

## 🎯 Quality Control

### Response Validation

```python
def validate_response(response):
    """Validate AI model response"""
    errors = []
    
    # Check choice
    if response.choice not in [1, 2]:
        errors.append(f"Invalid choice: {response.choice}")
    
    # Check confidence
    if response.confidence not in range(1, 7):
        errors.append(f"Invalid confidence: {response.confidence}")
    
    # Check response time
    if response.response_time < 0.1:
        errors.append("Response time too fast (likely cached)")
    
    if response.response_time > 60:
        errors.append("Response time too slow (likely timeout)")
    
    return errors

def is_valid_response(response):
    """Check if response is valid for analysis"""
    return len(validate_response(response)) == 0
```

### Data Integrity

```python
def check_data_integrity(trial_data):
    """Ensure trial data is complete and consistent"""
    checks = {
        'stimulus_consistent': 
            trial_data['target_interval'] in [1, 2],
        'locations_different': 
            trial_data['first_location'] != trial_data['second_location'],
        'contrasts_reasonable': 
            0.0 <= trial_data['first_contrast'] <= 1.0,
        'responses_present': 
            len(trial_data['responses']) > 0,
        'timing_reasonable': 
            all(0.1 < r['response_time'] < 60 for r in trial_data['responses'])
    }
    
    return all(checks.values()), checks
```

---

## 🔄 Experimental Flow

### Complete Session Workflow

```python
async def run_experimental_session():
    # 1. Initialize
    session = initialize_session()
    staircase = AdaptiveStaircase(target_performance=0.71)
    
    # 2. Run trials
    for trial_num in range(1, n_trials + 1):
        # Generate stimulus
        trial_info = generate_trial_stimulus(trial_num, staircase)
        
        # Query models
        responses = await query_all_models(trial_info)
        
        # Validate responses
        valid_responses = [r for r in responses if is_valid_response(r)]
        
        # Update staircase
        if valid_responses:
            staircase.update(valid_responses[0].correct)
        
        # Save data
        save_trial_data(trial_info, responses)
        
        # Check convergence
        if staircase.is_converged() and trial_num > 20:
            logger.info(f"Staircase converged at trial {trial_num}")
    
    # 3. Finalize
    generate_session_summary(session)
    return session
```

This experimental design ensures **rigorous scientific methodology** while maintaining **practical feasibility** for AI model testing! 🔬✨ 