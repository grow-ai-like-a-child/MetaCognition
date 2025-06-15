# ⚙️ Experiment Parameters Explained

This document provides comprehensive explanations of every parameter in the Gabor experiment configuration.

## 📋 Configuration File Structure

The main configuration is in `experiment_config.json`:

```json
{
  "n_trials_per_session": 50,
  "target_performance": 0.71,
  "models_to_test": ["claude-opus-4-20250514", "claude-3-5-sonnet-20240620"],
  "save_images": true,
  "staircase_params": { ... },
  "stimulus_params": { ... }
}
```

---

## 🎯 Core Experiment Parameters

### `n_trials_per_session` (integer)
**Default**: `50`  
**Range**: `10-500`  
**Purpose**: Number of trials per experimental session

**Guidelines:**
- **Research studies**: 50-100 trials (statistical power)
- **Quick testing**: 10-25 trials (rapid feedback)
- **Extensive analysis**: 100+ trials (detailed patterns)

**Effects:**
- **More trials** = Better statistical reliability, longer runtime
- **Fewer trials** = Faster results, less reliable patterns

### `target_performance` (float)
**Default**: `0.71`  
**Range**: `0.55-0.85`  
**Purpose**: Target accuracy level maintained by staircase

**Scientific Rationale:**
- **0.71 (71%)**: Optimal for measuring confidence differences
- **Above 0.80**: Too easy, ceiling effects in confidence
- **Below 0.65**: Too hard, floor effects, models may give up

**Effects on Analysis:**
- **Higher targets**: More conservative confidence, smaller sensitivity
- **Lower targets**: More liberal confidence, larger variability

### `models_to_test` (array of strings)
**Default**: `["claude-opus-4-20250514", "claude-3-5-sonnet-20240620"]`  
**Purpose**: List of AI models to test simultaneously

**Available Models:**
```json
// Latest Anthropic Models (2025)
"claude-opus-4-20250514"        // Most powerful
"claude-sonnet-4-20250514"      // High performance
"claude-3-7-sonnet-20250219"    // Extended thinking
"claude-3-5-sonnet-20241022"    // Updated version
"claude-3-5-sonnet-20240620"    // Original version
"claude-3-5-haiku-20241022"     // Fastest
```

**Comparison Strategies:**
- **Version comparison**: Same model family, different versions
- **Capability comparison**: Different model classes (Opus vs Sonnet)
- **Speed comparison**: Fast vs thorough models

### `save_images` (boolean)
**Default**: `true`  
**Purpose**: Whether to save stimulus images for inspection

**Storage Impact:**
- **true**: ~50-100MB per session (debugging valuable)
- **false**: Minimal storage (faster, production use)

---

## 📊 Staircase Procedure Parameters

```json
"staircase_params": {
  "initial_contrast": 0.5,
  "step_size_up": 0.05,
  "step_size_down": 0.02,
  "min_contrast": 0.1,
  "max_contrast": 1.0
}
```

### `initial_contrast` (float)
**Default**: `0.5`  
**Range**: `0.2-0.8`  
**Purpose**: Starting contrast level for the staircase

**Selection Guidelines:**
- **Conservative start**: `0.3-0.4` (for difficult models)
- **Standard start**: `0.5` (works for most models)
- **Aggressive start**: `0.6-0.7` (for very capable models)

### `step_size_up` (float)
**Default**: `0.05`  
**Range**: `0.02-0.10`  
**Purpose**: Contrast increase after incorrect responses

**Tuning:**
- **Larger steps** (0.08-0.10): Faster convergence, less precision
- **Smaller steps** (0.02-0.04): Slower convergence, higher precision
- **Standard** (0.05): Good balance for most applications

### `step_size_down` (float)
**Default**: `0.02`  
**Range**: `0.01-0.05`  
**Purpose**: Contrast decrease after correct responses

**Asymmetric Design:**
```
step_size_up > step_size_down
```
**Rationale**: Easier to recover from making task too hard than too easy

### `min_contrast` (float)
**Default**: `0.1`  
**Range**: `0.05-0.2`  
**Purpose**: Minimum allowed contrast level

**Considerations:**
- **Too low** (<0.05): May be imperceptible even to humans
- **Too high** (>0.2): May not challenge capable models
- **Standard** (0.1): Reasonable floor for most vision systems

### `max_contrast` (float)
**Default**: `1.0`  
**Range**: `0.8-1.0`  
**Purpose**: Maximum allowed contrast level

**Notes:**
- **1.0**: Full contrast (recommended)
- **<1.0**: Artificial ceiling (rarely needed)

---

## 🎨 Visual Stimulus Parameters

```json
"stimulus_params": {
  "image_size": 400,
  "patch_size": 60,
  "spatial_freq": 0.1
}
```

### `image_size` (integer)
**Default**: `400`  
**Range**: `300-800`  
**Purpose**: Image dimensions in pixels (square images)

**Trade-offs:**
- **Larger images** (600-800px): Higher resolution, slower processing
- **Smaller images** (300-400px): Faster processing, sufficient detail
- **Standard** (400px): Good balance for most models

### `patch_size` (integer)
**Default**: `60`  
**Range**: `30-120`  
**Purpose**: Gabor patch diameter in pixels

**Visual Angle Considerations:**
```
Patch size relative to image:
- 60/400 = 15% of image width
- Approximately 3-4 degrees visual angle
```

**Guidelines:**
- **Smaller patches** (30-45px): More challenging, requires precise vision
- **Larger patches** (80-120px): Easier detection, may be too obvious
- **Standard** (60px): Validated size from human studies

### `spatial_freq` (float)
**Default**: `0.1`  
**Range**: `0.05-0.3`  
**Purpose**: Spatial frequency of Gabor stripes (cycles per pixel)

**Visual Effects:**
- **Lower frequency** (0.05-0.08): Wider stripes, easier to see
- **Higher frequency** (0.15-0.3): Narrower stripes, more challenging
- **Standard** (0.1): Optimal for most vision systems

**Formula:**
```
cycles_per_patch = spatial_freq × patch_size
0.1 × 60 = 6 cycles per patch
```

---

## 🎯 Spatial Layout Parameters (Fixed)

The experiment uses **6 fixed spatial locations** arranged around the image center:

```
     [1]
 [0]  +  [2]
     [3]
 [5]     [4]
```

**Location Coordinates:**
```python
locations = [
    (-100, 0),    # Left
    (0, -100),    # Top
    (100, 0),     # Right
    (0, 100),     # Bottom
    (70, 70),     # Bottom-right
    (-70, 70)     # Bottom-left
]
```

**Why 6 Locations:**
- **Sufficient variety**: Tests spatial processing
- **Manageable complexity**: Not overwhelming for analysis
- **Symmetrical**: Balanced across visual field
- **Research validated**: Standard in psychophysics

---

## 🔧 Advanced Configuration

### Custom Model Addition
To add new models, update the model list:

```json
"models_to_test": [
  "claude-opus-4-20250514",
  "your-custom-model-name"
]
```

Then implement the model interface in `vision_models.py`.

### Session Management
```json
"session_params": {
  "break_interval": 25,     // Optional breaks every N trials
  "randomize_trials": true, // Randomize trial order
  "warm_up_trials": 5      // Practice trials (not analyzed)
}
```

### Data Collection
```json
"data_params": {
  "collect_response_text": true,  // Save full AI responses
  "collect_timing": true,          // Detailed timing data
  "collect_metadata": true         // System information
}
```

---

## 📊 Parameter Validation

The system automatically validates parameters:

### Error Checking
- **Range validation**: Parameters within acceptable bounds
- **Type checking**: Correct data types
- **Logical consistency**: Compatible parameter combinations

### Warnings
- **Non-standard values**: Parameters outside typical ranges
- **Performance implications**: Settings that may affect results
- **Compatibility issues**: Model-specific considerations

### Example Validation Output
```
✓ n_trials_per_session: 50 (valid)
✓ target_performance: 0.71 (optimal)
⚠ initial_contrast: 0.8 (high, may converge slowly)
✗ min_contrast: 0.05 (too low, may be imperceptible)
```

---

## 🎯 Recommended Configurations

### Quick Testing
```json
{
  "n_trials_per_session": 20,
  "target_performance": 0.71,
  "save_images": true,
  "staircase_params": {
    "initial_contrast": 0.5,
    "step_size_up": 0.08,
    "step_size_down": 0.03
  }
}
```

### Research Study
```json
{
  "n_trials_per_session": 100,
  "target_performance": 0.71,
  "save_images": true,
  "staircase_params": {
    "initial_contrast": 0.4,
    "step_size_up": 0.05,
    "step_size_down": 0.02
  }
}
```

### High-Precision Analysis
```json
{
  "n_trials_per_session": 200,
  "target_performance": 0.71,
  "save_images": false,
  "staircase_params": {
    "initial_contrast": 0.35,
    "step_size_up": 0.03,
    "step_size_down": 0.01
  }
}
```

---

Understanding these parameters allows you to customize the experiment for your specific research questions and AI models! 🎯✨ 