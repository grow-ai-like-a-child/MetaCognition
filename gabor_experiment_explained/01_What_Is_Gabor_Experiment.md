# 🎯 What Is the Gabor Experiment?

## Overview

The **Gabor Experiment** is a scientifically rigorous test of **metacognitive abilities** in AI vision models. It asks a fundamental question: **Do AI models have awareness of their own perceptual decisions?**

## 🧠 The Core Question

### Human Metacognition
When humans make perceptual decisions, they typically:
- **Make a choice** (e.g., "I think the left circle is brighter")
- **Feel confident or uncertain** about that choice
- **Adjust behavior** based on their confidence

### AI Metacognition (Unknown)
Do AI models exhibit similar patterns?
- **Can they report appropriate confidence levels?**
- **Does their confidence correlate with accuracy?**
- **Do they show human-like metacognitive patterns?**

## 🔬 The Experimental Approach

### Two-Alternative Forced Choice (2AFC) Task
```
[FIRST INTERVAL]     [SECOND INTERVAL]     [RESPONSE]
      ○                    ◉                Choice: 2
   (dimmer)             (brighter)          Confidence: 4/6
```

**The AI must:**
1. **Decide** which interval had the higher contrast stimulus
2. **Report confidence** on a 1-6 scale
3. **Explain reasoning** (optional)

### Why This Design?
- **Proven methodology**: Used in human studies for decades
- **Quantifiable**: Produces clear metrics
- **Comparable**: Direct comparison to human data
- **Sensitive**: Detects subtle differences in metacognitive ability

## 🎨 Visual Stimulus: Gabor Patches

### What Are Gabor Patches?
- **Sinusoidal gratings** (stripes) in a **Gaussian envelope** (circular fade)
- Think: **Circular striped patterns** with adjustable visibility
- **Mathematically precise**: Controllable contrast, frequency, orientation

### Why Gabor Patches?
1. **Psychophysically validated**: Extensively studied in vision research
2. **Parametric control**: Precise manipulation of difficulty
3. **Minimal complexity**: Focus on basic perceptual processes
4. **Cross-species**: Used in human, animal, and now AI studies

### Visual Example
```
Low Contrast Gabor          High Contrast Gabor
       ░░░░░                      ████████
     ░░░░░░░░░                  ██████████
   ░░░░░░░░░░░░░              ████████████
     ░░░░░░░░░                  ██████████
       ░░░░░                      ████████
```

## ⚙️ The Staircase Procedure

### Adaptive Difficulty
The experiment automatically adjusts difficulty to:
- **Maintain ~71% accuracy** (optimal for measuring confidence)
- **Avoid ceiling/floor effects**
- **Personalize to each model's capabilities**

### How It Works
```
Trial 1: Easy (high contrast) → Correct → Make harder
Trial 2: Medium contrast → Wrong → Make easier  
Trial 3: Medium-easy → Correct → Make harder
... continues until performance stabilizes
```

## 📊 What We Measure

### Primary Measures
1. **Accuracy**: Percentage of correct responses
2. **Confidence**: Mean confidence ratings (1-6 scale)
3. **Metacognitive Sensitivity**: Confidence difference between correct/incorrect trials
4. **Calibration**: How well confidence predicts accuracy

### Advanced Measures
- **Response consistency**: Variability across trials
- **Spatial biases**: Performance differences by location
- **Temporal effects**: Changes over time
- **Model comparisons**: Relative performance between AI systems

## 🎯 Example Trial Walkthrough

### Trial Setup
```
Current threshold: 0.45 contrast
Locations: Position 2 vs Position 5
Higher contrast interval: 2 (random)
```

### Stimulus Generation
```
Interval 1: Gabor at position 2, contrast = 0.32 (0.45 × 0.7)
Interval 2: Gabor at position 5, contrast = 0.45 (target)
```

### AI Response
```
Model: Claude Opus 4
Choice: 2 (correct!)
Confidence: 4/6
Reasoning: "The second interval appeared slightly more visible, 
           though the difference was subtle."
```

### Data Recording
```
✓ Correct response
✓ Moderate confidence (appropriate for difficulty)
✓ Staircase adjustment: Decrease contrast to 0.43
✓ Metacognitive sensitivity: +1 point
```

## 🔬 Scientific Significance

### Bridging AI and Cognitive Science
This experiment represents the **first systematic application** of established cognitive science methods to AI systems.

### Key Innovations
1. **Quantitative metacognition**: Precise measurement of AI self-awareness
2. **Comparative framework**: Direct comparison between models
3. **Human-comparable metrics**: Results interpretable in context of human studies
4. **Standardized methodology**: Reproducible across labs and models

### Research Applications
- **AI Safety**: Understanding AI uncertainty and confidence
- **Human-AI Interaction**: Building appropriately confident AI systems
- **Cognitive Science**: Testing theories of metacognition in artificial systems
- **Model Evaluation**: New dimension for assessing AI capabilities

## 🎪 Real-World Implications

### Why This Matters
- **AI Transparency**: Models that can report their uncertainty
- **Safer AI Systems**: Knowing when AI is unsure
- **Better Human-AI Teams**: Appropriate confidence communication
- **Scientific Understanding**: Insights into the nature of machine cognition

### Future Directions
- **Clinical Applications**: AI diagnostic systems with calibrated confidence
- **Educational AI**: Tutoring systems that know their limitations
- **Autonomous Systems**: Self-aware AI that can request human help
- **Research Tools**: Standardized measures for AI metacognition

---

This experiment opens a new frontier in AI evaluation, moving beyond simple accuracy to measure the **self-awareness** that makes AI systems truly intelligent partners for humans. 🧠✨ 