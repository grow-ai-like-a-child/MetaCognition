import numpy as np
from collections import deque
import logging

class AdaptiveStaircase:
    """
    Implements a staircase procedure to maintain ~71% performance.
    Based on the Quest+ algorithm and simple up-down rules.
    """
    
    def __init__(self, initial_contrast=0.5, target_performance=0.71,
                 step_size_up=0.05, step_size_down=0.02, 
                 min_contrast=0.1, max_contrast=1.0,
                 n_reversals_for_threshold=6, history_length=10):
        """
        Initialize staircase parameters.
        
        Args:
            initial_contrast: Starting contrast level
            target_performance: Target performance level (0.71 for ~71%)
            step_size_up: Step size for increasing difficulty (decreasing contrast)
            step_size_down: Step size for decreasing difficulty (increasing contrast)
            min_contrast: Minimum contrast level
            max_contrast: Maximum contrast level
            n_reversals_for_threshold: Number of reversals to calculate threshold
            history_length: Number of recent trials to track for performance
        """
        self.initial_contrast = initial_contrast
        self.current_contrast = initial_contrast
        self.target_performance = target_performance
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.n_reversals_for_threshold = n_reversals_for_threshold
        
        # Track trial history
        self.trial_history = []
        self.recent_performance = deque(maxlen=history_length)
        self.reversals = []
        self.last_direction = None
        
        # Performance tracking
        self.n_correct = 0
        self.n_trials = 0
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def update(self, correct):
        """
        Update staircase based on trial outcome.
        
        Args:
            correct: Boolean indicating if response was correct
        """
        self.trial_history.append({
            'trial': len(self.trial_history) + 1,
            'contrast': self.current_contrast,
            'correct': correct
        })
        
        self.recent_performance.append(correct)
        self.n_trials += 1
        if correct:
            self.n_correct += 1
        
        # Calculate recent performance
        if len(self.recent_performance) >= 3:  # Need at least 3 trials
            recent_perf = np.mean(self.recent_performance)
            
            # Determine direction for contrast adjustment
            if recent_perf > self.target_performance:
                # Performance too high, make task harder (decrease contrast)
                new_direction = 'down'
                new_contrast = max(self.min_contrast, 
                                 self.current_contrast - self.step_size_up)
            else:
                # Performance too low, make task easier (increase contrast)
                new_direction = 'up'
                new_contrast = min(self.max_contrast,
                                 self.current_contrast + self.step_size_down)
            
            # Check for reversal
            if self.last_direction is not None and new_direction != self.last_direction:
                self.reversals.append({
                    'trial': self.n_trials,
                    'contrast': self.current_contrast,
                    'performance': recent_perf
                })
                self.logger.info(f"Reversal {len(self.reversals)} at trial {self.n_trials}, "
                               f"contrast={self.current_contrast:.3f}, perf={recent_perf:.3f}")
            
            self.last_direction = new_direction
            self.current_contrast = new_contrast
        
        self.logger.info(f"Trial {self.n_trials}: correct={correct}, "
                        f"contrast={self.current_contrast:.3f}, "
                        f"recent_perf={np.mean(self.recent_performance):.3f}")
    
    def get_current_contrast(self):
        """Get current contrast level for next trial."""
        return self.current_contrast
    
    def get_threshold_estimate(self):
        """
        Calculate threshold estimate based on recent reversals.
        Returns None if insufficient reversals.
        """
        if len(self.reversals) < self.n_reversals_for_threshold:
            return None
        
        # Use last N reversals for threshold
        recent_reversals = self.reversals[-self.n_reversals_for_threshold:]
        threshold = np.mean([r['contrast'] for r in recent_reversals])
        return threshold
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        overall_performance = self.n_correct / self.n_trials if self.n_trials > 0 else 0
        recent_performance = np.mean(self.recent_performance) if self.recent_performance else 0
        
        return {
            'overall_performance': overall_performance,
            'recent_performance': recent_performance,
            'n_trials': self.n_trials,
            'n_correct': self.n_correct,
            'current_contrast': self.current_contrast,
            'n_reversals': len(self.reversals),
            'threshold_estimate': self.get_threshold_estimate()
        }
    
    def is_converged(self, min_trials=20, min_reversals=6):
        """
        Check if staircase has converged to stable performance.
        
        Args:
            min_trials: Minimum number of trials
            min_reversals: Minimum number of reversals
        """
        if self.n_trials < min_trials or len(self.reversals) < min_reversals:
            return False
        
        # Check if recent performance is close to target
        if len(self.recent_performance) >= 10:
            recent_perf = np.mean(list(self.recent_performance)[-10:])
            performance_diff = abs(recent_perf - self.target_performance)
            
            # Consider converged if within 5% of target performance
            return performance_diff < 0.05
        
        return False
    
    def reset(self):
        """Reset staircase to initial state."""
        self.current_contrast = self.initial_contrast
        self.trial_history = []
        self.recent_performance = deque(maxlen=self.recent_performance.maxlen)
        self.reversals = []
        self.last_direction = None
        self.n_correct = 0
        self.n_trials = 0
    
    def get_contrast_for_difficulty(self, target_performance_level):
        """
        Get contrast level for a specific target performance.
        Useful for creating stimuli of different difficulties.
        
        Args:
            target_performance_level: Desired performance level (0-1)
        """
        threshold = self.get_threshold_estimate()
        if threshold is None:
            return self.current_contrast
        
        # Simple linear mapping based on psychometric function
        # This is a simplified model - could be improved with proper psychometric fitting
        if target_performance_level > self.target_performance:
            # Easier than current threshold
            multiplier = 1 + (target_performance_level - self.target_performance) * 2
        else:
            # Harder than current threshold  
            multiplier = target_performance_level / self.target_performance
        
        contrast = threshold * multiplier
        return np.clip(contrast, self.min_contrast, self.max_contrast)

class MultiLocationStaircase:
    """
    Manages separate staircases for each of the 6 Gabor locations.
    This allows for location-specific threshold estimation.
    """
    
    def __init__(self, n_locations=6, **staircase_kwargs):
        self.n_locations = n_locations
        self.staircases = {}
        
        for loc in range(n_locations):
            self.staircases[loc] = AdaptiveStaircase(**staircase_kwargs)
    
    def update(self, location, correct):
        """Update the staircase for a specific location."""
        self.staircases[location].update(correct)
    
    def get_contrast(self, location):
        """Get current contrast for a specific location."""
        return self.staircases[location].get_current_contrast()
    
    def get_global_stats(self):
        """Get performance statistics across all locations."""
        all_stats = {}
        for loc, staircase in self.staircases.items():
            all_stats[f'location_{loc}'] = staircase.get_performance_stats()
        
        # Calculate global performance
        total_trials = sum(s.n_trials for s in self.staircases.values())
        total_correct = sum(s.n_correct for s in self.staircases.values())
        global_performance = total_correct / total_trials if total_trials > 0 else 0
        
        all_stats['global'] = {
            'performance': global_performance,
            'total_trials': total_trials,
            'total_correct': total_correct
        }
        
        return all_stats
    
    def is_converged(self):
        """Check if all staircases have converged."""
        return all(s.is_converged() for s in self.staircases.values())
    
    def reset_all(self):
        """Reset all staircases."""
        for staircase in self.staircases.values():
            staircase.reset()

# Test and simulation functions
def simulate_staircase(n_trials=100, true_threshold=0.4, noise_level=0.1):
    """
    Simulate staircase procedure with known psychometric function.
    
    Args:
        n_trials: Number of trials to simulate
        true_threshold: True threshold contrast
        noise_level: Amount of noise in responses
    """
    staircase = AdaptiveStaircase(initial_contrast=0.5)
    
    def psychometric_function(contrast, threshold=true_threshold, slope=3):
        """Simulate psychometric function."""
        # Logistic function
        x = slope * (contrast - threshold)
        return 1 / (1 + np.exp(-x))
    
    results = []
    
    for trial in range(n_trials):
        current_contrast = staircase.get_current_contrast()
        
        # Simulate response based on psychometric function + noise
        p_correct = psychometric_function(current_contrast)
        p_correct += np.random.normal(0, noise_level)
        p_correct = np.clip(p_correct, 0, 1)
        
        correct = np.random.random() < p_correct
        staircase.update(correct)
        
        results.append({
            'trial': trial + 1,
            'contrast': current_contrast,
            'p_correct': p_correct,
            'correct': correct,
            'performance': staircase.get_performance_stats()['recent_performance']
        })
    
    return staircase, results

if __name__ == "__main__":
    # Test staircase simulation
    staircase, results = simulate_staircase(n_trials=50)
    
    print("Staircase simulation complete!")
    print(f"Final performance stats: {staircase.get_performance_stats()}")
    print(f"Threshold estimate: {staircase.get_threshold_estimate()}")
    print(f"Converged: {staircase.is_converged()}") 