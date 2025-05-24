import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from PIL import Image
import io
import base64

class GaborStimulus:
    """
    Generates Gabor patch stimuli for metacognition experiments.
    Based on the behavioral task described in the research paper.
    """
    
    def __init__(self, image_size=400, patch_size=60, spatial_freq=0.1):
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_freq = spatial_freq
        self.center = image_size // 2
        
        # Six locations around central fixation (60 degrees apart)
        self.locations = self._generate_locations()
        
    def _generate_locations(self):
        """Generate 6 locations around central fixation point."""
        radius = self.image_size // 4  # Distance from center
        locations = []
        for i in range(6):
            angle = i * np.pi / 3  # 60 degrees apart
            x = self.center + radius * np.cos(angle)
            y = self.center + radius * np.sin(angle)
            locations.append((int(x), int(y)))
        return locations
    
    def create_gabor_patch(self, contrast=1.0, orientation=0, phase=0):
        """
        Create a Gabor patch with specified parameters.
        
        Args:
            contrast: Contrast level (0-1)
            orientation: Orientation in radians
            phase: Phase offset
        """
        # Create coordinate grids
        x = np.linspace(-1, 1, self.patch_size)
        y = np.linspace(-1, 1, self.patch_size)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
        Y_rot = -X * np.sin(orientation) + Y * np.cos(orientation)
        
        # Gaussian envelope
        sigma = 0.3
        gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
        
        # Sinusoidal grating
        grating = np.cos(2 * np.pi * self.spatial_freq * X_rot + phase)
        
        # Combine Gaussian and grating with contrast
        gabor = contrast * gaussian * grating
        
        # Normalize to 0-255 range
        gabor = ((gabor + 1) / 2 * 255).astype(np.uint8)
        
        return gabor
    
    def create_stimulus_display(self, target_location, target_contrast, 
                              distractor_contrast=0.3, orientation=0):
        """
        Create a stimulus display with Gabor patches at specified locations.
        
        Args:
            target_location: Index (0-5) of target location
            target_contrast: Contrast of target patch
            distractor_contrast: Contrast of distractor patches
            orientation: Orientation of all patches
        """
        # Create background
        background = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 128
        
        # Add fixation point
        cv2.circle(background, (self.center, self.center), 3, (0, 0, 0), -1)
        
        # Add Gabor patches
        for i, (x, y) in enumerate(self.locations):
            if i == target_location:
                contrast = target_contrast
            else:
                contrast = distractor_contrast
                
            gabor = self.create_gabor_patch(contrast, orientation)
            
            # Convert to 3-channel
            gabor_rgb = np.stack([gabor, gabor, gabor], axis=2)
            
            # Calculate patch boundaries
            half_size = self.patch_size // 2
            x_start = max(0, x - half_size)
            x_end = min(self.image_size, x + half_size)
            y_start = max(0, y - half_size)
            y_end = min(self.image_size, y + half_size)
            
            # Calculate gabor boundaries
            gx_start = max(0, half_size - x)
            gx_end = gx_start + (x_end - x_start)
            gy_start = max(0, half_size - y)
            gy_end = gy_start + (y_end - y_start)
            
            # Blend Gabor patch with background
            alpha = 0.7
            background[y_start:y_end, x_start:x_end] = (
                alpha * gabor_rgb[gy_start:gy_end, gx_start:gx_end] +
                (1 - alpha) * background[y_start:y_end, x_start:x_end]
            ).astype(np.uint8)
        
        return background
    
    def create_temporal_sequence(self, first_target_loc, second_target_loc,
                               first_contrast, second_contrast,
                               interval_duration=0.5):
        """
        Create two temporal intervals for the 2AFC task.
        
        Args:
            first_target_loc: Target location in first interval
            second_target_loc: Target location in second interval  
            first_contrast: Target contrast in first interval
            second_contrast: Target contrast in second interval
            interval_duration: Duration of each interval in seconds
        """
        first_interval = self.create_stimulus_display(
            first_target_loc, first_contrast
        )
        
        second_interval = self.create_stimulus_display(
            second_target_loc, second_contrast
        )
        
        return first_interval, second_interval
    
    def image_to_base64(self, image):
        """Convert numpy image to base64 string for API transmission."""
        if len(image.shape) == 3:
            # Convert BGR to RGB if needed
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def visualize_stimulus(self, image, title="Gabor Stimulus"):
        """Visualize the stimulus for debugging."""
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        
        # Add location markers for reference
        for i, (x, y) in enumerate(self.locations):
            circle = Circle((x, y), self.patch_size//2, 
                          fill=False, color='red', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            plt.text(x, y-self.patch_size//2-10, f'L{i}', 
                    ha='center', va='top', color='red', fontsize=10)
        
        plt.tight_layout()
        return plt.gcf()

# Test functions
def test_gabor_generation():
    """Test Gabor patch generation."""
    stimulus = GaborStimulus()
    
    # Create test stimulus
    test_image = stimulus.create_stimulus_display(
        target_location=2, 
        target_contrast=0.8,
        distractor_contrast=0.3
    )
    
    # Visualize
    fig = stimulus.visualize_stimulus(test_image, "Test Gabor Stimulus")
    plt.show()
    
    return stimulus, test_image

if __name__ == "__main__":
    test_gabor_generation() 