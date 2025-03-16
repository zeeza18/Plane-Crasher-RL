# Import pygame library for graphics and animation capabilities
import pygame
# Import math module for trigonometric functions and mathematical operations
import math
# Import random module for generating randomized effect properties
import random

# Define the SunEffect class that creates an animated sun in the corner
class SunEffect:
    """
    Animated effect that appears as a sun in the top-left corner of the screen.
    Features rays, pulsing, and a gentle wobble effect.
    """
    # Constructor method initializes the sun animation
    # Parameters:
    #   screen_width: width of game screen in pixels
    #   screen_height: height of game screen in pixels
    def __init__(self, screen_width, screen_height):
        """Initialize the sun effect."""
        # Store screen dimensions for reference
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Position - exactly at the corner (0,0)
        # Larger radius to create a proper half-sun appearance
        self.radius = 70  # Size of the sun circle
        # Corner coordinates (0,0) to place sun in top-left
        self.center_x = 0
        self.center_y = 0
        
        # Define sun color parameters for realistic appearance
        self.base_color = (255, 215, 0)     # Golden yellow for main sun body
        self.outer_color = (255, 165, 0)    # Orange for outer edge and rays
        self.highlight_color = (255, 255, 200)  # Bright yellow for sun highlight
        
        # Animation parameters to control motion effects
        self.time = 0  # Tracks animation time
        self.pulse_speed = 0.03  # Controls how fast the sun pulses (grows/shrinks)
        self.wobble_speed = 0.01  # Controls how fast the sun wobbles (moves slightly)
        self.pulse_amplitude = 3  # Maximum size change during pulsing
        self.wobble_amplitude = 1  # Maximum position change during wobbling
        
        # Sun rays configuration
        self.num_rays = 8  # Number of light rays emanating from the sun
        self.rays = []  # Will store ray angles
        self.ray_lengths = []  # Will store ray length values
        self.ray_speeds = []  # Will store ray animation speeds
        
        # Initialize rays with random properties
        # Only create rays in the visible quadrant (between 0 and 90 degrees)
        for i in range(self.num_rays):
            # Calculate angle in radians (evenly distributed from 0 to 90 degrees)
            angle = (math.pi / 2 / (self.num_rays - 1)) * i
            # Random ray length for varied appearance
            length = random.uniform(self.radius * 0.8, self.radius * 1.2)
            # Random animation speed for each ray
            speed = random.uniform(0.02, 0.05)
            # Store values in their respective lists
            self.rays.append(angle)
            self.ray_lengths.append(length)
            self.ray_speeds.append(speed)
    
    # Method to update sun animation state
    # No parameters or return value
    def update(self):
        """Update the sun animation."""
        # Increment time value (assumes approximately 60 FPS)
        self.time += 0.016  # ~16ms per frame
        
        # Update ray animations to create pulsing effect
        for i in range(self.num_rays):
            # Calculate new ray length using sine wave for smooth oscillation
            # Base length is radius, with ±30% variation based on sine wave
            self.ray_lengths[i] = (self.radius * 1.0 + 
                                math.sin(self.time * self.ray_speeds[i]) * 
                                self.radius * 0.3)
    
    # Method to draw the sun effect on screen
    # Parameters:
    #   screen: pygame surface to draw on
    # No return value
    def draw(self, screen):
        """Draw the half-sun effect on the screen."""
        # Calculate current pulsing and wobbling effects
        # Use sine wave for smooth cyclical change
        pulse = math.sin(self.time * self.pulse_speed) * self.pulse_amplitude
        # Limit wobble to ensure sun stays anchored to corner (no negative values)
        wobble_x = max(0, math.sin(self.time * self.wobble_speed) * self.wobble_amplitude)
        wobble_y = max(0, math.cos(self.time * self.wobble_speed) * self.wobble_amplitude)
        
        # Apply pulse effect to current radius
        current_radius = self.radius + pulse
        # Apply wobble effect to current position
        current_x = self.center_x + wobble_x
        current_y = self.center_y + wobble_y
        
        # Draw sun rays emanating from the sun
        for i in range(self.num_rays):
            # Get this ray's properties
            angle = self.rays[i]
            length = self.ray_lengths[i]
            
            # Calculate ray start point (at sun's edge)
            # Using polar coordinates (angle + radius) converted to Cartesian (x,y)
            start_x = current_x + (current_radius - 5) * math.cos(angle)
            start_y = current_y + (current_radius - 5) * math.sin(angle)
            # Calculate ray end point
            end_x = current_x + length * math.cos(angle)
            end_y = current_y + length * math.sin(angle)
            
            # Draw a ray as a line with the outer sun color
            pygame.draw.line(
                screen,               # Surface to draw on
                self.outer_color,     # Orange color for rays
                (start_x, start_y),   # Start point at sun's edge
                (end_x, end_y),       # End point extending outward
                3                     # Line width (3 pixels)
            )
        
        # Draw outer glow as a series of translucent circles
        # Creates gradient effect from largest to smallest
        for r in range(int(current_radius) + 15, int(current_radius) - 5, -5):
            # Calculate alpha (transparency) for this circle
            # Further from center = more transparent
            alpha = 200 - (int(current_radius) + 15 - r) * 10
            # Ensure alpha stays within valid range (0-200)
            alpha = max(0, min(alpha, 200))
            
            # Create a transparent surface for this glow circle
            glow_surface = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            # Draw circle with calculated alpha value
            pygame.draw.circle(
                glow_surface,             # Surface to draw on
                (*self.outer_color, alpha),  # RGBA color with transparency
                (0, 0),                  # Center at top-left corner
                r                        # Circle radius
            )
            # Draw only the top-left quadrant by positioning at corner
            screen.blit(glow_surface, (current_x - 0, current_y - 0))
        
        # Draw main sun body as a quarter circle
        # Create a transparent surface for the sun body
        body_surface = pygame.Surface((int(current_radius*2), int(current_radius*2)), pygame.SRCALPHA)
        
        # Draw the full circle on the surface at the top-left corner
        pygame.draw.circle(
            body_surface,         # Surface to draw on
            self.base_color,      # Golden yellow for main sun
            (0, 0),              # Center at top-left corner
            int(current_radius)   # Current sun radius
        )
        
        # Add highlight circle to create realistic lighting effect
        # Offset from center to create illusion of depth/light source
        pygame.draw.circle(
            body_surface,           # Surface to draw on
            self.highlight_color,   # Bright yellow for highlight
            (int(-current_radius * 0.2),   # Offset X from center
             int(-current_radius * 0.2)),  # Offset Y from center
            int(current_radius * 0.5)      # Smaller radius (50% of sun)
        )
        
        # Draw only the visible quarter by positioning at corner
        # This creates the effect of a sun rising from the corner
        screen.blit(body_surface, (current_x, current_y))