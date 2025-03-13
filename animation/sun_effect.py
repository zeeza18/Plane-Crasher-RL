import pygame
import math
import random

class SunEffect:
    """
    Animated effect that appears as a sun in the top-left corner of the screen.
    Features rays, pulsing, and a gentle wobble effect.
    """
    def __init__(self, screen_width, screen_height):
        """Initialize the sun effect."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Position - exactly at the corner (0,0)
        self.radius = 70  # Larger radius to make half-sun look right
        self.center_x = 0
        self.center_y = 0
        
        # Sun parameters
        self.base_color = (255, 215, 0)  # Golden yellow
        self.outer_color = (255, 165, 0)  # Orange
        self.highlight_color = (255, 255, 200)  # Bright yellow
        
        # Animation parameters
        self.time = 0
        self.pulse_speed = 0.03
        self.wobble_speed = 0.01
        self.pulse_amplitude = 3
        self.wobble_amplitude = 1
        
        # Sun rays
        self.num_rays = 8
        self.rays = []
        self.ray_lengths = []
        self.ray_speeds = []
        
        # Initialize rays with random lengths and speeds
        # Only use angles that would be visible (between 0 and 90 degrees)
        for i in range(self.num_rays):
            angle = (math.pi / 2 / (self.num_rays - 1)) * i
            length = random.uniform(self.radius * 0.8, self.radius * 1.2)
            speed = random.uniform(0.02, 0.05)
            self.rays.append(angle)
            self.ray_lengths.append(length)
            self.ray_speeds.append(speed)
    
    def update(self):
        """Update the sun animation."""
        # Update time
        self.time += 0.016  # Roughly 60 FPS
        
        # Update ray animations
        for i in range(self.num_rays):
            # Make the rays expand and contract
            self.ray_lengths[i] = (self.radius * 1.0 + 
                                math.sin(self.time * self.ray_speeds[i]) * 
                                self.radius * 0.3)
    
    def draw(self, screen):
        """Draw the half-sun effect on the screen."""
        # Calculate pulsing and wobbling effects
        pulse = math.sin(self.time * self.pulse_speed) * self.pulse_amplitude
        # Limit wobble to ensure sun stays anchored to corner
        wobble_x = max(0, math.sin(self.time * self.wobble_speed) * self.wobble_amplitude)
        wobble_y = max(0, math.cos(self.time * self.wobble_speed) * self.wobble_amplitude)
        
        current_radius = self.radius + pulse
        current_x = self.center_x + wobble_x
        current_y = self.center_y + wobble_y
        
        # Draw sun rays
        for i in range(self.num_rays):
            angle = self.rays[i]
            length = self.ray_lengths[i]
            
            # Calculate ray start and end points
            start_x = current_x + (current_radius - 5) * math.cos(angle)
            start_y = current_y + (current_radius - 5) * math.sin(angle)
            end_x = current_x + length * math.cos(angle)
            end_y = current_y + length * math.sin(angle)
            
            # Draw a ray with gradient color
            pygame.draw.line(
                screen, 
                self.outer_color,
                (start_x, start_y), 
                (end_x, end_y), 
                3
            )
        
        # Draw outer glow (gradient circles from largest to smallest)
        for r in range(int(current_radius) + 15, int(current_radius) - 5, -5):
            alpha = 200 - (int(current_radius) + 15 - r) * 10
            alpha = max(0, min(alpha, 200))
            
            # Draw only a quarter circle for the glow (top-left quadrant)
            glow_surface = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surface,
                (*self.outer_color, alpha),
                (0, 0),
                r
            )
            # Only blit the top-left quadrant
            screen.blit(glow_surface, (current_x - 0, current_y - 0))
        
        # Draw main sun body (quarter circle)
        # Create a surface for the sun
        body_surface = pygame.Surface((int(current_radius*2), int(current_radius*2)), pygame.SRCALPHA)
        
        # Draw the full circle on the surface
        pygame.draw.circle(
            body_surface, 
            self.base_color,
            (0, 0), 
            int(current_radius)
        )
        
        # Add highlight to the sun
        pygame.draw.circle(
            body_surface, 
            self.highlight_color,
            (int(-current_radius * 0.2), 
             int(-current_radius * 0.2)), 
            int(current_radius * 0.5)
        )
        
        # Blit only the corner of the surface to get a quarter circle
        screen.blit(body_surface, (current_x, current_y))