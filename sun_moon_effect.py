import pygame
import math
import random

class SunMoonEffect:
    """
    Animated effect that appears as either a sun or moon in the top-left corner of the screen.
    Features rays, pulsing, and a gentle wobble effect.
    """
    def __init__(self, screen_width, screen_height):
        """Initialize the sun/moon effect."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Position - exactly at the corner (0,0)
        self.radius = 70  # Larger radius to make half-sun/moon look right
        self.center_x = 0
        self.center_y = 0
        
        # Sun parameters
        self.sun_base_color = (255, 215, 0)  # Golden yellow
        self.sun_outer_color = (255, 165, 0)  # Orange
        self.sun_highlight_color = (255, 255, 200)  # Bright yellow
        
        # Moon parameters
        self.moon_base_color = (220, 220, 220)  # Light gray
        self.moon_outer_color = (180, 180, 180)  # Slightly darker gray
        self.moon_highlight_color = (255, 255, 255)  # Pure white
        
        # Current colors (start with sun)
        self.base_color = self.sun_base_color
        self.outer_color = self.sun_outer_color
        self.highlight_color = self.sun_highlight_color
        
        # Mode tracking
        self.is_moon = False
        
        # Animation parameters
        self.time = 0
        self.pulse_speed = 0.03
        self.wobble_speed = 0.01
        self.pulse_amplitude = 3
        self.wobble_amplitude = 1
        
        # Sun/moon rays
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
        
        # Moon craters (only visible in moon mode)
        self.craters = []
        for _ in range(3):
            # Create random craters within the moon radius
            x_offset = random.uniform(self.radius * 0.2, self.radius * 0.8)
            y_offset = random.uniform(self.radius * 0.2, self.radius * 0.8)
            crater_size = random.uniform(5, 10)
            self.craters.append({
                'x': x_offset, 
                'y': y_offset, 
                'size': crater_size
            })
    
    def set_moon_mode(self, is_moon):
        """Switch between sun and moon mode"""
        if is_moon != self.is_moon:
            self.is_moon = is_moon
            if is_moon:
                self.base_color = self.moon_base_color
                self.outer_color = self.moon_outer_color
                self.highlight_color = self.moon_highlight_color
            else:
                self.base_color = self.sun_base_color
                self.outer_color = self.sun_outer_color
                self.highlight_color = self.sun_highlight_color
    
    def update(self):
        """Update the sun/moon animation."""
        # Update time
        self.time += 0.016  # Roughly 60 FPS
        
        # Update ray animations (rays only visible for sun)
        for i in range(self.num_rays):
            # Make the rays expand and contract
            self.ray_lengths[i] = (self.radius * 1.0 + 
                                math.sin(self.time * self.ray_speeds[i]) * 
                                self.radius * 0.3)
    
    def draw(self, screen):
        """Draw the half-sun/moon effect on the screen."""
        # Calculate pulsing and wobbling effects
        pulse = math.sin(self.time * self.pulse_speed) * self.pulse_amplitude
        # Limit wobble to ensure sun/moon stays anchored to corner
        wobble_x = max(0, math.sin(self.time * self.wobble_speed) * self.wobble_amplitude)
        wobble_y = max(0, math.cos(self.time * self.wobble_speed) * self.wobble_amplitude)
        
        current_radius = self.radius + pulse
        current_x = self.center_x + wobble_x
        current_y = self.center_y + wobble_y
        
        # Draw sun rays (only for sun mode)
        if not self.is_moon:
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
        
        # Draw main sun/moon body (quarter circle)
        # Create a surface for the sun/moon
        body_surface = pygame.Surface((int(current_radius*2), int(current_radius*2)), pygame.SRCALPHA)
        
        # Draw the full circle on the surface
        pygame.draw.circle(
            body_surface, 
            self.base_color,
            (0, 0), 
            int(current_radius)
        )
        
        # Add highlight to the sun/moon
        pygame.draw.circle(
            body_surface, 
            self.highlight_color,
            (int(-current_radius * 0.2), 
             int(-current_radius * 0.2)), 
            int(current_radius * 0.5)
        )
        
        # Add craters for moon mode
        if self.is_moon:
            for crater in self.craters:
                # Draw darker circles for craters
                pygame.draw.circle(
                    body_surface,
                    (150, 150, 150),  # Darker gray
                    (int(-current_radius + crater['x']), 
                     int(-current_radius + crater['y'])),
                    crater['size']
                )
        
        # Blit only the corner of the surface to get a quarter circle
        screen.blit(body_surface, (current_x, current_y))

# Test code
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    
    effect = SunMoonEffect(800, 600)
    
    running = True
    is_moon = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    # Toggle moon mode
                    is_moon = not is_moon
                    effect.set_moon_mode(is_moon)
        
        # Set background based on mode
        if is_moon:
            screen.fill((25, 25, 112))  # Dark blue for night
        else:
            screen.fill((135, 206, 235))  # Sky blue for day
        
        effect.update()
        effect.draw(screen)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()