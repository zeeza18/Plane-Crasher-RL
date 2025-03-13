import random
import pygame

class CloudEffect:
    """Generates decorative clouds evenly distributed across the entire background"""
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.clouds = []
        self.max_clouds = 15  # Maximum number of clouds
        
        # Generate initial clouds distributed across the screen
        self._generate_initial_clouds()
    
    def _generate_initial_clouds(self):
        """Create initial set of clouds evenly distributed across the entire screen"""
        # Divide the screen into sections for better distribution
        sections_x = 5
        sections_y = 6  # Increased to cover more vertical space
        
        section_width = self.screen_width // sections_x
        section_height = self.screen_height // sections_y  # No longer restricting to upper half
        
        # Create clouds in each section
        for x_section in range(sections_x):
            for y_section in range(sections_y):
                # Add some randomness to position within section
                x_pos = (x_section * section_width) + random.randint(0, section_width - 1)
                y_pos = (y_section * section_height) + random.randint(10, section_height - 1)
                
                # Don't create a cloud in every section (70% chance)
                if random.random() < 0.7:
                    self._create_cloud(x_pos, y_pos)
    
    def _create_cloud(self, x_pos=None, y_pos=None):
        """Create a new cloud"""
        if x_pos is None:
            x_pos = self.screen_width + random.randint(50, 200)
            
        if y_pos is None:
            # Allow clouds to appear across the full height of the screen
            # Keep a margin of 50px from the bottom to avoid overlapping with ground elements
            y_pos = random.randint(30, self.screen_height - 50)
        
        # Cloud properties - reduced sizes to 60%
        num_circles = random.randint(4, 8)  # Slightly reduced maximum number of circles
        size = random.randint(12, 27)       # Reduced from 20-45 to 12-27 (60%)
        speed = random.uniform(0.3, 1.5)    # Horizontal speed
        color_value = random.randint(230, 255)  # Whiteness (230-255)
        
        # Create the cloud structure (collection of circles)
        circles = []
        
        for i in range(num_circles):
            # Position circles relative to cloud center
            offset_x = random.randint(-size//2, size//2)
            offset_y = random.randint(-size//4, size//4)
            circle_size = random.randint(size//2, size)
            
            circles.append({
                'offset_x': offset_x,
                'offset_y': offset_y,
                'size': circle_size
            })
            
        # Create the cloud object
        cloud = {
            'x': x_pos,
            'y': y_pos,
            'speed': speed,
            'circles': circles,
            'color': (color_value, color_value, color_value),
            'size': size
        }
        
        self.clouds.append(cloud)
        
    def update(self):
        """Update cloud positions and create new ones as needed"""
        # Move clouds
        clouds_to_remove = []
        for cloud in self.clouds:
            cloud['x'] -= cloud['speed']
            
            # Mark clouds that moved off-screen for removal
            if cloud['x'] + cloud['size'] * 2 < 0:
                clouds_to_remove.append(cloud)
        
        # Remove off-screen clouds
        for cloud in clouds_to_remove:
            self.clouds.remove(cloud)
        
        # Add new clouds as needed - more frequent new clouds
        if len(self.clouds) < self.max_clouds and random.random() < 0.05:  # 5% chance per frame
            # Create new clouds at different heights across the entire screen
            y_pos = random.randint(30, self.screen_height - 50)
            self._create_cloud(None, y_pos)
            
        # Occasional extra cloud to fill gaps
        if len(self.clouds) < self.max_clouds - 3 and random.random() < 0.01:  # 1% chance for burst of clouds
            for _ in range(2):
                y_pos = random.randint(30, self.screen_height - 50)
                self._create_cloud(None, y_pos)
    
    def draw(self, screen):
        """Draw all clouds"""
        for cloud in self.clouds:
            # Draw each circle in the cloud
            for circle in cloud['circles']:
                pygame.draw.circle(
                    screen,
                    cloud['color'],
                    (
                        int(cloud['x'] + circle['offset_x']),
                        int(cloud['y'] + circle['offset_y'])
                    ),
                    circle['size']
                )