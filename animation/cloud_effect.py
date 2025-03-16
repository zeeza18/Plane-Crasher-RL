# Import the random module for generating random properties (positions, sizes, etc.)
import random
# Import pygame library for drawing graphics
import pygame

# Define CloudEffect class to create cloud background decoration
class CloudEffect:
    """Generates decorative clouds evenly distributed across the entire background"""
    # Constructor method initializes the cloud effect system
    # Parameters:
    #   screen_width: width of game screen in pixels
    #   screen_height: height of game screen in pixels
    def __init__(self, screen_width, screen_height):
        # Store screen dimensions for reference
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Initialize empty list to store all cloud objects
        self.clouds = []
        # Set maximum number of clouds to maintain performance
        self.max_clouds = 15  # Limits total clouds to prevent overcrowding and performance issues
        
        # Generate the initial set of clouds distributed across screen
        # This ensures screen has clouds from the start instead of waiting for them to appear
        self._generate_initial_clouds()
    
    # Private method to populate the screen with initial clouds
    # This creates a well-distributed starting set of clouds
    # No parameters or return value
    def _generate_initial_clouds(self):
        """Create initial set of clouds evenly distributed across the entire screen"""
        # Divide the screen into grid sections for even cloud distribution
        sections_x = 5  # Number of horizontal sections
        sections_y = 6  # Number of vertical sections (increased to cover more vertical space)
        
        # Calculate the size of each grid section
        section_width = self.screen_width // sections_x
        section_height = self.screen_height // sections_y  # Uses full screen height
        
        # Create clouds in each section of the grid
        for x_section in range(sections_x):
            for y_section in range(sections_y):
                # Add randomness to position within each section
                # This prevents clouds from appearing in a perfect grid pattern
                x_pos = (x_section * section_width) + random.randint(0, section_width - 1)
                y_pos = (y_section * section_height) + random.randint(10, section_height - 1)
                
                # Only place a cloud in 70% of sections to create natural variation
                # This simulates real cloud formations which aren't perfectly distributed
                if random.random() < 0.7:
                    # Create a cloud at the calculated position
                    self._create_cloud(x_pos, y_pos)
    
    # Private method to create a single new cloud
    # Parameters:
    #   x_pos: horizontal position (optional, defaults to off-screen right)
    #   y_pos: vertical position (optional, defaults to random height)
    # No return value
    def _create_cloud(self, x_pos=None, y_pos=None):
        """Create a new cloud"""
        # If no x position provided, place cloud off-screen to the right
        # This is used for new clouds that will enter during gameplay
        if x_pos is None:
            x_pos = self.screen_width + random.randint(50, 200)
            
        # If no y position provided, choose random height within screen bounds
        if y_pos is None:
            # Place clouds anywhere vertically with margins from top and bottom
            # 30px from top, 50px from bottom to avoid overlapping ground elements
            y_pos = random.randint(30, self.screen_height - 50)
        
        # Define cloud appearance properties with randomized values
        # Number of circles that make up this cloud (affects cloud shape)
        num_circles = random.randint(4, 8)  # Slightly reduced from typical cloud implementations
        # Base size of cloud circles (60% smaller than original implementation)
        size = random.randint(12, 27)  # Reduced size for more subtle clouds
        # Movement speed varies between clouds for natural appearance
        speed = random.uniform(0.3, 1.5)  # Pixels per frame
        # Color variation - all circles are white-to-light-gray
        color_value = random.randint(230, 255)  # RGB values (same for R,G,B creates white/gray)
        
        # Create the cloud structure as a collection of overlapping circles
        # This simulates the puffy appearance of real clouds
        circles = []
        
        # Generate each circle that makes up this cloud
        for i in range(num_circles):
            # Position circles with slight offsets from cloud center
            # This creates irregular cloud shapes instead of perfect circles
            offset_x = random.randint(-size//2, size//2)  # Horizontal offset
            offset_y = random.randint(-size//4, size//4)  # Vertical offset (less than horizontal)
            # Each circle has randomized size relative to base cloud size
            circle_size = random.randint(size//2, size)
            
            # Add this circle's properties to the cloud's circles list
            circles.append({
                'offset_x': offset_x,  # X offset from cloud center
                'offset_y': offset_y,  # Y offset from cloud center
                'size': circle_size    # Radius of this circle
            })
            
        # Create the complete cloud object as a dictionary
        cloud = {
            'x': x_pos,  # Horizontal position of cloud center
            'y': y_pos,  # Vertical position of cloud center
            'speed': speed,  # Movement speed (pixels per frame)
            'circles': circles,  # List of circle definitions that form the cloud
            'color': (color_value, color_value, color_value),  # RGB color (white/gray)
            'size': size  # Base size parameter (used for off-screen detection)
        }
        
        # Add this cloud to the clouds list for tracking and rendering
        self.clouds.append(cloud)
        
    # Method to update all clouds (called once per frame)
    # No parameters or return value
    def update(self):
        """Update cloud positions and create new ones as needed"""
        # Move all existing clouds leftward based on their speed
        clouds_to_remove = []  # Track clouds that move off-screen
        for cloud in self.clouds:
            # Update cloud position by subtracting its speed from x coordinate
            cloud['x'] -= cloud['speed']
            
            # Identify clouds that have moved completely off the left edge of screen
            # Uses 2x size to ensure cloud is fully off-screen before removal
            if cloud['x'] + cloud['size'] * 2 < 0:
                clouds_to_remove.append(cloud)
        
        # Remove clouds that moved off-screen to maintain performance
        # Prevents accumulation of invisible clouds
        for cloud in clouds_to_remove:
            self.clouds.remove(cloud)
        
        # Add new clouds as needed to maintain cloud coverage
        # 5% chance per frame to add a new cloud when below max count
        if len(self.clouds) < self.max_clouds and random.random() < 0.05:
            # Create new cloud at random height
            y_pos = random.randint(30, self.screen_height - 50)
            # Pass None for x_pos to place cloud off-screen right
            self._create_cloud(None, y_pos)
            
        # Occasionally add multiple clouds to fill visual gaps
        # 1% chance for burst of clouds when significantly below max
        if len(self.clouds) < self.max_clouds - 3 and random.random() < 0.01:
            # Create 2 clouds with different heights
            for _ in range(2):
                y_pos = random.randint(30, self.screen_height - 50)
                self._create_cloud(None, y_pos)
    
    # Method to draw all clouds to the screen
    # Parameters:
    #   screen: pygame surface to draw on
    # No return value
    def draw(self, screen):
        """Draw all clouds"""
        # Iterate through all cloud objects
        for cloud in self.clouds:
            # Draw each circle that makes up this cloud
            # Clouds are made of multiple overlapping circles
            for circle in cloud['circles']:
                # Draw a filled circle for each component
                pygame.draw.circle(
                    screen,  # Surface to draw on
                    cloud['color'],  # Color of this cloud (white/gray)
                    (
                        # X position (cloud center + circle offset)
                        int(cloud['x'] + circle['offset_x']),
                        # Y position (cloud center + circle offset)
                        int(cloud['y'] + circle['offset_y'])
                    ),
                    circle['size']  # Radius of this circle
                )