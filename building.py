import pygame
import random

class Building:
    # Light colors for buildings (excluding blue since it's the sky color)
    COLORS = [
        (255, 192, 203),  # Light pink
        (144, 238, 144),  # Light green
        (255, 255, 153),  # Light yellow
        (255, 160, 122),  # Light salmon
        (216, 191, 216),  # Light purple
        (173, 216, 230),  # Light cyan
    ]
    
    WINDOW_COLOR = (120, 120, 120)  # Gray windows
    DOOR_COLOR = (139, 69, 19)      # Brown door color
    WINDOW_MARGIN = 7  # Margin between windows
    WINDOW_SIZE = 10  # Size of each window
    
    def __init__(self, x, width, height, position='top'):
        self.x = x
        self.width = width
        self.height = height
        self.position = position  # 'top' or 'bottom'
        
        # Choose random color for this building
        self.color = random.choice(self.COLORS)
        
        # Door dimensions - calculate these BEFORE creating windows
        self.door_width = self.width // 3
        self.door_height = min(20, max(self.height // 6, 12))  # Ensure reasonable size
        
        # Calculate door position
        self.door_x = self.width // 2 - self.door_width // 2  # Center horizontally
        
        # For TOP buildings, door should be at the TOP (not bottom)
        # For BOTTOM buildings, door should be at the BOTTOM (not top)
        if self.position == 'top':
            self.door_y = 0  # Door at the top of top buildings
        else:  # bottom
            self.door_y = self.height - self.door_height  # Door at the bottom of bottom buildings
            
        # Create windows layout AFTER setting up door properties
        self.create_windows()
    
    def create_windows(self):
        """Create window positions throughout the building height"""
        self.windows = []
        
        # Fixed 3 windows per row
        windows_per_row = 3
        
        # Calculate horizontal margin to center the windows
        total_window_width = windows_per_row * self.WINDOW_SIZE
        total_margin_width = self.width - total_window_width
        margin_between = total_margin_width / (windows_per_row + 1)
        
        # Determine how many rows of windows can fit vertically
        # Leave margins at top and bottom of building
        usable_height = self.height - 2 * self.WINDOW_MARGIN
        if usable_height <= 0:  # Very short building
            return
            
        # Spacing between windows should be more even
        spacing = 6  # Space between window rows
        max_rows = usable_height // (self.WINDOW_SIZE + spacing)
        
        # For top buildings, start from the bottom and go up
        # For bottom buildings, start from the top and go down
        spacing = 6  # Same spacing as calculated above
        if self.position == 'top':
            start_y = self.height - self.WINDOW_MARGIN - self.WINDOW_SIZE
            row_increment = -(self.WINDOW_SIZE + spacing)
        else:  # bottom
            start_y = self.WINDOW_MARGIN
            row_increment = (self.WINDOW_SIZE + spacing)
        
        # Create all rows of windows
        for row in range(max_rows):
            window_y = start_y + row * row_increment
            
            # Skip if window would be outside the building
            if window_y < 0 or window_y + self.WINDOW_SIZE > self.height:
                continue
                
            for col in range(windows_per_row):
                window_x = margin_between + col * (self.WINDOW_SIZE + margin_between)
                
                # Skip window positions that would overlap with the door
                door_area = pygame.Rect(
                    self.door_x,
                    self.door_y,
                    self.door_width,
                    self.door_height
                )
                
                window_area = pygame.Rect(
                    window_x,
                    window_y,
                    self.WINDOW_SIZE,
                    self.WINDOW_SIZE
                )
                
                # Only add window if it doesn't overlap with the door
                if not door_area.colliderect(window_area):
                    self.windows.append((window_x, window_y))
    
    def draw(self, screen, screen_y=0):
        """Draw the building and its windows
        
        Args:
            screen: pygame surface to draw on
            screen_y: y-coordinate offset on screen
        """
        # Draw the main building
        pygame.draw.rect(
            screen, 
            self.color,
            (self.x, screen_y, self.width, self.height)
        )
        
        # Draw the windows
        for window_x, window_y in self.windows:
            pygame.draw.rect(
                screen,
                self.WINDOW_COLOR,
                (self.x + window_x, screen_y + window_y, self.WINDOW_SIZE, self.WINDOW_SIZE)
            )
        
        # Draw the door
        pygame.draw.rect(
            screen,
            self.DOOR_COLOR,
            (self.x + self.door_x, screen_y + self.door_y, self.door_width, self.door_height)
        )
        
        # Draw a slight darker door frame
        door_frame_color = (max(0, self.DOOR_COLOR[0] - 30), 
                           max(0, self.DOOR_COLOR[1] - 30),
                           max(0, self.DOOR_COLOR[2] - 30))
        
        pygame.draw.rect(
            screen,
            door_frame_color,
            (self.x + self.door_x, screen_y + self.door_y, self.door_width, self.door_height),
            1  # Width=1 for just the border
        )
        
        # Draw door handle
        handle_color = (200, 200, 200)  # Silver handle
        handle_x = self.x + self.door_x + self.door_width * 0.8
        
        if self.position == 'top':
            handle_y = screen_y + self.door_y + self.door_height * 0.5
        else:
            handle_y = screen_y + self.door_y + self.door_height * 0.5
            
        pygame.draw.circle(
            screen,
            handle_color,
            (int(handle_x), int(handle_y)),
            2  # Radius
        )

def generate_buildings(screen_width, screen_height, pipe_info, gap_size):
    """Generate top and bottom buildings based on pipe information
    
    Args:
        screen_width: Width of the game screen
        screen_height: Height of the game screen
        pipe_info: Dict with 'x' and 'top_height' keys
        gap_size: Gap between top and bottom buildings
    
    Returns:
        Tuple of (top_building, bottom_building)
    """
    x = pipe_info['x']
    width = 52  # Standard pipe width
    
    # Create top building
    top_height = pipe_info['top_height']
    top_building = Building(x, width, top_height, 'top')
    
    # Create bottom building
    bottom_y = top_height + gap_size
    bottom_height = screen_height - bottom_y
    bottom_building = Building(x, width, bottom_height, 'bottom')
    
    return (top_building, bottom_building)