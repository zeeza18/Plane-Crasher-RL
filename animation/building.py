# Import pygame library for game development and graphical functions
import pygame
# Import random module to generate random elements like building colors
import random

# Define the Building class that creates visual obstacles in the game
# Each building replaces the standard pipes with more detailed structures
class Building:
    # Define a set of light pastel colors for buildings to create visual variety
    # Blue is excluded since it would blend with the sky background
    COLORS = [
        (255, 192, 203),  # Light pink - Creates soft-colored buildings
        (144, 238, 144),  # Light green - Provides natural-looking variety
        (255, 255, 153),  # Light yellow - Adds sunny, bright option
        (255, 160, 122),  # Light salmon - Warm-toned building option
        (216, 191, 216),  # Light purple - Adds cool-toned variety
        (173, 216, 230),  # Light cyan - Lighter blue that still contrasts with sky
    ]
    
    # Define display properties for building details
    WINDOW_COLOR = (120, 120, 120)  # Gray windows - neutral color to contrast with buildings
    DOOR_COLOR = (139, 69, 19)      # Brown door color - wood-like appearance for realism
    WINDOW_MARGIN = 7  # Space between windows and building edges in pixels
    WINDOW_SIZE = 10  # Size of each window square in pixels
    
    # Constructor method that initializes a new building
    # Parameters:
    #   x: horizontal position on screen
    #   width: width of building in pixels
    #   height: height of building in pixels
    #   position: whether building extends from 'top' or 'bottom' of screen
    def __init__(self, x, width, height, position='top'):
        # Store the building's position and dimensions
        self.x = x                  # Horizontal position (will change as building moves)
        self.width = width          # Width of the building (typically 52 pixels)
        self.height = height        # Height of the building (varies based on game difficulty)
        self.position = position    # Position identifier - 'top' or 'bottom'
        
        # Randomly select a color from the predefined list for this building
        # This creates visual variety so each building looks slightly different
        self.color = random.choice(self.COLORS)
        
        # Calculate door dimensions proportional to building size
        # Door takes up 1/3 of building width for appropriate proportion
        self.door_width = self.width // 3
        # Door height is balanced between minimum viable size and building proportion
        # Uses min() and max() to set bounds - between 12-20px or 1/6 of building height
        self.door_height = min(20, max(self.height // 6, 12))
        
        # Calculate door position within the building
        # Center the door horizontally within the building
        self.door_x = self.width // 2 - self.door_width // 2
        
        # Set vertical door position based on building type
        # For top buildings: door at top edge (appears at bottom of downward building)
        # For bottom buildings: door at bottom edge (appears at ground level)
        if self.position == 'top':
            self.door_y = 0  # Place door at top edge for top buildings
        else:  # bottom
            self.door_y = self.height - self.door_height  # Place door at bottom edge
            
        # Generate the window layout after door is positioned
        # This ensures windows don't overlap with the door
        self.create_windows()
    
    # Method to create window positions throughout the building
    # Creates a realistic pattern of windows while avoiding the door area
    # No parameters or return value - populates self.windows list
    def create_windows(self):
        """Create window positions throughout the building height"""
        # Initialize empty list to store window positions
        self.windows = []
        
        # Set a fixed number of windows per row for consistent appearance
        # 3 windows creates a balanced facade that fits well in the building width
        windows_per_row = 3
        
        # Calculate horizontal spacing to distribute windows evenly
        # First, determine the total width occupied by all windows in a row
        total_window_width = windows_per_row * self.WINDOW_SIZE
        # Calculate the remaining width available for margins
        total_margin_width = self.width - total_window_width
        # Divide remaining space evenly (including outer margins)
        margin_between = total_margin_width / (windows_per_row + 1)
        
        # Determine how many rows of windows can fit vertically
        # Subtract top and bottom margins from total height
        usable_height = self.height - 2 * self.WINDOW_MARGIN
        # Handle very short buildings that can't fit windows
        if usable_height <= 0:
            return  # Exit method if building is too short for windows
            
        # Define spacing between window rows
        # 6 pixels provides a balanced, realistic spacing
        spacing = 6
        # Calculate maximum number of complete rows that will fit
        max_rows = usable_height // (self.WINDOW_SIZE + spacing)
        
        # Determine window placement direction based on building position
        # Windows need to be arranged differently for top vs bottom buildings
        if self.position == 'top':
            # For top buildings, start from bottom and work upward
            # This is because top buildings hang from the ceiling
            start_y = self.height - self.WINDOW_MARGIN - self.WINDOW_SIZE
            # Move upward for each row (negative increment)
            row_increment = -(self.WINDOW_SIZE + spacing)
        else:  # bottom
            # For bottom buildings, start from top and work downward
            # Bottom buildings rise from the ground up
            start_y = self.WINDOW_MARGIN
            # Move downward for each row (positive increment)
            row_increment = (self.WINDOW_SIZE + spacing)
        
        # Generate all rows of windows
        for row in range(max_rows):
            # Calculate y position for current row of windows
            window_y = start_y + row * row_increment
            
            # Skip if window would extend beyond building boundaries
            # This prevents graphical glitches with windows outside building
            if window_y < 0 or window_y + self.WINDOW_SIZE > self.height:
                continue
                
            # Place each window in the current row
            for col in range(windows_per_row):
                # Calculate x position for this specific window
                window_x = margin_between + col * (self.WINDOW_SIZE + margin_between)
                
                # Create rectangular areas to test for door overlap
                # Define door area as pygame Rect for collision detection
                door_area = pygame.Rect(
                    self.door_x,       # X position of door relative to building
                    self.door_y,       # Y position of door
                    self.door_width,   # Width of door
                    self.door_height   # Height of door
                )
                
                # Define potential window area as pygame Rect
                window_area = pygame.Rect(
                    window_x,          # X position of window relative to building
                    window_y,          # Y position of window
                    self.WINDOW_SIZE,  # Width of window
                    self.WINDOW_SIZE   # Height of window
                )
                
                # Only add window if it doesn't overlap with door
                # colliderect() returns True if rectangles overlap
                if not door_area.colliderect(window_area):
                    # Store window position if no overlap detected
                    self.windows.append((window_x, window_y))
    
    # Method to draw the building on screen with all its features
    # Parameters:
    #   screen: pygame surface to draw on
    #   screen_y: vertical offset in pixels (default 0)
    # No return value
    def draw(self, screen, screen_y=0):
        """Draw the building and its windows
        
        Args:
            screen: pygame surface to draw on
            screen_y: y-coordinate offset on screen
        """
        # Draw the main building rectangle with selected color
        # This creates the basic structure of the building
        pygame.draw.rect(
            screen,      # Surface to draw on
            self.color,  # Building color (randomly selected)
            (self.x, screen_y, self.width, self.height)  # Position (x,y) and size
        )
        
        # Draw all windows as small squares
        # Iterate through stored window positions
        for window_x, window_y in self.windows:
            pygame.draw.rect(
                screen,              # Surface to draw on
                self.WINDOW_COLOR,   # Gray window color
                # Position (adjusted for building position) and size
                (self.x + window_x, screen_y + window_y, self.WINDOW_SIZE, self.WINDOW_SIZE)
            )
        
        # Draw the door as a rectangle
        # This creates the entrance to the building
        pygame.draw.rect(
            screen,          # Surface to draw on
            self.DOOR_COLOR, # Brown door color
            # Position (adjusted for building position) and size
            (self.x + self.door_x, screen_y + self.door_y, self.door_width, self.door_height)
        )
        
        # Create a slightly darker frame around the door for visual depth
        # Calculate darker version of door color by reducing RGB values
        door_frame_color = (max(0, self.DOOR_COLOR[0] - 30), 
                           max(0, self.DOOR_COLOR[1] - 30),
                           max(0, self.DOOR_COLOR[2] - 30))
        
        # Draw just the outline/border of the door using the darker color
        pygame.draw.rect(
            screen,              # Surface to draw on
            door_frame_color,    # Darker door frame color
            # Same position and size as door
            (self.x + self.door_x, screen_y + self.door_y, self.door_width, self.door_height),
            1  # Width=1 means draw only the outline (not filled)
        )
        
        # Add a door handle for detail
        # Silver/metallic color for handle
        handle_color = (200, 200, 200)
        # Position handle on right side of door (80% across)
        handle_x = self.x + self.door_x + self.door_width * 0.8
        
        # Position handle vertically in middle of door for both building types
        if self.position == 'top':
            handle_y = screen_y + self.door_y + self.door_height * 0.5
        else:
            handle_y = screen_y + self.door_y + self.door_height * 0.5
            
        # Draw handle as a small circle
        pygame.draw.circle(
            screen,            # Surface to draw on
            handle_color,      # Silver handle color
            (int(handle_x), int(handle_y)),  # Position (convert to integers for drawing)
            2  # Radius in pixels (small for appropriate scale)
        )

# Function to generate a matched pair of buildings for game obstacles
# Parameters:
#   screen_width: width of game screen in pixels
#   screen_height: height of game screen in pixels
#   pipe_info: dictionary with pipe position data (x, top_height)
#   gap_size: vertical space between buildings for bird to fly through
# Returns: tuple containing (top_building, bottom_building)
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
    # Extract horizontal position from pipe info
    x = pipe_info['x']
    # Set standard width for all buildings (matches pipe width)
    width = 52
    
    # Create top building that extends down from top of screen
    # Height is determined by pipe information
    top_height = pipe_info['top_height']
    # Instantiate Building object with 'top' position
    top_building = Building(x, width, top_height, 'top')
    
    # Create bottom building that extends up from bottom of screen
    # Calculate vertical start position of bottom building (after gap)
    bottom_y = top_height + gap_size
    # Calculate height of bottom building (from bottom_y to screen bottom)
    bottom_height = screen_height - bottom_y
    # Instantiate Building object with 'bottom' position
    bottom_building = Building(x, width, bottom_height, 'bottom')
    
    # Return both buildings as a tuple for use in game
    # These buildings replace the standard pipes as obstacles
    return (top_building, bottom_building)