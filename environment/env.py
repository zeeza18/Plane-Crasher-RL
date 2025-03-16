# Import gymnasium library - a toolkit for developing and comparing reinforcement learning algorithms
# This provides standardized environments and interfaces for reinforcement learning research
import gymnasium as gym

# Import spaces module from gymnasium - used to define the action and observation spaces
# Spaces determine what actions the agent can take and what observations it receives
from gymnasium import spaces

# Import numpy - a fundamental package for scientific computing with Python
# Used for efficient array operations and mathematical functions
import numpy as np

# Import pygame - a set of Python modules designed for writing video games
# Provides functionality for graphics, sound, input handling, etc.
import pygame

# Import random module - used for generating random numbers
# Will be used for randomizing pipe positions and gaps
import random

# Import custom explosion animation classes - used for visual effects
# ExplosionEffect creates an explosion animation when bird crashes
# GrayExhaust creates exhaust trail effect behind the bird
from animation.procedural_explosion import ExplosionEffect, GrayExhaust

# Import Building class and generate_buildings function from building module
# These are used to create and manage the building graphics that replace standard pipes
from animation.building import Building, generate_buildings

# Import CloudEffect - a class that manages cloud animations in the background
# Adds visual interest and depth to the game scene
from animation.cloud_effect import CloudEffect

# Import SunEffect - a class that manages the sun animation in the background
# Creates a more visually appealing environment
from animation.sun_effect import SunEffect

# Import SoundManager - a class that handles all game sounds
# Manages background music and sound effects like crash sounds
from sound_manager import SoundManager

# Import GameOverEffect - a class that creates the game over animation
# Shows a visual "Game Over" when player loses
from animation.game_over_effect import GameOverEffect

# Define FlappyBirdEnv class that inherits from gym.Env
# This class implements the Gymnasium environment interface
class FlappyBirdEnv(gym.Env):
    def __init__(self):
        # Call the parent class constructor to initialize base environment functionality
        # This ensures proper Gymnasium environment initialization
        super(FlappyBirdEnv, self).__init__()
        
        # Initialize Pygame library for game development
        # This must be called before using any pygame functionality
        pygame.init()
        
        # Create a SoundManager instance to handle game audio
        # This centralizes all sound-related functionality
        self.sound_manager = SoundManager()
        # Start playing background music immediately
        self.sound_manager.play_background_music()
        
        # Initialize game over related variables
        # game_over_effect will hold the animation object when game ends
        self.game_over_effect = None
        # Flag to control when to show the game over screen
        self.show_game_over = False
        # Flag to track if the environment is waiting for player to restart
        self.waiting_for_restart = False
        
        # SCREEN DIMENSIONS
        # Set the width of the game window in pixels
        self.SCREEN_WIDTH = 288
        # Set the height of the game window in pixels
        self.SCREEN_HEIGHT = 512
        
        # OBSTACLE DIMENSIONS
        # Set the width of pipes/buildings in pixels
        self.PIPE_WIDTH = 52
        # Initial gap size between pipes - starts wider for easier gameplay
        self.INITIAL_PIPE_GAP = 180
        # Minimum gap size between pipes - difficulty increases by narrowing gap
        self.MIN_PIPE_GAP = 120
        # Current gap size - initialized to starting value
        self.PIPE_GAP = self.INITIAL_PIPE_GAP
        # Horizontal distance between consecutive pipe pairs in pixels
        self.PIPE_SPACING = 200
        
        # BIRD DIMENSIONS
        # Width of the bird (plane) sprite in pixels
        self.BIRD_WIDTH = 34
        # Height of the bird (plane) sprite in pixels
        self.BIRD_HEIGHT = 24
        
        # PHYSICS PARAMETERS
        # Gravity constant - determines rate of downward acceleration
        # Higher value makes the bird fall faster
        self.GRAVITY = 0.8
        # Upward velocity applied when flapping - negative because y-axis is inverted
        # More negative value makes the bird jump higher
        self.FLAP_VELOCITY = -8
        # Initial horizontal scroll speed - negative because objects move right to left
        # This is how fast the pipes/buildings move across the screen
        self.BASE_VELOCITY = -3
        # Maximum horizontal scroll speed - the fastest the game will go
        # Game will gradually speed up to this value as score increases
        self.MAX_VELOCITY = -8
        
        # DIFFICULTY PROGRESSION PARAMETERS
        # Current scroll speed - initialized to base velocity
        self.current_velocity = self.BASE_VELOCITY
        # How much to increase speed each time a milestone is reached
        # Higher value makes difficulty ramp up faster
        self.SPEED_INCREASE = 0.3
        # Score milestone for speed increases - speed increases every N points
        # Lower value makes game get harder more quickly
        self.SPEED_MILESTONE = 4
        # Amount to decrease gap size by at each milestone
        # Higher value makes gaps narrow more quickly
        self.GAP_DECREASE = 3
        # Score at which to start decreasing gap size
        # Allows player to get used to game before gap narrows
        self.MIN_GAP_MILESTONE = 9
        # Score at which to start the vertical wave pattern
        # This adds more complex pipe movements for higher difficulty
        self.WAVE_START_SCORE = 9
        
        # Create the pygame display window with specified dimensions
        # This creates the actual game window
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        )
        # Create a pygame Clock object to control game frame rate
        # This ensures consistent game speed across different computers
        self.clock = pygame.time.Clock()
        
        # Define the ACTION SPACE for reinforcement learning
        # Discrete(2) means two possible actions: 0 (don't flap) or 1 (flap)
        self.action_space = spaces.Discrete(2)
        
        # Define the OBSERVATION SPACE for reinforcement learning
        # This defines what the agent can observe about the environment
        # Array contains: [bird_y, bird_velocity, distance_to_next_pipe, 
        #                   pipe_top_height, pipe_bottom_y]
        self.observation_space = spaces.Box(
            # Minimum possible values for each observation
            low=np.array([0, -10, 0, 0, 0]), 
            # Maximum possible values for each observation
            high=np.array([self.SCREEN_HEIGHT, 10, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_HEIGHT]),
            # Data type for the observation values
            dtype=np.float32
        )
        
        # Set fixed horizontal position for the bird (1/3 of screen width)
        # Bird doesn't move horizontally, only pipes move to create scrolling effect
        self.bird_x = self.SCREEN_WIDTH // 3
        
        # Initialize list for visual effects (explosions, exhaust, etc.)
        self.effects = []
        # Timer to control how often exhaust effects are added
        # Creates periodic puffs of exhaust rather than continuous stream
        self.exhaust_timer = 0
        
        # Initialize list for building objects
        # Will store the graphical building elements that replace pipes
        self.buildings = []
        
        # Create cloud effect instance for background scenery
        # Adds animated clouds to make background more interesting
        self.cloud_effect = CloudEffect(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        
        # Create sun effect instance for background scenery
        # Adds animated sun to enhance visual appeal
        self.sun_effect = SunEffect(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        
        # DIFFICULTY TRACKING FLAGS
        # Flag to track when to update difficulty settings
        # True when difficulty should be updated for next pipe
        self.pending_difficulty_update = False
        # Track next score threshold for increasing speed
        # Will be incremented each time speed increases
        self.next_score_for_speed = self.SPEED_MILESTONE
        # Track next score threshold for decreasing gap
        # Will be incremented each time gap narrows
        self.next_score_for_gap = self.MIN_GAP_MILESTONE
        
        # Reset the environment to initialize all game values
        # This ensures game starts in a consistent initial state
        self.reset()

    # Reset function: returns the environment to initial state
    # Input:
    #   seed (optional) - random seed for reproducibility
    # Returns:
    #   initial observation and empty info dictionary
    def reset(self, seed=None):
        # Call parent reset method with provided seed if any
        # This ensures proper random number generation for reproducibility
        super().reset(seed=seed)
        
        # Reset bird position to middle height of screen
        # Consistent starting position for each new game
        self.bird_y = self.SCREEN_HEIGHT // 2
        # Reset bird vertical velocity to zero (not moving up or down)
        # Bird starts with no momentum
        self.bird_velocity = 0
        
        # Reset game elements to initial state
        # Clear all pipes from previous game
        self.pipes = []
        # Clear all buildings from previous game
        self.buildings = []
        # Reset score to zero
        self.score = 0
        # Reset scroll speed to initial value
        self.current_velocity = self.BASE_VELOCITY
        # Reset pipe gap size to initial (easier) value
        self.PIPE_GAP = self.INITIAL_PIPE_GAP
        
        # Reset difficulty progression tracking
        # Reset next score for speed increase
        self.next_score_for_speed = self.SPEED_MILESTONE
        # Reset next score for gap decrease
        self.next_score_for_gap = self.MIN_GAP_MILESTONE
        # Reset difficulty update flag
        self.pending_difficulty_update = False
        
        # Reset visual effects
        # Clear all active effects (explosions, exhaust, etc.)
        self.effects = []
        # Reset exhaust timer to zero
        self.exhaust_timer = 0
        
        # Reset game over state
        # Turn off game over screen
        self.show_game_over = False
        # Remove game over effect object
        self.game_over_effect = None
        # Reset waiting flag
        self.waiting_for_restart = False
        
        # Generate initial pipe to start game with
        # This ensures there's always at least one pipe on screen
        self._generate_pipe()
        
        # Handle background music
        # Check if sound manager exists and music is not playing
        if hasattr(self, 'sound_manager') and not pygame.mixer.music.get_busy():
            # Start playing background music
            self.sound_manager.play_background_music()
        # Or resume if it was paused during game over
        elif hasattr(self, 'sound_manager'):
            # Unpause music if it was paused
            self.sound_manager.unpause_background_music()
        
        # Get initial observation of game state for the agent
        observation = self._get_observation()
        
        # Return initial observation and empty info dictionary
        # Format required by Gymnasium API
        return observation, {}

    # Update difficulty parameters based on current score
    # No inputs, no return value
    def _update_difficulty(self):
        """Update difficulty parameters based on current score"""
        # Check if we need to increase speed based on score milestone
        if self.score >= self.next_score_for_speed:
            # Calculate new velocity by applying speed increase
            # Subtracting makes velocity more negative (faster)
            new_velocity = self.current_velocity - self.SPEED_INCREASE
            # Cap velocity at maximum speed
            # max() is used because velocities are negative (more negative = faster)
            self.current_velocity = max(new_velocity, self.MAX_VELOCITY)
            # Set next milestone for speed increase
            # Next speed increase will happen after another SPEED_MILESTONE points
            self.next_score_for_speed += self.SPEED_MILESTONE
        
        # Check if we need to decrease gap size based on score milestone
        # Only decrease if above gap milestone AND current gap is still above minimum
        if self.score >= self.next_score_for_gap and self.PIPE_GAP > self.MIN_PIPE_GAP:
            # Decrease gap size but don't go below minimum allowed gap
            self.PIPE_GAP = max(self.MIN_PIPE_GAP, self.PIPE_GAP - self.GAP_DECREASE)
            # Set next milestone for gap decrease
            # Gap decreases every point after MIN_GAP_MILESTONE
            self.next_score_for_gap += 1
        
        # Reset the pending update flag since update is now complete
        self.pending_difficulty_update = False

    # Generate a new pipe pair with vertical positioning
    # No inputs, no return value
    def _generate_pipe(self):
        """Generate a new pipe pair with more vertical variation"""
        # Apply any pending difficulty updates BEFORE generating the new pipe
        # This ensures new pipe uses updated difficulty parameters
        if self.pending_difficulty_update:
            self._update_difficulty()
        
        # Calculate variation range based on score
        # More variation as score increases, capped at 200
        # Higher variation means pipes can appear at more extreme positions
        variation_range = min(200, 100 + self.score * 2)
        
        # Calculate random gap center position within safe margins
        # This determines vertical position of the gap between pipes
        gap_center = random.randint(
            # Minimum distance from top (ensures gap isn't too high)
            self.PIPE_GAP // 2 + 50,
            # Minimum distance from bottom (ensures gap isn't too low)
            self.SCREEN_HEIGHT - (self.PIPE_GAP // 2) - 50
        )
        
        # Add oscillation pattern for advanced difficulty (starts at score 9)
        # This creates a wave pattern to pipe positioning for added challenge
        if self.score >= self.WAVE_START_SCORE:  
            # Wave amplitude increases with score (capped at 60)
            # Higher amplitude means larger vertical shifts in gap position
            wave_amplitude = min(60, (self.score - self.WAVE_START_SCORE + 1) * 3)
            # Wave frequency increases with score
            # Higher frequency means more rapid oscillations
            wave_frequency = 0.15 + min(0.3, self.score * 0.01)
            # Apply sine wave pattern to gap center
            # Sine function creates smooth oscillating pattern
            gap_center += int(wave_amplitude * np.sin(self.score * wave_frequency))
            # Ensure gap stays within screen boundaries with safe margins
            # np.clip constrains value between minimum and maximum
            gap_center = np.clip(gap_center, 
                               self.PIPE_GAP // 2 + 30,
                               self.SCREEN_HEIGHT - (self.PIPE_GAP // 2) - 30)
        
        # Calculate top pipe height based on gap center and gap size
        # Top pipe extends from top of screen to top of gap
        top_height = gap_center - (self.PIPE_GAP // 2)
        
        # Determine x position for new pipe
        if len(self.pipes) == 0:
            # First pipe starts at right edge of screen
            x_pos = self.SCREEN_WIDTH
        else:
            # Subsequent pipes spaced horizontally from previous pipe
            # Consistent spacing creates predictable rhythm
            x_pos = self.pipes[-1]['x'] + self.PIPE_SPACING
        
        # Create new pipe data structure as a dictionary
        new_pipe = {
            'x': x_pos,               # Horizontal position
            'top_height': top_height, # Height of top pipe
            'passed': False           # Flag to track if bird has passed this pipe for scoring
        }
        
        # Add new pipe to pipes list
        self.pipes.append(new_pipe)
        
        # Generate building graphics for this pipe
        # This creates visual representations of the collision objects
        top_building, bottom_building = generate_buildings(
            self.SCREEN_WIDTH,        # Screen width for sizing
            self.SCREEN_HEIGHT,       # Screen height for sizing 
            new_pipe,                 # Pipe data to base buildings on
            self.PIPE_GAP             # Gap size between buildings
        )
        
        # Add buildings to buildings list with reference to pipe index
        # This maintains association between pipes and their visual representations
        self.buildings.append({
            'pipe_idx': len(self.pipes) - 1,  # Index of associated pipe
            'top': top_building,              # Top building object
            'bottom': bottom_building         # Bottom building object
        })

    # Execute one step in the environment based on action
    # Input:
    #   action: 0 (do nothing) or 1 (flap)
    # Returns:
    #   observation, reward, terminated, truncated, info dictionary
    def step(self, action):
        # Check if waiting for restart after game over
        # This allows player to restart after game over animation completes
        if self.waiting_for_restart:
            # Reset the environment and return new state
            return self.reset()
            
        # Handle pygame events to keep window responsive
        # This processes window events like close button clicks
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Close pygame if window is closed
                pygame.quit()
                # Return dummy values since environment is closing
                return None, 0, True, False, {}
            # Check for key press to restart if game over animation completed
            elif event.type == pygame.KEYDOWN and self.show_game_over and self.game_over_effect and self.game_over_effect.is_complete():
                # Set waiting flag to trigger reset on next step
                self.waiting_for_restart = True

        # Small positive reward for surviving each frame
        # Encourages agent to stay alive longer
        reward = 0.1
        
        # Apply flap action if action is 1
        if action == 1:
            # Set upward velocity to make bird jump
            self.bird_velocity = self.FLAP_VELOCITY
            # Add exhaust effect at back of plane for visual feedback
            exhaust_x = self.bird_x
            exhaust_y = self.bird_y + self.BIRD_HEIGHT//2
            self.effects.append(GrayExhaust(exhaust_x, exhaust_y))
        
        # Update bird position based on physics
        # Apply gravity to increase downward velocity (positive y is downward)
        self.bird_velocity += self.GRAVITY
        # Update vertical position based on current velocity
        self.bird_y += self.bird_velocity
        
        # Update pipe positions based on current velocity
        # Move all pipes to the left at current scroll speed
        for pipe in self.pipes:
            pipe['x'] += self.current_velocity
            
        # Update building positions to match their associated pipes
        # This ensures visual elements match collision objects
        for building_info in self.buildings:
            pipe_idx = building_info['pipe_idx']
            if pipe_idx < len(self.pipes):  # Safety check to prevent index errors
                # Get x position of associated pipe
                pipe_x = self.pipes[pipe_idx]['x']
                # Set building x positions to match pipe positions
                building_info['top'].x = pipe_x
                building_info['bottom'].x = pipe_x
        
        # Update cloud animation in background
        # This keeps clouds moving for visual interest
        self.cloud_effect.update()
        
        # Update sun animation in background
        # This keeps sun animation running
        self.sun_effect.update()
        
        # Find indices of pipes that have moved off screen (left edge)
        # These pipes need to be removed to save memory
        off_screen_indices = [i for i, pipe in enumerate(self.pipes) 
                             if pipe['x'] + self.PIPE_WIDTH < 0]
        
        # Remove off-screen buildings first (in reverse to avoid index issues)
        # Removing in reverse order prevents index shifting problems
        for i in sorted(off_screen_indices, reverse=True):
            # Remove buildings associated with off-screen pipes
            self.buildings = [b for b in self.buildings if b['pipe_idx'] != i]
            
            # Update remaining building indices
            # This maintains correct association between pipes and buildings
            for b in self.buildings:
                # Decrement index reference for buildings after removed pipe
                if b['pipe_idx'] > i:
                    b['pipe_idx'] -= 1
        
        # Then remove off-screen pipes
        # This removes pipes that have completely passed the left edge
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + self.PIPE_WIDTH >= 0]
        
        # Add new pipes if needed (maintain at least 2 pipes in game)
        # This ensures there are always pipes visible or approaching
        while len(self.pipes) < 2:
            self._generate_pipe()
        
        # Track if score increased during this step
        score_increased = False
        
        # Check each pipe to see if bird has passed it
        for pipe in self.pipes:
            # Bird passes pipe when it moves beyond right edge of pipe
            # and pipe hasn't been counted yet
            if self.bird_x > pipe['x'] + self.PIPE_WIDTH and not pipe['passed']:
                pipe['passed'] = True        # Mark pipe as passed
                self.score += 1              # Increment score
                reward = 1.0                 # Give larger reward for passing pipe
                score_increased = True       # Flag that score increased
                
                # Flag that we need to update difficulty for the NEXT pipe
                # This delays difficulty updates until a new pipe is generated
                self.pending_difficulty_update = True
        
        # Additional reward for being in the middle third of screen (safer area)
        # This encourages agent to stay in safer vertical position
        ideal_zone = (self.SCREEN_HEIGHT / 3, 2 * self.SCREEN_HEIGHT / 3)
        if ideal_zone[0] <= self.bird_y <= ideal_zone[1]:
            reward += 0.1  # Small bonus reward for staying in safe zone
        
        # Update visual effects (exhaust, explosions, etc.)
        self._update_effects()
        
        # Check for collisions with pipes or boundaries
        # This determines if game should end
        terminated = self._check_collision()
        if terminated:
            reward = -1.0  # Negative reward for collision/game over
            # Add explosion effect at bird position for visual feedback
            self.effects.append(ExplosionEffect(self.bird_x + self.BIRD_WIDTH//2, self.bird_y + self.BIRD_HEIGHT//2))
            
            # Show game over effect
            self.show_game_over = True
            # Create game over effect animation with sound manager passed in
            # This allows GameOverEffect to play the crash sound directly
            self.game_over_effect = GameOverEffect(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.sound_manager)
        
        # Get current observation of game state for the agent
        observation = self._get_observation()
        
        # Return state, reward, termination status, truncation status, and info
        # Format required by Gymnasium API
        return observation, reward, terminated, False, {'score': self.score}

    # Update and manage visual effects
    # No inputs, no return value
    def _update_effects(self):
        """Update and manage visual effects"""
        # Update existing effects and remove completed ones
        # Use list() to create a copy to avoid modification during iteration
        for effect in list(self.effects):
            effect.update()  # Update effect animation state
            if effect.is_complete():
                self.effects.remove(effect)  # Remove completed effects
        
        # Add engine exhaust effect periodically
        # Increment timer each frame
        self.exhaust_timer += 1
        if self.exhaust_timer >= 6:  # Add exhaust every 6 frames
            self.exhaust_timer = 0  # Reset timer
            # Position exhaust at the back of the plane
            exhaust_x = self.bird_x
            exhaust_y = self.bird_y + self.BIRD_HEIGHT//2
            # Add new exhaust effect to list
            self.effects.append(GrayExhaust(exhaust_x, exhaust_y))

    # Render the current environment state to the screen
    # No inputs, no return value
    def render(self):
        """Render the current environment state"""
        # Handle pygame events to keep window responsive
        # Process window events like close button clicks
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Close pygame if window is closed
                pygame.quit()
                return
            # Check for key press to restart if game over animation completed
            elif event.type == pygame.KEYDOWN and self.show_game_over and self.game_over_effect and self.game_over_effect.is_complete():
                # Set waiting flag to trigger reset on next step
                self.waiting_for_restart = True
        
        # Clear screen with sky blue background
        # RGB value (135, 206, 235) is a light blue color
        self.screen.fill((135, 206, 235))
        
        # Draw sun in background (before clouds)
        # This ensures sun appears behind clouds for correct visual layering
        self.sun_effect.draw(self.screen)
        
        # Draw clouds in background
        # Clouds are part of the background scenery
        self.cloud_effect.draw(self.screen)
        
        # Draw buildings instead of pipes
        # Buildings are the visual representation of obstacles
        for building_info in self.buildings:
            # Draw top building at position 0 (top of screen)
            building_info['top'].draw(self.screen, 0)
            # Draw bottom building below gap
            # Position is calculated based on top building height plus gap size
            building_info['bottom'].draw(self.screen, building_info['top'].height + self.PIPE_GAP)
        
        # Draw exhaust effects behind the plane
        # Exhaust should appear behind plane for correct layering
        for effect in self.effects:
            if isinstance(effect, GrayExhaust):
                effect.draw(self.screen)
        
        # Draw airplane instead of bird rectangle
        # This creates a more detailed visual than a simple rectangle
        
        # Draw plane body (white/silver ellipse)
        pygame.draw.ellipse(self.screen, (220, 220, 220), 
                        (self.bird_x, self.bird_y + 2, 
                        self.BIRD_WIDTH - 10, self.BIRD_HEIGHT - 4))
        
        # Draw plane nose (gray triangle)
        # Creates pointed front of plane
        pygame.draw.polygon(self.screen, (180, 180, 180), [
            (self.bird_x + self.BIRD_WIDTH - 10, self.bird_y + self.BIRD_HEIGHT//2),
            (self.bird_x + self.BIRD_WIDTH, self.bird_y + self.BIRD_HEIGHT//2 - 2),
            (self.bird_x + self.BIRD_WIDTH, self.bird_y + self.BIRD_HEIGHT//2 + 2)
        ])
        
        # Draw wings (blue polygon)
        # Creates top wings of plane
        pygame.draw.polygon(self.screen, (70, 130, 180), [
            (self.bird_x + self.BIRD_WIDTH//2 - 5, self.bird_y + self.BIRD_HEIGHT//2),
            (self.bird_x + self.BIRD_WIDTH//2 - 10, self.bird_y),
            (self.bird_x + self.BIRD_WIDTH//2 + 5, self.bird_y),
            (self.bird_x + self.BIRD_WIDTH//2 + 5, self.bird_y + self.BIRD_HEIGHT//2)
        ])
        
        # Draw tail (blue triangle)
        # Creates tail fin at back of plane
        pygame.draw.polygon(self.screen, (70, 130, 180), [
            (self.bird_x + 2, self.bird_y + self.BIRD_HEIGHT//2 - 2),
            (self.bird_x - 3, self.bird_y + self.BIRD_HEIGHT//2 - 6),
            (self.bird_x + 2, self.bird_y + self.BIRD_HEIGHT//2 + 2)
        ])
        
        # Draw windows (light blue rectangles)
        # Creates small window details on plane body
        for i in range(2):
            pygame.draw.rect(self.screen, (200, 230, 255),
                            (self.bird_x + 10 + i*6, self.bird_y + 5, 4, 3))
        
        # Draw explosion effects over everything
        # Explosions should appear on top of all other elements
        for effect in self.effects:
            if isinstance(effect, ExplosionEffect):
                effect.draw(self.screen)
        
        # Draw score at the top middle of screen
        # Create font for score display
        font = pygame.font.SysFont('Arial', 24, bold=True)
        # Render score text as white text
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        # Center text at top of screen
        text_rect = score_text.get_rect(center=(self.SCREEN_WIDTH//2, 20))
        # Draw text to screen
        self.screen.blit(score_text, text_rect)
        
        # Draw game over effect if active
        # Only shown after collision
        if self.show_game_over and self.game_over_effect:
            # Update game over animation
            self.game_over_effect.update()
            # Draw game over animation
            self.game_over_effect.draw(self.screen)
        
        # Update display to show all drawn elements
        # This refreshes the screen with all new drawings
        pygame.display.flip()
        # Limit frame rate to 30 FPS for consistent gameplay speed
        self.clock.tick(30)

    # Check for collisions with pipes or boundaries
    # No inputs
    # Returns: boolean indicating whether collision occurred
    def _check_collision(self):
        # Check if bird hit top or bottom boundaries of screen
        if self.bird_y <= 0 or self.bird_y + self.BIRD_HEIGHT >= self.SCREEN_HEIGHT:
            # Return True indicating collision occurred
            return True
        
        # Create a rectangle representing bird hitbox for collision detection
        # Parameters: x, y, width, height
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, 
                              self.BIRD_WIDTH, self.BIRD_HEIGHT)
        
        # Check collisions with each pipe
        for pipe in self.pipes:
            # Create rectangle for top pipe (extends from top of screen)
            # Parameters: x, y, width, height
            top_pipe_rect = pygame.Rect(pipe['x'], 0,
                                      self.PIPE_WIDTH, pipe['top_height'])
            
            # Create rectangle for bottom pipe (extends to bottom of screen)
            # Parameters: x, y, width, height
            bottom_pipe_rect = pygame.Rect(pipe['x'], 
                                         pipe['top_height'] + self.PIPE_GAP,
                                         self.PIPE_WIDTH, 
                                         self.SCREEN_HEIGHT)
            
            # Check if bird rectangle intersects with either pipe rectangle
            # colliderect returns True if rectangles overlap
            if bird_rect.colliderect(top_pipe_rect) or \
               bird_rect.colliderect(bottom_pipe_rect):
                # Return True indicating collision occurred
                return True
        
        # No collision detected
        return False
    # Get the current observation state for reinforcement learning
    # No inputs
    # Returns: numpy array with observation values
    def _get_observation(self):
        # Find the next pipe ahead of the bird
        # This is the pipe the bird needs to navigate through next
        next_pipe = None
        for pipe in self.pipes:
            # Find first pipe that bird hasn't passed yet (pipe's right edge is ahead of bird's position)
            if pipe['x'] + self.PIPE_WIDTH > self.bird_x:
                next_pipe = pipe
                break
        
        # If no pipe ahead (unlikely), use the first pipe
        # This is a fallback to prevent errors if all pipes are somehow behind the bird
        if next_pipe is None:
            next_pipe = self.pipes[0]
        
        # Create and return observation array with 5 key game state parameters:
        return np.array([
            self.bird_y,                          # Bird's current vertical position
            self.bird_velocity,                   # Bird's current vertical velocity
            next_pipe['x'] - self.bird_x,         # Horizontal distance from bird to next pipe
            next_pipe['top_height'],              # Height of top pipe (how low it extends)
            next_pipe['top_height'] + self.PIPE_GAP  # Bottom of gap (top of bottom pipe)
        ], dtype=np.float32)  # Ensure consistent data type for observation space

    # Clean up resources when environment is closed
    # No inputs
    # No return value
    def close(self):
        # Stop sounds before quitting to prevent audio glitches
        if hasattr(self, 'sound_manager'):
            self.sound_manager.stop_background_music()
        # Quit pygame to release all resources and close window
        pygame.quit()