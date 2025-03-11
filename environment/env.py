import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from procedural_explosion import ExplosionEffect, GrayExhaust
from building import Building, generate_buildings

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        
        # Initialize Pygame
        pygame.init()
        
        # Constants
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.PIPE_WIDTH = 52
        self.INITIAL_PIPE_GAP = 180  # Starting with wider gaps
        self.MIN_PIPE_GAP = 120      # Will narrow down to this
        self.PIPE_GAP = self.INITIAL_PIPE_GAP  # Current gap size
        self.PIPE_SPACING = 200  # Horizontal spacing between pipes
        self.BIRD_WIDTH = 34
        self.BIRD_HEIGHT = 24
        self.GRAVITY = 0.8
        self.FLAP_VELOCITY = -8
        self.BASE_VELOCITY = -3      # Starting slower
        self.MAX_VELOCITY = -8       # Will speed up to this
        
        # Difficulty parameters
        self.current_velocity = self.BASE_VELOCITY
        self.SPEED_INCREASE = 0.2    # Smaller increments but more frequent
        self.SPEED_MILESTONE = 5     # Increase difficulty every 5 points
        self.GAP_DECREASE = 2        # How much to decrease gap size
        self.MIN_GAP_MILESTONE = 10  # Start decreasing gap after this score
        
        # Pygame setup with NOFRAME flag
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        )
        self.clock = pygame.time.Clock()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0]), 
            high=np.array([self.SCREEN_HEIGHT, 10, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_HEIGHT]),
            dtype=np.float32
        )
        
        # Initialize game state
        self.bird_x = self.SCREEN_WIDTH // 3
        
        # Visual effects
        self.effects = []
        self.exhaust_timer = 0
        
        # Building storage
        self.buildings = []
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset bird position and velocity
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        
        # Reset pipes and game state
        self.pipes = []
        self.buildings = []
        self.score = 0
        self.current_velocity = self.BASE_VELOCITY
        self.PIPE_GAP = self.INITIAL_PIPE_GAP  # Reset gap size
        
        # Reset visual effects
        self.effects = []
        self.exhaust_timer = 0
        
        # Initialize with first pipe
        self._generate_pipe()
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}

    def _generate_pipe(self):
        """Generate a new pipe pair with more vertical variation"""
        # Increase vertical variation based on score
        variation_range = min(200, 100 + self.score * 2)  # More variation as score increases
        
        # Calculate gap center with increasing variation
        gap_center = random.randint(
            self.PIPE_GAP // 2 + 50,  # Minimum from top
            self.SCREEN_HEIGHT - (self.PIPE_GAP // 2) - 50  # Minimum from bottom
        )
        
        # Add some oscillation to make it more challenging
        if self.score > 20:  # Start adding wave pattern after score 20
            wave_amplitude = min(50, self.score)  # Increasing amplitude with score
            gap_center += int(wave_amplitude * np.sin(self.score * 0.1))
            gap_center = np.clip(gap_center, 
                               self.PIPE_GAP // 2 + 30,
                               self.SCREEN_HEIGHT - (self.PIPE_GAP // 2) - 30)
        
        top_height = gap_center - (self.PIPE_GAP // 2)
        
        if len(self.pipes) == 0:
            x_pos = self.SCREEN_WIDTH
        else:
            x_pos = self.pipes[-1]['x'] + self.PIPE_SPACING
        
        new_pipe = {
            'x': x_pos,
            'top_height': top_height,
            'passed': False
        }
        
        self.pipes.append(new_pipe)
        
        # Generate buildings for this pipe
        top_building, bottom_building = generate_buildings(
            self.SCREEN_WIDTH, 
            self.SCREEN_HEIGHT, 
            new_pipe, 
            self.PIPE_GAP
        )
        
        self.buildings.append({
            'pipe_idx': len(self.pipes) - 1,
            'top': top_building,
            'bottom': bottom_building
        })

    def step(self, action):
        # Handle events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, 0, True, False, {}

        reward = 0.1  # Small reward for staying alive
        
        # Apply action
        if action == 1:
            self.bird_velocity = self.FLAP_VELOCITY
            # Add more exhaust when flapping
            exhaust_x = self.bird_x
            exhaust_y = self.bird_y + self.BIRD_HEIGHT//2
            self.effects.append(GrayExhaust(exhaust_x, exhaust_y))
        
        # Update bird
        self.bird_velocity += self.GRAVITY
        self.bird_y += self.bird_velocity
        
        # Update pipes and buildings
        for pipe in self.pipes:
            pipe['x'] += self.current_velocity
            
        # Update building positions to match pipes
        for building_info in self.buildings:
            pipe_idx = building_info['pipe_idx']
            if pipe_idx < len(self.pipes):  # Safety check
                pipe_x = self.pipes[pipe_idx]['x']
                building_info['top'].x = pipe_x
                building_info['bottom'].x = pipe_x
        
        # Remove off-screen pipes and buildings
        off_screen_indices = [i for i, pipe in enumerate(self.pipes) 
                             if pipe['x'] + self.PIPE_WIDTH < 0]
        
        # Remove buildings first (in reverse to avoid index issues)
        for i in sorted(off_screen_indices, reverse=True):
            self.buildings = [b for b in self.buildings if b['pipe_idx'] != i]
            
            # Update remaining building indices
            for b in self.buildings:
                if b['pipe_idx'] > i:
                    b['pipe_idx'] -= 1
        
        # Then remove pipes
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + self.PIPE_WIDTH >= 0]
        
        # Add new pipes if needed
        while len(self.pipes) < 2:
            self._generate_pipe()
        
        # Score update and difficulty increase
        for pipe in self.pipes:
            if self.bird_x > pipe['x'] + self.PIPE_WIDTH and not pipe['passed']:
                pipe['passed'] = True
                self.score += 1
                reward = 1.0
                
                # Increase speed every SPEED_MILESTONE points
                if self.score % self.SPEED_MILESTONE == 0:
                    # Speed up but cap at max velocity
                    new_velocity = self.current_velocity - self.SPEED_INCREASE
                    self.current_velocity = max(new_velocity, self.MAX_VELOCITY)
                
                # Decrease gap size after MIN_GAP_MILESTONE points
                if self.score >= self.MIN_GAP_MILESTONE and self.PIPE_GAP > self.MIN_PIPE_GAP:
                    self.PIPE_GAP = max(self.MIN_PIPE_GAP, 
                                      self.PIPE_GAP - self.GAP_DECREASE)
        
        # Additional reward for being in the middle third of the screen
        ideal_zone = (self.SCREEN_HEIGHT / 3, 2 * self.SCREEN_HEIGHT / 3)
        if ideal_zone[0] <= self.bird_y <= ideal_zone[1]:
            reward += 0.1
        
        # Update visual effects
        self._update_effects()
        
        # Check collisions
        terminated = self._check_collision()
        if terminated:
            reward = -1.0
            # Add explosion effect at bird position when collision happens
            self.effects.append(ExplosionEffect(self.bird_x + self.BIRD_WIDTH//2, self.bird_y + self.BIRD_HEIGHT//2))
        
        observation = self._get_observation()
        return observation, reward, terminated, False, {'score': self.score}

    def _update_effects(self):
        """Update and manage visual effects"""
        # Update existing effects
        for effect in list(self.effects):
            effect.update()
            if effect.is_complete():
                self.effects.remove(effect)
        
        # Add engine exhaust effect periodically
        self.exhaust_timer += 1
        if self.exhaust_timer >= 6:  # Add exhaust every 6 frames
            self.exhaust_timer = 0
            # Position exhaust at the back of the plane
            exhaust_x = self.bird_x
            exhaust_y = self.bird_y + self.BIRD_HEIGHT//2
            self.effects.append(GrayExhaust(exhaust_x, exhaust_y))

    def render(self):
        """Render the current environment state"""
        # Handle events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Clear screen
        self.screen.fill((135, 206, 235))  # Sky blue background
        
        # Draw buildings instead of pipes
        for building_info in self.buildings:
            building_info['top'].draw(self.screen, 0)
            building_info['bottom'].draw(self.screen, building_info['top'].height + self.PIPE_GAP)
        
        # Draw exhaust effects (behind the plane)
        for effect in self.effects:
            if isinstance(effect, GrayExhaust):
                effect.draw(self.screen)
        
        # Draw plane instead of bird rectangle
        # Plane body (white/silver)
        pygame.draw.ellipse(self.screen, (220, 220, 220), 
                        (self.bird_x, self.bird_y + 2, 
                        self.BIRD_WIDTH - 10, self.BIRD_HEIGHT - 4))
        
        # Plane nose (gray)
        pygame.draw.polygon(self.screen, (180, 180, 180), [
            (self.bird_x + self.BIRD_WIDTH - 10, self.bird_y + self.BIRD_HEIGHT//2),
            (self.bird_x + self.BIRD_WIDTH, self.bird_y + self.BIRD_HEIGHT//2 - 2),
            (self.bird_x + self.BIRD_WIDTH, self.bird_y + self.BIRD_HEIGHT//2 + 2)
        ])
        
        # Wings (blue)
        pygame.draw.polygon(self.screen, (70, 130, 180), [
            (self.bird_x + self.BIRD_WIDTH//2 - 5, self.bird_y + self.BIRD_HEIGHT//2),
            (self.bird_x + self.BIRD_WIDTH//2 - 10, self.bird_y),
            (self.bird_x + self.BIRD_WIDTH//2 + 5, self.bird_y),
            (self.bird_x + self.BIRD_WIDTH//2 + 5, self.bird_y + self.BIRD_HEIGHT//2)
        ])
        
        # Tail (blue)
        pygame.draw.polygon(self.screen, (70, 130, 180), [
            (self.bird_x + 2, self.bird_y + self.BIRD_HEIGHT//2 - 2),
            (self.bird_x - 3, self.bird_y + self.BIRD_HEIGHT//2 - 6),
            (self.bird_x + 2, self.bird_y + self.BIRD_HEIGHT//2 + 2)
        ])
        
        # Windows (light blue)
        for i in range(2):
            pygame.draw.rect(self.screen, (200, 230, 255),
                            (self.bird_x + 10 + i*6, self.bird_y + 5, 4, 3))
        
        # Draw explosion effects (over everything)
        for effect in self.effects:
            if isinstance(effect, ExplosionEffect):
                effect.draw(self.screen)
        
        # Draw score at the top middle
        font = pygame.font.SysFont('Arial', 24, bold=True)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        text_rect = score_text.get_rect(center=(self.SCREEN_WIDTH//2, 20))
        self.screen.blit(score_text, text_rect)
        
        pygame.display.flip()
        self.clock.tick(30)

    def _check_collision(self):
        # Check boundaries
        if self.bird_y <= 0 or self.bird_y + self.BIRD_HEIGHT >= self.SCREEN_HEIGHT:
            return True
        
        # Check pipe collisions (same as before, but we're checking building hitboxes)
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, 
                              self.BIRD_WIDTH, self.BIRD_HEIGHT)
        
        for pipe in self.pipes:
            top_pipe_rect = pygame.Rect(pipe['x'], 0,
                                      self.PIPE_WIDTH, pipe['top_height'])
            bottom_pipe_rect = pygame.Rect(pipe['x'], 
                                         pipe['top_height'] + self.PIPE_GAP,
                                         self.PIPE_WIDTH, 
                                         self.SCREEN_HEIGHT)
            
            if bird_rect.colliderect(top_pipe_rect) or \
               bird_rect.colliderect(bottom_pipe_rect):
                return True
        
        return False

    def _get_observation(self):
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.PIPE_WIDTH > self.bird_x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = self.pipes[0]
        
        return np.array([
            self.bird_y,
            self.bird_velocity,
            next_pipe['x'] - self.bird_x,
            next_pipe['top_height'],
            next_pipe['top_height'] + self.PIPE_GAP
        ], dtype=np.float32)

    def close(self):
        pygame.quit()