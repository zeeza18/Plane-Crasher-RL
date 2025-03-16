import pygame

class GameOverEffect:
    """A stylish 'Game Over' animation that matches the building/plane style"""
    def __init__(self, screen_width, screen_height, sound_manager=None):
        # Basic dimensions for positioning
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Store sound manager reference if provided
        self.sound_manager = sound_manager
        
        # Animation parameters
        self.alpha = 0                 # Start fully transparent
        self.fade_in_speed = 5         # How quickly the effect appears
        self.text_scale = 0.5          # Start small
        self.scale_speed = 0.05        # How quickly the text grows
        self.complete = False          # Whether animation is complete
        self.max_scale = 1.2           # Maximum size before shrinking back
        self.final_scale = 1.0         # Final stable size
        self.scaling_up = True         # Whether we're in the growing phase
        
        # Visual styling
        self.text_color = (200, 50, 50)  # Dark red
        self.shadow_offset = 2           # Pixel offset for text shadow
        
        # Sound has been played flag
        self.sound_played = False
        
    def update(self):
        # Play sound on the first update frame (ensures sound is synchronized with visuals)
        if not self.sound_played and self.sound_manager:
            # Pause any existing background music
            if hasattr(self.sound_manager, 'pause_background_music'):
                self.sound_manager.pause_background_music()
                
            # Play crash sound - ensure it's the full version, not just a short effect
            if hasattr(self.sound_manager, 'play_crash_sound'):
                # Set volume higher to make it more prominent
                if hasattr(self.sound_manager.crash_sound, 'set_volume'):
                    self.sound_manager.crash_sound.set_volume(1.0)  # Maximum volume
                self.sound_manager.play_crash_sound()
                
            self.sound_played = True
        
        # Increase alpha (fade in)
        if self.alpha < 255:
            self.alpha += self.fade_in_speed
            self.alpha = min(255, self.alpha)
            
        # Scale animation
        if self.scaling_up:
            self.text_scale += self.scale_speed
            if self.text_scale >= self.max_scale:
                self.scaling_up = False
        else:
            self.text_scale -= self.scale_speed
            if self.text_scale <= self.final_scale:
                self.text_scale = self.final_scale
                self.complete = True
    
    def draw(self, screen):
        # Create font for game over text - smaller size to fit the game window
        try:
            # Use a smaller base font size appropriate for the small game window
            font_size = int(30 * self.text_scale)  # Reduced from 50 to 30
            font = pygame.font.SysFont('Arial Black', font_size, bold=True)
            
            # Render text with shadow
            text_shadow = font.render('GAME OVER', True, (0, 0, 0))
            text_shadow_rect = text_shadow.get_rect(center=(
                self.screen_width//2 + self.shadow_offset, 
                self.screen_height//2 + self.shadow_offset
            ))
            
            text = font.render('GAME OVER', True, self.text_color)
            text_rect = text.get_rect(center=(self.screen_width//2, self.screen_height//2))
            
            # Create a semi-transparent overlay for the background
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, min(120, self.alpha // 2)))  # Semi-transparent black
            
            # Apply overlay and text with current alpha
            screen.blit(overlay, (0, 0))
            
            # Apply alpha to texts
            text_shadow.set_alpha(self.alpha)
            text.set_alpha(self.alpha)
            
            # Draw shadow then text
            screen.blit(text_shadow, text_shadow_rect)
            screen.blit(text, text_rect)
                
        except Exception as e:
            print(f"Error rendering game over text: {e}")
    
    def is_complete(self):
        return self.complete