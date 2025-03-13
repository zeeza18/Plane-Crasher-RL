import pygame
from procedural_explosion import ExplosionEffect

class CollisionHandler:
    def __init__(self, screen):
        # Store the screen to draw on
        self.screen = screen
        self.explosion = None
        self.game_over = False
        self.show_game_over_text = True
        
    def check_collision(self, bird_x, bird_y, bird_width, bird_height, pipes, pipe_width, pipe_gap, screen_height):
        """Check for any collision and create explosion if needed"""
        # Create rectangles for collision detection
        bird_rect = pygame.Rect(bird_x, bird_y, bird_width, bird_height)
        
        # Check for boundary collisions
        if bird_y <= 0 or bird_y + bird_height >= screen_height:
            if not self.game_over:
                self.game_over = True
                # Position explosion at the center of the plane
                explosion_x = bird_x + (bird_width // 2)
                explosion_y = bird_y + (bird_height // 2)
                self.explosion = ExplosionEffect(explosion_x, explosion_y)
            return True
            
        # Check pipe collisions
        for pipe in pipes:
            top_pipe_rect = pygame.Rect(pipe['x'], 0, pipe_width, pipe['top_height'])
            bottom_pipe_rect = pygame.Rect(
                pipe['x'], 
                pipe['top_height'] + pipe_gap,
                pipe_width, 
                screen_height
            )
            
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                if not self.game_over:
                    self.game_over = True
                    # Position explosion at point of impact
                    if bird_rect.colliderect(top_pipe_rect):
                        # Collision with top pipe
                        explosion_x = max(bird_x + (bird_width // 2), pipe['x'])
                        explosion_y = max(bird_y, pipe['top_height'] // 2)
                    else:
                        # Collision with bottom pipe
                        explosion_x = max(bird_x + (bird_width // 2), pipe['x'])
                        explosion_y = min(bird_y + bird_height, pipe['top_height'] + pipe_gap + 10)
                    
                    self.explosion = ExplosionEffect(explosion_x, explosion_y)
                return True
                
        return False
        
    def update(self):
        """Update the explosion if it exists"""
        if self.explosion:
            self.explosion.update()
            
    def reset(self):
        """Reset the collision handler"""
        self.explosion = None
        self.game_over = False
            
    def render(self, score):
        """Render the explosion and game over text"""
        if self.game_over and self.explosion:
            # Draw explosion
            self.explosion.draw(self.screen)
            
            # Draw "Game Over" text
            if self.show_game_over_text:
                font = pygame.font.Font(None, 72)
                game_over_text = font.render("GAME OVER", True, (255, 0, 0))
                text_rect = game_over_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
                self.screen.blit(game_over_text, text_rect)
                
                # Draw final score
                score_font = pygame.font.Font(None, 48)
                final_score_text = score_font.render(f"Final Score: {score}", True, (255, 255, 255))
                score_rect = final_score_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 60))
                self.screen.blit(final_score_text, score_rect)
                
                # Draw restart instruction
                restart_font = pygame.font.Font(None, 36)
                restart_text = restart_font.render("Press 'R' to restart", True, (255, 255, 255))
                restart_rect = restart_text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2 + 110))
                self.screen.blit(restart_text, restart_rect)
            
            return True
        return False