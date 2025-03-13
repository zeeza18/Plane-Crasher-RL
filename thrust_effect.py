import random
import pygame

class ThrustEffect:
    """A thruster effect that appears behind the plane when it's moving"""
    def __init__(self, owner):
        self.owner = owner  # Reference to the environment
        self.particles = []
        self.max_particles = 30
        self.last_thrust_time = 0
        
    def update(self, action):
        """Update the thrust particles"""
        # Create new particles when the plane flaps (action=1)
        if action == 1:
            self.last_thrust_time = 15  # Show thrust for 15 frames
            
            # Add new particles at the back of the plane
            for _ in range(8):  # Add 8 particles
                self.particles.append({
                    'x': self.owner.bird_x - 10,  # Behind the plane
                    'y': self.owner.bird_y + self.owner.BIRD_HEIGHT // 2 + random.randint(-8, 8),
                    'size': random.randint(5, 12),
                    'life': random.randint(15, 25),
                    'dx': random.uniform(-4.5, -2),  # Move left
                    'dy': random.uniform(-0.5, 0.5),  # Slight vertical movement
                    'color': (
                        random.randint(220, 255),  # Red
                        random.randint(100, 200),  # Green
                        random.randint(0, 50)      # Blue
                    )
                })
        
        # Show thrust for a few frames after flapping
        if self.last_thrust_time > 0:
            self.last_thrust_time -= 1
            
            # Add small "idle" particles
            if random.random() < 0.6:  # 60% chance per frame
                self.particles.append({
                    'x': self.owner.bird_x - 5,  # Behind the plane
                    'y': self.owner.bird_y + self.owner.BIRD_HEIGHT // 2 + random.randint(-5, 5),
                    'size': random.randint(3, 8),
                    'life': random.randint(10, 20),
                    'dx': random.uniform(-3, -1),  # Move left
                    'dy': random.uniform(-0.4, 0.4),  # Slight vertical movement
                    'color': (
                        random.randint(180, 255),  # Red
                        random.randint(80, 180),   # Green
                        random.randint(0, 40)      # Blue
                    )
                })
        
        # Update existing particles
        particles_to_remove = []
        for particle in self.particles:
            # Update position
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            
            # Reduce size
            particle['size'] -= 0.15
            
            # Reduce life
            particle['life'] -= 1
            
            # Fade color
            r, g, b = particle['color']
            fade_speed = 10
            r = max(0, r - fade_speed)
            g = max(0, g - fade_speed)
            particle['color'] = (r, g, b)
            
            # Mark for removal if too small or expired
            if particle['size'] <= 0 or particle['life'] <= 0:
                particles_to_remove.append(particle)
        
        # Remove dead particles
        for particle in particles_to_remove:
            if particle in self.particles:
                self.particles.remove(particle)
        
        # Limit the number of particles
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]
    
    def draw(self, screen):
        """Draw the thrust particles"""
        for particle in self.particles:
            size = max(1, int(particle['size']))
            
            # Draw main particle
            pygame.draw.circle(
                screen,
                particle['color'],
                (int(particle['x']), int(particle['y'])),
                size
            )
            
            # Add a glowing core for more visible effect
            if size > 3:
                inner_size = size * 0.6
                inner_color = (255, min(255, particle['color'][1] + 50), 50)
                pygame.draw.circle(
                    screen,
                    inner_color,
                    (int(particle['x']), int(particle['y'])),
                    int(inner_size)
                )