import pygame
import random

class ExplosionEffect:
    """A larger explosion effect with bright red and yellow colors"""
    def __init__(self, pos_x, pos_y):
        self.particles = []
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.duration = 120  # Even longer duration for explosion
        self.frame = 0
        self.create_particles()
    
    def create_particles(self):
        """Create explosion particles in all directions with bright colors"""
        # Very large number of particles for a massive explosion
        num_particles = random.randint(90, 120)
        
        for i in range(num_particles):
            # Explode in all directions with greater spread
            dx = random.uniform(-12, 12)  # Much wider spread
            dy = random.uniform(-12, 12)  # Much wider spread
            
            # Much larger particles for explosion
            size = random.uniform(8, 25)  # Much bigger particles
            lifetime = random.randint(60, self.duration)
            
            # Decide whether to make this a red or yellow particle
            if random.random() < 0.6:  # 60% red
                # Bright red particles
                r = 255
                g = random.randint(0, 60)  # Very little green
                b = random.randint(0, 20)  # Almost no blue
            else:  # 40% yellow
                # Bright yellow particles
                r = 255
                g = random.randint(200, 255)  # High green for yellow
                b = random.randint(0, 40)  # Very little blue
            
            self.particles.append({
                'x': self.pos_x,
                'y': self.pos_y,
                'dx': dx,
                'dy': dy,
                'size': size,
                'lifetime': lifetime,
                'age': 0,
                'color': (r, g, b),
                'alpha': 255
            })
        
        # Add some outer glow particles
        num_glow = random.randint(40, 60)
        for i in range(num_glow):
            # Spread further for glow effect
            dx = random.uniform(-15, 15)
            dy = random.uniform(-15, 15)
            
            # Larger glow particles
            size = random.uniform(15, 30)
            lifetime = random.randint(30, 90)
            
            # Orange-yellow glow
            r = 255
            g = random.randint(100, 180)
            b = random.randint(0, 30)
            
            self.particles.append({
                'x': self.pos_x,
                'y': self.pos_y,
                'dx': dx,
                'dy': dy,
                'size': size,
                'lifetime': lifetime,
                'age': 0,
                'color': (r, g, b),
                'alpha': random.randint(100, 180)  # Semi-transparent for glow
            })
    
    def update(self):
        """Update particle positions and ages"""
        self.frame += 1
        
        for particle in self.particles:
            # Update position
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            
            # Add gravity and slowdown
            particle['dy'] += 0.1
            particle['dx'] *= 0.98
            
            # Age the particle
            particle['age'] += 1
            
            # Fade out as it ages
            if particle['age'] > particle['lifetime'] * 0.7:
                fade_factor = (particle['lifetime'] - particle['age']) / (particle['lifetime'] * 0.3)
                particle['alpha'] = max(0, int(255 * fade_factor))
            
            # Reduce size gradually
            if particle['age'] > particle['lifetime'] * 0.5:
                particle['size'] = max(0, particle['size'] * 0.98)
    
    def draw(self, screen):
        """Draw the explosion particles"""
        # Sort particles by size to draw larger ones first (behind smaller ones)
        sorted_particles = sorted(self.particles, key=lambda p: p['size'], reverse=True)
        
        for particle in sorted_particles:
            # Skip dead particles
            if particle['age'] >= particle['lifetime'] or particle['alpha'] <= 0:
                continue
                
            # Create a surface for the particle
            s = int(particle['size'] * 2)
            if s <= 0:
                continue
                
            particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            # Create the particle with alpha
            pygame.draw.circle(
                particle_surface,
                (*particle['color'], particle['alpha']),
                (s // 2, s // 2),
                max(1, int(particle['size']))
            )
            
            # Draw it on screen
            screen.blit(
                particle_surface,
                (int(particle['x'] - particle['size']), 
                 int(particle['y'] - particle['size']))
            )
    
    def is_complete(self):
        """Check if all particles are dead or effect duration is over"""
        return self.frame >= (self.duration + 60)  # Allow even more time for lingering particles

class GrayExhaust:
    """A simple gray exhaust effect for airplanes"""
    def __init__(self, pos_x, pos_y):
        self.particles = []
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.duration = 25
        self.frame = 0
        self.create_particles()
    
    def create_particles(self):
        """Create a small number of gray particles"""
        num_particles = random.randint(3, 5)
        
        for i in range(num_particles):
            # Mainly going left (behind the plane)
            dx = random.uniform(-2.5, -0.8)
            # Very small vertical spread
            dy = random.uniform(-0.3, 0.3)
            
            # Small gray particles
            size = random.uniform(2, 4)
            lifetime = random.randint(10, self.duration)
            
            # Gray exhaust colors
            gray = random.randint(180, 220)
            
            self.particles.append({
                'x': self.pos_x,
                'y': self.pos_y,
                'dx': dx,
                'dy': dy,
                'size': size,
                'lifetime': lifetime,
                'age': 0,
                'color': (gray, gray, gray),
                'alpha': random.randint(150, 200)
            })
    
    def update(self):
        """Update particle positions and ages"""
        self.frame += 1
        
        for particle in self.particles:
            # Update position
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            
            # Add minor gravity and slowdown
            particle['dy'] += 0.03
            particle['dx'] *= 0.98
            
            # Age the particle
            particle['age'] += 1
            
            # Fade out as it ages
            if particle['age'] > particle['lifetime'] * 0.5:
                particle['alpha'] = max(0, particle['alpha'] - 10)
                particle['size'] = max(0, particle['size'] - 0.1)
    
    def draw(self, screen):
        """Draw the exhaust particles"""
        for particle in self.particles:
            # Skip dead particles
            if particle['age'] >= particle['lifetime'] or particle['alpha'] <= 0:
                continue
                
            # Create a surface for the exhaust particle
            s = int(particle['size'] * 2)
            if s <= 0:
                continue
                
            particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            # Create gray particle with alpha
            pygame.draw.circle(
                particle_surface,
                (*particle['color'], particle['alpha']),
                (s // 2, s // 2),
                max(1, int(particle['size']))
            )
            
            # Draw it on screen
            screen.blit(
                particle_surface,
                (int(particle['x'] - particle['size']), 
                 int(particle['y'] - particle['size']))
            )
    
    def is_complete(self):
        """Check if all particles are dead or effect duration is over"""
        if self.frame >= self.duration:
            return True
            
        # Also check if all particles are effectively invisible
        return all(p['age'] >= p['lifetime'] or p['alpha'] <= 10 for p in self.particles)