import pygame
import random
import math

class ExplosionEffect:
    """A realistic aircraft explosion effect with massive flames and debris"""
    def __init__(self, pos_x, pos_y):
        self.particles = []
        self.debris = []
        self.shockwaves = []
        self.flames = []
        self.smoke_clouds = []
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.duration = 180  # Longer duration for full explosion sequence
        self.frame = 0
        
        # Create the initial blast
        self.create_initial_blast()
        
        # Schedule secondary explosions
        self.secondary_explosions = []
        for _ in range(5):  # More secondary explosions
            self.secondary_explosions.append({
                'x': self.pos_x + random.uniform(-25, 25),
                'y': self.pos_y + random.uniform(-25, 25),
                'delay': random.randint(5, 30),
                'size': random.uniform(0.6, 1.2)  # Varying sizes
            })
    
    def create_initial_blast(self):
        """Create the initial explosion with massive flames"""
        # Initial fireball (bright orange/yellow center)
        num_core = random.randint(60, 80)
        for i in range(num_core):
            # Initial burst particles move faster
            speed = random.uniform(5, 15)
            angle = random.uniform(0, 2 * math.pi)
            
            # Hot center - brighter orange/yellow
            r = 255
            g = random.randint(180, 240)
            b = random.randint(0, 60)
            
            self.particles.append({
                'x': self.pos_x,
                'y': self.pos_y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed,
                'size': random.uniform(15, 35),  # Larger particles
                'lifetime': random.randint(20, 40),
                'age': 0,
                'color': (r, g, b),
                'alpha': 255,
                'type': 'core'
            })
        
        # Create large flame particles (deep oranges, reds)
        num_flames = random.randint(120, 180)  # Many more flames
        for i in range(num_flames):
            speed = random.uniform(2, 12)
            angle = random.uniform(0, 2 * math.pi)
            
            # Create a gradient from orange to deep red
            r = 255
            g = random.randint(20, 150)  # Various oranges and reds
            b = random.randint(0, 20)    # Very little blue
            
            self.particles.append({
                'x': self.pos_x,
                'y': self.pos_y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed - random.uniform(0, 1.5),  # Slight upward bias
                'size': random.uniform(12, 30),  # Larger flames
                'lifetime': random.randint(40, 80),  # Longer lasting
                'age': 0,
                'color': (r, g, b),
                'alpha': random.randint(220, 255),
                'type': 'fire'
            })
        
        # Create persistent flame effects
        num_persistent_flames = random.randint(8, 12)
        for i in range(num_persistent_flames):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(5, 20)
            
            self.flames.append({
                'x': self.pos_x + math.cos(angle) * distance,
                'y': self.pos_y + math.sin(angle) * distance,
                'size': random.uniform(15, 35),
                'lifetime': random.randint(60, 120),
                'age': 0,
                'flicker_speed': random.uniform(0.1, 0.2),
                'flicker_offset': random.uniform(0, 3.14)  # Random offset
            })
        
        # Create smoke particles (dark grays to black)
        num_smoke = random.randint(80, 140)
        for i in range(num_smoke):
            speed = random.uniform(1, 6)
            angle = random.uniform(0, 2 * math.pi)
            
            # Dark smoke colors
            gray = random.randint(20, 70)  # Darker smoke
            
            self.particles.append({
                'x': self.pos_x,
                'y': self.pos_y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed - random.uniform(0.8, 2.5),  # Stronger upward bias for smoke
                'size': random.uniform(20, 40),  # Larger smoke particles
                'lifetime': random.randint(90, 150),  # Lingering smoke
                'age': 0,
                'color': (gray, gray, gray),
                'alpha': random.randint(150, 220),
                'type': 'smoke'
            })
        
        # Create large smoke clouds (slower moving, expanding)
        num_clouds = random.randint(6, 10)
        for i in range(num_clouds):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(5, 25)
            
            # Dark smoke colors with slight variation
            gray_base = random.randint(30, 60)
            gray = (
                gray_base + random.randint(-10, 10),
                gray_base + random.randint(-10, 10),
                gray_base + random.randint(-5, 5)
            )
            
            self.smoke_clouds.append({
                'x': self.pos_x + math.cos(angle) * distance,
                'y': self.pos_y + math.sin(angle) * distance,
                'dx': math.cos(angle) * random.uniform(0.3, 1.2),
                'dy': -random.uniform(0.5, 1.5),  # Rise upward
                'size': random.uniform(25, 40),
                'max_size': random.uniform(50, 80),
                'growth_rate': random.uniform(0.2, 0.5),
                'lifetime': random.randint(100, 180),
                'age': 0,
                'color': gray,
                'alpha': random.randint(150, 200)
            })
        
        # Create metal debris (plane parts)
        num_debris = random.randint(30, 50)  # More debris
        for i in range(num_debris):
            speed = random.uniform(4, 15)  # Faster moving debris
            angle = random.uniform(0, 2 * math.pi)
            
            # Debris colors (metallic grays, blues, whites)
            if random.random() < 0.6:  # Metal parts
                r = g = b = random.randint(150, 220)  # Silver/metallic
            elif random.random() < 0.8:  # Blue parts (from plane)
                r = random.randint(50, 80)
                g = random.randint(80, 130)
                b = random.randint(150, 220)
            else:  # White parts (from plane body)
                r = g = b = random.randint(220, 255)
            
            # Debris are small metal chunks with longer trails
            self.debris.append({
                'x': self.pos_x,
                'y': self.pos_y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed,
                'size': random.uniform(2, 8),
                'lifetime': random.randint(40, 120),
                'age': 0,
                'color': (r, g, b),
                'alpha': 255,
                'trail': [],  # Store positions for trail effect
                'trail_color': (min(r + 50, 255), min(g + 50, 255), min(b + 20, 255)) if random.random() < 0.5 else (255, random.randint(100, 200), 0)  # Some debris trails are flaming
            })
        
        # Create expanding shockwave
        self.shockwaves.append({
            'x': self.pos_x,
            'y': self.pos_y,
            'radius': 5,
            'max_radius': random.randint(80, 120),  # Larger radius
            'growth_rate': random.uniform(4, 7),  # Faster expansion
            'alpha': 200,
            'color': (255, 255, 255)
        })
    
    def create_secondary_explosion(self, x, y, size_factor=1.0):
        """Create a smaller secondary explosion"""
        num_particles = random.randint(40, 80)
        
        # Fire particles
        for i in range(num_particles):
            speed = random.uniform(2, 8) * size_factor
            angle = random.uniform(0, 2 * math.pi)
            
            # Orange/red colors - more varied
            r = 255
            g = random.randint(30, 170)
            b = random.randint(0, 30)
            
            self.particles.append({
                'x': x,
                'y': y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed,
                'size': random.uniform(10, 20) * size_factor,
                'lifetime': random.randint(20, 50),
                'age': 0,
                'color': (r, g, b),
                'alpha': random.randint(200, 255),
                'type': 'fire'
            })
        
        # Smoke from secondary explosion
        num_smoke = random.randint(20, 40)
        for i in range(num_smoke):
            speed = random.uniform(1, 3) * size_factor
            angle = random.uniform(0, 2 * math.pi)
            
            # Dark smoke
            gray = random.randint(20, 70)
            
            self.particles.append({
                'x': x,
                'y': y,
                'dx': math.cos(angle) * speed,
                'dy': math.sin(angle) * speed - random.uniform(0.5, 1.5),
                'size': random.uniform(15, 25) * size_factor,
                'lifetime': random.randint(40, 80),
                'age': 0,
                'color': (gray, gray, gray),
                'alpha': random.randint(150, 200),
                'type': 'smoke'
            })
        
        # Small shockwave
        self.shockwaves.append({
            'x': x,
            'y': y,
            'radius': 3,
            'max_radius': random.randint(30, 60) * size_factor,
            'growth_rate': random.uniform(3, 5),
            'alpha': 180,
            'color': (255, 255, 220)
        })
        
        # Add flames at the secondary explosion site
        num_flames = random.randint(2, 5)
        for i in range(num_flames):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(2, 8) * size_factor
            
            self.flames.append({
                'x': x + math.cos(angle) * distance,
                'y': y + math.sin(angle) * distance,
                'size': random.uniform(8, 20) * size_factor,
                'lifetime': random.randint(30, 70),
                'age': 0,
                'flicker_speed': random.uniform(0.1, 0.3),
                'flicker_offset': random.uniform(0, 3.14)
            })
    
    def update(self):
        """Update all explosion components"""
        self.frame += 1
        
        # Check for scheduled secondary explosions
        for explosion in self.secondary_explosions[:]:
            if explosion['delay'] <= self.frame:
                self.create_secondary_explosion(explosion['x'], explosion['y'], explosion['size'])
                self.secondary_explosions.remove(explosion)
        
        # Update particles
        for particle in self.particles[:]:
            # Update position
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            
            # Different physics based on particle type
            if particle['type'] == 'core':
                # Core particles slow down quickly
                particle['dx'] *= 0.92
                particle['dy'] *= 0.92
            elif particle['type'] == 'fire':
                # Fire rises and spreads
                particle['dy'] -= 0.08  # More upward force
                particle['dx'] *= 0.98
                particle['dy'] *= 0.98
            elif particle['type'] == 'smoke':
                # Smoke rises more and lasts longer
                particle['dy'] -= 0.12  # Stronger upward
                particle['dx'] *= 0.99
                particle['dy'] *= 0.99
                
                # Smoke expands
                if particle['age'] < particle['lifetime'] * 0.6:
                    particle['size'] *= 1.01
            
            # Age the particle
            particle['age'] += 1
            
            # Fade out as it ages
            progress = particle['age'] / particle['lifetime']
            if progress > 0.7:
                fade_factor = 1 - (progress - 0.7) / 0.3
                particle['alpha'] = max(0, int(255 * fade_factor))
            
            # Remove dead particles
            if particle['age'] >= particle['lifetime']:
                self.particles.remove(particle)
        
        # Update persistent flames
        for flame in self.flames[:]:
            flame['age'] += 1
            
            # Remove dead flames
            if flame['age'] >= flame['lifetime']:
                self.flames.remove(flame)
        
        # Update smoke clouds
        for cloud in self.smoke_clouds[:]:
            # Move the cloud
            cloud['x'] += cloud['dx']
            cloud['y'] += cloud['dy']
            
            # Clouds slow down over time
            cloud['dx'] *= 0.98
            cloud['dy'] *= 0.98
            
            # Clouds grow in size until they reach max_size
            if cloud['size'] < cloud['max_size']:
                cloud['size'] += cloud['growth_rate']
            
            # Age the cloud
            cloud['age'] += 1
            
            # Fade out as it ages
            progress = cloud['age'] / cloud['lifetime']
            if progress > 0.5:
                fade_factor = 1 - (progress - 0.5) / 0.5
                cloud['alpha'] = max(0, int(cloud['alpha'] * fade_factor))
            
            # Remove dead clouds
            if cloud['age'] >= cloud['lifetime']:
                self.smoke_clouds.remove(cloud)
        
        # Update debris
        for debris in self.debris[:]:
            # Store previous position for trail
            debris['trail'].append((debris['x'], debris['y']))
            if len(debris['trail']) > 15:  # Longer trails
                debris['trail'].pop(0)
            
            # Update position with gravity
            debris['x'] += debris['dx']
            debris['y'] += debris['dy']
            debris['dy'] += 0.25  # Gravity
            
            # Slight air resistance
            debris['dx'] *= 0.99
            debris['dy'] *= 0.99
            
            # Age the debris
            debris['age'] += 1
            
            # Remove old debris
            if debris['age'] >= debris['lifetime']:
                self.debris.remove(debris)
        
        # Update shockwaves
        for wave in self.shockwaves[:]:
            wave['radius'] += wave['growth_rate']
            
            # Fade out as it expands
            progress = wave['radius'] / wave['max_radius']
            wave['alpha'] = max(0, int(wave['alpha'] * (1 - progress)))
            
            # Remove completed shockwaves
            if wave['radius'] >= wave['max_radius']:
                self.shockwaves.remove(wave)
    
    def draw(self, screen):
        """Draw all explosion components in the correct order"""
        # Draw shockwaves first (behind everything)
        for wave in self.shockwaves:
            # Create a surface for the shockwave
            size = int(wave['radius'] * 2)
            wave_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # Draw the shockwave ring
            pygame.draw.circle(
                wave_surface,
                (*wave['color'], wave['alpha']),
                (size // 2, size // 2),
                int(wave['radius']),
                width=2  # Just draw the outline for ring effect
            )
            
            # Draw it on screen
            screen.blit(
                wave_surface,
                (int(wave['x'] - wave['radius']), int(wave['y'] - wave['radius']))
            )
        
        # Draw smoke clouds (behind everything else)
        for cloud in self.smoke_clouds:
            # Create a surface for the cloud
            size = int(cloud['size'] * 2)
            cloud_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # Draw the cloud (multiple overlapping circles for puffy look)
            center = (size // 2, size // 2)
            num_puffs = random.randint(5, 8)
            for _ in range(num_puffs):
                offset_x = random.uniform(-cloud['size'] * 0.3, cloud['size'] * 0.3)
                offset_y = random.uniform(-cloud['size'] * 0.3, cloud['size'] * 0.3)
                puff_size = random.uniform(cloud['size'] * 0.5, cloud['size'] * 0.8)
                
                pygame.draw.circle(
                    cloud_surface,
                    (*cloud['color'], cloud['alpha']),
                    (int(center[0] + offset_x), int(center[1] + offset_y)),
                    int(puff_size)
                )
            
            # Draw it on screen
            screen.blit(
                cloud_surface,
                (int(cloud['x'] - cloud['size']), int(cloud['y'] - cloud['size']))
            )
        
        # Draw smoke particles (behind fire)
        for particle in [p for p in self.particles if p['type'] == 'smoke']:
            s = int(particle['size'] * 2)
            if s <= 0:
                continue
                
            particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            pygame.draw.circle(
                particle_surface,
                (*particle['color'], particle['alpha']),
                (s // 2, s // 2),
                max(1, int(particle['size']))
            )
            
            screen.blit(
                particle_surface,
                (int(particle['x'] - particle['size']), int(particle['y'] - particle['size']))
            )
        
        # Draw debris with trails
        for debris in self.debris:
            # Draw trail first
            if len(debris['trail']) > 1:
                # Create fading trail effect
                for i in range(len(debris['trail']) - 1):
                    alpha = int(150 * (i / len(debris['trail'])))  # Fade out older trail points
                    start_pos = debris['trail'][i]
                    end_pos = debris['trail'][i + 1]
                    
                    # Draw trail segment with debris color or flame color
                    pygame.draw.line(
                        screen,
                        (*debris['trail_color'], alpha),
                        start_pos,
                        end_pos,
                        max(1, int(debris['size'] // 2))
                    )
            
            # Draw debris particle
            s = int(debris['size'] * 2)
            if s <= 0:
                continue
                
            debris_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            pygame.draw.circle(
                debris_surface,
                (*debris['color'], debris['alpha']),
                (s // 2, s // 2),
                max(1, int(debris['size']))
            )
            
            screen.blit(
                debris_surface,
                (int(debris['x'] - debris['size']), int(debris['y'] - debris['size']))
            )
        
        # Draw persistent flames
        for flame in self.flames:
            # Flames flicker by varying size
            flicker = math.sin(self.frame * flame['flicker_speed'] + flame['flicker_offset']) * 0.2 + 0.8
            current_size = flame['size'] * flicker
            
            # Calculate flame color (from yellow at center to red at edges)
            # Based on age - flames get darker over time
            age_factor = 1 - (flame['age'] / flame['lifetime'])
            
            # Create a flame shape
            s = int(current_size * 3)  # Surface larger than flame to accommodate glow
            if s <= 0:
                continue
                
            flame_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            center = (s // 2, s // 2)
            
            # Draw layers of the flame from outside in
            # Outer glow (red)
            outer_color = (255, 40 * age_factor, 0, 100)
            pygame.draw.circle(
                flame_surface,
                outer_color,
                center,
                int(current_size * 1.2)
            )
            
            # Main flame (orange)
            mid_color = (255, int(128 * age_factor), 0, 180)
            pygame.draw.circle(
                flame_surface,
                mid_color,
                center,
                int(current_size * 0.9)
            )
            
            # Core (yellow)
            inner_color = (255, int(200 * age_factor), 0, 255)
            pygame.draw.circle(
                flame_surface,
                inner_color,
                center,
                int(current_size * 0.5)
            )
            
            # Draw the flame
            screen.blit(
                flame_surface,
                (int(flame['x'] - current_size * 1.5), int(flame['y'] - current_size * 1.5))
            )
        
        # Draw fire particles
        for particle in [p for p in self.particles if p['type'] == 'fire']:
            s = int(particle['size'] * 2)
            if s <= 0:
                continue
                
            particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            # Draw with a glow effect for fire particles
            glow_radius = int(particle['size'] * 1.3)
            core_radius = int(particle['size'] * 0.7)
            
            # Outer glow (more transparent)
            pygame.draw.circle(
                particle_surface,
                (*particle['color'], particle['alpha'] // 2),
                (s // 2, s // 2),
                glow_radius
            )
            
            # Core (full color)
            pygame.draw.circle(
                particle_surface,
                (*particle['color'], particle['alpha']),
                (s // 2, s // 2),
                core_radius
            )
            
            screen.blit(
                particle_surface,
                (int(particle['x'] - particle['size']), int(particle['y'] - particle['size']))
            )
        
        # Draw core particles (brightest, on top)
        for particle in [p for p in self.particles if p['type'] == 'core']:
            s = int(particle['size'] * 2)
            if s <= 0:
                continue
                
            particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            # Bright core with glow
            pygame.draw.circle(
                particle_surface,
                (*particle['color'], particle['alpha']),
                (s // 2, s // 2),
                max(1, int(particle['size']))
            )
            
            # Add a bright center
            pygame.draw.circle(
                particle_surface,
                (255, 255, 180, particle['alpha']),
                (s // 2, s // 2),
                max(1, int(particle['size'] * 0.5))
            )
            
            screen.blit(
                particle_surface,
                (int(particle['x'] - particle['size']), int(particle['y'] - particle['size']))
            )
    
    def is_complete(self):
        """Check if the explosion effect is complete"""
        return (self.frame >= self.duration and 
                len(self.particles) == 0 and 
                len(self.debris) == 0 and 
                len(self.shockwaves) == 0 and
                len(self.flames) == 0 and
                len(self.smoke_clouds) == 0)


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