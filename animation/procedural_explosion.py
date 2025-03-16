# Import pygame for graphics and animation capabilities
import pygame
# Import random module for generating randomized effect properties
import random
# Import math module for trigonometric functions and mathematical calculations
import math

# Define the ExplosionEffect class that creates a realistic aircraft explosion animation
class ExplosionEffect:
    """A realistic aircraft explosion effect with massive flames and debris"""
    # Constructor initializes a new explosion at the specified coordinates
    # Parameters:
    #   pos_x: horizontal position of explosion center
    #   pos_y: vertical position of explosion center
    def __init__(self, pos_x, pos_y):
        # Initialize empty lists to store different types of explosion elements
        self.particles = []      # General particles (core, fire, smoke)
        self.debris = []         # Metal fragments with trails
        self.shockwaves = []     # Expanding circular shock waves
        self.flames = []         # Persistent flame effects
        self.smoke_clouds = []   # Large smoke clouds
        # Store explosion center position
        self.pos_x = pos_x
        self.pos_y = pos_y
        # Set total duration of explosion effect in frames
        self.duration = 180  # 6 seconds at 30fps
        # Initialize frame counter
        self.frame = 0
        
        # Create the main explosion immediately
        self.create_initial_blast()
        
        # Schedule delayed secondary explosions for more realism
        self.secondary_explosions = []
        # Create 5 secondary explosions with random positions, delays, and sizes
        for _ in range(5):
            self.secondary_explosions.append({
                'x': self.pos_x + random.uniform(-25, 25),        # Offset from main explosion
                'y': self.pos_y + random.uniform(-25, 25),        # Offset from main explosion
                'delay': random.randint(5, 30),                   # Frames until this explosion triggers
                'size': random.uniform(0.6, 1.2)                  # Relative size (0.6-1.2x)
            })
    
    # Method to create the initial main explosion
    # No parameters or return value
    def create_initial_blast(self):
        """Create the initial explosion with massive flames"""
        # Initial fireball (bright orange/yellow center)
        # Create 60-80 core particles for the bright center
        num_core = random.randint(60, 80)
        for i in range(num_core):
            # Initial burst particles move faster with random directions
            speed = random.uniform(5, 15)  # Random speed
            angle = random.uniform(0, 2 * math.pi)  # Random direction (0-360 degrees)
            
            # Hot center - brighter orange/yellow colors
            r = 255  # Full red
            g = random.randint(180, 240)  # Variable green for orange/yellow variation
            b = random.randint(0, 60)  # Low blue for warmer colors
            
            # Add core particle with properties
            self.particles.append({
                'x': self.pos_x,  # Start at explosion center
                'y': self.pos_y,  # Start at explosion center
                'dx': math.cos(angle) * speed,  # Horizontal velocity component
                'dy': math.sin(angle) * speed,  # Vertical velocity component
                'size': random.uniform(15, 35),  # Random size (large for visibility)
                'lifetime': random.randint(20, 40),  # How long this particle lives
                'age': 0,  # Current age (frames)
                'color': (r, g, b),  # RGB color
                'alpha': 255,  # Full opacity
                'type': 'core'  # Particle type identifier
            })
        
        # Create large flame particles (deep oranges, reds)
        # Many more flame particles (120-180) for thick fire effect
        num_flames = random.randint(120, 180)
        for i in range(num_flames):
            speed = random.uniform(2, 12)  # Random speed
            angle = random.uniform(0, 2 * math.pi)  # Random direction
            
            # Create a gradient from orange to deep red
            r = 255  # Full red
            g = random.randint(20, 150)  # Lower green for deeper oranges and reds
            b = random.randint(0, 20)    # Very little blue (keeps colors warm)
            
            # Add flame particle with properties
            self.particles.append({
                'x': self.pos_x,  # Start at explosion center
                'y': self.pos_y,  # Start at explosion center
                'dx': math.cos(angle) * speed,  # Horizontal velocity
                'dy': math.sin(angle) * speed - random.uniform(0, 1.5),  # Vertical velocity with upward bias
                'size': random.uniform(12, 30),  # Random size
                'lifetime': random.randint(40, 80),  # Longer lasting than core
                'age': 0,  # Current age
                'color': (r, g, b),  # RGB color
                'alpha': random.randint(220, 255),  # Slight transparency variation
                'type': 'fire'  # Particle type identifier
            })
        
        # Create persistent flame effects for longer-burning areas
        num_persistent_flames = random.randint(8, 12)  # Number of persistent flames
        for i in range(num_persistent_flames):
            # Place flames in a circle around explosion center
            angle = random.uniform(0, 2 * math.pi)  # Random angle
            distance = random.uniform(5, 20)  # Random distance from center
            
            # Add persistent flame with properties
            self.flames.append({
                'x': self.pos_x + math.cos(angle) * distance,  # Positioned around center
                'y': self.pos_y + math.sin(angle) * distance,  # Positioned around center
                'size': random.uniform(15, 35),  # Random size
                'lifetime': random.randint(60, 120),  # Long-lasting flames
                'age': 0,  # Current age
                'flicker_speed': random.uniform(0.1, 0.2),  # Controls flame animation speed
                'flicker_offset': random.uniform(0, 3.14)  # Random phase offset for varied animation
            })
        
        # Create smoke particles (dark grays to black)
        num_smoke = random.randint(80, 140)  # Number of smoke particles
        for i in range(num_smoke):
            speed = random.uniform(1, 6)  # Slower than flames
            angle = random.uniform(0, 2 * math.pi)  # Random direction
            
            # Dark smoke colors (grays)
            gray = random.randint(20, 70)  # Dark gray values
            
            # Add smoke particle with properties
            self.particles.append({
                'x': self.pos_x,  # Start at explosion center
                'y': self.pos_y,  # Start at explosion center
                'dx': math.cos(angle) * speed,  # Horizontal velocity
                'dy': math.sin(angle) * speed - random.uniform(0.8, 2.5),  # Vertical with stronger upward bias
                'size': random.uniform(20, 40),  # Larger particles for smoke
                'lifetime': random.randint(90, 150),  # Smoke lingers longer
                'age': 0,  # Current age
                'color': (gray, gray, gray),  # Gray RGB values
                'alpha': random.randint(150, 220),  # Semi-transparent
                'type': 'smoke'  # Particle type identifier
            })
        
        # Create large smoke clouds (slower moving, expanding)
        num_clouds = random.randint(6, 10)  # Number of large smoke clouds
        for i in range(num_clouds):
            # Position clouds around explosion center
            angle = random.uniform(0, 2 * math.pi)  # Random angle
            distance = random.uniform(5, 25)  # Random distance from center
            
            # Dark smoke colors with slight variation between RGB channels
            gray_base = random.randint(30, 60)  # Base gray value
            gray = (
                gray_base + random.randint(-10, 10),  # R with variation
                gray_base + random.randint(-10, 10),  # G with variation
                gray_base + random.randint(-5, 5)     # B with less variation
            )
            
            # Add smoke cloud with properties
            self.smoke_clouds.append({
                'x': self.pos_x + math.cos(angle) * distance,  # Positioned around center
                'y': self.pos_y + math.sin(angle) * distance,  # Positioned around center
                'dx': math.cos(angle) * random.uniform(0.3, 1.2),  # Horizontal velocity
                'dy': -random.uniform(0.5, 1.5),  # Upward velocity only
                'size': random.uniform(25, 40),  # Starting size
                'max_size': random.uniform(50, 80),  # Maximum size (clouds expand)
                'growth_rate': random.uniform(0.2, 0.5),  # Speed of expansion
                'lifetime': random.randint(100, 180),  # Long-lasting clouds
                'age': 0,  # Current age
                'color': gray,  # RGB color tuple
                'alpha': random.randint(150, 200)  # Semi-transparent
            })
        
        # Create metal debris (plane parts)
        num_debris = random.randint(30, 50)  # Number of debris particles
        for i in range(num_debris):
            speed = random.uniform(4, 15)  # Fast-moving debris
            angle = random.uniform(0, 2 * math.pi)  # Random direction
            
            # Debris colors with probability weights
            if random.random() < 0.6:  # 60% chance: Metal parts
                r = g = b = random.randint(150, 220)  # Silver/metallic gray
            elif random.random() < 0.8:  # 20% chance: Blue parts (from plane)
                r = random.randint(50, 80)  # Low red
                g = random.randint(80, 130)  # Medium green
                b = random.randint(150, 220)  # High blue
            else:  # 20% chance: White parts (from plane body)
                r = g = b = random.randint(220, 255)  # Light gray/white
            
            # Add debris particle with properties
            self.debris.append({
                'x': self.pos_x,  # Start at explosion center
                'y': self.pos_y,  # Start at explosion center
                'dx': math.cos(angle) * speed,  # Horizontal velocity
                'dy': math.sin(angle) * speed,  # Vertical velocity
                'size': random.uniform(2, 8),  # Small metal pieces
                'lifetime': random.randint(40, 120),  # Variable lifetime
                'age': 0,  # Current age
                'color': (r, g, b),  # RGB color
                'alpha': 255,  # Fully opaque
                'trail': [],  # Empty list to store trail positions
                'trail_color': (min(r + 50, 255), min(g + 50, 255), min(b + 20, 255)) if random.random() < 0.5 else (255, random.randint(100, 200), 0)  # Brightened color or flame color
            })
        
        # Create expanding shockwave
        self.shockwaves.append({
            'x': self.pos_x,  # Center at explosion
            'y': self.pos_y,  # Center at explosion
            'radius': 5,  # Starting radius
            'max_radius': random.randint(80, 120),  # Maximum expansion radius
            'growth_rate': random.uniform(4, 7),  # Expansion speed
            'alpha': 200,  # Starting transparency
            'color': (255, 255, 255)  # White color
        })
    
    # Method to create smaller secondary explosions
    # Parameters:
    #   x: horizontal position
    #   y: vertical position
    #   size_factor: scaling factor for explosion size (default 1.0)
    # No return value
    def create_secondary_explosion(self, x, y, size_factor=1.0):
        """Create a smaller secondary explosion"""
        # Create 40-80 particles for this secondary explosion
        num_particles = random.randint(40, 80)
        
        # Fire particles for secondary explosion
        for i in range(num_particles):
            speed = random.uniform(2, 8) * size_factor  # Scale speed by size factor
            angle = random.uniform(0, 2 * math.pi)  # Random direction
            
            # Orange/red colors with more variation
            r = 255  # Full red
            g = random.randint(30, 170)  # Wide range of green for color variation
            b = random.randint(0, 30)  # Low blue for warmth
            
            # Add fire particle
            self.particles.append({
                'x': x,  # Start at specified position
                'y': y,  # Start at specified position
                'dx': math.cos(angle) * speed,  # Horizontal velocity
                'dy': math.sin(angle) * speed,  # Vertical velocity
                'size': random.uniform(10, 20) * size_factor,  # Scaled size
                'lifetime': random.randint(20, 50),  # Shorter lifetime than main explosion
                'age': 0,  # Current age
                'color': (r, g, b),  # RGB color
                'alpha': random.randint(200, 255),  # Near-opaque
                'type': 'fire'  # Particle type identifier
            })
        
        # Smoke particles for secondary explosion
        num_smoke = random.randint(20, 40)  # Fewer smoke particles than main explosion
        for i in range(num_smoke):
            speed = random.uniform(1, 3) * size_factor  # Scaled speed
            angle = random.uniform(0, 2 * math.pi)  # Random direction
            
            # Dark smoke color
            gray = random.randint(20, 70)  # Dark gray
            
            # Add smoke particle
            self.particles.append({
                'x': x,  # Start at specified position
                'y': y,  # Start at specified position
                'dx': math.cos(angle) * speed,  # Horizontal velocity
                'dy': math.sin(angle) * speed - random.uniform(0.5, 1.5),  # With upward bias
                'size': random.uniform(15, 25) * size_factor,  # Scaled size
                'lifetime': random.randint(40, 80),  # Medium lifetime
                'age': 0,  # Current age
                'color': (gray, gray, gray),  # Gray RGB values
                'alpha': random.randint(150, 200),  # Semi-transparent
                'type': 'smoke'  # Particle type identifier
            })
        
        # Small shockwave for secondary explosion
        self.shockwaves.append({
            'x': x,  # Center at specified position
            'y': y,  # Center at specified position
            'radius': 3,  # Starting radius (smaller than main)
            'max_radius': random.randint(30, 60) * size_factor,  # Scaled maximum radius
            'growth_rate': random.uniform(3, 5),  # Expansion speed
            'alpha': 180,  # Starting transparency
            'color': (255, 255, 220)  # Slight yellow tint
        })
        
        # Add flames at the secondary explosion site
        num_flames = random.randint(2, 5)  # Fewer flames than main explosion
        for i in range(num_flames):
            angle = random.uniform(0, 2 * math.pi)  # Random angle
            distance = random.uniform(2, 8) * size_factor  # Scaled distance from center
            
            # Add persistent flame
            self.flames.append({
                'x': x + math.cos(angle) * distance,  # Positioned around center
                'y': y + math.sin(angle) * distance,  # Positioned around center
                'size': random.uniform(8, 20) * size_factor,  # Scaled size
                'lifetime': random.randint(30, 70),  # Medium lifetime
                'age': 0,  # Current age
                'flicker_speed': random.uniform(0.1, 0.3),  # Animation speed
                'flicker_offset': random.uniform(0, 3.14)  # Animation phase offset
            })
    
    # Method to update all explosion components (called once per frame)
    # No parameters or return value
    def update(self):
        """Update all explosion components"""
        # Increment frame counter
        self.frame += 1
        
        # Check for and trigger scheduled secondary explosions
        for explosion in self.secondary_explosions[:]:  # Use slice copy to safely modify during iteration
            if explosion['delay'] <= self.frame:  # Check if it's time for this explosion
                # Create the secondary explosion
                self.create_secondary_explosion(explosion['x'], explosion['y'], explosion['size'])
                # Remove from list after triggering
                self.secondary_explosions.remove(explosion)
        
        # Update particles (core, fire, smoke)
        for particle in self.particles[:]:  # Use slice copy to safely modify during iteration
            # Update position based on velocity
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            
            # Apply different physics based on particle type
            if particle['type'] == 'core':
                # Core particles slow down quickly for concentrated explosion
                particle['dx'] *= 0.92  # Horizontal velocity decay
                particle['dy'] *= 0.92  # Vertical velocity decay
            elif particle['type'] == 'fire':
                # Fire rises and spreads
                particle['dy'] -= 0.08  # Upward force (fire rises)
                particle['dx'] *= 0.98  # Slow horizontal decay
                particle['dy'] *= 0.98  # Slow vertical decay
            elif particle['type'] == 'smoke':
                # Smoke rises more and lasts longer
                particle['dy'] -= 0.12  # Stronger upward force (smoke rises faster)
                particle['dx'] *= 0.99  # Very slow horizontal decay
                particle['dy'] *= 0.99  # Very slow vertical decay
                
                # Smoke expands during first 60% of its life
                if particle['age'] < particle['lifetime'] * 0.6:
                    particle['size'] *= 1.01  # Grow by 1% each frame
            
            # Increment particle age
            particle['age'] += 1
            
            # Fade out alpha as particle nears end of life
            progress = particle['age'] / particle['lifetime']  # Life progress (0-1)
            if progress > 0.7:  # Start fading at 70% of lifetime
                fade_factor = 1 - (progress - 0.7) / 0.3  # Linear fade to zero
                particle['alpha'] = max(0, int(255 * fade_factor))  # Apply fade to alpha
            
            # Remove dead particles
            if particle['age'] >= particle['lifetime']:
                self.particles.remove(particle)
        
        # Update persistent flames
        for flame in self.flames[:]:  # Use slice copy to safely modify during iteration
            # Increment flame age
            flame['age'] += 1
            
            # Remove flames that have expired
            if flame['age'] >= flame['lifetime']:
                self.flames.remove(flame)
        
        # Update smoke clouds
        for cloud in self.smoke_clouds[:]:  # Use slice copy to safely modify during iteration
            # Move the cloud based on velocity
            cloud['x'] += cloud['dx']
            cloud['y'] += cloud['dy']
            
            # Clouds slow down over time
            cloud['dx'] *= 0.98  # Horizontal velocity decay
            cloud['dy'] *= 0.98  # Vertical velocity decay
            
            # Clouds grow in size until they reach max_size
            if cloud['size'] < cloud['max_size']:
                cloud['size'] += cloud['growth_rate']  # Increase size each frame
            
            # Increment cloud age
            cloud['age'] += 1
            
            # Fade out alpha as cloud nears end of life
            progress = cloud['age'] / cloud['lifetime']  # Life progress (0-1)
            if progress > 0.5:  # Start fading at 50% of lifetime
                fade_factor = 1 - (progress - 0.5) / 0.5  # Linear fade to zero
                cloud['alpha'] = max(0, int(cloud['alpha'] * fade_factor))  # Apply fade
            
            # Remove expired clouds
            if cloud['age'] >= cloud['lifetime']:
                self.smoke_clouds.remove(cloud)
        
        # Update debris particles
        for debris in self.debris[:]:  # Use slice copy to safely modify during iteration
            # Store previous position for trail effect
            debris['trail'].append((debris['x'], debris['y']))
            # Limit trail length to 15 positions
            if len(debris['trail']) > 15:
                debris['trail'].pop(0)  # Remove oldest position
            
            # Update position with gravity effect
            debris['x'] += debris['dx']  # Horizontal movement
            debris['y'] += debris['dy']  # Vertical movement
            debris['dy'] += 0.25  # Apply gravity (accelerate downward)
            
            # Apply slight air resistance
            debris['dx'] *= 0.99  # Horizontal velocity decay
            debris['dy'] *= 0.99  # Vertical velocity decay
            
            # Increment debris age
            debris['age'] += 1
            
            # Remove expired debris
            if debris['age'] >= debris['lifetime']:
                self.debris.remove(debris)
        
        # Update shockwaves
        for wave in self.shockwaves[:]:  # Use slice copy to safely modify during iteration
            # Increase radius by growth rate
            wave['radius'] += wave['growth_rate']
            
            # Fade out alpha as wave expands
            progress = wave['radius'] / wave['max_radius']  # Expansion progress (0-1)
            wave['alpha'] = max(0, int(wave['alpha'] * (1 - progress)))  # Linear fade to zero
            
            # Remove completed shockwaves
            if wave['radius'] >= wave['max_radius']:
                self.shockwaves.remove(wave)
    
    # Method to draw all explosion components to the screen
    # Parameters:
    #   screen: pygame surface to draw on
    # No return value
    def draw(self, screen):
        """Draw all explosion components in the correct order"""
        # Draw shockwaves first (behind everything)
        for wave in self.shockwaves:
            # Create a surface for the shockwave with transparency
            size = int(wave['radius'] * 2)  # Surface size based on diameter
            wave_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # Draw the shockwave as a ring
            pygame.draw.circle(
                wave_surface,  # Surface to draw on
                (*wave['color'], wave['alpha']),  # RGBA color
                (size // 2, size // 2),  # Center of circle
                int(wave['radius']),  # Radius
                width=2  # Draw as 2px wide ring
            )
            
            # Draw wave surface onto main screen
            screen.blit(
                wave_surface,  # Surface to draw
                (int(wave['x'] - wave['radius']), int(wave['y'] - wave['radius']))  # Position
            )
        
        # Draw smoke clouds (behind everything else)
        for cloud in self.smoke_clouds:
            # Create a surface for the cloud with transparency
            size = int(cloud['size'] * 2)  # Surface size based on diameter
            cloud_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # Draw the cloud as multiple overlapping circles for puffy look
            center = (size // 2, size // 2)  # Center point
            num_puffs = random.randint(5, 8)  # Number of overlapping circles
            
            # Draw each puff in the cloud
            for _ in range(num_puffs):
                # Random offsets from center
                offset_x = random.uniform(-cloud['size'] * 0.3, cloud['size'] * 0.3)
                offset_y = random.uniform(-cloud['size'] * 0.3, cloud['size'] * 0.3)
                # Random puff size
                puff_size = random.uniform(cloud['size'] * 0.5, cloud['size'] * 0.8)
                
                # Draw the puff circle
                pygame.draw.circle(
                    cloud_surface,  # Surface to draw on
                    (*cloud['color'], cloud['alpha']),  # RGBA color
                    (int(center[0] + offset_x), int(center[1] + offset_y)),  # Position
                    int(puff_size)  # Radius
                )
            
            # Draw cloud surface onto main screen
            screen.blit(
                cloud_surface,  # Surface to draw
                (int(cloud['x'] - cloud['size']), int(cloud['y'] - cloud['size']))  # Position
            )
        
        # Draw smoke particles (behind fire)
        for particle in [p for p in self.particles if p['type'] == 'smoke']:
            # Calculate surface size based on particle diameter
            s = int(particle['size'] * 2)
            if s <= 0:
                continue  # Skip invalid sizes
                
            # Create a surface for the smoke particle with transparency
            particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            # Draw the smoke particle as a circle
            pygame.draw.circle(
                particle_surface,  # Surface to draw on
                (*particle['color'], particle['alpha']),  # RGBA color
                (s // 2, s // 2),  # Center of circle
                max(1, int(particle['size']))  # Radius (minimum 1px)
            )
            
            # Draw smoke particle onto main screen
            screen.blit(
                particle_surface,  # Surface to draw
                (int(particle['x'] - particle['size']), int(particle['y'] - particle['size']))  # Position
            )
        
        # Draw debris with trails
        for debris in self.debris:
            # Draw trail first if it exists
            if len(debris['trail']) > 1:
                # Create fading trail effect with multiple line segments
                for i in range(len(debris['trail']) - 1):
                    # Fade alpha based on position in trail (older = more transparent)
                    alpha = int(150 * (i / len(debris['trail'])))
                    # Get positions for line segment
                    start_pos = debris['trail'][i]  # Start point
                    end_pos = debris['trail'][i + 1]  # End point
                    
                    # Draw trail segment with debris trail color
                    pygame.draw.line(
                        screen,  # Surface to draw on
                        (*debris['trail_color'], alpha),  # RGBA color
                        start_pos,  # Start position
                        end_pos,  # End position
                        max(1, int(debris['size'] // 2))  # Line width
                    )
            
            # Draw debris particle
            s = int(debris['size'] * 2)  # Surface size based on diameter
            if s <= 0:
                continue  # Skip invalid sizes
                
            # Create a surface for the debris particle with transparency
            debris_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            # Draw the debris as a circle
            pygame.draw.circle(
                debris_surface,  # Surface to draw on
                (*debris['color'], debris['alpha']),  # RGBA color
                (s // 2, s // 2),  # Center of circle
                max(1, int(debris['size']))  # Radius (minimum 1px)
            )
            
            # Draw debris particle onto main screen
            screen.blit(
                debris_surface,  # Surface to draw
                (int(debris['x'] - debris['size']), int(debris['y'] - debris['size']))  # Position
            )
        
        # Draw persistent flames
        for flame in self.flames:
            # Flames flicker by varying size with sine wave
            flicker = math.sin(self.frame * flame['flicker_speed'] + flame['flicker_offset']) * 0.2 + 0.8
            current_size = flame['size'] * flicker  # Apply flicker to size
            
            # Calculate flame color based on age (flames get darker over time)
            age_factor = 1 - (flame['age'] / flame['lifetime'])  # 1.0 to 0.0 as flame ages
            
            # Create a flame surface with transparency
            s = int(current_size * 3)  # Surface larger than flame to accommodate glow
            if s <= 0:
                continue  # Skip invalid sizes
                
            flame_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            center = (s // 2, s // 2)  # Center point
            
            # Draw layers of the flame from outside in for realistic look
            # Outer glow (red with transparency)
            outer_color = (255, 40 * age_factor, 0, 100)  # Reddish with 40% opacity
            pygame.draw.circle(
                flame_surface,  # Surface to draw on
                outer_color,  # RGBA color
                center,  # Center position
                int(current_size * 1.2)  # Radius (20% larger than flame)
            )
            
            # Main flame (orange with more opacity)
            mid_color = (255, int(128 * age_factor), 0, 180)  # Orange with 70% opacity
            pygame.draw.circle(
                flame_surface,  # Surface to draw on
                mid_color,  # RGBA color
                center,  # Center position
                int(current_size * 0.9)  # Radius (90% of flame size)
            )
            
            # Core (yellow, fully opaque)
            inner_color = (255, int(200 * age_factor), 0, 255)  # Yellow, fully opaque
            pygame.draw.circle(
                flame_surface,  # Surface to draw on
                inner_color,  # RGBA color
                center,  # Center position
                int(current_size * 0.5)  # Radius (50% of flame size)
            )
            
            # Draw the flame onto main screen
            screen.blit(
                flame_surface,  # Surface to draw
                (int(flame['x'] - current_size * 1.5), int(flame['y'] - current_size * 1.5))  # Position
            )
        
        # Draw fire particles
        for particle in [p for p in self.particles if p['type'] == 'fire']:
            s = int(particle['size'] * 2)  # Surface size based on diameter
            if s <= 0:
                continue  # Skip invalid sizes
                
            # Create a surface for the fire particle with transparency
            particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
            
            # Draw with a glow effect for fire particles
            glow_radius = int(particle['size'] * 1.3)  # Outer glow larger than particle
            core_radius = int(particle['size'] * 0.7)  # Inner core smaller than particle
            
            # Outer glow (more transparent)
            pygame.draw.circle(
                particle_surface,  # Surface to draw on
                (*particle['color'], particle['alpha'] // 2),  # RGBA color at half opacity
                (s // 2, s // 2),  # Center of circle
                glow_radius  # Radius of glow
            )
            
            # Core (full color)
            pygame.draw.circle(
                particle_surface,  # Surface to draw on
                (*particle['color'], particle['alpha']),  # RGBA color at full particle opacity
                (s // 2, s // 2),  # Center of circle
                core_radius  # Radius of core
            )       
        # Draw core particles (brightest, on top of everything)
        # Draw core particles (brightest, on top of everything)
        for particle in [p for p in self.particles if p['type'] == 'core']:
           s = int(particle['size'] * 2)  # Surface size based on diameter
           if s <= 0:
               continue  # Skip invalid sizes
               
           # Create a surface for the core particle with transparency
           particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
           
           # Bright core with glow - main explosion center
           pygame.draw.circle(
               particle_surface,  # Surface to draw on
               (*particle['color'], particle['alpha']),  # RGBA color at full particle opacity
               (s // 2, s // 2),  # Center of circle
               max(1, int(particle['size']))  # Radius (minimum 1px)
           )
           
           # Add a bright center (yellow-white) to emphasize intensity
           pygame.draw.circle(
               particle_surface,  # Surface to draw on
               (255, 255, 180, particle['alpha']),  # RGBA very bright yellow-white
               (s // 2, s // 2),  # Center of circle
               max(1, int(particle['size'] * 0.5))  # Half the radius of outer circle
           )
           
           # Draw core particle onto main screen
           screen.blit(
               particle_surface,  # Surface to draw
               (int(particle['x'] - particle['size']), int(particle['y'] - particle['size']))  # Position
           )
   
   # Method to check if explosion animation is complete
   # No parameters
   # Returns: boolean indicating if effect is finished
    def is_complete(self):
       """Check if the explosion effect is complete"""
       # Explosion is complete when all components are gone AND duration has passed
       return (self.frame >= self.duration and 
               len(self.particles) == 0 and 
               len(self.debris) == 0 and 
               len(self.shockwaves) == 0 and
               len(self.flames) == 0 and
               len(self.smoke_clouds) == 0)


# Define a simpler exhaust effect class for airplane engine exhaust
class GrayExhaust:
   """A simple gray exhaust effect for airplanes"""
   # Constructor initializes exhaust effect at specified position
   # Parameters:
   #   pos_x: horizontal position of exhaust origin
   #   pos_y: vertical position of exhaust origin
   def __init__(self, pos_x, pos_y):
       # Initialize list to store exhaust particles
       self.particles = []
       # Store origin position
       self.pos_x = pos_x
       self.pos_y = pos_y
       # Set total duration in frames
       self.duration = 25  # Short effect (less than 1 second at 30fps)
       # Initialize frame counter
       self.frame = 0
       # Create initial particles immediately
       self.create_particles()
   
   # Method to create the exhaust particles
   # No parameters or return value
   def create_particles(self):
       """Create a small number of gray particles"""
       # Create 3-5 particles per exhaust puff
       num_particles = random.randint(3, 5)
       
       # Generate each particle
       for i in range(num_particles):
           # Particles move mainly left (behind the plane)
           # Negative x velocity means moving left
           dx = random.uniform(-2.5, -0.8)
           # Very small vertical spread for streamlined appearance
           dy = random.uniform(-0.3, 0.3)
           
           # Small gray particles for subtle effect
           size = random.uniform(2, 4)  # Small particles
           lifetime = random.randint(10, self.duration)  # Variable lifetime
           
           # Gray exhaust colors (light to medium gray)
           gray = random.randint(180, 220)  # Light to medium gray value
           
           # Add exhaust particle with properties
           self.particles.append({
               'x': self.pos_x,  # Start at exhaust origin
               'y': self.pos_y,  # Start at exhaust origin
               'dx': dx,  # Horizontal velocity (leftward)
               'dy': dy,  # Vertical velocity (minimal)
               'size': size,  # Particle size
               'lifetime': lifetime,  # How long this particle lives
               'age': 0,  # Current age
               'color': (gray, gray, gray),  # Gray RGB values
               'alpha': random.randint(150, 200)  # Semi-transparent
           })
   
   # Method to update exhaust particles (called once per frame)
   # No parameters or return value
   def update(self):
       """Update particle positions and ages"""
       # Increment frame counter
       self.frame += 1
       
       # Update each particle
       for particle in self.particles:
           # Update position based on velocity
           particle['x'] += particle['dx']
           particle['y'] += particle['dy']
           
           # Add minor gravity and slowdown effects
           particle['dy'] += 0.03  # Slight downward acceleration
           particle['dx'] *= 0.98  # Slight horizontal slowdown
           
           # Increment particle age
           particle['age'] += 1
           
           # Fade out and shrink as particle ages (after half its lifetime)
           if particle['age'] > particle['lifetime'] * 0.5:
               particle['alpha'] = max(0, particle['alpha'] - 10)  # Fade by reducing alpha
               particle['size'] = max(0, particle['size'] - 0.1)  # Shrink slightly
   
   # Method to draw exhaust particles to the screen
   # Parameters:
   #   screen: pygame surface to draw on
   # No return value
   def draw(self, screen):
       """Draw the exhaust particles"""
       # Process each particle
       for particle in self.particles:
           # Skip dead or invisible particles
           if particle['age'] >= particle['lifetime'] or particle['alpha'] <= 0:
               continue
               
           # Create a surface for the exhaust particle with transparency
           s = int(particle['size'] * 2)  # Surface size based on diameter
           if s <= 0:
               continue  # Skip invalid sizes
               
           # Create surface with transparency
           particle_surface = pygame.Surface((s, s), pygame.SRCALPHA)
           
           # Draw gray particle circle with alpha
           pygame.draw.circle(
               particle_surface,  # Surface to draw on
               (*particle['color'], particle['alpha']),  # RGBA color
               (s // 2, s // 2),  # Center of circle
               max(1, int(particle['size']))  # Radius (minimum 1px)
           )
           
           # Draw particle onto main screen
           screen.blit(
               particle_surface,  # Surface to draw
               (int(particle['x'] - particle['size']), 
                int(particle['y'] - particle['size']))  # Position
           )
   
   # Method to check if exhaust effect is complete
   # No parameters
   # Returns: boolean indicating if effect is finished
   def is_complete(self):
       """Check if all particles are dead or effect duration is over"""
       # Complete if duration has passed
       if self.frame >= self.duration:
           return True
           
       # Also check if all particles are effectively invisible
       # Function returns True if ALL particles meet the condition
       return all(p['age'] >= p['lifetime'] or p['alpha'] <= 10 for p in self.particles)