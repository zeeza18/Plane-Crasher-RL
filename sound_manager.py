import pygame

class SoundManager:
    """
    Manages sounds and music for the game
    """
    def __init__(self):
        # Initialize the mixer
        try:
            pygame.mixer.init()
            
            # Sound file paths
            self.background_music_path = 'assets/sprites/background_music.mp3'
            self.crash_sound_path = 'assets/sprites/crash_sound.mp3'
            
            # Load sounds
            try:
                # Background music
                pygame.mixer.music.load(self.background_music_path)
                pygame.mixer.music.set_volume(0.3)  # Lower volume for background music
                
                # Sound effects
                self.crash_sound = pygame.mixer.Sound(self.crash_sound_path)
                self.crash_sound.set_volume(0.8)  # Higher volume for crash
                
                self.loaded = True
                print("Sound assets loaded successfully")
            except Exception as e:
                print(f"Error loading sound assets: {e}")
                self.loaded = False
        except Exception as e:
            print(f"Error initializing sound mixer: {e}")
            self.loaded = False
    
    def play_background_music(self, loops=-1):
        """
        Play background music. Loops indefinitely by default.
        """
        if self.loaded:
            try:
                pygame.mixer.music.play(loops)
            except Exception as e:
                print(f"Error playing background music: {e}")
    
    def stop_background_music(self):
        """
        Stop the background music
        """
        if self.loaded:
            try:
                pygame.mixer.music.stop()
            except Exception as e:
                print(f"Error stopping background music: {e}")
    
    def pause_background_music(self):
        """
        Pause the background music
        """
        if self.loaded:
            try:
                pygame.mixer.music.pause()
            except Exception as e:
                print(f"Error pausing background music: {e}")
    
    def unpause_background_music(self):
        """
        Unpause the background music
        """
        if self.loaded:
            try:
                pygame.mixer.music.unpause()
            except Exception as e:
                print(f"Error unpausing background music: {e}")
    
    def play_crash_sound(self):
        """
        Play the crash sound effect
        """
        if self.loaded:
            try:
                self.crash_sound.play()
            except Exception as e:
                print(f"Error playing crash sound: {e}")
    
    def play_crash_sound_as_background(self):
        """
        Play the crash sound effect as background music for the game over screen
        """
        if self.loaded:
            try:
                # Pause the regular background music
                pygame.mixer.music.pause()
                
                # Play crash sound - possibly loop it for longer effect
                self.crash_sound.play()
            except Exception as e:
                print(f"Error playing crash sound as background: {e}")