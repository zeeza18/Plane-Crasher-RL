# Import pygame library for audio functionality
import pygame

# Define the SoundManager class to handle game audio
class SoundManager:
    """
    Manages sounds and music for the game
    """
    def __init__(self):
        # Initialize the pygame mixer system for audio playback
        try:
            # Attempt to initialize the audio mixer
            pygame.mixer.init()
            
            # Define file paths for audio assets
            self.background_music_path = 'assets/sprites/background_music.mp3'  # Path to background music
            self.crash_sound_path = 'assets/sprites/crash_sound.mp3'            # Path to crash sound effect
            
            # Load sound files into memory
            try:
                # Load background music into the music player
                # Using music.load for longer audio that will loop
                pygame.mixer.music.load(self.background_music_path)
                pygame.mixer.music.set_volume(0.3)  # Set to 30% volume (quieter for background)
                
                # Load sound effects as Sound objects
                # Using Sound for shorter effects that may overlap
                self.crash_sound = pygame.mixer.Sound(self.crash_sound_path)
                self.crash_sound.set_volume(0.8)  # Set to 80% volume (louder for impact)
                
                # Set flag indicating successful loading
                self.loaded = True
                print("Sound assets loaded successfully")
            except Exception as e:
                # Handle errors during sound file loading
                print(f"Error loading sound assets: {e}")
                self.loaded = False  # Mark as not loaded for safety
        except Exception as e:
            # Handle errors during mixer initialization
            print(f"Error initializing sound mixer: {e}")
            self.loaded = False  # Mark as not loaded for safety
    
    def play_background_music(self, loops=-1):
        """
        Play background music. Loops indefinitely by default.
        
        Args:
            loops: Number of times to repeat (-1 means infinite looping)
        """
        if self.loaded:
            try:
                # Start playing background music with specified number of loops
                # Default -1 means infinite looping
                pygame.mixer.music.play(loops)
            except Exception as e:
                # Handle errors during playback attempt
                print(f"Error playing background music: {e}")
    
    def stop_background_music(self):
        """
        Stop the background music completely
        """
        if self.loaded:
            try:
                # Stop the music player (cannot be resumed, must be restarted)
                pygame.mixer.music.stop()
            except Exception as e:
                # Handle errors during stop attempt
                print(f"Error stopping background music: {e}")
    
    def pause_background_music(self):
        """
        Pause the background music (can be resumed later)
        """
        if self.loaded:
            try:
                # Pause music playback (can be resumed with unpause)
                pygame.mixer.music.pause()
            except Exception as e:
                # Handle errors during pause attempt
                print(f"Error pausing background music: {e}")
    
    def unpause_background_music(self):
        """
        Unpause the background music (resumes from pause point)
        """
        if self.loaded:
            try:
                # Resume music from the point it was paused
                pygame.mixer.music.unpause()
            except Exception as e:
                # Handle errors during unpause attempt
                print(f"Error unpausing background music: {e}")
    
    def play_crash_sound(self):
        """
        Play the crash sound effect once
        """
        if self.loaded:
            try:
                # Play the crash sound once
                # Sound objects can be played multiple times simultaneously
                self.crash_sound.play()
            except Exception as e:
                # Handle errors during sound playback attempt
                print(f"Error playing crash sound: {e}")
    
    def play_crash_sound_as_background(self):
        """
        Play the crash sound effect as background music for the game over screen
        """
        if self.loaded:
            try:
                # Pause current background music to avoid overlap
                pygame.mixer.music.pause()
                
                # Play crash sound (will play once by default)
                # This could be extended to loop the crash sound if needed
                self.crash_sound.play()
            except Exception as e:
                # Handle errors during sound switching attempt
                print(f"Error playing crash sound as background: {e}")