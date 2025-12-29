import speech_recognition as sr
import pyttsx3
import logging
import time

# Import the response logic from your existing app
try:
    from app import respond
    print("Successfully imported logic from app.py")
except ImportError as e:
    print(f"Error importing app.py: {e}")
    print("Make sure app.py is in the same directory.")
    exit(1)

# ==========================
# INITIALIZATION
# ==========================
# Initialize Recognizer
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = 3000

# Removed global engine init to prevent state issues

def clean_text_for_speech(text):
    """Remove markdown and special characters for better speech."""
    # Remove bold/italic markers
    text = text.replace("**", "").replace("__", "").replace("*", "")
    # Remove code blocks if any (simple approach)
    text = text.replace("`", "")
    return text

def speak(text):
    """Convert text to speech."""
    try:
        clean_text = clean_text_for_speech(text)
        print(f"Robot: {text}") # Print original with formatting
        
        # Re-initialize engine each time to avoid 'runAndWait' hangs in loops
        engine = pyttsx3.init()
        
        # Configure Voice
        voices = engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()
        del engine # Clean up
    except Exception as e:
        print(f"TTS Error: {e}")

def listen_for_command():
    """Listen to the microphone and return recognized text."""
    with sr.Microphone() as source:
        print("\nAdjusting for ambient noise... (Please wait)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        # Increase pause threshold to allow for gaps in speech (default is 0.8)
        recognizer.pause_threshold = 1.2
        print("Listening... (Say 'Bujji' to wake me up)")
        
        try:
            # Listen with a timeout to prevent hanging forever
            # Increased phrase_time_limit to capture long prediction queries
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
            
            print("Recognizing...")
            command = recognizer.recognize_google(audio)
            print(f"User said: {command}")
            return command.lower()
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            speak("I am having trouble connecting to the internet.")
            return None
        except Exception as e:
            print(f"Microphone Error: {e}")
            return None

def main():
    speak("Hello, I am ready. Say Bujji to start.")
    
    listening = True
    while listening:
        command = listen_for_command()
        
        if command:
            # Wake Word Detection
            if "bujji" in command:
                # Remove the wake word to get the actual query
                # Example: "Bujji who is the principal" -> "who is the principal"
                query = command.replace("bujji", "").strip()
                
                if not query:
                    speak("Yes? How can I help you?")
                    # Optional: Listen again immediately for the actual query
                    continue

                if "exit" in query or "quit" in query or "stop" in query:
                    speak("Goodbye!")
                    listening = False
                    break
                
                # Get Response from the AI Brain (app.py)
                try:
                    # app.respond expects (message, history)
                    # We pass empty history [] as this loop manages state simply for now
                    response_text = respond(query, [])
                    
                    # Speak the response
                    speak(response_text)
                    
                except Exception as e:
                    print(f"Processing Error: {e}")
                    speak("I encountered an error processing your request.")
            
            else:
                print("Ignored (No wake word)")

if __name__ == "__main__":
    main()
