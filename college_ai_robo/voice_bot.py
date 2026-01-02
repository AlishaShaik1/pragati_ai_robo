import speech_recognition as sr
import pyttsx3
import logging
import time
import sys

# ==========================
# IMPORT LOGIC
# ==========================
try:
    # Trying to import from ex.py as that corresponds to the current workspace file.
    from ex import respond
    print("Successfully imported logic from ex.py")
except Exception as e:
    print(f"Error importing ex.py: {e}")
    print("Make sure ex.py is in the same directory and dependencies are correct.")
    sys.exit(1)

# ==========================
# INITIALIZATION
# ==========================
# Initialize Recognizer
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = 3000

# Initialize Engine Globally (Pi-Safe)
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
    # Optional: Set voice if available (Linux/Pi might not have 'zira')
    voices = engine.getProperty('voices')
    if voices:
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
except Exception as e:
    print(f"Engine Initialization Warning: {e}")

def clean_text_for_speech(text):
    """Remove markdown and special characters for better speech."""
    if not text: return ""
    # Remove bold/italic markers
    text = text.replace("**", "").replace("__", "").replace("*", "")
    # Remove code blocks if any (simple approach)
    text = text.replace("`", "")
    return text

def speak(text):
    """Convert text to speech (Pi-Safe Version)."""
    try:
        clean_text = clean_text_for_speech(text)
        print(f"Robot: {text}") # Print original with formatting

        if engine:
            engine.say(clean_text)
            engine.runAndWait()
        else:
            print("TTS Engine not initialized.")

    except Exception as e:
        print(f"TTS Error: {e}")

def listen_for_command():
    """Listen to the microphone and return recognized text."""
    with sr.Microphone() as source:
        print("\nAdjusting for ambient noise... (Please wait)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        # Increase pause threshold to allow for gaps in speech
        recognizer.pause_threshold = 1.2
        print("Listening... (Say 'jarvis' to wake me up)")
        
        try:
            # Listen with a timeout to prevent hanging forever
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
    speak("Hello, I am ready. Say jarvis to start.")
    
    listening = True
    while listening:
        command = listen_for_command()
        
        if command:
            # Wake Word Detection
            if "jarvis" in command:
                # Remove the wake word to get the actual query
                query = command.replace("jarvis", "").strip()
                
                if not query:
                    speak("Yes? How can I help you?")
                    continue

                if "exit" in query or "quit" in query or "stop" in query:
                    speak("Goodbye!")
                    listening = False
                    break
                
                # Get Response from the AI Brain
                try:
                    # respond expects (message, history)
                    response_text = respond(query, [])
                    
                    # Fail-safe for None or empty response
                    if not response_text:
                        response_text = "I do not have that information."
                    
                    # Speak the response
                    speak(response_text)
                    
                except Exception as e:
                    print(f"Processing Error: {e}")
                    speak("I encountered an error processing your request.")
            
            else:
                print("Ignored (No wake word)")

if __name__ == "__main__":
    main()
