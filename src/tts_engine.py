import pyttsx3
import threading
from typing import Set


class TTSEngine:
    def __init__(self, language: str = "english"):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        self.set_language(language)
        self.spoken_labels: Set[str] = set()

    def set_language(self, language: str):
        voices = self.engine.getProperty("voices")
        for voice in voices:
            if language.lower() in voice.id.lower():
                self.engine.setProperty("voice", voice.id)
                break

    def announce(self, results):
        current_labels = {results.names[int(box.cls[0])] for box in results.boxes}
        new_labels = current_labels - self.spoken_labels

        if new_labels:
            threading.Thread(
                target=self._speak,
                args=(f"Detected: {', '.join(new_labels)}",),
                daemon=True,
            ).start()
            self.spoken_labels.update(new_labels)

    def _speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()
