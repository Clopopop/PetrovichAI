import os
import moviepy
from openai import OpenAI

class Transcriber:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, file_path):
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )
        return transcription

    def transcribe_video(self, video_path, temp_audio_path="temp_audio.ogg"):
        """Extract audio from a video, then transcribe it via Whisper."""
        try:
            # Load the video with MoviePy
            clip = moviepy.VideoFileClip(video_path)

            # Write audio to OGG (compressed)
            # (Vorbis is a typical codec in an OGG container)
            clip.audio.write_audiofile(
                temp_audio_path,
                codec="libvorbis",
                logger = None
            )

            # Transcribe the extracted OGG audio
            transcription_text = self.transcribe(temp_audio_path)

        finally:
            # Close the clip and clean up
            if 'clip' in locals():
                clip.close()
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        return transcription_text