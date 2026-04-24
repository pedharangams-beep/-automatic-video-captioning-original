from models.audio_model import transcribe, detect_sound

audio_path = "data/audio/audio.wav"

print("\n[INFO] Processing audio...\n")

speech = transcribe(audio_path)
sound = detect_sound(audio_path)

print("\n[Speech]")
print(speech)

print("\n[Sound Context]")
print(sound)