import os
from utils.video_utils import extract_frames
from utils.audio_utils import extract_audio
from models.image_caption import generate_caption
from models.audio_model import transcribe, detect_sound

# Paths
video_path = "input.mp4"
frames_folder = "data/frames"
audio_path = "data/audio/audio.wav"

# ================================
# Helper Functions
# ================================

def remove_similar_captions(captions):
    cleaned = []
    
    for cap in captions:
        if not cleaned or cap != cleaned[-1]:
            cleaned.append(cap)
    
    return cleaned


def format_time(seconds):
    return f"00:00:{seconds:02},000"


def save_as_srt(frame_captions, speech_text, output_file="outputs/output.srt"):
    with open(output_file, "w") as f:
        
        # Frame-based captions
        for i, cap in enumerate(frame_captions):
            start = format_time(i)
            end = format_time(i + 1)

            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(cap + "\n\n")

        # Add full speech at end
        f.write(f"{len(frame_captions)+1}\n")
        f.write(f"{format_time(len(frame_captions))} --> {format_time(len(frame_captions)+5)}\n")
        f.write("Full Speech: " + speech_text + "\n\n")


# ================================
# Step 1: Extract frames
# ================================
extract_frames(video_path, frames_folder, fps=1)

# ================================
# Step 2: Extract audio
# ================================
extract_audio(video_path, audio_path)

# ================================
# Step 3: Generate frame captions
# ================================

frame_files = sorted(
    os.listdir(frames_folder),
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

frame_captions = []

print("\n[INFO] Generating visual captions...\n")

for file in frame_files:
    path = os.path.join(frames_folder, file)
    caption = generate_caption(path)
    
    frame_captions.append(caption)
    print(f"[Time {len(frame_captions)-1}s] {caption}")

# Remove duplicates
frame_captions = remove_similar_captions(frame_captions)

# ================================
# Step 4: Audio processing
# ================================
print("\n[INFO] Processing audio...\n")

speech_text = transcribe(audio_path)
sound_label = detect_sound(audio_path)

print("\n[Sound Environment]")
print(sound_label)

# ================================
# Step 5: Final Output
# ================================

print("\n[INFO] Final Output\n")

print("\n--- Visual Timeline ---\n")
for i, cap in enumerate(frame_captions):
    print(f"[Time {i}s] {cap}")

print("\n--- Full Speech ---\n")
print(speech_text)

# ================================
# Save Outputs
# ================================

# Save TXT
with open("outputs/captions.txt", "w") as f:
    f.write("=== Visual Captions ===\n\n")
    for i, cap in enumerate(frame_captions):
        f.write(f"[Time {i}s] {cap}\n")
    
    f.write("\n=== Full Speech ===\n\n")
    f.write(speech_text)

# Save SRT
save_as_srt(frame_captions, speech_text)

print("\n[INFO] Captions saved to outputs/")