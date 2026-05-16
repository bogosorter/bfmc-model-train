import cv2
import os

video_folder = "videos"  # folder containing your videos
output_folder = "frames"
target_size = (512, 270)  # width x height

os.makedirs(output_folder, exist_ok=True)

for video_file in os.listdir(video_folder):
    if not video_file.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
        continue

    video_path = os.path.join(video_folder, video_file)
    video_name = os.path.splitext(video_file)[0]

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, target_size)
        frame_filename = os.path.join(
            output_folder, f"{video_name}_frame_{frame_count:05d}.jpg"
        )
        cv2.imwrite(frame_filename, frame_resized)
        frame_count += 1

    cap.release()
    print(f"Extracted and resized {frame_count} frames from '{video_file}'")
