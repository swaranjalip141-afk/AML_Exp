import cv2
import os

# Dataset path
dataset_path = "dataset"

# Output frames folder
output_path = "frames"

# Create output folder
os.makedirs(output_path, exist_ok=True)

# Classes
classes = os.listdir(dataset_path)

for cls in classes:

    class_path = os.path.join(dataset_path, cls)

    # Skip txt file
    if not os.path.isdir(class_path):
        continue

    save_path = os.path.join(output_path, cls)

    os.makedirs(save_path, exist_ok=True)

    videos = os.listdir(class_path)

    print(f"Processing class: {cls}")

    for video in videos:

        video_path = os.path.join(class_path, video)

        cap = cv2.VideoCapture(video_path)

        count = 0
        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Save every 10th frame
            if count % 10 == 0:

                frame = cv2.resize(frame, (128, 128))

                frame_name = f"{video}_{frame_count}.jpg"

                cv2.imwrite(
                    os.path.join(save_path, frame_name),
                    frame
                )

                frame_count += 1

            count += 1

        cap.release()

print("Frame extraction completed!")