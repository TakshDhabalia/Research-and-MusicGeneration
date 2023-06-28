import cv2
import os

def save_frames_from_video(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Save the frame as an image in the output folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

    print(f"Frames saved: {frame_count}")

# Example usage
video_path = "D:\MusicGeneration\Testing\I'm Just Riding On My Bike (ORIGINAL).mp4"
output_folder = "D:\MusicGeneration\Testing\data"

save_frames_from_video(video_path, output_folder)

#now we move on to the next file 

