import cv2 as cv
from tkinter import Tk, Canvas
from PIL import Image, ImageTk
import mediapipe as mp

# Set up webcam
video_source = 0
vid = cv.VideoCapture(video_source)
vid.set(3, 640)  # default width
vid.set(4, 480)  # default height

# Global variable for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# Function to update the canvas with the current webcam frame and hand tracking
def update_webcam_canvas():
    ret, frame = vid.read()
    if ret:
        # Horizontally flip the frame
        frame = cv.flip(frame, 1)
        
        # Convert the OpenCV BGR frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        frame_with_hand_tracking = process_hand_tracking(frame_rgb)

        # Convert the frame to a PhotoImage object
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_with_hand_tracking))

        # Clear previous content on the canvas
        webcam_canvas.delete("all")

        # Update the canvas with the new frame at the top
        webcam_canvas.create_image(0, 0, anchor='nw', image=photo)

        # Save a reference to avoid garbage collection
        webcam_canvas.photo = photo

        # Schedule the next update after a short time to avoid high CPU usage
        webcam_root.after(33, update_webcam_canvas)

# Function to process the frame for hand tracking
def process_hand_tracking(frame):
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Your hand tracking logic here
    # For example, draw a circle on the index finger
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 8:  # Assuming index finger landmark is at index 8
                    cv.circle(frame, (cx, cy), 20, (255, 0, 0), cv.FILLED)

        # mpHands.HAND_CONNECTIONS for mesh
        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    

    return frame

# Create tkinter window for the webcam feed
webcam_root = Tk()
webcam_root.title("Webcam Feed")
# Set the initial size of the window
webcam_root.geometry("640x960")  # Twice the height of the video
webcam_canvas = Canvas(webcam_root, width=640, height=480)
webcam_canvas.pack()

# Start updating the webcam canvas
update_webcam_canvas()

# Run the Tkinter event loop for the webcam window
webcam_root.mainloop()

# Release the webcam when the Tkinter window is closed
vid.release()