import cv2 as cv
import tkinter as tk
from tkinter import Tk, Canvas, Button
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import mediapipe as mp
import pytesseract

# Constants for mode
DRAW_MODE = "draw"
RECOGNIZE_MODE = "recognize"

# Set up webcam
video_source = 0
vid = cv.VideoCapture(video_source)
vid.set(3, 640)  # default width
vid.set(4, 480)  # default height

# set pathway to own local tesseract
personal_tesseract_path = '/usr/local/Cellar/tesseract/5.3.3/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = personal_tesseract_path

# Global variable for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Global variables for drawing
drawing = False
lines = [] 
lastX, lastY, index_finger_x, index_finger_y = 0, 0, 0, 0

# Mode variable
current_mode = DRAW_MODE  # Initial mode is drawing

# Function to update the canvas with the current webcam frame and hand tracking
def update_webcam_canvas():
    global lastX, lastY, index_finger_x, index_finger_y
    ret, frame = vid.read()
    if ret:
        # Horizontally flip the frame conditionally
        if current_mode == DRAW_MODE:
            frame = cv.flip(frame, 1)

        # Convert the OpenCV BGR frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame based on the current mode
        if current_mode == DRAW_MODE:
            frame_with_hand_tracking = process_hand_tracking(frame_rgb)
        else:
            frame_with_hand_tracking = frame_rgb

        # Convert the frame to a PhotoImage object
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_with_hand_tracking))

        # Clear previous content on the canvas
        webcam_canvas.delete("all")

        # Update the canvas with the new frame at the top
        webcam_canvas.create_image(0, 0, anchor='nw', image=photo)

        # Redraw all stored lines on the canvas if in drawing mode
        if current_mode == DRAW_MODE:
            for line in lines:
                webcam_canvas.create_line(line, width=20, fill='black')

        # Save a reference to avoid garbage collection
        webcam_canvas.photo = photo

        if drawing and current_mode == DRAW_MODE:
            # Draw a line on the canvas and store the coordinates
            line = (lastX, lastY, index_finger_x, index_finger_y)
            lines.append(line)
            lastX, lastY = index_finger_x, index_finger_y
            webcam_canvas.create_line(line, width=20, fill='black')

        # Schedule the next update after a short time to avoid high CPU usage
        webcam_root.after(33, update_webcam_canvas)


# Function to process the frame for hand tracking
def process_hand_tracking(frame):
    global drawing, lastX, lastY, index_finger_x, index_finger_y
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Your hand tracking logic here
    # For example, draw a circle on the index finger
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:  # Assuming index finger landmark is at index 8
                    cv.circle(frame, (cx, cy), 10, (255, 0, 0), cv.FILLED)
                    index_finger_x, index_finger_y = cx, cy
        # mpHands.HAND_CONNECTIONS for mesh
        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    return frame


# Function to handle mouse click event (drawing toggle)
def draw(event):
    global drawing, lastX, lastY, index_finger_x, index_finger_y
    drawing = not drawing 
    lastX, lastY = index_finger_x, index_finger_y


# Function to clear the canvas
def clear_canvas():
    global lines
    lines = []
    webcam_canvas.delete("all")


# Function to perform text recognition using Tesseract
def recognize_text():
    # Capture the content of just the webcam
    x, y, w, h = 0, 60, 640, 480 
    canvas_image = ImageGrab.grab(bbox=(x, y, x + w, y + h))

    # Convert the Image to an OpenCV BGR image
    canvas_image_cv = cv.cvtColor(np.array(canvas_image), cv.COLOR_RGB2BGR)

    # Perform additional processing on the image for better recognition
    gray = cv.cvtColor(canvas_image_cv, cv.COLOR_BGR2GRAY)
    enhanced = cv.equalizeHist(gray)

    # Save the captured image as a JPEG file
    image_filename = "captured_image.jpg"
    cv.imwrite(image_filename, enhanced)

    # Perform text recognition using Tesseract
    text = pytesseract.image_to_string(enhanced)
    
    print("Recognized Text:", text)
    print("Image saved as:", image_filename)


# Function to switch modes
def switch_mode(new_mode):
    global current_mode
    current_mode = new_mode
    clear_canvas()
    # Enable or disable buttons based on the mode
    if current_mode == DRAW_MODE:
        recognize_button.config(state=tk.DISABLED)
    else:
        recognize_button.config(state=tk.NORMAL)


# Create tkinter window for the webcam feed
webcam_root = Tk()
webcam_root.title("Webcam Feed")
# Set the initial size of the window
webcam_root.geometry("640x960")  # Twice the height of the video
webcam_canvas = Canvas(webcam_root, width=640, height=480)
webcam_canvas.pack()

# Create mode buttons
draw_button_mode = Button(webcam_root, text="Drawing Mode", command=lambda: switch_mode(DRAW_MODE))
draw_button_mode.pack(side=tk.LEFT)
recognize_button_mode = Button(webcam_root, text="Text Recognition Mode", command=lambda: switch_mode(RECOGNIZE_MODE))
recognize_button_mode.pack(side=tk.RIGHT)

# Create a "Clear Canvas" button
clear_button = Button(webcam_root, text="Clear Canvas", command=clear_canvas)
clear_button.pack()

# Create a "Recognize Text" button
recognize_button = Button(webcam_root, text="Recognize Text", command=recognize_text)
recognize_button.config(state=tk.DISABLED)
recognize_button.pack()

# Bind mouse click event to toggle_drawing function
webcam_canvas.bind("<ButtonPress-1>", draw)

# Start updating the webcam canvas
update_webcam_canvas()

# Run the Tkinter event loop for the webcam window
webcam_root.mainloop()

# Release the webcam when the Tkinter window is closed
vid.release()
