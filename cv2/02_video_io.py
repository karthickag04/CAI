"""Video I/O Basics (Webcam)
- Open default webcam
- Display frames
- Quit with 'q'

Run: python 02_video_io.py
"""
import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import os

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # OpenCV capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        # Label for video frames
        self.label = Label(window)
        self.label.pack()

        # Buttons
        self.capture_btn = Button(window, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack(side="left", padx=10, pady=10)

        self.quit_btn = Button(window, text="Quit", command=self.quit_app)
        self.quit_btn.pack(side="right", padx=10, pady=10)

        # Update video frames
        self.update_frame()

        # Proper close handling
        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.img_counter = 0

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (Tkinter)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        # Keep looping
        self.window.after(10, self.update_frame)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            os.makedirs("images", exist_ok=True)
            filename = os.path.join("images", f"capture_{self.img_counter}.png")
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
            self.img_counter += 1

    def quit_app(self):
        print("Closing app...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

# Run
root = tk.Tk()
app = WebcamApp(root, "Webcam Capture App")
root.mainloop()
