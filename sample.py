import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Initialize camera indices
cam_indices = [0, 2]

# Open cameras
cams = [cv2.VideoCapture(i) for i in cam_indices]

# Check if cameras are opened
if not all(cam.isOpened() for cam in cams):
    print("Error: Could not open one or more cameras.")
    for cam in cams:
        cam.release()
    exit()

# Create a counter for image filenames
counter = 0

def capture_images():
    global counter
    # Capture images from both cameras
    ret1, frame1 = cams[0].read()
    ret2, frame2 = cams[1].read()

    if ret1 and ret2:
        # Save images
        cv2.imwrite(f'left/{counter}.jpg', frame1)
        cv2.imwrite(f'right/{counter}.jpg', frame2)
        # Update the text area with the save message
        text_area.insert(tk.END, f"Images saved as left/{counter}.jpg and right/{counter}.jpg\n")
        counter += 1
    else:
        text_area.insert(tk.END, "Failed to capture images from both cameras.\n")

def update_frames():
    # Read frames from both cameras
    ret1, frame1 = cams[0].read()
    ret2, frame2 = cams[1].read()

    if ret1 and ret2:
        # Convert images to PhotoImage format
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (320, 240))
        img2 = cv2.resize(img2, (320, 240))
        img1 = ImageTk.PhotoImage(image=Image.fromarray(img1))
        img2 = ImageTk.PhotoImage(image=Image.fromarray(img2))

        # Update labels
        label1.config(image=img1)
        label1.image = img1
        label2.config(image=img2)
        label2.image = img2

    # Schedule the next frame update
    root.after(10, update_frames)

# Create main window
root = tk.Tk()
root.title("Camera Viewer")

# Create labels to display camera feeds
label1 = tk.Label(root)
label1.pack(side="left")
label2 = tk.Label(root)
label2.pack(side="right")

# Create capture button
capture_button = tk.Button(root, text="Capture", command=capture_images)
capture_button.pack()

# Create a text area to display messages
text_area = tk.Text(root, height=5, width=50)
text_area.pack()

# Start updating frames
update_frames()

# Start the GUI loop
root.mainloop()

# Release cameras when the GUI is closed
for cam in cams:
    cam.release()
cv2.destroyAllWindows()

