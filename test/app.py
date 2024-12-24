import tkinter as tk
from tkinter import Text
from threading import Thread
import cv2
from PIL import Image, ImageTk
from recognition import recognize_face, get_face_embedding, detector, db_name
from database import load_embeddings_from_db, add_embedding_to_db, process_images
import numpy as np

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Face Recognition System")
        self.root.geometry("1000x800")
        
        # Camera and recognition state
        self.running = False
        self.capture_mode = False
        self.known_faces = load_embeddings_from_db(db_name)

        # Set up UI
        self.create_ui()

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def create_ui(self):
        # Camera frame
        self.camera_frame = tk.Frame(self.root, width=960, height=720)
        self.camera_frame.pack_propagate(0)
        self.camera_frame.pack()
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()

        # Controls frame
        self.controls_frame = tk.Frame(self.root, height=150)
        self.controls_frame.pack()

        # Start button
        self.start_button = tk.Button(self.controls_frame, text="Start", command=self.toggle_start, width=12, height=3)
        self.start_button.pack(side=tk.LEFT, padx=20)

        # Capture button
        self.capture_button = tk.Button(self.controls_frame, text="Capture", command=self.capture, width=12, height=3, state=tk.DISABLED)
        self.capture_button.pack(side=tk.LEFT, padx=20)

        # Add Student button
        self.add_student_button = tk.Button(self.controls_frame, text="Add Student", command=self.add_student, width=12, height=3)
        self.add_student_button.pack(side=tk.LEFT, padx=20)

        # Log area
        self.log_text = Text(self.controls_frame, height=4, width=80)
        self.log_text.pack(side=tk.LEFT, padx=20)

    def toggle_start(self):
        self.running = not self.running
        if self.running:
            self.start_button.config(text="Pause")
            self.capture_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(text="Start")
            self.capture_button.config(state=tk.DISABLED)

    def capture(self):
        if not self.running:
            return

        self.log_text.delete(1.0, tk.END)
        recognized = {}
        unknown_count = 0
        known_count = 0

        ret, frame = self.cap.read()
        if not ret:
            self.log_text.insert(tk.END, "Failed to capture frame.\n")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(frame_rgb)

        for result in results:
            x, y, width, height = result['box']
            face = frame_rgb[y:y + height, x:x + width]
            face_embedding = get_face_embedding(face)

            if face_embedding is not None:
                name, confidence = recognize_face(face_embedding, self.known_faces)
                if name == "Unknown":
                    unknown_count += 1
                else:
                    recognized[name] = confidence

        for name, confidence in recognized.items():
            self.log_text.insert(tk.END, f"{known_count + 1}. {name}: {confidence:.2f}\n")
            known_count += 1

        self.log_text.insert(tk.END, f"Unknown count: {unknown_count}\n")

    def add_student(self):
        # Dialog for adding a new student
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Student")

        tk.Label(dialog, text="Name:").grid(row=0, column=0, padx=10, pady=10)
        name_entry = tk.Entry(dialog, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(dialog, text="Path to Picture:").grid(row=1, column=0, padx=10, pady=10)
        path_entry = tk.Entry(dialog, width=30)
        path_entry.grid(row=1, column=1, padx=10, pady=10)

        def handle_add():
            name = name_entry.get()
            path = path_entry.get()
            try: 
                process_images(name = name, image_folder = path)
                dialog.destroy()
            except Exception as e:
                pass
            

        tk.Button(dialog, text="Add", command=handle_add).grid(row=2, column=0, columnspan=2, pady=10)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (960, 720))
            frame_bgr = frame.copy()

            if self.running:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(frame_rgb)
                for result in results:
                    x, y, width, height = result['box']
                    face = frame_rgb[y:y + height, x:x + width]
                    face_embedding = get_face_embedding(face)

                    if face_embedding is not None:
                        name, confidence = recognize_face(face_embedding, self.known_faces)

                        label = f"{name} ({confidence:.2f})"
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame_bgr, (x, y), (x + width, y + height), color, 2)
                        cv2.putText(frame_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.camera_label.imgtk = img
            self.camera_label.configure(image=img)
        
        self.root.after(10, self.update_frame)

    def close_app(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()
