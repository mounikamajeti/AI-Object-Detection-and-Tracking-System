import os
import threading
import time
import cv2
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from detector.yolo_detector import ObjectDetector
from tracker.deep_sort_tracker import ObjectTracker
from database.database import DatabaseManager
from utils.video_utils import VideoUtils
from utils.counter import ObjectCounter
from utils.logger import AppLogger

class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title('AI Object Detection and Tracking System')
        self.root.geometry('1280x780')
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('dark-blue')

        self.video_source = None
        self.capture = None
        self.running = False
        self.recording = False
        self.should_stop = False
        self.frame = None
        self.video_writer = None
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = 0.0
        self.db_manager = DatabaseManager('output/detections.db')
        self.detector = ObjectDetector('models/yolov8n.pt')
        self.tracker = ObjectTracker()
        self.counter = ObjectCounter()
        self.available_classes = self.detector.class_names

        self.build_ui()

    def build_ui(self) -> None:
        main_frame = ctk.CTkFrame(self.root, fg_color='#1f222d')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        header = ctk.CTkLabel(main_frame, text='AI Object Detection and Tracking System', font=ctk.CTkFont(size=20, weight='bold'))
        header.pack(pady=(10, 5))

        button_frame = ctk.CTkFrame(main_frame, fg_color='#272b38')
        button_frame.pack(fill='x', padx=10, pady=10)

        self.open_camera_btn = ctk.CTkButton(button_frame, text='Open Webcam', command=self.open_webcam)
        self.open_camera_btn.pack(side='left', padx=8, pady=8)

        self.upload_video_btn = ctk.CTkButton(button_frame, text='Upload Video', command=self.upload_video)
        self.upload_video_btn.pack(side='left', padx=8, pady=8)

        self.capture_btn = ctk.CTkButton(button_frame, text='Capture Frame', command=self.capture_frame, state='disabled')
        self.capture_btn.pack(side='left', padx=8, pady=8)

        self.record_btn = ctk.CTkButton(button_frame, text='Start Recording', command=self.toggle_recording, state='disabled')
        self.record_btn.pack(side='left', padx=8, pady=8)

        self.stop_btn = ctk.CTkButton(button_frame, text='Stop', command=self.stop_video, state='disabled')
        self.stop_btn.pack(side='left', padx=8, pady=8)

        display_frame = ctk.CTkFrame(main_frame, fg_color='#2a2f3d')
        display_frame.pack(fill='both', expand=True, padx=10, pady=8)

        self.video_panel = ctk.CTkLabel(display_frame, text='Video Display Area', width=840, height=480, fg_color='#171923')
        self.video_panel.pack(side='left', padx=10, pady=10)

        stats_frame = ctk.CTkFrame(display_frame, fg_color='#222534', width=380)
        stats_frame.pack(side='left', fill='y', padx=10, pady=10)

        self.stats_label = ctk.CTkLabel(stats_frame, text='Statistics', font=ctk.CTkFont(size=16, weight='bold'))
        self.stats_label.pack(padx=10, pady=(10, 5))

        self.objects_detected_label = ctk.CTkLabel(stats_frame, text='Objects Entered: 0', anchor='w')
        self.objects_detected_label.pack(fill='x', padx=10, pady=4)

        self.active_tracks_label = ctk.CTkLabel(stats_frame, text='Active Tracks: 0', anchor='w')
        self.active_tracks_label.pack(fill='x', padx=10, pady=4)

        self.left_objects_label = ctk.CTkLabel(stats_frame, text='Objects Left: 0', anchor='w')
        self.left_objects_label.pack(fill='x', padx=10, pady=4)

        self.fps_label = ctk.CTkLabel(stats_frame, text='FPS: 0.00', anchor='w')
        self.fps_label.pack(fill='x', padx=10, pady=4)

        self.processing_label = ctk.CTkLabel(stats_frame, text='Processing Time: 0.00 sec', anchor='w')
        self.processing_label.pack(fill='x', padx=10, pady=4)

        self.search_label = ctk.CTkLabel(stats_frame, text='Search Detection History', font=ctk.CTkFont(size=14, weight='bold'))
        self.search_label.pack(padx=10, pady=(20, 4))

        self.search_name = ctk.CTkEntry(stats_frame, placeholder_text='Object Name')
        self.search_name.pack(fill='x', padx=10, pady=4)

        self.search_id = ctk.CTkEntry(stats_frame, placeholder_text='Tracking ID')
        self.search_id.pack(fill='x', padx=10, pady=4)

        self.search_date = ctk.CTkEntry(stats_frame, placeholder_text='Date YYYY-MM-DD')
        self.search_date.pack(fill='x', padx=10, pady=4)

        self.search_btn = ctk.CTkButton(stats_frame, text='Search', command=self.search_history)
        self.search_btn.pack(fill='x', padx=10, pady=8)

        self.history_table = ttk.Treeview(main_frame, columns=('ID', 'Object', 'Track ID', 'Confidence', 'Timestamp', 'Frame'), show='headings', height=7)
        for col in ('ID', 'Object', 'Track ID', 'Confidence', 'Timestamp', 'Frame'):
            self.history_table.heading(col, text=col)
            self.history_table.column(col, anchor='center')
        self.history_table.pack(fill='x', padx=10, pady=(0, 10))

        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.update_history_table()

    def open_webcam(self) -> None:
        try:
            self.video_source = 0
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError('Webcam is not available.')
            self.start_video_loop()
        except Exception as exc:
            AppLogger.error(str(exc))
            messagebox.showerror('Camera Error', str(exc))

    def upload_video(self) -> None:
        try:
            file_path = filedialog.askopenfilename(filetypes=[('Video Files', '*.mp4 *.avi *.mov')])
            if not file_path:
                return
            self.video_source = file_path
            self.capture = cv2.VideoCapture(file_path)
            if not self.capture.isOpened():
                raise RuntimeError('Cannot open selected video file.')
            self.start_video_loop()
        except Exception as exc:
            AppLogger.error(str(exc))
            messagebox.showerror('Video Error', str(exc))

    def start_video_loop(self) -> None:
        self.running = True
        self.should_stop = False
        self.capture_btn.configure(state='normal')
        self.record_btn.configure(state='normal')
        self.stop_btn.configure(state='normal')
        self.open_camera_btn.configure(state='disabled')
        self.upload_video_btn.configure(state='disabled')
        self.start_time = time.time()
        self.frame_count = 0
        threading.Thread(target=self.process_frames, daemon=True).start()

    def process_frames(self) -> None:
        try:
            while self.running and self.capture is not None and self.capture.isOpened() and not self.should_stop:
                started = time.time()
                ret, frame = self.capture.read()
                if not ret:
                    break
                frame = VideoUtils.resize_frame(frame)
                self.frame_count += 1
                detections = self.detector.detect_objects(frame)
                tracks = self.tracker.update_tracks(detections, frame)
                self.counter.update([track['track_id'] for track in tracks])

                for track in tracks:
                    VideoUtils.draw_detection(
                        frame,
                        track['bbox'],
                        track['class_name'],
                        track['confidence'],
                        track['track_id'],
                    )
                    self.db_manager.save_detection(
                        object_name=track['class_name'],
                        tracking_id=track['track_id'],
                        confidence=track['confidence'],
                        frame_number=self.frame_count,
                    )

                if self.recording and self.video_writer is not None:
                    self.video_writer.write(frame)

                processing_time = time.time() - started
                self.fps = 1.0 / processing_time if processing_time > 0 else 0.0
                self.frame = frame.copy()
                self.root.after(1, lambda f=frame.copy(), pt=processing_time: self.update_and_display(f, pt))

            self.stop_video()
        except Exception as exc:
            AppLogger.error(str(exc))
            messagebox.showerror('Processing Error', str(exc))
            self.stop_video()

    def display_frame(self, frame) -> None:
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (840, 480))
            image = Image.fromarray(image)
            photo = ctk.CTkImage(light_image=image, size=(840, 480))
            self.video_panel.configure(image=photo)
            self.video_panel.image = photo
        except Exception as exc:
            AppLogger.warning(f'Failed to display frame: {exc}')

    def update_and_display(self, frame, processing_time: float) -> None:
        self.display_frame(frame)
        self.update_statistics(processing_time)

    def update_statistics(self, processing_time: float) -> None:
        self.objects_detected_label.configure(text=f'Objects Entered: {self.counter.get_total()}')
        self.active_tracks_label.configure(text=f'Active Tracks: {self.counter.get_current()}')
        self.left_objects_label.configure(text=f'Objects Left: {self.counter.get_left()}')
        self.fps_label.configure(text=f'FPS: {self.fps:.2f}')
        self.processing_label.configure(text=f'Processing Time: {processing_time:.2f} sec')

    def capture_frame(self) -> None:
        try:
            if self.frame is None:
                return
            screenshot_path = Path('output/screenshots') / f'screenshot_{int(time.time())}.jpg'
            cv2.imwrite(str(screenshot_path), self.frame)
            messagebox.showinfo('Capture Saved', f'Screenshot saved to {screenshot_path}')
        except Exception as exc:
            AppLogger.error(str(exc))
            messagebox.showerror('Capture Error', str(exc))

    def toggle_recording(self) -> None:
        if not self.recording:
            if self.capture is None or not self.capture.isOpened():
                return
            output_path = Path('output/recordings') / f'recording_{int(time.time())}.mp4'
            fps = self.capture.get(cv2.CAP_PROP_FPS) or 20.0
            self.video_writer = VideoUtils.get_video_writer(str(output_path), fps, (1024, 576))
            self.recording = True
            self.record_btn.configure(text='Stop Recording')
        else:
            self.recording = False
            self.record_btn.configure(text='Start Recording')
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                messagebox.showinfo('Recording Saved', 'Video recording has been saved.')

    def stop_video(self) -> None:
        self.should_stop = True
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.record_btn.configure(text='Start Recording')
        self.open_camera_btn.configure(state='normal')
        self.upload_video_btn.configure(state='normal')
        self.capture_btn.configure(state='disabled')
        self.record_btn.configure(state='disabled')
        self.stop_btn.configure(state='disabled')

    def search_history(self) -> None:
        try:
            object_name = self.search_name.get().strip()
            tracking_id = self.search_id.get().strip()
            date = self.search_date.get().strip()
            if tracking_id and not tracking_id.isdigit():
                raise ValueError('Tracking ID must be a number.')
            records = self.db_manager.get_history(object_name, tracking_id, date)
            self.history_table.delete(*self.history_table.get_children())
            for record in records:
                self.history_table.insert('', 'end', values=record)
        except Exception as exc:
            AppLogger.error(str(exc))
            messagebox.showerror('Search Error', str(exc))

    def update_history_table(self) -> None:
        try:
            records = self.db_manager.get_history()
            self.history_table.delete(*self.history_table.get_children())
            for record in records[-50:]:
                self.history_table.insert('', 'end', values=record)
        except Exception as exc:
            AppLogger.error(str(exc))

    def on_close(self) -> None:
        self.stop_video()
        self.db_manager.close()
        self.root.destroy()
