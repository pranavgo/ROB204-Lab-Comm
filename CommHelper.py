# Written by Skye Krzykwa during Summer 2024 for the ROB204 Communication Lab
# krzyk@umich.edu

import os, sys, math, threading, time, logging
os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
logging.disable(sys.maxsize)
import warnings
warnings.filterwarnings("ignore")

import cv2
from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import ipywidgets as widgets
from IPython import display
from gpiozero import AngularServo as Servo
from gpiozero.pins.pigpio import PiGPIOFactory

# FUNCTION GENERATED WITH CHATGPT
def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# --- USB CAMERA SHIM ---
import cv2
import time

class USBPicamera2:
    def __init__(self):
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def create_video_configuration(self, main=None, **kwargs):
        return {}

    def configure(self, config):
        # Do not force the Pi-camera resolution onto the USB webcam
        pass

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open USB camera /dev/video0")

    def capture_array(self, *args, **kwargs):
        # USB webcams can need a few frames to warm up
        for _ in range(10):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.1)

        raise RuntimeError("Could not read frame from USB camera /dev/video0")

Picamera2 = USBPicamera2
# --- END USB CAMERA SHIM ---

class CommLab(object):
    os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
    pigpio_factory = PiGPIOFactory()
    EmbodimentYawServo = Servo(13, pin_factory=pigpio_factory, min_angle = 90, max_angle = -90, min_pulse_width=0.0005, max_pulse_width=0.0024)
    
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    tracking = None
    tracking_run = False
    
    tracking_status = widgets.HTML(value="Tracking not started")
    
    file = open("images/placeholder.jpg", "rb")
    image = file.read()
    tracking_image  = widgets.Image(value=image,format='jpeg',width=640,height=480)
    
    tracking_output = widgets.HTML(value="🙈")
    tracking_fps = widgets.HTML(value="FPS: 0")
    
    disp_container = widgets.VBox([tracking_status, tracking_output, tracking_fps]) #Miranda: Try deleteing tracking_image from this list.
    output = widgets.Output()
    TARGET_FPS = 10        
    
    def setEmbodimentYaw(self, degrees):
        self.EmbodimentYawServo.angle = (degrees)
    def setText(self, string):
        self.tracking_output.value = string
        
    def process_rotation(self, yaw):
        if yaw > 20:
            self.setText("👈")
            self.setEmbodimentYaw(-45)
        elif yaw < -20:
            self.setText("👉")
            self.setEmbodimentYaw(45)
        else:
            self.setText("🫵")
            self.setEmbodimentYaw(0)
    
    def tracking_function(self, tracking_status, tracking_image, tracking_output, tracking_fps):
        self.setText("Tracking starting...")
        with Picamera2() as picam:
            picam.configure(picam.create_video_configuration(main={"format": 'RGB888', "size": (1296, 730)}))
            picam.start()
            self.setText("Tracking started.")
            while self.tracking_run:
                start_time = time.time()
                image = picam.capture_array("main")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                detection_result = self.detector.detect(mp_image)
    
                resize_image = cv2.flip(image, 1)
                resize_frame = cv2.resize(resize_image, (0, 0), fx = 0.25, fy = 0.25)
                _,ret_array = cv2.imencode('.jpg', cv2.cvtColor(resize_frame, cv2.COLOR_RGB2BGR))
                self.tracking_image.value = ret_array
                
                try:
                    transformation_matrix = detection_result.facial_transformation_matrixes[0]
                    rotation_angles = rotation_matrix_to_euler_angles(transformation_matrix)
                    yaw_angle = np.degrees(rotation_angles[1])
                    self.process_rotation(yaw_angle)
        
                except Exception as e:
                    self.setText("🙈")
                    pass
    
                fps = 1.0 / (time.time() - start_time)
                self.tracking_fps.value = "FPS: " + str(fps)
                sleep_time = (1.0 / self.TARGET_FPS) - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                adj_fps = 1.0 / (time.time() - start_time)
                self.tracking_fps.value = "FPS: " + str(adj_fps)
    
    def start_tracking(self, b):
        with self.output:
            if self.tracking is None or not self.tracking.is_alive():
                self.tracking_run = True
                self.tracking = threading.Thread(target=self.tracking_function, args=(self.tracking_status, self.tracking_image, self.tracking_output, self.tracking_fps, ))
                self.tracking.start()
    
    def stop_tracking(self, b):
        with self.output:
            if self.tracking is not None and self.tracking.is_alive():
                self.tracking_run = False
                self.setText("Tracking stopped.")
    
    def update_func(self, new_func):
        self.process_rotation = new_func
    
    def run(self):
        self.setEmbodimentYaw(0)
        display.clear_output()
        start_btn = widgets.Button(description="Start Tracking", button_style='success')
        start_btn.on_click(self.start_tracking)
        
        stop_btn = widgets.Button(description="Stop Tracking", button_style='danger')
        stop_btn.on_click(self.stop_tracking)
        
        toolbar = widgets.HBox([start_btn, stop_btn])
        disp = widgets.Box(children = [widgets.VBox([toolbar, self.tracking_image, self.disp_container])], layout=widgets.Layout(justify_content='center'))
        display.display(disp, self.output)
