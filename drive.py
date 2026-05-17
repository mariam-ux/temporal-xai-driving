import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

from utils import preprocess

from collections import deque

frame_buffer = deque(maxlen=5)

sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 5
controller.set_desired(set_speed)


prev_steering = 0.0  # global smoothing memory

@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering

    print("🔥 TELEMETRY RECEIVED")

    if data:

        # -------------------------
        # SPEED
        # -------------------------
        speed = float(data["speed"])

        # -------------------------
        # IMAGE PROCESSING
        # -------------------------
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        image_array = preprocess(image_array)

        print("Shape after preprocess:", image_array.shape)
       
        # -------------------------
        # TEMPORAL BUFFER
        # -------------------------
        frame_buffer.append(image_array)

        if len(frame_buffer) < 5:
            send_control(0.0, 0.0)
            return

        model_input = np.array(frame_buffer)
        model_input = np.expand_dims(model_input, axis=0)  # (1, 5, 66, 200, 3)

        # -------------------------
        # PREDICTION
        # -------------------------
        print("BUFFER SIZE:", len(frame_buffer))
        pred = model.predict(model_input, verbose=0)[0][0]

        # 🔥 scale steering (VERY IMPORTANT)
        steering_angle = float(pred) * 0.8
        steering_angle = max(-1.0, min(1.0, steering_angle))
        # -------------------------
        # SMOOTHING (reduces zig-zag crashes)
        # -------------------------
        alpha = 0.5
        steering_angle = alpha * prev_steering + (1 - alpha) * steering_angle
        prev_steering = steering_angle

        # -------------------------
        # THROTTLE CONTROL (SAFE)
        # -------------------------
        # reduce speed during turns
        if abs(steering_angle) > 0.25:
            target_speed = 4
        else:
            target_speed = 6

        controller.set_desired(target_speed)

        if speed < target_speed:
            throttle = 0.3
        else:
            throttle = controller.update(speed)

        # safer clamp
        throttle = max(0.05, min(0.3, throttle))
        print("MIN:", np.min(image_array), "MAX:", np.max(image_array))
        print("PRED:", pred)
        print("STEERING:", steering_angle)
        print("SPEED:", speed)
        print("THROTTLE:", throttle)

        # -------------------------
        # SEND CONTROL
        # -------------------------
        send_control(steering_angle, throttle)

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("CLIENT CONNECTED:", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    model = load_model(args.model, compile=False)
    print("Model loaded successfully")
    print(model.input_shape)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.WSGIApp(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
