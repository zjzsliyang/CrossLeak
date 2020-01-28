#! /usr/bin/python3
import io
import time
import picamera
import picamera.array
import numpy as np
import yaml
import os

with open("setup.yaml", "r") as f:
    setup = yaml.load(f, Loader=yaml.FullLoader)

FOLDER = setup["raw_video_folder"]
os.makedirs(FOLDER, exist_ok=True)

FILE_PATTERN = FOLDER + 'motion%02d.h264'
FILE_BUFFER = setup["V_FILE_BUFFER"] #1048576

REC_SECONDS = setup["V_REC_SECONDS"] #10
REC_BITRATE = setup["V_REC_BITRATE"] #1000000

MOTION_MAGNITUDE = setup["V_MOTION_MAGNITUDE"] #70
MOTION_VECTORS = setup["V_MOTION_VECTORS"] #10


class MotionDetector(picamera.array.PiMotionAnalysis):
    def __init__(self, camera, size=None):
        super(MotionDetector, self).__init__(camera, size)
        self.vector_count = 0
        self.detected = 0

    def analyse(self, a):
        a = np.sqrt(np.square(a['x'].astype(np.float)) + np.square(a['y'].astype(np.float))).clip(0, 255).astype(np.uint8)
        vector_count = (a > MOTION_MAGNITUDE).sum()
        if vector_count > MOTION_VECTORS:
            self.detected = time.time()

def main():
    with picamera.PiCamera() as camera:
        camera.resolution = (setup["V_RESOLUTION"][0],setup["V_RESOLUTION"][1])
        #(1640,1232)
        camera.framerate = setup["V_FRAMERATE"] #30
        camera.rotation = setup["V_ROTATION"] #180 as the camera is upside down
        time.sleep(2)


        ring_buffer = picamera.PiCameraCircularIO(camera, seconds=REC_SECONDS, bitrate=REC_BITRATE)
        file_number = 1
        file_output = io.open(FILE_PATTERN % file_number, 'wb', buffering=FILE_BUFFER)
        motion_detector = MotionDetector(camera)
        camera.start_recording(ring_buffer, format='h264', bitrate=REC_BITRATE,intra_period=30, motion_output=motion_detector)
        try:
            while True:
                print("Waiting for motion")
                while motion_detector.detected < time.time() - 1:
                    camera.wait_recording(1)
                print('Motion detected')
                print('Recording to %s' % file_output.name)
                with ring_buffer.lock:
                    for frame in ring_buffer.frames:
                        if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                            ring_buffer.seek(frame.position)
                            break
                    while True:
                        buf = ring_buffer.read1()
                        if not buf:
                            break
                        file_output.write(buf)
                camera.split_recording(file_output)
                ring_buffer = picamera.PiCameraCircularIO(camera, seconds=REC_SECONDS, bitrate=REC_BITRATE)
                while motion_detector.detected > time.time() - REC_SECONDS:
                    camera.wait_recording(1)
                camera.split_recording(ring_buffer)
                file_number += 1
                file_output.close()
                file_output = io.open(FILE_PATTERN % file_number, 'wb', buffering=FILE_BUFFER)
        finally:
            camera.stop_recording()

if __name__ == '__main__':
    main()




