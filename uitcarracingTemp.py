from client_lib import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import math

CHECKPOINT = 160

# PID Controller class for smoothing the steering angle
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    
    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# Preprocess the image with Gaussian blur to reduce noise
def preprocess_image(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur
    return blurred_image

# Calculate the steering angle based on the image data
def AngCal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray*(255/np.max(gray))).astype(np.uint8)

    h, w = gray.shape

    # Average multiple rows around the checkpoint for better stability
    rows_to_average = 5
    line_rows = gray[CHECKPOINT:CHECKPOINT+rows_to_average, :]
    avg_row = np.mean(line_rows, axis=0)

    min_x, max_x = 0, 0
    flag = True

    for x, y in enumerate(avg_row):
        if y == 255 and flag:
            flag = False
            min_x = x
        elif y == 255:
            max_x = x

    center_row = int((max_x + min_x) / 2)
    
    x0, y0 = int(w / 2), h
    x1, y1 = center_row, CHECKPOINT

    # Calculate the angle using arctangent
    value = (x1 - x0) / (y0 - y1)
    angle = math.degrees(math.atan(value))

    # Constrain the angle within [-25, 25] degrees
    if angle > 25:
        angle = 25
    elif angle < -25:
        angle = -25

    # Return the error (difference from center) for the PID controller
    error = (center_row - int(w / 2))
    return error

if __name__ == "__main__":
    pid_steer = PID(kp=1.0, ki=0.01, kd=0.05)  # Initialize PID controller
    
    try:
        while True:
            state = GetStatus()
            segment_image = GetSeg()

            # Preprocess the image
            segment_image = preprocess_image(segment_image)

            # Show the preprocessed segmentation image
            cv2.imshow('segment_image', segment_image)

            # Calculate the error (difference from the center) using AngCal
            error = AngCal(segment_image)

            # Get the PID controller output
            pid_output = pid_steer.compute(error)

            # Clamp the PID output within [-25, 25] degrees
            angle = max(min(pid_output, 25), -25)

            # Dynamically adjust speed based on the steering angle
            if abs(angle) > 15:
                speed = 15
            else:
                speed = 20

            # Send the control command to the car
            AVControl(speed=speed, angle=angle)

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
