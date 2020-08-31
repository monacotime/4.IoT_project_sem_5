# pylint: disable=line-too-long

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

classes = ["ok", "call"]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

height, width, channels = (416,416,3)

interpreter = tf.lite.Interpreter(model_path=r"C:\Users\monac\Documents\Projects\train_yolo_to_detect_custom_object (online gpu)\tflite implement\yolov3-tiny_custom.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# input_shape = input_details[0]['shape']

input_data = np.array(Image.open(r'C:\Users\monac\Documents\Projects\train_yolo_to_detect_custom_object (online gpu)\tflite implement\test.jpg').resize((416,416)), dtype="float32")
input_data = np.expand_dims(input_data, axis = 0)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

outs = interpreter.get_tensor(output_details[0]['index'])

# print(outs[0][0])

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5: