# Instructions for conversion

1. ### **Clone this repository:**
    [hunglc007's yolo converter](https://github.com/hunglc007/tensorflow-yolov4-tflite)

1. ### **!!! Immediately change the data/classes/coco.names -> your classes**

3. ### **Write these commands to the terminal (replace names with your files):**
    
    - ### convert to checkpoint

        python save_model.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny-416 --input_size 416 --model yolov3 --tiny --framework tflite


    - ### convert to tflite

        python convert_tflite.py --weights ./checkpoints/yolov3-tiny-416 --output ./checkpoints/yolov3-tiny-416-int8.tflite --quantize_mode full_int8



    - ### demo tflite

        python detect.py --weights ./checkpoints/yolov3-tiny-416-int8.tflite --size 416 --model yolov3 --tiny true --image ./data/test.jpg --framework tflite


YESSSS IT WORKSSSS!!!



