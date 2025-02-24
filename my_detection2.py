import jetson.inference
import jetson.utils
from jetson.inference import detectNet
net = detectNet("ssd-mobilenet-v2", threshold=0.5)

input_image_path = "/home/nvidia/jetson-inference/examples/human.jpeg"
output_image_path = "/home/nvidia/jetson-inference/examples/human_finish.jpeg"

img = jetson.utils.loadImage(input_image_path)

detections = net.Detect(img)

print("Detected {:d} objects in image".format(len(detections)))
for detection in detections:
    print(detection)

jetson.utils.saveImage(output_image_path, img)
