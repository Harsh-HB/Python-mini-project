import cv2
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

config_file = r'C:\Users\Harsh\Desktop\pbl\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = r'C:\Users\Harsh\Desktop\pbl\frozen_inference_graph.pb'

# TensorFlow object detection model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = r'C:\Users\Harsh\Desktop\pbl\yolo3.txt'
with open(filename, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print("Number of Classes:", len(classLabels))
print("Class labels:", classLabels)

# Set up model input parameters
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Read the image
img = cv2.imread(r'C:\Users\Harsh\Desktop\pbl\sample.jpg')

# Perform object detection
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

print("Confidence:", confidence)
print("Class Index:", ClassIndex)
print("Bounding Boxes:", bbox)

# Draw bounding boxes and labels on the image
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (0, 255, 0), 1)
    cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 0, 255), thickness=1)

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display image using matplotlib
plt.imshow(img_rgb)
plt.axis('off')  # To hide axes
plt.show()  # Ensure the image is displayed
