# Face-Detector
# Detects how many faces are there in the picture 

# Install dependencies if not installed
# !pip install opencv-python matplotlib

import cv2
import matplotlib.pyplot as plt
# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image_path = r"C:\\Users\\Anushka\\OneDrive\\Pictures\\Uploads\\FB_IMG_1677685619574.jpg"  # <--- change this to your image name cpy the path of the image and keep here
image = cv2.imread(image_path)

if image is None:
    print("⚠️ Local image not found. Using online test image instead...")

    # =============== OPTION 2: Download sample Lena image from OpenCV repo ============
    import urllib.request
    import numpy as np

    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(f"Number of faces detected: {len(faces)}")
# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Show result
plt.figure(figsize=(8,6))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
[AI.ipynb](https://github.com/user-attachments/files/22637251/AI.ipynb)
