import cv2
import os

# Load the group picture
image_path = 'Data\Misc\class_clear.jpg'
image = cv2.imread(image_path)

# Create a LBP face detector
lbp_detector_path = 'Data\Misc\haarcascade_frontalface_default.xml'
lbp_detector = cv2.CascadeClassifier(lbp_detector_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces using the LBP classifier
rects = lbp_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,minSize=(10,10))

# Create the "faces" folder if it doesn't exist
if not os.path.exists('faces'):
    os.makedirs('faces')

# Loop over the detected faces
for (i, (x, y, w, h)) in enumerate(rects):
    # Adjust the size of the face rectangle
    factor = 0.2
    x -= int(w * factor)
    y -= int(h * factor)
    w = int(w * (1 + factor * 2))
    h = int(h * (1 + factor * 2))

    # Crop the face from the image
    face = image[y:y+h, x:x+w]

    # Save the cropped face as a file
    filename = f'face_{i}.jpg'
    cv2.imwrite(os.path.join('faces', filename), face)
