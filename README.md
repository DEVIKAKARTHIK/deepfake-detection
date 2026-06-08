# Deepfake Video Detection System

## 1. Project Objective

The aim of this project is to determine whether a video is:

* **Real Video**
* **Deepfake Video**

using Deep Learning techniques.

A deepfake is a video that has been manipulated using Artificial Intelligence to replace or modify a person's face.

---

# 2. Dataset Preparation

### Dataset Structure

```text
dataset/
│
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│
└── fake/
    ├── video1.mp4
    ├── video2.mp4
```

### Why?

The model needs examples of both categories:

* Real videos
* Fake videos

to learn the difference.

---

# 3. Importing Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
```

## OpenCV

```python
import cv2
```

Used for:

* Reading videos
* Extracting frames
* Face detection
* Image processing

---

## NumPy

```python
import numpy as np
```

Used for:

* Numerical calculations
* Storing image data in arrays

---

## Matplotlib

```python
import matplotlib.pyplot as plt
```

Used for:

* Displaying images
* Plotting graphs

---

## OS

```python
import os
```

Used for:

* Accessing files
* Navigating folders

---

# 4. Reading Videos

```python
cap = cv2.VideoCapture(video_file)
```

### Purpose

Opens a video file.

Example:

```python
cap = cv2.VideoCapture("video1.mp4")
```

---

# 5. Extracting Frames

```python
success, frame = cap.read()
```

### Purpose

Extracts one frame from the video.

Output:

```python
success = True
```

means frame extracted successfully.

---

# Why Extract Frames?

CNN models work on images, not videos.

Therefore:

```text
Video
 ↓
Frames
 ↓
Images
 ↓
CNN
```

---

# 6. Displaying a Frame

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.imshow(frame_rgb)
plt.show()
```

### Why?

OpenCV uses:

```text
BGR
```

Matplotlib uses:

```text
RGB
```

So conversion is required.

---

# 7. Face Detection

```python
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
```

### Purpose

Loads Haar Cascade face detector.

---

# Detecting Faces

```python
faces = face_detector.detectMultiScale(
    gray,
    1.1,
    4
)
```

### Parameters

#### gray

Grayscale image

#### 1.1

Scale factor

#### 4

Minimum neighbors

Reduces false detections.

---

# 8. Drawing Face Rectangle

```python
cv2.rectangle(
    img,
    (x,y),
    (x+w,y+h),
    (255,0,0),
    2
)
```

### Purpose

Draws a box around the detected face.

---

# 9. Face Cropping

```python
face = frame[y:y+h, x:x+w]
```

### Purpose

Extracts only the face.

Before:

```text
Entire frame
```

After:

```text
Only face
```

---

# Why Crop Faces?

Deepfake manipulation mainly occurs on faces.

Background information is unnecessary.

---

# 10. Image Resizing

```python
face = cv2.resize(face,(224,224))
```

### Purpose

Makes all images the same size.

Output:

```text
224 × 224
```

---

# Why 224×224?

CNN requires fixed-size input.

---

# 11. Creating Labels

For Real:

```python
labels.append(0)
```

For Fake:

```python
labels.append(1)
```

### Meaning

```text
0 = Real
1 = Fake
```

---

# 12. Building Dataset

```python
data.append(face)
```

Stores face images.

```python
labels.append(...)
```

Stores corresponding labels.

---

# 13. Converting to NumPy Arrays

```python
X = np.array(data)

y = np.array(labels)
```

### Purpose

Machine learning models require arrays.

Result:

```python
X.shape
```

Output:

```text
(2277,224,224,3)
```

Meaning:

```text
2277 images
224 height
224 width
3 color channels
```

---

# 14. Data Normalization

```python
X = X / 255.0
```

### Purpose

Converts pixel values:

```text
0-255
```

into:

```text
0-1
```

Example:

```text
255 → 1.0
128 → 0.50
```

---

# Why Normalize?

Faster training.

Better convergence.

More stable learning.

---

# 15. Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

```python
X_train,X_test,y_train,y_test =
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Purpose

Splits data:

```text
80% Training
20% Testing
```

---

# Why?

Training set:

```text
Model learns
```

Testing set:

```text
Model evaluation
```

---

# 16. CNN Model

```python
model = Sequential()
```

Creates CNN network.

---

## First Convolution Layer

```python
Conv2D(
    32,
    (3,3),
    activation='relu'
)
```

### Purpose

Detects:

* edges
* textures
* patterns

---

## Max Pooling

```python
MaxPooling2D()
```

### Purpose

Reduces image dimensions.

Makes training faster.

---

## Second Convolution Layer

```python
Conv2D(
    64,
    (3,3),
    activation='relu'
)
```

Learns more complex features.

---

## Flatten Layer

```python
Flatten()
```

Converts:

```text
Feature Maps
```

into:

```text
1D Vector
```

---

## Dense Layer

```python
Dense(
    128,
    activation='relu'
)
```

Learns classification patterns.

---

## Output Layer

```python
Dense(
    1,
    activation='sigmoid'
)
```

Produces:

```text
0 → Real
1 → Fake
```

---

# 17. Compiling Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## Adam Optimizer

Updates weights efficiently.

---

## Binary Crossentropy

Used because:

```text
Real
Fake
```

Only two classes.

---

## Accuracy

Measures prediction correctness.

---

# 18. Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test,y_test)
)
```

### Purpose

Trains CNN.

---

## Epoch

```text
1 Epoch
```

means entire dataset seen once.

You used:

```text
10 Epochs
```

---

## Batch Size

```text
16 images
```

processed together.

---

# 19. Results

Your final dataset:

```text
2277 images
```

Accuracy:

```text
Training Accuracy:
98.41%

Validation Accuracy:
98.68%
```

---

# 20. Saving Model

```python
model.save(
    "deepfake_model.h5"
)
```

### Purpose

Stores trained weights.

No need to retrain every time.

---

# 21. Loading Model

```python
from tensorflow.keras.models import load_model

model =
load_model(
    "deepfake_model.h5"
)
```

Loads trained model.

---

# 22. Gradio Interface

```python
import gradio as gr
```

### Purpose

Creates a web application.

---

## Interface

```python
gr.Interface(
    fn=predict_video,
    inputs=gr.File(type="filepath"),
    outputs="text"
)
```

### Purpose

Allows user to:

```text
Upload Video
↓
Get Prediction
```

---

# 23. Prediction Process

```text
Upload Video
       ↓
Read Video
       ↓
Extract Frame
       ↓
Detect Face
       ↓
Crop Face
       ↓
Resize Face
       ↓
Normalize
       ↓
CNN Prediction
       ↓
Real / Fake
```

---
