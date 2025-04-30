# Face-Mask-Detection-Using-CNN
Face Mask Detection Using CNN: A Deep Learning Approach

Face Mask Detection using CNN

This project implements a binary image classification model using  Convolutional Neural Networks (CNN) to detect whether a person in an image is wearing a mask or not. Built using Python, TensorFlow, Keras, and Kaggle datasets , this project covers a complete deep learning pipeline from data collection to model training and prediction.
 Project Workflow & Programming Breakdown

 1. Dataset Download using Kaggle API
!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d omkargurav/face-mask-dataset
Explanation:
•	Uses the Kaggle API to automate dataset download.
•	Sets up authentication by configuring kaggle.json.
________________________________________
 2. Extract and Organize Dataset
from zipfile import ZipFile
with ZipFile('/content/face-mask-dataset.zip','r') as zip:
    zip.extractall()

Explanation:
•	Extracts the dataset for further preprocessing and labeling.
________________________________________3. Image Preprocessing
IMG_SIZE = 100
data = []

for category in ['with_mask', 'without_mask']:
    path = os.path.join(DATA_DIR, category)
    label = [1, 0] if category == "with_mask" else [0, 1]
      for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append([new_array, label])
Explanation:
•	Reads each image and resizes it to 100x100.
•	Assigns a one-hot encoded label.
•	Appends image-label pair to the dataset.
________________________________________
4. Data Shuffling and Normalization
random.shuffle(data)
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X) / 255.0
y = np.array(y)
Explanation:
•	Shuffles dataset to prevent ordering bias.
•	Normalizes pixel values to [0, 1].
________________________________________
 5. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
Explanation:
•	Splits data into training (90%) and testing (10%) sets for evaluation.
________________________________________

6. CNN Model Architecture (TensorFlow/Keras)
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

Explanation:
•	2 Convolutional layers extract spatial features.
•	MaxPooling reduces spatial dimensions.
•	Flatten prepares features for the Dense layers.
•	Final Dense layer uses softmax for binary classification (2 classes).
________________________________________
7. Model Compilation and Training
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
Explanation:
•	Adam optimizer for adaptive learning.
•	Categorical Crossentropy for one-hot classification.
•	Trains for 10 epochs and validates on test data.
________________________________________
8. Evaluation and Prediction
model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
Explanation:
•	Evaluates accuracy and loss.
•	Predicts labels for test set.
________________________________________
Key Features
CNN from scratch using Keras
Dataset download and config using Kaggle API
One-hot encoding for classification
Data preprocessing and augmentation
Model training, validation, and evaluation
Reproducible and modular notebook structure
________________________________________
What I Learn from This Project
•	End-to-end deep learning workflow
•	Image classification using CNN
•	Preprocessing real-world data
•	Handling multi-class labels (via one-hot encoding)
•	Building scalable Keras models
•	Evaluation and interpretation of model predictions

