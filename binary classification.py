import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Folder paths
POSITIVE_FOLDER = r"D:\FYP Material\new data set from kaggle\random data set for traning"  # Handwriting images
NEGATIVE_FOLDER = r"D:\FYP Material\new data set from kaggle\Not Hand Writting Img"         # Non-handwriting images

# Supported image extensions to filter
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

IMG_SIZE = 224

def load_images(folder, label):
    images = []
    labels = []
    for img_name in os.listdir(folder):
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue  # skip non-image files
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
            else:
                print(f"Warning: Could not read image {img_path}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return images, labels

print("Loading handwriting images (positive class)...")
X_pos, y_pos = load_images(POSITIVE_FOLDER, 1)

print("Loading non-handwriting images (negative class)...")
X_neg, y_neg = load_images(NEGATIVE_FOLDER, 0)

# Combine datasets
X = np.array(X_pos + X_neg).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y_pos + y_neg)

if len(X) == 0:
    raise ValueError("No images found. Check your folders and contents!")

print(f"Total images: {len(X)} (Handwriting: {len(X_pos)}, Non-handwriting: {len(X_neg)})")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build CNN model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("binary_handwriting_classifier.h5",include_optimizer=False)
print("Model saved as binary_handwriting_classifier.h5")

# Evaluate
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Training accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")
