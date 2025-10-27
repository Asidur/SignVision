# Importing necessary libraries
import numpy as np  # Used to handle numerical array data
import tensorflow as tf  # Deep learning framework for model creation and training
from sklearn.model_selection import train_test_split  # To split dataset into training/testing
import os  # Handling directories and file paths

# --- Location of Dataset Folder (A-Z folders inside it) ---
DATA_DIR = './asl_dataset_fast'

# Read folder names automatically as label names (A, B, C, ..., Z)
LABELS = sorted(os.listdir(DATA_DIR))
NUM_CLASSES = len(LABELS)  # Number of letters/classes

print(f"Found {NUM_CLASSES} classes: {LABELS}")

# ----------------------------
# 1️⃣ Loading the Data
# ----------------------------
print("Loading data from disk...")
X = []  # To store landmark features
y = []  # To store numeric class labels

for i, label in enumerate(LABELS):  # Loop through each label (A-Z with index i)
    label_dir = os.path.join(DATA_DIR, label)

    # Ensure only valid folders processed
    if os.path.isdir(label_dir):
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)

            # Each .npy file contains normalized (x,y) hand coordinates
            data = np.load(file_path)
            X.append(data)
            y.append(i)  # Store label index (0-25)

# Convert lists to numpy arrays for training
X = np.array(X)
y = np.array(y)

print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")
# X shape example: (13000, 42) → 13000 samples, 42 features (21 landmarks * 2 coordinates)

# ----------------------------
# 2️⃣ Train/Test Split
# ----------------------------
# stratify=y → this makes sure each class appears evenly in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert labels to one-hot vectors (e.g., A=[1,0,0,...])
y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

# ----------------------------
# 3️⃣ Building the Neural Network Model
# ----------------------------
model = tf.keras.models.Sequential([
    # ✅ Input layer: must match number of features in each sample
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    # ✅ Hidden Layers — Learn sign features
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),  # Helps prevent overfitting

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),

    # ✅ Output Layer — 26 neurons for 26 letters
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Configure the model: What optimizer/method to learn?
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # multi-class classification
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ----------------------------
# 4️⃣ Training the Model
# ----------------------------
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=50,  # Higher = better learning (if dataset is large)
    batch_size=32,  # Number of samples processed before updating weights
    validation_data=(X_test, y_test)  # To check accuracy while training
)

# ----------------------------
# 5️⃣ Saving the Model
# ----------------------------
MODEL_NAME = 'asl_recognizer_model.h5'
model.save(MODEL_NAME)
print(f"\nTraining complete! Model saved as {MODEL_NAME}")

# ----------------------------
# 6️⃣ Evaluate Accuracy on Test Data
# ----------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nFinal Test Accuracy: {accuracy * 100:.2f}%')
