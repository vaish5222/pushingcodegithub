import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = load_model('model.h5')

# Define your actions and other parameters
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
no_sequences = 20
sequence_length = 30
DATA_PATH = os.path.join('MP_Data')

# Create a label map
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Load test data and create sequences
for action in actions:
    for sequence in range(no_sequences, no_sequences + 5):  # Using last 5 sequences for testing
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
                if res.shape != (63,):
                    print(f"Frame {frame_num} in sequence {sequence} for action {action} has incorrect shape: {res.shape}")
                    res = np.zeros((63,))
            except FileNotFoundError:
                print(f"Frame {frame_num} in sequence {sequence} for action {action} not found. Filling with zeros.")
                res = np.zeros((63,))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences to numpy arrays
X_test = np.array(sequences, dtype=np.float32)
y_test = np.eye(len(actions))[labels]

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Make predictions on test data
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# Get class labels
class_labels = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]

# Print classification report
print("Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=class_labels))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))
