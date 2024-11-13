import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.callbacks import TensorBoard

# Define your actions and other parameters
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

no_sequences = 30

sequence_length = 30

DATA_PATH = os.path.join('MP_Data')

# Create a label map
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Load data and create sequences
for action in actions:
    
    for sequence in range(no_sequences):
        window = []
    
        for frame_num in range(sequence_length):
    
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
    
                if res.shape != (63,):  # Check shape consistency
                    print(f"Frame {frame_num} in sequence {sequence} for action {action} has incorrect shape: {res.shape}")
                    res = np.zeros((63,))  # Fill with zeros if shape is incorrect
    
            except FileNotFoundError:
                print(f"Frame {frame_num} in sequence {sequence} for action {action} not found. Filling with zeros.")
                res = np.zeros((63,))  # Fill with zeros if file not found
    
            window.append(res)
    
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences to numpy arrays
X = np.array(sequences, dtype=np.float32)
y = to_categorical(labels).astype(int)


# Check the shapes to ensure consistency
print(f"Number of sequences: {len(X)}")
print(f"Shape of first sequence: {X[0].shape}")
print(f"Shape of each frame: {X[0][0].shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define the model architecture

model = Sequential()
model.add(Input(shape=(sequence_length, X[0][0].shape[0])))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Callbacks for tensorboard logging
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))

model.summary()

# Save the model
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')