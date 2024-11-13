from function import *
from time import sleep
import numpy as np
import os
import cv2
import mediapipe as mp

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize mediapipe
mp_hands = mp.solutions.hands

# Function to process mediapipe results and ensure the output is always the same shape
def extract_keypoints(results):
    rh = np.zeros((63,))  # Default to zeros
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            
            if rh.shape != (63,):
                rh = np.zeros((63,))  # Ensure shape consistency
            break  # Only consider the first detected hand
    return rh

# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Loop through actions
    for action in actions:

        # Loop through sequences aka videos
        for sequence in range(no_sequences):

            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                frame = cv2.imread('Image/{}/{}.png'.format(action, sequence))

                if frame is None:
                    print(f"Frame Image/{action}/{sequence}.png not found.")
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply wait logic
                if frame_num == 0:
                    
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                   
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
