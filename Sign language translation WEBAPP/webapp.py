from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from keras.models import model_from_json
import mediapipe as mp
from function import mediapipe_detection, extract_keypoints
from googletrans import Translator

app = Flask(__name__)

# Load the trained model
with open("model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("model.h5")

# Initialize variables
sentence = []
accuracy = []
sequence = []
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands

# Initialize Translator
translator = Translator()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global sentence, accuracy, sequence

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        try:

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:

                    break

                cropframe = frame[40:400, 0:300]
                frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

                image, results = mediapipe_detection(cropframe, hands)
                keypoints = extract_keypoints(results)

                if keypoints is not None and None not in keypoints:
                    sequence.append(keypoints)
                    sequence = sequence[-20:]

                if len(sequence) == 20:

                    try:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        action = actions[np.argmax(res)]
                        
                        if len(sentence) == 0 or (action != sentence[-1]):
                            sentence.append(action)
                            accuracy.append(f"{res[np.argmax(res)] * 100:.2f}%")

                        if len(sentence) > 1:
                            sentence = sentence[-1:]
                            accuracy = accuracy[-1:]

                    except Exception as e:
                        print(f"Prediction error: {e}")

                cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
                cv2.putText(frame, f"Output: {' '.join(sentence)} Accuracy: {' '.join(accuracy)}", (3, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', frame)

                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:

            print(f"Error: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_output')
def get_output():
    global sentence, accuracy
    return jsonify(sentence=' '.join(sentence), accuracy=' '.join(accuracy))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
