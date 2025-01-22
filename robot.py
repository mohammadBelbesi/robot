import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
import RPi.GPIO as GPIO
import time
from sklearn.model_selection import train_test_split

# Motor GPIO Pins (replace with your actual pin numbers)
motor_pins = [17, 27, 22, 5, 6, 13]
GPIO.setmode(GPIO.BCM)
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)


# Step 1: Load and Train the Model
def train_model_from_csv(csv_path):
    """Train a classification model using dataset defined in a CSV file."""
    # Load dataset
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for index, row in df.iterrows():
        img_path = row['ImagePath']
        label = row['Label']

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Resize to model input size
            images.append(img)
            labels.append(1 if label == "Organic" else 0)
            # Encode: Organic=1, Non-Organic=0

    # Convert to NumPy arrays
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    # Save the model
    model.save("organic_classifier_model.h5")
    print("Model trained and saved as 'organic_classifier_model.h5'")

    return model


# Step 2: Move the Robotic Arm
def move_arm(action):
    """Control the robot arm based on the action (PICK or DROP)."""
    if action == "PICK":
        print("Picking up item...")
        GPIO.output(motor_pins[0], GPIO.HIGH)
        time.sleep(1)
        GPIO.output(motor_pins[0], GPIO.LOW)
    elif action == "DROP":
        print("Dropping item into the container...")
        GPIO.output(motor_pins[1], GPIO.HIGH)
        time.sleep(1)
        GPIO.output(motor_pins[1], GPIO.LOW)


# Step 3: Real-Time Classification and Robot Control
def classify_and_control_robot(model):
    """Capture video, classify objects, and control the robotic arm."""
    # Set up camera
    cap = cv2.VideoCapture(0)
    print("Starting robot... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Resize and preprocess the frame
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img / 255.0, axis=0)

            # Make predictions
            prediction = model.predict(img)
            label = "Organic" if prediction[0] > 0.5 else "Non-Organic"
            confidence = (prediction[0][0]
                          if prediction[0] > 0.5
                          else 1 - prediction[0][0])

            # Display the result
            print(f"Detected: {label} with confidence {confidence:.2f}")

            # Perform action based on classification
            if label == "Organic" and confidence > 0.8:
                move_arm("PICK")
                time.sleep(1)  # Simulate pick up
                move_arm("DROP")

            # Show the camera feed
            cv2.imshow("Robot Camera Feed", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Cleaned up resources. Exiting...")


# Main Program
if __name__ == "__main__":
    # Train the model (uncomment this line if training is needed)
    # model = train_model_from_csv("dataset.csv")

    # Load the pre-trained model
    model = tf.keras.models.load_model("organic_classifier_model.h5")

    # Start the robot
    classify_and_control_robot(model)
