# Importing necessary libraries
from tensorflow.keras.preprocessing import image as Image
import numpy as np
from keras.models import load_model
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import glob, random



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define arguments for the Model
height = 300
width = 300
channels = 3
batch_size = 512
target_shape = (height, width, channels)
target_size = (height, width)
class_names = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h',
 'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
num_classes = len(class_names)

# Background image to draw hand landmarks on
background_img = cv2.imread("mod_background_black.jpg")

# Load the Model
model = load_model("trained_models/sign_lang_detect_inceptv3_model_segmented_large_dataset_plus5.h5")


# Function to segment images using MediaPipe
def mp_segment(img_path, IMAGE_FILES, new_files_location):
    # Detect hand landmarks in images
    with mp_hands.Hands(
        static_image_mode=True,
        # Detecting one hand only
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
            for idx, file in enumerate(IMAGE_FILES):
                # Read an image, flip it around y-axis for correct handedness
                image = cv2.flip(cv2.imread(img_path + "/" + file), 1)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print handedness and draw hand landmarks on the background image.
                print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                # Image dimensions
                image_height, image_width, image_channel = image.shape
                target_size = (image_height, image_width)
                annotated_image = cv2.resize(background_img.copy(), target_size)

                for hand_landmarks in results.multi_hand_landmarks:
                    print('hand_landmarks:', hand_landmarks)
                    
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                # Save the processed image
                cv2.imwrite(new_files_location + "/" + file, cv2.flip(annotated_image, 1))



# Function to predict an image
def prediction(image_data, model):
    # Resize the image
    img_arr = Image.smart_resize(image_data, target_size)

    processed_old = np.expand_dims(img_arr, axis=0)
    processed = processed_old / 255.

    # Predicting the image
    predicted_class = model.predict(processed)

    # Get the index of the prediction
    index = np.argmax(predicted_class)

    # Convert list of predictions to float at 2 decimal places
    predicted_probabilities = list(
        map('{:.2f}'.format,
        predicted_class[0]*100)
    )

    # Show only predictions with greather than 50% probability
    if (float(predicted_probabilities[index]) > 80):
        prediction_result = "Prediction-Confident : {} \n Probability : {}%".format(str(class_names[index]).title(), predicted_probabilities[index])
        plt.title(prediction_result, 
            size=18, 
            color='green'
        )
        plt.imshow(image_data)
        #print(predicted_class)
    elif (float(predicted_probabilities[index]) < 81  and float(predicted_probabilities[index]) > 50):
        prediction_result = "Prediction-Moderate : {} \n Probability : {}%".format(str(class_names[index]).title(), predicted_probabilities[index])
        plt.title(prediction_result, 
            size=18, 
            color='blue'
        )
        plt.imshow(image_data)
    elif(float(predicted_probabilities[index]) < 51  and float(predicted_probabilities[index]) > 30):
        prediction_result = "Prediction-Low Confidence : {} \n Probability : {}%".format(str(class_names[index]).title(), predicted_probabilities[index])
        plt.title(prediction_result, 
            size=18, 
            color='red'
        )
        plt.imshow(image_data)
    else:
        prediction_result = "hum... I didn't quite get that, please try again"
        plt.title(prediction_result, 
            size=18, 
            color='red'
        )
        plt.imshow(image_data)

    return prediction_result


# Select random unseen test image
file_path_type = "datasets/asl_processed_test_dataset/*.jpeg"
def rand_img():
    img = glob.glob(file_path_type)
    rand_img = random.choice(img)
    rand_img = Image.load_img(rand_img, target_size = target_size)
    return rand_img

# Show 10 predictions

def show_rand_pred(num):
    plt.figure(figsize=(15, 20))
    for x in range(num):
        ax = plt.subplot(4, 3, x + 1)
        prediction(rand_img(), model)
        plt.axis("off")


@app.route("/predict", methods=["POST"])
def main():
    try:
        length = int(request.form.get('length'))

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    prediction = show_rand_pred(length)

    return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")