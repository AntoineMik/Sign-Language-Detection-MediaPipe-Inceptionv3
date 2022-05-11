
# Importing necessary libraries
from tensorflow.keras.preprocessing import image as Image
from PIL import Image as pilimage
import numpy as np
from keras.models import load_model
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import glob, random

# Path tho the dataset of images
original_path = "datasets/asl_test_dataset/*.jpeg"
# Processed dataset path
original_processed_path = "datasets/asl_processed_test_dataset/*.jpeg"

# Test user raw data location
user_test_path = "datasets/user_test_data"

# Location of the new processed user test data
user_processed_path = "datasets/user_processed_test_dataset"


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
model = load_model("sign_lang_detect_inceptv3_model_segmented_large_dataset_plus5.h5")


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

                if not results.multi_hand_landmarks:
                    continue
                # Image dimensions
                image_height, image_width, image_channel = image.shape
                target_size = (image_height, image_width)
                annotated_image = cv2.resize(background_img.copy(), target_size)

                for hand_landmarks in results.multi_hand_landmarks:
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
file_path_type = original_path
def rand_img(path):
    img = glob.glob(path)
    path_img = random.choice(img)
    _, name = os.path.split(path_img)
    rand_img = Image.load_img(path_img, target_size = target_size)

    return (name, rand_img)


# Show random predictions
def show_rand_pred(num, file_path):
    fig = plt.figure(figsize=(15, 20))
    for x in range(num):
        ax = plt.subplot(4, 3, x + 1)
        _, img = rand_img(file_path)
        prediction(img, model)
        plt.axis("off")
    return fig

def create_empty_dir(path):
    # If path exist and is not a file
    if os.path.exists(path) and not os.path.isfile(path):
        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        else: # remove files if non empty
            files = os.listdir(path)
            for file in files:
                os.remove(os.path.join(path, file))
            return True
    else: # If the path does not exist, create the folder
        try: 
            os.mkdir(path)
            return True
        except OSError as error: 
            print(error)
    return False


def generate_user_raw(size):
    # Random select user images
    if create_empty_dir(user_test_path):
        for x in range(size):
            img_name, img = rand_img(original_path)
            img.save(os.path.join(user_test_path, img_name))
        return True

    return False

def generate_user_processed(size):
    
    # Get the test images 
    user_test_images = os.listdir(user_test_path)
    if not user_test_images:
        if generate_user_raw(size):
            # Creating the test data directory if it does not exist and process user images
            if create_empty_dir(user_processed_path):
                # Process user rand images with mediapipe
                mp_segment(user_test_path, user_test_images, user_processed_path)
                return True
    else:
        # Creating the test data directory if it does not exist and process user images
        if create_empty_dir(user_processed_path):
            # Process user rand images with mediapipe
            mp_segment(user_test_path, user_test_images, user_processed_path)
            return True
    
    return False


# Display images in a location
def display_img(img_path):
    fig = plt.figure(figsize=(15, 20))
    files = os.listdir(img_path)
    if files:
        for x, file in zip(range(len(files)), files):
            ax = plt.subplot(4, 3, x + 1)
            img = Image.load_img(os.path.join(img_path, file), target_size = target_size)
            plt.imshow(img)
            plt.axis("off")
        return fig
    return False


# Predict processed images from location
def predict_img(img_path):
    fig = plt.figure(figsize=(15, 20))
    files = os.listdir(img_path)
    if files:
        for x, file in zip(range(len(files)), files):
            ax = plt.subplot(4, 3, x + 1)
            img = Image.load_img(os.path.join(img_path, file), target_size = target_size)
            prediction(img, model)
            plt.axis("off")
        return fig
    return False