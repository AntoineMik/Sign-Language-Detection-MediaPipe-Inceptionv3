
from asyncore import write
import streamlit as st
import requests
import json
from PIL import Image
import os

#API endpoint
api_url = "http://10.0.1.5:5000"

# Get server status
def send_request(length):
    files = {
        'length': (None, length),
    }
    response = requests.post(api_url, files=files)
    status_code = response.status_code

    return status_code, response

# Get raw random images
def request_rand_img(length):
    files = {
        'length': (None, length),
    }
    response = requests.post(api_url + '/random/images', files=files)
    status_code = response.status_code

    return status_code, response

# Process the random images with MediaPipe
def request_process_img(length):
    files = {
        'length': (None, length),
    }
    response = requests.post(api_url + '/process/images', files=files)
    status_code = response.status_code

    return status_code, response

# Predict the processed images
def request_predict_img(length):
    files = {
        'length': (None, length),
    }
    response = requests.post(api_url + '/predict/images', files=files)
    status_code = response.status_code

    return status_code, response


# Get random predictions
def request_rand_process_img(length):
    files = {
        'length': (None, length),
    }
    response = requests.post(api_url + '/random', files=files)
    status_code = response.status_code

    return status_code, response



test_data_dir = "datasets/asl_test_dataset"
processed_test_data_dir = "datasets/asl_processed_test_dataset"

size = len(os.listdir(processed_test_data_dir))

st.title("American Sign Language Detection")
st.header("Detect sign language using MediaPipe and Inceptionv3 model")

length_slider = st.sidebar.slider("Number of Random Images", 1, (size - 1))

if st.sidebar.button("Generate Random Images"):
    
    status_code, response = request_rand_img(length_slider)
    if status_code == 200:
        pred = response.text
        image = Image.open('user_rand.jpg')
        st.success(
            st.image(image, caption='ASL Random Images')
        )
    else:
        st.error(str(status_code) + " Error")


if st.sidebar.button("Process Generated Images"):
    
    status_code, response = request_process_img(length_slider)
    if status_code == 200:
        pred = response.text
        image = Image.open('process_rand.jpg')
        st.success(
            st.image(image, caption='ASL Random Image Processed MediaPipe')
        )
    else:
        st.error(str(status_code) + " Error")


if st.sidebar.button("Predict Processed Images"):
    
    status_code, response = request_predict_img(length_slider)
    if status_code == 200:
        pred = response.text
        image = Image.open('user_pred.jpg')
        st.success(
            st.image(image, caption='ASL Image Prediction')
        )
    else:
        st.error(str(status_code) + " Error")



if st.sidebar.button("Predict Random Processed Images"):
    
    status_code, response = request_rand_process_img(length_slider)
    if status_code == 200:
        pred = response.text
        image = Image.open('rand_pred.jpg')
        st.success(
            st.image(image, caption='ASL Random Image Prediction')
        )
    else:
        st.error(str(status_code) + " Error")

# Check server status
if st.sidebar.button("Check Server Status"):
    
    status_code, response = send_request(length_slider)
    if status_code == 200:
        st.success(
            response.text
        )
    else:
        st.error(str(status_code) + " Error")