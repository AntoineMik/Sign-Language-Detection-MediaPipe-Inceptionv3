# Sign-Language-Detection-MediaPipe-Inceptionv3

## Introduction

The ability of a system to perceive and detect sign language is a vital component of improving accessibility across a variety of technological domains and platforms. For example, it can understand sign language and hand gestures and overlay the translation on top of the physical world in augmented or virtual reality. The real-time hand gesture perception has been solved by Google's MediaPipe and open-source, cross-platform framework for building pipelines to process data of different natures such as video and audio. The MediaPipe hand landmarks detection approach allows us to detect the position of individual fingers, making it ideal for building a machine learning model to identify sign language numbers and alphabet.

## Project Steps:

Step 1: Collect image datasets of American Sign Language.

Step 2: Process the image dataset with MediaPipe to detect hand landmarks and features such as fingers and palm.

Step 3: Prepare the new image dataset for training using Keras Image Data Generator.

Step 4: Build the machine learning model using Google's Inceptionv3 as a baseline to take advantage of transfer learning.

Step 4: Use the model to predict unseen hand sign images.

Step 5: Use the Model to build a live hand sign transcription.

## Sample

1 - Generate random images from the dataset.

![Alt text](/images/user_rand.jpg?raw=true "Random Generated Images")

2 - Process the images with MediaPipe to detect hand landmarks.
(Note: Not all images can be processed with MediaPipe. Bad images are ignored during this step)

![Alt text](/images/process_rand.jpg?raw=true "Processed Images")

2 - Predict the processed images using the build model

![Alt text](/images/user_pred.jpg?raw=true "Processed Images")

#### Technical concepts

Computer Vision, 
TensorFlow,
Keras,
MediaPipe,
InceptionV3, 
Convolutional Neural Network (CNN)

##### For More info and Tutorial:
https://antoinemik.github.io/Sign-Language-Detection-MediaPipe-Inceptionv3/
