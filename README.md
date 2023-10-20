# Sign Language Detection with MediaPipe and Inceptionv3

Sign language is a crucial mode of communication for many individuals, and enabling technology to understand and interpret it is a significant step toward improving accessibility across various technological domains. This project focuses on building a system that can perceive and detect sign language using Google's MediaPipe and the powerful Inceptionv3 model. The potential applications range from real-time sign language translation in augmented or virtual reality to enhancing communication and accessibility for the hearing-impaired.

## Project Overview

### MediaPipe and Hand Landmarks

Google's MediaPipe provides an open-source, cross-platform framework for processing different data types, including video and audio. In this project, we leverage MediaPipe's hand landmarks detection approach, which enables us to detect the position of individual fingers and palms. This is a critical step in understanding and interpreting hand gestures in sign language.

## Key Steps (Detailed in Python Notebook)

**For detailed project steps and code, please refer to the included Python notebook file.**

The project follows these key steps:

1. **Data Collection (Step 1)**: We collect image datasets of American Sign Language to create a diverse and representative dataset.

2. **Hand Landmarks Detection (Step 2)**: We utilize MediaPipe to process the image dataset and detect hand landmarks and features, such as fingers and palm positions.

3. **Data Preparation (Step 3)**: We prepare the processed image dataset for model training using Keras Image Data Generator.

4. **Machine Learning Model (Step 4)**: We build a machine learning model, leveraging Google's Inceptionv3 as a baseline. Transfer learning is used to take advantage of the pre-trained Inceptionv3 model.

5. **Predictions (Step 5)**: We use the trained model to predict and classify unseen hand sign images, enabling the system to understand sign language.

6. **Live Transcription (Step 6)**: We take the project one step further by creating a live hand-sign transcription system. This allows real-time interpretation and translation of sign language gestures.

## Prerequisites

Before running this project, ensure you have the necessary dependencies and libraries installed. The primary requirements include Python, TensorFlow, Keras, and MediaPipe. You can find detailed installation instructions in the project documentation.

## Usage

You can use this project for various applications, including real-time sign language translation, gesture recognition, and accessibility improvements for the hearing-impaired. Detailed instructions on using and integrating the model for your specific use case can be found in the project documentation and the provided Python notebook.

## Contributing

We welcome contributions and improvements from the community. If you have ideas suggestions or want to enhance the project, please feel free to contribute. Together, we can make technology more inclusive and accessible.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Thank you for your interest in the Sign Language Detection project. We hope it brings positive changes in accessibility and communication. If you have any questions or need assistance, please don't hesitate to reach out.

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
