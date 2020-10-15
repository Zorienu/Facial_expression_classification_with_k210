# Facial expression classification with k210 (using MobileNet)
Facial expression classification using the k210 processor and MobileNet

## Description of the files

- k210 upload file (folder): contains the .kmodel files ('2_emotions.kmodel' for face expression classification and 'facedetect.kmodel' for face detection) for the k210 processor and the 'flash-list.json' configuration file for flashing the files and two '.bin' firmware files, using the '...minimum_with_kmodel_v4_support.bin' in the '.kfpkg' flashing files, saving some memory space in the Sipeed (about 1Mb less)
- 2_emotions.h5: saved model trained using the 'keras_mobilenet.py' file for classifying between happy and sad expressions
- 2_emotions.tflite: saved model, converted from the '2_emotions.h5' model, using the 'h52tlite.py' file
- FER_and_face_detection_k210.py: performs the face detection and face expression classification in the k210 processor, detecting the face, cuting and resizing to (128 * 128), expected for the face expression classification model.
- FER_k210.py: test for the face expression classification model accuracy in the k210 board
- keras_mobilenet.py: Preprocessing of the dataset, modification of the mobilenet output and train of the model, '.h5' model is exported
- test_mobilenet.py: Load the model, preprocess the image to check if it is working properly
