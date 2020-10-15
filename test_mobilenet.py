import numpy as np
import keras
from keras.preprocessing import image
from keras.models import load_model

train_dataset_directory = 'train'
validation_dataset_directory = 'validation'
batch_size = 32

# We have to pre process the image to pass to the model
def prepare_image(file):
    img = image.load_img(file, target_size = (128, 128)) # Size that mobilenet expects
    #img = np.repeat(img, 3, 2) # Convert img from grayscale to RGB
    img.show()
    img_array = image.img_to_array(img)
    print(img_array.shape)
    img_array_expanded_dims = np.expand_dims(img_array, axis = 0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

preprocessed_image = prepare_image('validation/happy/167.jpg')
model = load_model('2_emotions.h5')
predictions = model.predict(preprocessed_image)
print(predictions)
