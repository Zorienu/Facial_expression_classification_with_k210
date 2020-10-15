import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, Flatten, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np

train_dataset_directory = 'train'
validation_dataset_directory = 'validation'
batch_size = 32

# We have to pre process the image to pass to the model
def prepare_image(file):
    #print('name of the file', file)
    #img = image.load_img(file, target_size = (128, 128)) # Size that mobilenet expects
    img = file
    #img = np.repeat(img, 3, 2) # Convert img from grayscale to RGB
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis = 0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# Base model for our DNN, it will download the pretrained weights' file
base_model = keras.applications.mobilenet.MobileNet(
    input_shape = (128, 128, 3),
    alpha = 0.75,
    depth_multiplier = 1,
    dropout = 0.001,
    pooling = 'avg',
    include_top = False, # This argument must be True if we want to predict using the entire network
    weights = "imagenet",
    classes = 1000
)

# Testing the original network
#preprocessed_image = prepare_image('american_chamaleon.jpg')
#predictions = base_model.predict(preprocessed_image)
#results = imagenet_utils.decode_predictions(predictions)
#print(results)

x = base_model.output
x = Dropout(0.001, name = 'dropout')(x)
preds = Dense(2, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = preds)

for i, layer in enumerate(model.layers):
    print(i, layer.name)

print(model.layers[-1].output) # name of the output tensor

print('setting trainable and no trainable layers')
for layer in model.layers[:86]:
    layer.trainable = False

for layer in model.layers[86:]:
    layer.trainable = True

print('generating train data')
train_data_generator = ImageDataGenerator(preprocessing_function = prepare_image)
train_generator = train_data_generator.flow_from_directory(
    train_dataset_directory,
    target_size = (128, 128),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

validation_data_generator = ImageDataGenerator(preprocessing_function = prepare_image)
validation_generator = validation_data_generator.flow_from_directory(
    validation_dataset_directory,
    target_size = (128, 128),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

model.summary()
model.compile(
    optimizer = Adam(lr = 0.001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('2_emotions.h5',
                             monitor = 'val_loss',
                             mode = 'min',
                             save_best_only = True,
                             verbose = 1)

early_stop = EarlyStopping(monitor = 'val_loss',
                           min_delta = 0,
                           patience = 5, # Model accuracy not improving for 3 rounds, then stop
                           verbose = 1,
                           restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.0001) # Model accuracy not improving, then reduce the learning rate 

callbacks = [checkpoint, early_stop, reduce_lr]

nb_train_samples = 12102
nb_validation_samples = 2964
epochs = 10

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size
)

print('finish thesis madafaca...')