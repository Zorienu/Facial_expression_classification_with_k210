from keras.models import load_model
import tensorflow as tf

model_name = '2_emotions'
model = load_model(model_name + '.h5')
model.summary()

output_tensor = model.layers[-1].output
print(output_tensor)

tf.compat.v1.disable_eager_execution()
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5', output_arrays = ['dense/Softmax'])

tfmodel = converter.convert()
file = open(model_name + '.tflite', 'wb')
file.write(tfmodel)
file.close()
print('tflite file created...')