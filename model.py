import keras
from keras.utils import plot_model
import conv_net_sound

model = conv_net_sound.conv_net(input_shape=[32768], class_n=32)
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

