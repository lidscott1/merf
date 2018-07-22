"""
Model specifier for a Mixed Effects Deep Learning model

author: Liam Scott

Credit for MERF code base, and this is heavily influenced by:
:author: Sourav Dey <sdey@manifold.ai>

"""
from keras import Sequential
from keras.layers import Dense

layers = [

    Dense(1, input_shape=(16, ))
]

model = Sequential(layers)
