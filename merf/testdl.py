
from merf.medl import MEDL
from keras import Sequential
from keras.layers import Dense
import numpy as np
from merf.utils import MERFDataGenerator
from merf.merf import MERF

dgm = MERFDataGenerator(m=.6, sigma_b=np.sqrt(4.5), sigma_e=1)

num_clusters_each_size  = 20
train_sizes = [1, 3, 5, 7, 9]
known_sizes = [9, 27, 45, 63, 81]
new_sizes = [10, 30, 50, 70, 90]

train_cluster_sizes  = MERFDataGenerator.create_cluster_sizes_array(train_sizes, num_clusters_each_size)
known_cluster_sizes = MERFDataGenerator.create_cluster_sizes_array(known_sizes, num_clusters_each_size)
new_cluster_sizes = MERFDataGenerator.create_cluster_sizes_array(new_sizes, num_clusters_each_size)
train, test_known, test_new, training_cluster_ids, ptev, prev = dgm.generate_split_samples(train_cluster_sizes, known_cluster_sizes, new_cluster_sizes)

X_train = train[['X_0', 'X_1', 'X_2']]
Z_train = train[['Z']]
clusters_train = train['cluster']
y_train = train['y']


model = Sequential([

    Dense(10, input_shape=(X_train.shape[1], ), activation="relu"),
    Dense(3, input_shape=(X_train.shape[1],), activation="relu"),
    Dense(1, )


])


model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["mse"])

fit_params = {'batch_size': X_train.shape[0], 'epochs': 10, 'verbose': 0}


mixed_deep_learn = MEDL(model, fit_params)


mixed_deep_learn.fit(X_train, Z_train, clusters_train, y_train)

rf = MERF()

rf.fit(X_train, Z_train, clusters_train, y_train)