from tensorflow import keras as keras
# import keras
import numpy as np
from augementation import augmentation_image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_fps, labels1, bands=None, batch_size=16, dim=(512, 512),
                                n_channels=11, n_classes1=3, n_classes2=19, no_data_value=-999, augumentation=True, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels1 = labels1
        self.list_fps = list_fps
        self.bands = bands
        self.n_channels = n_channels
        self.n_classes1 = n_classes1
        self.n_classes2 = n_classes2
        self.shuffle = shuffle
        self.augumentation = augumentation
        self.on_epoch_end()
        self.no_data_value = no_data_value

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_fps) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Generate indexes of the batch
        list_fps_temp = [self.list_fps[k] for k in indexes] # Find list of IDs
        labels1_temp = [self.labels1[k] for k in indexes] # Find list of IDs
        # labels2_temp = [self.labels2[k] for k in indexes] # Find list of IDs
        X, y = self.__data_generation(list_fps_temp, labels1_temp) # Generate data
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_fps))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_data(self, fp):
        try:
            x = np.load(fp)
        except:
            print(fp)
            exit(-1)
        if x.shape != (512, 512, 17):
            print(f'fp is {fp}, shape is {x.shape}')
        if self.augumentation:
            x = augmentation_image(x)
        return x

    def __data_generation(self, list_fps_temp, labels1_temp, labels2_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y1 = np.empty((self.batch_size), dtype=int)
        y2 = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, (fp, label1, label2) in enumerate(zip(list_fps_temp, labels1_temp, labels2_temp)):
            x = self.read_data(fp)
            X[i,] = x  # Store sample
            y1[i] = label1 # Store class
            y2[i] = label2 # Store class

        # if np.sum(np.isnan(X)):
        #     print('nan warning X')
        #     print('nan warning X')
        return [X], [keras.utils.to_categorical(y1, num_classes=self.n_classes1), keras.utils.to_categorical(y2, num_classes=self.n_classes2)]


#         if bn in ['dem', 'aspect', 'roughness', 'slope', 'TPI', 'TRI', 'hillshade']:
#             hist = (b[b!=0]//0.005).astype(np.int).ravel()
#             bin_counts_st = np.bincount((b[b!=0]//0.005).astype(np.int).ravel())
#             bin_counts = np.zeros(int(1//(0.005-10**-6)), dtype=np.int)
#             bin_counts[:len(bin_counts_st)] = bin_counts_st


class DataGenerator_Leu(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_fps, labels1, bands=None, batch_size=16, dim=(512, 512),
                                n_channels=11, n_classes1=3, no_data_value=-999, augumentation=True, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels1 = labels1
        self.list_fps = list_fps
        self.bands = bands
        self.n_channels = n_channels
        self.n_classes1 = n_classes1
        # self.n_classes2 = n_classes2
        self.shuffle = shuffle
        self.augumentation = augumentation
        self.on_epoch_end()
        self.no_data_value = no_data_value

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_fps) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Generate indexes of the batch
        list_fps_temp = [self.list_fps[k] for k in indexes] # Find list of IDs
        labels1_temp = [self.labels1[k] for k in indexes] # Find list of IDs
        X, y = self.__data_generation(list_fps_temp, labels1_temp) # Generate data
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_fps))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_data(self, fp):
        try:
            x = np.load(fp)
        except:
            print(fp)
            exit(-1)
        if x.shape != (512, 512, 17):
            print(f'fp is {fp}, shape is {x.shape}')
        if self.augumentation:
            x = augmentation_image(x)
        return x

    def __data_generation(self, list_fps_temp, labels1_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y1 = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, (fp, label1) in enumerate(zip(list_fps_temp, labels1_temp)):
            x = self.read_data(fp)
            X[i,] = x  # Store sample
            y1[i] = label1 # Store class
        print(f'x is {X.shape}, y1 {y1}, n cls {self.n_classes1}')
        return [X], [keras.utils.to_categorical(y1, num_classes=self.n_classes1)]
