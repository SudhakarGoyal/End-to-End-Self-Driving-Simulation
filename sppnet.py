from keras.layers import Flatten, concatenate, MaxPooling2D
import math

def spplayer(x, layers):
    print(x.shape)
    spp = 0
    for i in range(len(layers)):
        dim_x = math.ceil(int(x.shape[1]) / layers[i])
        stride_x = math.floor(int(x.shape[1]) / layers[i])
        dim_y = math.ceil(int(x.shape[2]) / layers[i])
        stride_y = math.floor(int(x.shape[2]) / layers[i])
        # print(math.ceil(dim_x), math.ceil(dim_y), (stride_x, stride_y))
        maxpool = MaxPooling2D((dim_x, dim_y), strides=(stride_x, stride_y))(x)

        if i == 0:
            spp = Flatten()(maxpool)
        else:
            spp = concatenate([spp, Flatten()(maxpool)], axis=-1)

    return spp
