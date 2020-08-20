from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

def create_model(encoder_only=False):
    """
    Defines the AutoEncoder model.

    Arguments:
    encoder_only -- Boolean value used as a flag. If true, only Encoder part is
    returned. Otherwise, complete AutoEncoder is returned.

    Returns:
    Model -- Instance of Model for AutoEncoder or Encoder only depending upon the
    'encoder_only' flag.

    """

    input = Input(shape=(256, 256, 3)) # input is an 256x256 RGB image

    ### ENCODER ###
    X = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_regularizer=l2())(input)
    X = MaxPool2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_regularizer=l2())(X)
    X = MaxPool2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_regularizer=l2())(X)
    encoded = MaxPool2D(pool_size=(2, 2), name='encoder')(X)

    ### DECODER ###
    X = Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_regularizer=l2())(encoded)
    X = UpSampling2D(size=(2, 2), interpolation='bilinear')(X)

    X = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_regularizer=l2())(X)
    X = UpSampling2D(size=(2, 2), interpolation='bilinear')(X)

    X = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
            kernel_regularizer=l2())(X)
    X = UpSampling2D(size=(2, 2), interpolation='bilinear')(X)

    decoded = Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                    activation='sigmoid', kernel_regularizer=l2())(X)

    # create model instance
    if encoder_only:
        return Model(inputs=input, outputs=encoded)
    else:
        return Model(inputs=input, outputs=decoded)

def create_autoencoder():
    """
    Compiles the AutoEncoder, shows its summary and saves it in H5 format.

    Arguments:
    None

    Returns:
    None

    """

    model = create_model()
    model.compile(optimizer=Adam(), loss=binary_crossentropy)
    model.summary()
    model.save('autoencoder.h5')

def separate_encoder():
    """
    Compiles the Encoder, shows its summary and saves it in H5 format.

    Arguments:
    None

    Returns:
    None

    """

    model = create_model(encoder_only=True)
    model.compile(optimizer=Adam(), loss=binary_crossentropy)
    model.load_weights('trained_autoencoder.h5', by_name=True, skip_mismatch=True)
    model.summary()
    model.save('trained_encoder.h5')

if __name__ == '__main__':

    encoder_only = False # set this to True if you only want the encoder part

    if encoder_only:
        model = separate_encoder()
    else:
        model = create_autoencoder()
