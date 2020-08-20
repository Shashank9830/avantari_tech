import tensorflow as tf

total_images = 4738 # total images in the dataset
source_dir = 'dataset' # original dataset directory
target_dir = 'resized_256' # resized dataset directory

for i in range(total_images):

    # load the image and resize it using bilinear interpolation
    resized_image = tf.keras.preprocessing.image.load_img(f"{source_dir}/{i}.jpg",
                        target_size=(256, 256), interpolation='bilinear')

    # convert the image to a numpy array
    np_image = tf.keras.preprocessing.image.img_to_array(resized_image)

    # save the resized image in the target directory
    tf.keras.preprocessing.image.save_img(f"{target_dir}/{i}.jpg", np_image)
