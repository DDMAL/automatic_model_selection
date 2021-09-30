from __future__ import division

import cv2
import numpy as np
import random as rd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, UpSampling2D, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Masking, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import image_data_format
#import keras
import tensorflow as tf
import threading
from enum import Enum

kPIXEL_VALUE_FOR_MASKING = -100

class FileSelectionMode(Enum):
    RANDOM,     \
    SHUFFLE,    \
    DEFAULT     \
    = range(3)

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return FileSelectionMode[s]
        except KeyError:
            raise ValueError()

class SampleExtractionMode(Enum):
    RANDOM,     \
    SEQUENTIAL, \
    RESIZING    \
    = range(3)

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return SampleExtractionMode[s]
        except KeyError:
            raise ValueError()

# ===========================
#       SETTINGS
# ===========================

# gpu_options = tf.GPUOptions(
#     allow_growth=True,
#     per_process_gpu_memory_fraction=0.40
# )
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# keras.backend.tensorflow_backend.set_session(sess)
VALIDATION_SPLIT = 0.2
# BATCH_SIZE = 16
# ===========================


# ===========================
#       CONSTANTS
# ===========================
KEY_SELECTED_REGIONS = "rgba PNG - Selected regions"
KEY_RESOURCE_PATH = "resource_path"
KEY_IMAGES = "Image"
KEY_VALIDATION_RATIO = "Val"
# ===========================

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_input_shape(height, width, channels=3):
    if image_data_format() == "channels_first":
        return (channels, height, width)
    else:
        return (height, width, channels)


def get_domain_classifier(num_domains, height, width, pretrained_weights=None):
    ff = 32

    inputs = Input(shape=get_input_shape(height, width))
    mask = Masking(mask_value=kPIXEL_VALUE_FOR_MASKING)(inputs)

    conv1 = Conv2D(
        ff, 5, activation="relu", padding="same", kernel_initializer="he_normal"
    )(mask)
    #conv1 = Conv2D(
    #    ff, 5, activation="relu", padding="same", kernel_initializer="he_normal"
    #)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(
        ff * 2, 5, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    #conv2 = Conv2D(
    #    ff * 2, 5, activation="relu", padding="same", kernel_initializer="he_normal"
    #)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(
        ff * 4, 5, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    #conv3 = Conv2D(
    #    ff * 4, 5, activation="relu", padding="same", kernel_initializer="he_normal"
    #)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.2)(pool3)

    flatten = tf.keras.layers.Flatten() (pool3)
    dense1 = Dense(64, activation="relu")(flatten)
    dense2 = Dense(num_domains, activation="softmax")(dense1)

    model = Model(inputs=inputs, outputs=dense2)

    model.compile(
        optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    #model.summary()
    return model


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

#Load a Ground-truth image and apply the region mask if it is given
def load_gt_image(path_file, regions_mask=None):
    file_obj = cv2.imread(path_file, cv2.IMREAD_UNCHANGED,)  # 4-channel
    if file_obj is None : 
        raise Exception(
            'It is not possible to load the image\n'
            "Path: " + str(path_file)
        )

    TRANSPARENCY = 3
    bg_mask = (file_obj[:, :, TRANSPARENCY] == 255)
    
    if regions_mask is not None:
        masked = np.logical_and(bg_mask, regions_mask) * 1.0
        l = np.where((regions_mask == 0))
        masked[l] = kPIXEL_VALUE_FOR_MASKING
        return masked
    else:
        return bg_mask


def get_image_with_gt(inputs, image_paths, labels, region_paths, idx_file):

    # Required input ports
    # TODO assert that all layers have the same number of inputs (otherwise it will crack afterwards)
    number_of_training_pages = len(image_paths)
    if idx_file >= number_of_training_pages : # If we try to access to an non-existing layer 
        raise Exception(
            'The index of the file does not exist\n'
            "input images: " + str(number_of_training_pages) + " index acceded: " + str(idx_file)
        )

    gr = (cv2.imread(image_paths[idx_file], cv2.IMREAD_COLOR)) / 255.  # 3-channel
    gt = labels[idx_file]

    return gr, gt

def appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks):
    gr_sample = gr[
            row : row + patch_height, col : col + patch_width
        ]  # Greyscale image
    gr_chunks.append(gr_sample)
    gt_chunks.append(gt)



def createGeneratorSingleFileSequentialExtraction(inputs, image_paths, labels, region_paths, idx_file, row, col, patch_height, patch_width, batch_size):
    gr, gt = get_image_with_gt(inputs, image_paths, labels, region_paths, idx_file)

    gr_chunks = []
    gt_chunks = []

    hstride = patch_height // 2
    wstride = patch_width // 2
    
    count = 0
    for r in range(row, gr.shape[0] - patch_height, hstride):
        for c in range(col, gr.shape[1] - patch_width, wstride):
            appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks)
            count +=1
            if count % batch_size == 0:
                gr_chunks_arr = np.array(gr_chunks)
                gt_chunks_arr = np.array(gt_chunks)
                # convert gr_chunks and gt_chunks to the numpy arrays that are yield below

                return gr_chunks_arr, gt_chunks_arr, r, c  # convert into npy before yielding



def extractRandomSamples(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size, sample_extraction_mode):
    gr, gt = get_image_with_gt(inputs, image_paths, labels, region_paths, idx_file)

    gr_chunks = []
    gt_chunks = []

    for i in range(batch_size):
        row = np.random.randint(0, gr.shape[0] - patch_height)
        col = np.random.randint(0, gr.shape[1] - patch_width)
        appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks)

    gr_chunks_arr = np.array(gr_chunks)
    gt_chunks_arr = np.array(gt_chunks)
    # convert gr_chunks and gt_chunks to the numpy arrays that are yield below

    return gr_chunks_arr, gt_chunks_arr  # convert into npy before yielding


def get_stride(patch_height, patch_width):
    return patch_height // 2, patch_width // 2

def resizeImage(img, height, width, interpolation = cv2.INTER_LINEAR):
    img2 = img.copy()
    return cv2.resize(img2,(width,height), interpolation=interpolation)

#@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGeneratorResizingImages(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size):
    hstride, wstride = get_stride(patch_height, patch_width)
    
    gr_chunks = []
    gt_chunks = []

    count = 0
    number_of_training_pages = len(image_paths)

    while (True):
        idx_file = np.random.randint(number_of_training_pages)
        gr, gt = get_image_with_gt(inputs, image_paths, labels, region_paths, idx_file)
        gr = resizeImage(gr, patch_height, patch_width)

        gr_chunks.append(gr)
        gt_chunks.append(gt)

        count +=1
        if count % batch_size == 0:
            gr_chunks_arr = np.array(gr_chunks)
            gt_chunks_arr = np.array(gt_chunks)
            # convert gr_chunks and gt_chunks to the numpy arrays that are yield below

            return gr_chunks_arr, gt_chunks_arr  # convert into npy before yielding
            gr_chunks = []
            gt_chunks = []
            count = 0
                
@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGeneratorSequentialExtraction(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size):
    
    hstride, wstride = get_stride(patch_height, patch_width)
    
    gr_chunks = []
    gt_chunks = []

    gr, gt = get_image_with_gt(inputs, image_paths, labels, region_paths, idx_file)
    count = 0
    for row in range(0, gr.shape[0] - patch_height, hstride):
        for col in range(0, gr.shape[1] - patch_width, wstride):

            appendNewSample(gr, gt, row, col, patch_height, patch_width, gr_chunks, gt_chunks)
            count +=1
            if count % batch_size == 0:
                gr_chunks_arr = np.array(gr_chunks)
                gt_chunks_arr = np.array(gt_chunks)
                # convert gr_chunks and gt_chunks to the numpy arrays that are yield below

                yield gr_chunks_arr, gt_chunks_arr  # convert into npy before yielding
                gr_chunks = []
                gt_chunks = []
                count = 0


@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGeneratorDefault(inputs, image_paths, labels, region_paths, patch_height, patch_width, batch_size, sample_extraction_mode):
    print("Creating default generator...")
    
    number_of_training_pages = len(image_paths)

    while True:
        for idx_file in range(number_of_training_pages):
            if sample_extraction_mode == SampleExtractionMode.RANDOM:
                yield extractRandomSamples(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size, sample_extraction_mode)
            elif sample_extraction_mode == SampleExtractionMode.SEQUENTIAL:
                yield createGeneratorSequentialExtraction(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size)
            elif sample_extraction_mode == SampleExtractionMode.RESIZING:
                yield createGeneratorResizingImages(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size)
            else:
                raise Exception(
                    'The sample extraction mode does not exist.\n'
                )


@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGeneratorShuffle(inputs, image_paths, labels, region_paths, patch_height, patch_width, batch_size, sample_extraction_mode):
    print("Creating shuffle generator...")
    
    list_shuffle_idx_files = list(range(len(image_paths)))
        
    while True:
        rd.shuffle(list_shuffle_idx_files)
        
        for idx_file in list_shuffle_idx_files:
            if sample_extraction_mode == SampleExtractionMode.RANDOM:
                yield extractRandomSamples(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size, sample_extraction_mode)
            elif sample_extraction_mode == SampleExtractionMode.SEQUENTIAL:
                yield createGeneratorSequentialExtraction(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size)
            elif sample_extraction_mode == SampleExtractionMode.RESIZING:
                yield createGeneratorResizingImages(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size)
            else:
                raise Exception(
                    'The sample extraction mode does not exist.\n'
                )

@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGeneratorRandom(inputs, image_paths, labels, region_paths, patch_height, patch_width, batch_size, sample_extraction_mode):
    print("Creating random generator...")

    number_of_training_pages = len(image_paths)

    while True:
        idx_file = np.random.randint(number_of_training_pages) 
        if sample_extraction_mode == SampleExtractionMode.RANDOM:
            yield extractRandomSamples(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size, sample_extraction_mode)
        elif sample_extraction_mode == SampleExtractionMode.SEQUENTIAL:
            yield createGeneratorSequentialExtraction(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size)
        elif sample_extraction_mode == SampleExtractionMode.RESIZING:
            yield createGeneratorResizingImages(inputs, image_paths, labels, region_paths, idx_file, patch_height, patch_width, batch_size)
        else:
            raise Exception(
                'The sample extraction mode does not exist.\n'
            )



@threadsafe_generator  # Credit: https://anandology.com/blog/using-iterators-and-generators/
def createGenerator(inputs, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode, is_training, val_ratio):
    
    num_domains = len(inputs[KEY_IMAGES])
    
    if is_training: #For training
        image_paths = [img_path for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_IMAGES]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image >= int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]
        region_paths = [img_path for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_SELECTED_REGIONS]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image >= int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]
        labels = [idx_dataset for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_IMAGES]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image >= int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]

    else: #For validation
        image_paths = [img_path for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_IMAGES]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image < int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]
        region_paths = [img_path for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_SELECTED_REGIONS]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image < int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]
        labels = [idx_dataset for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_IMAGES]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image < int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]

    labels = np.eye(num_domains)[labels]
    
    if file_selection_mode == FileSelectionMode.DEFAULT:
        return createGeneratorDefault(inputs, image_paths, labels, region_paths, patch_height, patch_width, batch_size, sample_extraction_mode)
    elif file_selection_mode == FileSelectionMode.SHUFFLE:
        return createGeneratorShuffle(inputs, image_paths, labels, region_paths, patch_height, patch_width, batch_size, sample_extraction_mode)
    elif file_selection_mode == FileSelectionMode.RANDOM:
        return createGeneratorRandom(inputs, image_paths, labels, region_paths, patch_height, patch_width, batch_size, sample_extraction_mode)
    else:
        raise Exception(
            'The file extraction mode does not exist.\n'
        ) 



def getTrain(inputs, num_labels, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode, is_training, val_ratio = 0.):
    generator_label = createGenerator(
        inputs, patch_height, patch_width, batch_size, file_selection_mode, sample_extraction_mode, is_training, val_ratio
    )
    print(generator_label)

    return generator_label



def get_number_samples_sequential(image_paths, patch_height, patch_width):
    hstride, wstride = get_stride(patch_height, patch_width)
    number_samples = 0

    for path_file in image_paths:
        gr = cv2.imread(path_file, cv2.IMREAD_COLOR)  # 3-channel
        number_samples += ((gr.shape[0] - patch_height) // hstride) * ((gr.shape[1] - patch_width) // wstride)

    return number_samples

def get_steps_per_epoch(inputs, number_samples_per_class, patch_height, patch_width, batch_size, sample_extraction_mode, is_training, val_ratio):

    if is_training: #For training
        image_paths = [img_path for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_IMAGES]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image >= int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]
    else: #For validation
        image_paths = [img_path for idx_dataset, dict_imgs_folder in enumerate(inputs[KEY_IMAGES]) for idx_image, img_path in enumerate(dict_imgs_folder[KEY_RESOURCE_PATH]) if idx_image < int(np.ceil(len(dict_imgs_folder[KEY_RESOURCE_PATH])*val_ratio))]

    if sample_extraction_mode == SampleExtractionMode.RANDOM:
        return number_samples_per_class // batch_size
    elif sample_extraction_mode == SampleExtractionMode.SEQUENTIAL:
        return get_number_samples_sequential(image_paths, patch_height, patch_width)
    elif sample_extraction_mode == SampleExtractionMode.RESIZING:
        return np.ceil(len(image_paths) / batch_size)
    else:
        raise Exception(
            'The sample extraction mode does not exist.\n'
        )
        
    

def train_domain_classifier(
    inputs,
    num_domains,
    height,
    width,
    output_path,
    file_selection_mode,
    sample_extraction_mode,
    epochs,
    number_samples_per_class,
    batch_size=16,
):

    val_file_selection_mode = FileSelectionMode.DEFAULT
    val_sample_extraction_mode = SampleExtractionMode.RESIZING

    # Create ground_truth
    print("Creating data generators (training and validation)...")
    generator = getTrain(inputs, num_domains, height, width, batch_size, file_selection_mode, sample_extraction_mode, True, inputs[KEY_VALIDATION_RATIO])
    generator_validation = getTrain(inputs, num_domains, height, width, batch_size, val_file_selection_mode, val_sample_extraction_mode, False, inputs[KEY_VALIDATION_RATIO])

    
    print("Training a new domain classification model")
    model = get_domain_classifier(num_domains=len(inputs[KEY_IMAGES]), height=height, width=width)
    # model.summary()

    callbacks_list = [
        ModelCheckpoint(
            output_path,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
            mode="max",
        ),
        EarlyStopping(monitor="val_accuracy", patience=3, verbose=0, mode="max"),
    ]

    steps_per_epoch_train = get_steps_per_epoch(inputs, number_samples_per_class, height, width, batch_size, sample_extraction_mode, True, inputs[KEY_VALIDATION_RATIO])
    steps_per_epoch_val = get_steps_per_epoch(inputs, number_samples_per_class, height, width, batch_size, val_sample_extraction_mode, False, inputs[KEY_VALIDATION_RATIO])

    # Training stage
    model.fit(
        generator,
        verbose=2,
        steps_per_epoch=steps_per_epoch_train,
        validation_data=generator_validation,
        validation_steps=steps_per_epoch_val,
        callbacks=callbacks_list,
        epochs=epochs,
    )

    return 0


# Debugging code
if __name__ == "__main__":
    print("Must be run from Rodan")