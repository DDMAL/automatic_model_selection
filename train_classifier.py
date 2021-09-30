"""Local Fast Trainer
This is the file for running Calvo Fast Trainer loaclly. Make sure
to have an 'Images' folder with the correct inputs in the same directory.
If not, you can change the values in 'inputs' and 'outputs'.

Simply run `python local_fast_trainer.py` to see the output.
This will call `training_engine_sae.py`.

It should generate 3 files in its current state. A background model,
a Model 0, and a Log File.

If you're running it in a Rodan container, this will be located in code/Rodan/rodan/jobs/Calvo_classifier
If the container is already running, try `docker exec -it [container_name] bash` to run the script without
stopping.
"""

import logging
import os
import sys
import cv2
import numpy as np
import training_engine_classifier as training
import pdb
import argparse
import json


# ===========================
#       CONSTANTS
# ===========================
KEY_BACKGROUND_LAYER = "rgba PNG - Layer 0 (Background)"
KEY_SELECTED_REGIONS = "rgba PNG - Selected regions"
KEY_RESOURCE_PATH = "resource_path"
KEY_LAYERS = "layers"
KEY_IMAGES = "Image"
KEY_VALIDATION_RATIO = "Val"


kPATH_IMAGES_DEFAULT = ["datasets/MS73/images"]
kPATH_REGION_MASKS_DEFAULT = ["datasets/MS73/regions"]
kPATH_BACKGROUND_DEFAULT = ["datasets/MS73/layers/background"]
kPATH_LAYERS_DEFAULT = ["datasets/MS73/layers/staff", "datasets/MS73/layers/neumes"]
kPATH_OUTPUT_MODELS_DEFAULT = ["Models/MS73/model_background.hdf5", "Models/MS73/model_staff_dom_class.hdf5", "Models/MS73/model_neumes_dom_class.hdf5"]
kBATCH_SIZE_DEFAULT = 8
kPATCH_HEIGHT_DEFAULT = 256
kPATCH_WIDTH_DEFAULT = 256
kMAX_NUMBER_OF_EPOCHS_DEFAULT = 50
kNUMBER_SAMPLES_PER_CLASS_DEFAULT = 1000
kFILE_SELECTION_MODE_DEFAULT = training.FileSelectionMode.SHUFFLE
kSAMPLE_EXTRACTION_MODE_DEFAULT = training.SampleExtractionMode.RESIZING
kVALIDATION_RATIO_DEFAULT = 0.2
# ===========================


def menu():
    parser = argparse.ArgumentParser(description='Fast trainer')

    parser.add_argument(
                    '-psr',  
                    dest='path_src', 
                    help='List of paths of the source folders that contain the original images.',
                    action='append'
                    )

    parser.add_argument(
                    '-prg',  
                    dest='path_regions', 
                    help='Path of the folder that contains the region masks.',
                    action='append'
                    )

    parser.add_argument(
                    '-pbg',  
                    dest='path_bg', 
                    help='Path of the folder with the background layer data.',
                    action='append'
                    )

    parser.add_argument(
                    '-pgt',
                    dest='path_layer', 
                    help='Paths of the ground-truth folders to be considered (one per layer).', 
                    action='append'
                    )
    
    parser.add_argument(
                    '-out',
                    dest='path_out', 
                    help='Paths for the models saved after the training.',
                    default=kPATH_OUTPUT_MODELS_DEFAULT
                    )

    parser.add_argument(
                    '-width',
                    default=kPATCH_HEIGHT_DEFAULT,
                    dest='patch_width',
                    type=int,
                    help='Patch width'
                    )

    parser.add_argument(
                    '-height',
                    default=kPATCH_WIDTH_DEFAULT,
                    dest='patch_height',
                    type=int,
                    help='Patch height'
                    )

    parser.add_argument(
                    '-b',
                    default=kBATCH_SIZE_DEFAULT,
                    dest='batch_size',
                    type=int,
                    help='Batch size'
                    )

    parser.add_argument(
                    '-e',
                    default=kMAX_NUMBER_OF_EPOCHS_DEFAULT,
                    dest='max_epochs',
                    type=int,
                    help='Maximum number of epochs'
                    )

    parser.add_argument(
                    '-n',
                    default=kNUMBER_SAMPLES_PER_CLASS_DEFAULT,
                    dest='number_samples_per_class',
                    type=int,
                    help='Number of samples per class to be extracted'
                    )

    parser.add_argument(
                    '-fm',
                    default=kFILE_SELECTION_MODE_DEFAULT, 
                    dest='file_selection_mode',
                    type=training.FileSelectionMode.from_string, 
                    choices=list(training.FileSelectionMode), 
                    help='Mode of selecting images in the training process'
                    )

    parser.add_argument(
                    '-sm',
                    default=kSAMPLE_EXTRACTION_MODE_DEFAULT, 
                    dest='sample_extraction_mode',
                    type=training.SampleExtractionMode.from_string, 
                    choices=list(training.SampleExtractionMode), 
                    help='Mode of extracing samples for each image in the training process'
                    )

    parser.add_argument(
                    '-val',
                    default=kVALIDATION_RATIO_DEFAULT,
                    dest='validation_ratio',
                    type=float,
                    help='Ratio of validation images used for training the models'
                    )

    args = parser.parse_args()

    args.path_src = args.path_src if args.path_src is not None else kPATH_IMAGES_DEFAULT
    args.path_regions = args.path_regions if args.path_regions is not None else kPATH_REGION_MASKS_DEFAULT
    args.path_bg = args.path_bg if args.path_bg is not None else kPATH_BACKGROUND_DEFAULT
    args.path_layer = args.path_layer if args.path_layer is not None else kPATH_LAYERS_DEFAULT
    
    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args

# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    list_files =  [os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

    return sorted(list_files)

# Return the list of files for each folder in a list of directories.
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files_per_path(list_folders, ext=None):
    
    files = [list_files(path_folder, ext=ext) for path_folder in list_folders]
    return files

#Initialize the dictionary with the inputs
def init_input_dictionary(config):
    inputs = {}

    inputs["Image"] = []
    inputs[KEY_BACKGROUND_LAYER] = []
    inputs[KEY_SELECTED_REGIONS] = []
    inputs[KEY_LAYERS] = []
    inputs[KEY_VALIDATION_RATIO] = config.validation_ratio
    list_src_files = list_files_per_path(config.path_src)


    for idx_folder in range(len(list_src_files)):

        path_imgs = list_src_files[idx_folder]
        parent_path_bg = config.path_bg[idx_folder]
        parent_path_regions = config.path_regions[idx_folder]

        print (path_imgs)
        dict_img = {}
        dict_img[KEY_RESOURCE_PATH] = path_imgs
        inputs[KEY_IMAGES].append(dict_img)

        path_bgs = [ os.path.join(parent_path_bg, os.path.basename(path_imgs_i)) for path_imgs_i in path_imgs]
        dict_img = {}
        dict_img[KEY_RESOURCE_PATH] = path_bgs
        inputs[KEY_BACKGROUND_LAYER].append(dict_img)

        path_regions = [ os.path.join(parent_path_regions, os.path.splitext(os.path.basename(path_imgs_i))[0] + ".png") for path_imgs_i in path_imgs]
        dict_img = {}
        dict_img[KEY_RESOURCE_PATH] = path_regions
        inputs[KEY_SELECTED_REGIONS].append(dict_img)
        
        for path_layer in config.path_layer:
            path_layers = [ os.path.join(path_layer, os.path.basename(path_imgs_i)) for path_imgs_i in path_imgs]
            dict_img = {}
            dict_img[KEY_RESOURCE_PATH] = path_layers
            inputs[KEY_LAYERS].append(dict_img)
    
    return inputs

#########################################################################

config = menu()

# Fail if arbitrary layers are not equal before training occurs.

inputs = init_input_dictionary(config)
outputs = config.path_out

print(json.dumps(inputs, indent=2))
print(json.dumps(outputs, indent=2))

num_domains = len(inputs[KEY_IMAGES])

# Call in training function
status = training.train_domain_classifier(
    inputs=inputs,
    num_domains=num_domains,
    height=config.patch_height,
    width=config.patch_width,
    output_path=outputs,
    file_selection_mode=config.file_selection_mode,
    sample_extraction_mode=config.sample_extraction_mode,
    epochs=config.max_epochs,
    number_samples_per_class=config.number_samples_per_class,
    batch_size=config.batch_size,
)

print("Finishing the Fast CM trainer job.")

