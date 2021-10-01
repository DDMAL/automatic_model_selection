"""Predict domain
This is a file for using the domain classifier mechanism to decide to which domain belongs each input image.
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
from keras.models import load_model


# ===========================
#       CONSTANTS
# ===========================
KEY_BACKGROUND_LAYER = "rgba PNG - Layer 0 (Background)"
KEY_SELECTED_REGIONS = "rgba PNG - Selected regions"
KEY_RESOURCE_PATH = "resource_path"
KEY_LAYERS = "layers"
KEY_IMAGES = "Image"


kPATH_IMAGES_DEFAULT = "datasets/experiments/MS73/test/images"
kPATH_REGION_MASKS_DEFAULT = "datasets/experiments/MS73/test/regions"
kPATH_MODEL_DEFAULT = "Models/DomClass/MS73_cap.hdf5"
kPATH_BACKGROUND_DEFAULT = "datasets/MS73/layers/background"
kPATH_LAYERS_DEFAULT = ["datasets/MS73/layers/staff", "datasets/MS73/layers/neumes"]
kPATH_OUTPUT_DEFAULT = "Results/MS73_cap"
kBATCH_SIZE_DEFAULT = 8
kPATCH_HEIGHT_DEFAULT = 256
kPATCH_WIDTH_DEFAULT = 256
# ===========================


def menu():
    parser = argparse.ArgumentParser(description='Prediction by domain classifier')

    parser.add_argument(
                    '-psr',
                    dest='path_src', 
                    help='List of paths of the source folders that contain the original images.',
                    required=True
                    )

    parser.add_argument(
                    '-prg',  
                    dest='path_regions', 
                    help='Path of the folder that contains the region masks.'
                    )

    parser.add_argument(
                    '-pbg',  
                    dest='path_bg', 
                    help='Path of the folder with the background layer data.'
                    )

    parser.add_argument(
                    '-pgt',
                    dest='path_layer', 
                    help='Path of the ground-truth folder to be considered (one per layer).', 
                    action='append'
                    )
    
    parser.add_argument(
                    '-json-models',
                    dest='json_models', 
                    help='Path to the json file with the paths of the models for each domain considered in the domain classifier.', 
                    required=True
                    )

    parser.add_argument(
                    '-m',
                    dest='path_model_dom_classifier',
                    help='Path for the domain classification model.',
                    default=kPATH_MODEL_DEFAULT
                    )

    parser.add_argument(
                    '-out',
                    dest='path_out', 
                    help='Path for saving the predictions.',
                    default=kPATH_OUTPUT_DEFAULT
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

def resizeImage(img, height, width, interpolation = cv2.INTER_LINEAR):
    img2 = img.copy()
    return cv2.resize(img2,(width,height), interpolation=interpolation)

#Initialize the dictionary with the inputs
def init_input_dictionary(config):
    inputs = {}

    inputs["Image"] = []
    inputs[KEY_BACKGROUND_LAYER] = []
    inputs[KEY_SELECTED_REGIONS] = []
    inputs[KEY_LAYERS] = []
    list_src_files = list_files(config.path_src)

    parent_path_bg = config.path_bg
    parent_path_regions = config.path_regions

    for idx_folder in range(len(list_src_files)):

        path_imgs = list_src_files[idx_folder]

        print (path_imgs)
        inputs[KEY_IMAGES].append(path_imgs)

        path_bgs = os.path.join(parent_path_bg, os.path.basename(path_imgs))
        inputs[KEY_BACKGROUND_LAYER].append(path_bgs)

        path_regions = os.path.join(parent_path_regions, os.path.splitext(os.path.basename(path_imgs))[0] + ".png")
        inputs[KEY_SELECTED_REGIONS].append(path_regions)


        list_path_layers = []
        for path_layer in config.path_layer:
            path_layers = os.path.join(path_layer, os.path.basename(path_imgs))
            list_path_layers.append(path_layers)
        inputs[KEY_LAYERS].append(list_path_layers)
    
    return inputs


def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


#########################################################################

config = menu()

with open(config.json_models) as json_file:
    data_models = json.load(json_file)

# Fail if arbitrary layers are not equal before training occurs.

inputs = init_input_dictionary(config)
outputs = config.path_out

print(json.dumps(inputs, indent=2))
print(json.dumps(outputs, indent=2))
print(json.dumps(data_models, indent=2))

num_imgs = len(inputs[KEY_IMAGES])
num_domains = len(data_models)

# Call in training function

model = training.get_domain_classifier(num_domains=num_domains, height=config.patch_height, width=config.patch_width, pretrained_weights=config.path_model_dom_classifier)


grs = []
for path_img in inputs[KEY_IMAGES]:
    print ('-'*40)
    print (path_img)
    gr = (cv2.imread(path_img, cv2.IMREAD_COLOR)) / 255.  # 3-channel

    gr_resized = resizeImage(gr, config.patch_height, config.patch_width)

    grs.append(gr_resized) 
    grs = np.array(grs)
    
    predictions = model.predict(grs, batch_size=config.batch_size, verbose=0)
    print (predictions)

    idx_domain = np.argmax(predictions[0])

    models_selected = data_models[str(idx_domain)]
    print("Selected domain: " + models_selected["name"] + " (label " + str(idx_domain) + ")")

    filename = os.path.basename(path_img)

    threshold = 0.5
    
    
    for layer in models_selected["layers"]:

        path_prob_out = os.path.join(config.path_out, layer, "probability", os.path.basename(path_img))
        path_result_out = os.path.join(config.path_out, layer, "result", os.path.basename(path_img))

        mkdirp(os.path.dirname(path_prob_out))
        mkdirp(os.path.dirname(path_result_out))

        sae_model = load_model(models_selected["layers"][layer])
        sae_predictions = sae_model.predict(grs, batch_size=config.batch_size, verbose=0)

        print("Layer: " + layer + ": Saving probability map in " + path_prob_out)
        print("Layer: " + layer + ": Saving result in " + path_result_out)

        cv2.imwrite(path_prob_out, sae_predictions[0]*255)
        cv2.imwrite(path_result_out, (sae_predictions[0]>threshold)*255)

    
    grs = []    
    print ('-'*40)



print("Finishing the prediction job.")

