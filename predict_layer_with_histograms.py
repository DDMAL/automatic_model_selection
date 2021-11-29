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
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import math

# ===========================
#       CONSTANTS
# ===========================
KEY_IMAGES = "Image"

kPATH_IMAGES_DEFAULT = "datasets/experiments/MS73/test/images"
kPATH_MODEL_DEFAULT = "Models/DomClass/MS73_cap.hdf5"
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
    list_src_files = list_files(config.path_src)
    
    for idx_folder in range(len(list_src_files)):
        path_imgs = list_src_files[idx_folder]
        print (path_imgs)
        inputs[KEY_IMAGES].append(path_imgs)

    return inputs


def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def get_stride(patch_height, patch_width):
    return patch_height // 2, patch_width // 2


def predictBatch(batch_data_list, coords_batch, model, batch_size, margin, predicted_layer):
    batch_data_array = np.array(batch_data_list)
    sample_prediction = model.predict(batch_data_array, batch_size=batch_size)

    for idx_batch in range(len(batch_data_list)):
        (row_sample, col_sample) = coords_batch[idx_batch]
        predicted_layer[row_sample+margin : row_sample + patch_height, col_sample+margin : col_sample + patch_width] = sample_prediction[idx_batch,margin:,margin:,0]

    return predicted_layer
        
def extractLayer(image, model, patch_height, patch_width, batch_size=2, margin = 4):

    predicted_layer = np.zeros((image.shape[0], image.shape[1]))
    hstride, wstride = get_stride(patch_height, patch_width)
    
    sample_list = []
    count = 0
    coords_batch = []
    for row in range(0, image.shape[0]-hstride+1, hstride):
        for col in range(0, image.shape[1]-wstride+1, wstride):

            row = min(row, image.shape[0] - patch_height - 1)
            col = min(col, image.shape[1] - patch_width - 1)
            coords_batch.append((row, col))
            sample = image[row : row + patch_height, col : col + patch_width]
            sample_list.append(sample) 

            if len(sample_list) == batch_size:
                predicted_layer = predictBatch(sample_list, coords_batch, model, batch_size, margin, predicted_layer)
                coords_batch = []
                sample_list = []

    if len(sample_list) > 0:
        predicted_layer = predictBatch(sample_list, coords_batch, model, batch_size, margin, predicted_layer)

    return predicted_layer

def get_gt_path(config, name_layer_folder):
    if name_layer_folder in config.path_bg:
        path_img_gt = os.path.join(config.path_bg, os.path.basename(path_img))
        idx_layer = 0
    else:
        path_img_gt = None
        idx_layer = 1
        for path in config.path_layer:
            if name_layer_folder in path:
                path_img_gt = os.path.join(path, os.path.basename(path_img))
                break
            idx_layer+=1
    return path_img_gt, idx_layer


def getHistogramBins(sample_image, num_decimal):
    tuple_sample = tuple(sample_image.reshape(1,-1)[0])

    if num_decimal is not None:
        tuple_sample_round = []
        for num in tuple_sample:
            if num > 0.01:
                tuple_sample_round.append(round(num, num_decimal))
            
        tuple_sample = tuple_sample_round

        precision = 1.
        for i in range(num_decimal):
            precision /= 10.

        value = 0.
        value = round(value, num_decimal)
        while value <= 1:
            tuple_sample.append(value)
            value += precision
            value = round(value, num_decimal)
        
    histogram_prediction = Counter(tuple_sample)
    return histogram_prediction

def getNormalizedHistogram(histogram):
    print ("---------------------")
    print (histogram)
    values = getHistogramValuesSorted(histogram)
    print(values)
    total = np.sum(values)
    values = [v/float(total) for v in values]
    print (values)
    print ("---------------------")
    return values

def getHistogramValuesSorted(histogram):
    values = []
    items_histogram = sorted(histogram.items())
    
    for prob, value in items_histogram:
        values.append(value)    
    return values

def getHistogramsTraining(data_models):
    histograms_each_folder = {}

    for model_selected in data_models:
        models_selected = data_models [model_selected]
        name_model = data_models [model_selected]["name"]

        training_folder = models_selected["training_folder"]
        list_training_files = list_files(training_folder)

        for layer in models_selected["layers"]:
            print(layer)
            
            if model_selected not in histograms_each_folder:
                histograms_each_folder[name_model] = {}
            
            if layer not in histograms_each_folder[name_model]:
                histograms_each_folder[name_model][layer] = {}    

            for path_img in list_training_files:

                gr = cv2.imread(path_img, cv2.IMREAD_COLOR)  # 3-channel
                gr_normalized = (255. - gr) / 255.

                sae_model = load_model(data_models[model_selected]["layers"][layer])
                patch_height = models_selected["windows_shape"][0]
                patch_width = models_selected["windows_shape"][1]
                predicted_layer = extractLayer(gr_normalized, sae_model, patch_height, patch_width)

                histogram = getHistogramBins(predicted_layer, 2)

                normalized_histogram = getNormalizedHistogram(histogram)

                histograms_each_folder[name_model][layer][path_img] = normalized_histogram


    return histograms_each_folder


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

#model = training.get_domain_classifier(num_domains=num_domains, height=config.patch_height, width=config.patch_width, pretrained_weights=config.path_model_dom_classifier)


grs = []

summary_results = {}

summary_results["models"] = data_models
#summary_results["domain_classifier"] = config.path_model_dom_classifier
summary_results["test"] = []


histograms_train = getHistogramsTraining(data_models)

for path_img in inputs[KEY_IMAGES]:
    print ('-'*40)
    print (path_img)

    summary_img = {}
    summary_img["image"] = path_img

    gr = cv2.imread(path_img, cv2.IMREAD_COLOR)  # 3-channel
    gr_normalized = (255. - gr) / 255.

    gr_resized = resizeImage(gr_normalized, config.patch_height, config.patch_width)

    filename = os.path.basename(path_img)

    threshold = 0.5
    
    for model_selected in data_models:
        models_selected = data_models [model_selected]

        layer_predictions = []
        list_layer_gt = []
        for layer in data_models[model_selected]["layers"]:
            str_layer_folder = "/" + layer.lower()
            path_img_gt, idx_layer = get_gt_path(config, str_layer_folder)

            print(layer)
            print ("GT extracted from: " + path_img_gt)

            gt = cv2.imread(path_img_gt, cv2.IMREAD_UNCHANGED,)  # 4-channel
            if gt is None:
                continue
            TRANSPARENCY = 3
            gt = (gt[:, :, TRANSPARENCY] == 255)

            list_layer_gt.append(gt)
            
            path_prob_out = os.path.join(config.path_out, layer, "probability", os.path.basename(data_models[model_selected]["layers"][layer]), os.path.basename(path_img))
            path_result_out = os.path.join(config.path_out, layer, "result", os.path.basename(data_models[model_selected]["layers"][layer]), os.path.basename(path_img))
            path_gt_out = os.path.join(config.path_out, layer, "GT", os.path.basename(data_models[model_selected]["layers"][layer]), os.path.basename(path_img))
            

            mkdirp(os.path.dirname(path_prob_out))
            mkdirp(os.path.dirname(path_result_out))
            mkdirp(os.path.dirname(path_gt_out))

            sae_model = load_model(data_models[model_selected]["layers"][layer])

            patch_height = models_selected["windows_shape"][0]
            patch_width = models_selected["windows_shape"][1]

            predicted_layer = extractLayer(gr_normalized, sae_model, patch_height, patch_width)
            layer_predictions.append(predicted_layer)

            print("Layer: " + layer + ": Saving probability map in " + path_prob_out)
            print("Layer: " + layer + ": Saving result in " + path_result_out)
            print("Layer: " + layer + ": Saving result in " + path_gt_out)

            cv2.imwrite(path_prob_out, predicted_layer*255)
            cv2.imwrite(path_result_out, (predicted_layer>threshold)*255)
            cv2.imwrite(path_gt_out, gt*255)


            arr_prediction = np.array(predicted_layer>threshold)
            arr_gt = np.array(gt)
            results = precision_recall_fscore_support(arr_gt.flatten()*1, arr_prediction.flatten()*1, average='binary', pos_label=1)
            print(results)

            if "metrics" not in summary_results:
                summary_results["metrics"] = {}

            if layer not in summary_results["metrics"]:
                summary_results["metrics"][layer] = []
            
            results_layer = []
            results_layer.append(path_img)
            results_layer.append(data_models[model_selected]["layers"][layer])
            results_layer.append(str(results[0]))
            results_layer.append(str(results[1]))
            results_layer.append(str(results[2]))
            
            summary_results["metrics"][layer].append(results_layer)
            

        arr_layer_predictions = np.array(layer_predictions)
        arr_layer_gt = np.array(list_layer_gt)

        argmax_layer_predictions = np.argmax(arr_layer_predictions, axis=0)

        for layer in data_models[model_selected]["layers"]:
            str_layer_folder = "/" + layer.lower()
            path_img_gt, idx_layer = get_gt_path(config, str_layer_folder)

            print(layer)
            print ("IDX: " + str(idx_layer))
            print ("GT extracted from: " + path_img_gt)

            gt = cv2.imread(path_img_gt, cv2.IMREAD_UNCHANGED,)  # 4-channel
            if gt is None:
                continue
            TRANSPARENCY = 3
            gt = (gt[:, :, TRANSPARENCY] == 255)
            

            path_result_out = os.path.join(config.path_out, "combination", layer, "result", os.path.basename(data_models[model_selected]["layers"][layer]), os.path.basename(path_img))
            mkdirp(os.path.dirname(path_result_out))

            predicted_layer = (argmax_layer_predictions == idx_layer)

            cv2.imwrite(path_result_out, predicted_layer*255)
            idx_layer+=1

            arr_prediction = np.array(predicted_layer)
            arr_gt = np.array(gt)

            results = precision_recall_fscore_support(arr_gt.flatten()*1, arr_prediction.flatten()*1, average='binary', pos_label=1)

            if "metrics_combined" not in summary_results:
                summary_results["metrics_combined"] = {}

            if layer not in summary_results["metrics_combined"]:
                summary_results["metrics_combined"][layer] = []
            
            results_layer = []
            results_layer.append(path_img)
            results_layer.append(data_models[model_selected]["layers"][layer])
            results_layer.append(str(results[0]))
            results_layer.append(str(results[1]))
            results_layer.append(str(results[2]))
            
            summary_results["metrics_combined"][layer].append(results_layer)

        
    grs = []    
    summary_results["test"].append(summary_img)
    print ('-'*40)


print (summary_results)

path_results = os.path.join(config.path_out, "results.txt")
mkdirp(os.path.dirname(path_results))
with open(path_results, 'w') as f:
    json.dump(summary_results, f, indent=2)

print("Finishing the prediction job.")

