import torch.nn as nn
import torch
import os,csv
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
        
        """
    # if iter % lr_decay_iter or iter > max_iter:
    #     return optimizer
    
    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
# return lr
def get_label_info(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label
def get_label_info1(csv_path):
    """
        Retrieve the class names and label values for the selected dataset.
        Must be in CSV format!
        
        # Arguments
        csv_path: The file path of the class dictionairy
        
        # Returns
        Two lists: one for the class names and the other for the label values
        """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")
    
    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    # print(class_dict)
return class_names, label_values


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map



def reverse_one_hot(image):
    """
        Transform a 2D array in one-hot format (depth is num_classes),
        to a 2D array with only 1 channel, where each pixel value is
        the classified class key.
        
        # Arguments
        image: The one-hot format image
        
        # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
        """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])
    
    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
        Given a 1-channel array of class keys, colour code the segmentation results.
        
        # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
        # Returns
        Colour coded image for segmentation visualization
        """
    
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    label_values = [label_values[key] for key in label_values]
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    
    return x

def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())
    #print(total)
    count = [0.0] * num_classes
    #print(pred)
    #print(label)
    for i in range(len(label)):
        if pred[i] == label[i]:
            #print(label[i])
            count[int(pred[i])] = count[int(pred[i])] + 1.0

# If there are no pixels from a certain class in the GT,
# it returns NAN because of divide by zero
# Replace the nans with a 1.0.
accuracies = []
for i in range(len(total)):
    if total[i] == 0:
        accuracies.append(1.0)
        else:
            accuracies.append(float(count[i]) / float(total[i]))
#print(accuracies)
return accuracies
def compute_mean_iou(pred, label):
    
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);
    
    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val
        #print(label_i)
        label_i=label_i.numpy()
        #print(label_i)
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
    
    
    mean_iou = np.mean(I / U)
    return mean_iou
def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()
    #print(flat_pred)
    #print(flat_label)
    #print(len(flat_pred))
    #print(len(flat_label))
    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)
    csv_path = os.path.join('/content/segment/dataset/CamVid', 'class_dict.csv')
    label_info = get_label_info(csv_path)
    label = colour_code_segmentation(label.cpu().detach().numpy(), label_info)
    b=torch.from_numpy(label)
    flat_label = b.flatten()
    #print(flat_pred)
    #print(flat_label)
    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)
    
    iou = compute_mean_iou(flat_pred, flat_label)
    
    return global_accuracy, class_accuracies, prec,rec,f1,iou
