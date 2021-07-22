import argparse
import numpy as np
import os
import cv2
import keras

from ModelObjects import Sequential,VGG16Scratch, VGG16Pretrained
from read_digit_location import get_arrays_from_image_dir

WORKING_DIR = os.getcwd()
global_size = (48, 48)
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'


def preprocess_images(image_array, digit_locations, digit_labels, string_name=None):
    print('starting preprocessing for {}'.format(string_name))
    nondigit_data = []
    image_rows = image_array.shape[0]
    left = np.clip(digit_locations[:,2].astype(float).astype(int), a_min=0, a_max=None)
    top = np.clip(digit_locations[:,3].astype(float).astype(int), a_min=0, a_max=None)
    width = digit_locations[:,4].astype(float).astype(int)
    height = digit_locations[:,5].astype(float).astype(int)
    resized_images = []
    for i in range(image_rows):
        real_top = top[i]
        real_bottom = top[i] + height[i]
        real_left = left[i]
        real_right = left[i] + width[i]
        cropped_digits = image_array[i][real_top:real_bottom, real_left:real_right]
        resized_images.append(cv2.resize(cropped_digits, global_size))
    images_array = np.asarray(resized_images)
    mser_object = cv2.MSER_create(_min_area=25, _max_variation=0.73)
    image_rows = image_array.shape[0]
    for i in range(image_rows):
        chosen_image = images_array[i]
        gray = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2GRAY)
        regions, _ = mser_object.detectRegions(gray)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        boxes = [cv2.boundingRect(p) for p in hulls]
        boxes = np.asarray(boxes).astype("float")
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.5)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        boxes = boxes[pick].astype("int")
        # If the bounding box width and height is within +- 10 pixels of our cropped digit width and height
        for box in boxes:
            if int(float(digit_locations[i][4])) - 8 <= box[2] <= int(float(digit_locations[i][4])) + 8 and int(float(digit_locations[i][5])) - 8 <= box[3] <= int(float(digit_locations[i][5])) + 8:
                new_data = cv2.resize(chosen_image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]], global_size)
                nondigit_data.append(new_data)
                break

    nondigit_data = np.asarray(nondigit_data)
    nondigit_labels = np.zeros(nondigit_data.shape[0], dtype=np.uint8)

    final_data = np.concatenate((images_array, nondigit_data), axis=0)
    final_labels = np.concatenate((digit_labels, nondigit_labels), axis=0)

    print("Done preprocessing images for {}".format(string_name))
    return final_data, final_labels

def read_data(training_dir, testing_dir):
    train_digit_locations, test_digit_locations = get_arrays_from_image_dir(WORKING_DIR)
    # train_digit_locations = train_digit_locations.astype(float).astype(int)
    # test_digit_locations = train_digit_locations.astype(float).astype(int)

    train_digit_labels = np.asarray(train_digit_locations[:, 1].astype(float).astype(int), dtype=np.uint8)
    test_digit_labels = np.asarray(test_digit_locations[:, 1].astype(float).astype(int), dtype=np.uint8)
    print('all arrays built, reading images')
    # for every label '1','2', read the image
    testing_images = np.array([cv2.imread(os.path.join(testing_dir, test_digit_locations[i, 0])) for i in range(test_digit_locations.shape[0])])
    training_images = np.array([cv2.imread(os.path.join(training_dir, train_digit_locations[i, 0])) for i in range(train_digit_locations.shape[0])])
    # preprocess by finding ROI
    xtrain, ytrain = preprocess_images(training_images, train_digit_locations, train_digit_labels, 'training')
    xtest, ytest = preprocess_images(testing_images, test_digit_locations, test_digit_labels, 'testing')
    xtrain = xtrain.astype(np.float32) / 255.
    xtest = xtest.astype(np.float32) / 255.
    ytrain = keras.utils.to_categorical(ytrain, 11)
    ytest = keras.utils.to_categorical(ytest, 11)
    return (xtrain, ytrain), (xtest, ytest)

def model_builder():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequential', action='store_true', help='sequential keras model')
    parser.add_argument('--vgg16_pretrained', action='store_true', help='pretrained VGG16 keras model')
    parser.add_argument('--vgg16_scratch', action='store_true', help='VGG16 model without weights ')
    args = parser.parse_args()
    if args.sequential and not args.vgg16_pretrained and not args.vgg16_scratch:
        (Xtrain, Ytrain), (Xtest, Ytest) = read_data(TRAIN_DIR, TEST_DIR)
        print("Creating Sequential Model")
        sequential_model = Sequential(Xtrain, Ytrain, Xtest, Ytest)
        sequential_model.setup()
        sequential_model.train()
    elif args.vgg16_pretrained and not args.sequential and not args.vgg16_scratch:
        (Xtrain, Ytrain), (Xtest, Ytest) = read_data(TRAIN_DIR, TEST_DIR)
        print("Creating VGG16 Model with weights")
        vgg16_pretrained_model = VGG16Pretrained(Xtrain, Ytrain, Xtest, Ytest)
        vgg16_pretrained_model.setup()
        vgg16_pretrained_model.train()
    elif args.vgg16_scratch and not args.sequential and not args.vgg16_pretrained:
        (Xtrain, Ytrain), (Xtest, Ytest) = read_data(TRAIN_DIR, TEST_DIR)
        print("Creating VGG16 model without weights")

        vgg16_scratch_model = VGG16Scratch(Xtrain, Ytrain, Xtest, Ytest)
        vgg16_scratch_model.setup()
        vgg16_scratch_model.train()
    else:
        print('no argument specified. Exiting.')
        exit()

if __name__ == "__main__":
    model_builder()

