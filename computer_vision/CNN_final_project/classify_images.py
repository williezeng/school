import os
import numpy as np
import cv2
import keras

MODELS_DIR = 'models'
MODEL_FILENAME = 'vgg16_pretrained.h5'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
SIZE = (48,48)


def visualize(image, predictions, locs):
    """Takes predictions and digit_locations and places them in order and plots the final address and all of the rectangles
    """
    final_image = np.copy(image)
    for i,label in enumerate(predictions):
        number = np.argwhere(label==1)
        if number:
            num_val = number[0][0]
            if num_val != 0:
                if num_val == 10:  # 10 is the digit 0 in our array
                    num_val = 0
                # if saved_box:  # check if x coord are very similar or y coord
                #     if locs[i][0] - 5 <= saved_box[0][0] <= locs[i][0] + 5 or locs[i][1] - 5 <= saved_box[0][1] <= locs[i][1] + 5:
                #         cv2.rectangle(vis, (locs[i][0], locs[i][1]), (locs[i][0] + locs[i][2], locs[i][1] + locs[i][3]),
                #                       (0, 255, 0), thickness=2)
                #         cv2.putText(vis, str(num_val), (locs[i][0], locs[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #                     fontScale=1, color=(0, 0, 255), thickness=2)
                #
                #         cv2.rectangle(vis, (saved_box[0][0], saved_box[0][1]), (saved_box[1][0], saved_box[1][1]),
                #                       (0, 255, 0), thickness=2)
                #         cv2.putText(vis, saved_box[2], (saved_box[0][0], saved_box[0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #                     fontScale=1, color=(0, 0, 255), thickness=2)
                #         saved_box = (locs[i][0], locs[i][1]), (locs[i][0] + locs[i][2], locs[i][1] + locs[i][3]), str(num_val)  # hold for examination
                #
                #     else:
                #         saved_box = (locs[i][0], locs[i][1]), (locs[i][0] + locs[i][2], locs[i][1] + locs[i][3]), str(num_val)  # hold for examination
                # else:
                #     saved_box = (locs[i][0],locs[i][1]),(locs[i][0]+locs[i][2],locs[i][1]+locs[i][3]), str(num_val)  # hold for examination
    # if saved_box:  # at this point you know it has the similar coord
                cv2.rectangle(final_image, (locs[i][0], locs[i][1]), (locs[i][0] + locs[i][2], locs[i][1] + locs[i][3]),
                              (0, 255, 0), thickness=2)
                cv2.putText(final_image, str(num_val), (locs[i][0], locs[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255), thickness=2)
    return final_image



def filtering_noise(input_image, voted_prediction_array, roi_locations):
    """Removes digits in areas with less than m votes.
    """
    vis = None
    found_something = False
    saved_locations = []
    saved_predictions = []

    if len(voted_prediction_array) == 3:
        found_something = True
        vis = visualize(input_image, voted_prediction_array, roi_locations)

    elif len(voted_prediction_array) > 4:

        extracted_heights = [xx[3] for xx in roi_locations]
        extracted_heights = np.asarray(extracted_heights)
        tweaker = 3
        for i, label in enumerate(voted_prediction_array):
            # go through predictions and grab similar heights boxes
            if np.argwhere(label == 1):
                # Find other boxes with similar heights
                # find height values less than the predicted height value + 2
                # find indices of height values greater than predicted  height - 2
                # grab height values
                similar_heights = extracted_heights[list(extracted_heights) <= roi_locations[i][3] + tweaker]
                if len(similar_heights) > 0:
                    similar_width_indices = np.nonzero(similar_heights >= roi_locations[i][3] - tweaker)[0]
                    interesting = similar_heights[list(similar_heights) >= roi_locations[i][3] - tweaker]
                    # we found at least 4 similar heights, get out
                    if len(interesting) == 4:
                        for index in similar_width_indices:
                            saved_locations.append(roi_locations[index])
                            saved_predictions.append(voted_prediction_array[index])
                            found_something = True
                        break
        if found_something:
            vis = visualize(input_image, saved_predictions, saved_locations)

    elif len(voted_prediction_array) == 4:
        # vis = visualize(input_image, saved_predictions, saved_locations)
        # return vis
        extracted_width = [xx[2] for xx in roi_locations]
        extracted_width = np.asarray(extracted_width)
        extracted_heights = [xx[3] for xx in roi_locations]
        extracted_heights = np.asarray(extracted_heights)
        area_tweak = 8
        for i, label in enumerate(voted_prediction_array):
            # go through predictions and grab similar heights boxes
            if np.argwhere(label == 1):
                box_width = roi_locations[i][2]
                box_height = roi_locations[i][3]
                similar_heights = extracted_heights[list(extracted_heights) <= box_height + area_tweak]
                similar_widths = extracted_width[list(extracted_width) <= box_width + area_tweak]
                if len(similar_widths) > 0:
                    similar_width_indices = np.nonzero(similar_widths >= box_width - area_tweak)[0]
                    similar_height_indices = np.nonzero(similar_heights >= box_height - area_tweak)[0]
                    interesting_height = similar_heights[list(similar_heights) >= box_height - area_tweak]
                    interesting = similar_widths[list(similar_widths) >= box_width - area_tweak]
                    # we found at least 3 similar heights, get out
                    if len(interesting) >= 4 and len(interesting_height) >= 3:
                        for index in similar_height_indices:
                            saved_locations.append(roi_locations[index])
                            saved_predictions.append(voted_prediction_array[index])
                            found_something = True
                        break
        if found_something:
            vis = visualize(input_image, saved_predictions, saved_locations)
    if not found_something:
        extracted_leftx = [xx[0] for xx in roi_locations]
        extracted_leftx = np.asarray(extracted_leftx)
        min_index = np.where(extracted_leftx == min(extracted_leftx))
        modified = np.delete(extracted_leftx, min_index)
        max_index = np.where(modified == max(modified))
        modified = np.delete(modified, max_index)
        min_index = np.where(modified == min(modified))
        modified = np.delete(modified, min_index)
        for i, label in enumerate(voted_prediction_array):
            # go through predictions and grab similar heights boxes
            if np.argwhere(label == 1):
                box_leftx = roi_locations[i][0]
                if box_leftx in modified:
                    saved_locations.append(roi_locations[i])
                    saved_predictions.append(voted_prediction_array[i])
        vis = visualize(input_image, saved_predictions, saved_locations)

    return vis


def classification(digit_array_holder, predictions, roi_locations, number_of_predictions):
    voted_indecies = []
    chosen_index = None
    for i, label in enumerate(predictions):
        number = np.argwhere(label==1)
        if number.size>0:
            numval = number[0][0]
            digit_array_holder[numval].append(i)

    for digit in digit_array_holder:
        votes = np.zeros(len(digit))
        if len(digit) > number_of_predictions:
            d_locations = np.array([roi_locations[idx] for idx in digit])
            for holder in range(len(digit)):
                for queried_digit in range(len(digit)):
                    x_candidate = d_locations[holder][0] + d_locations[holder][2]//2
                    x_query = d_locations[queried_digit][0] + d_locations[queried_digit][2]//2
                    y_candidate = d_locations[holder][1] + d_locations[holder][3]//2
                    y_query = d_locations[queried_digit][1] + d_locations[queried_digit][3]//2
                    distance = np.sqrt((x_candidate-x_query)**2 + (y_candidate-y_query)**2)
                    if distance <= 1:
                        votes[holder] += 1
            if np.max(votes) > 3:
                chosen_index = np.argmax(votes)
            elif np.max(votes) > 2:
                chosen_index = np.argmax(votes)
            elif np.max(votes) > 1:
                chosen_index = np.argmax(votes)
            voted_indecies.append(digit[chosen_index])
    voted_predictions_array = [predictions[idx] for idx in voted_indecies]
    box_locations = [roi_locations[idx] for idx in voted_indecies]
    return voted_predictions_array, box_locations

def preprocess_image(image):
    invGamma = 1.0 / 1.8
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    adjusted = cv2.LUT(image, table)
    # show_image(adjusted)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    mser_object = cv2.MSER_create(_min_area=90, _max_variation=0.90, _max_area=4000)
    regions, _ = mser_object.detectRegions(gray)

    vis = np.copy(image)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    boxes = [cv2.boundingRect(p) for p in hulls]
    boxes = np.asarray(boxes).astype("float")

    # initialize the list of picked indexes
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
    # keep looping while some indexes still remain in the indexes
    # list
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
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.3)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    boxes = boxes[pick].astype("int")
    for box in boxes:
        cv2.rectangle(vis,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,255,255))
    img = np.copy(image)
    windows = []
    for box in boxes:
        new_window = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]
        new_window = cv2.resize(new_window, (SIZE))
        windows.append(new_window)
    return np.asarray(windows), np.asarray(boxes)


if __name__ == "__main__":
    vis = None
    model_file = os.path.join(MODELS_DIR, MODEL_FILENAME)
    model = keras.models.load_model(model_file)
    for filename in os.listdir(INPUT_DIR):
        digit_array = []
        for i in range(11):
            digit_array.append([])
        if filename.endswith('png'):
            image_file = os.path.join(INPUT_DIR, filename)
            print(filename)
            original_read_image = cv2.imread(image_file)
            img = np.copy(original_read_image)
            ROI, ROI_location = preprocess_image(img.astype(np.uint8))
            predictions = model.predict(x=ROI, batch_size=128)
            voted_predictions_array, box_locations = classification(digit_array, predictions, ROI_location, 10)
            final_output = filtering_noise(img, voted_predictions_array, box_locations)
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_output)