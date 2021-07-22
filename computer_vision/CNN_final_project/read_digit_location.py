import h5py
import os
import numpy as np
import concurrent.futures

def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])


def create_arrays(working_dir, train_or_test):
    array_of_pic_box = None
    full_dir = os.path.join(working_dir, train_or_test, 'digitStruct.mat')
    if not os.path.isfile(full_dir):
        print('digitStruct.mat not found in {path}. Unable to create arrays.'.format(path=full_dir))
        return array_of_pic_box
    mat_data = h5py.File(full_dir, mode='r')
    size = mat_data['/digitStruct/name'].size
    list_of_pic_box = []
    for _i in range(size):
        pic = get_name(_i, mat_data)
        box = get_box_data(_i, mat_data)
        for x in range(len(box['label'])):
            list_of_pic_box.append([pic, box['label'][x], box['left'][x], box['top'][x], box['width'][x], box['height'][x]])
    array_of_pic_box = np.asarray(list_of_pic_box)
    return (train_or_test, array_of_pic_box)

def get_arrays_from_image_dir(some_dir):
    train_array = None
    test_array = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in ('train', 'test'):
            futures.append(executor.submit(create_arrays, working_dir=some_dir, train_or_test=i))
        for future in concurrent.futures.as_completed(futures):
            if len(future.result()) > 0 and future.result()[0] == 'train':
                print('train array finished')
                train_array = future.result()[1]
            elif len(future.result()) > 0 and future.result()[0] == 'test':
                print('test array finished')
                test_array = future.result()[1]
            else:
                print('failed at grabbing arrays.')
    return train_array, test_array



if __name__ == "__main__":
    working_dir = os.getcwd()
    get_arrays_from_image_dir(working_dir)


