import os
import time
import skimage.io
import numpy as np
import pandas as pd

if os.name == 'nt':
    src_dir = r"C:/Users/Eduardo Montesuma/Desktop/InnovR/reduced_dataset/train/in_mul/"
    aug_dir = r"C:/Users/Eduardo Montesuma/Desktop/InnovR/reduced_dataset/train/aug_data/"
    ref_dir = r"C:/Users/Eduardo Montesuma/Desktop/InnovR/reduced_dataset/train/ref_mul/"
elif os.name == 'posix':
    src_dir = "/home/efernand/InnovR_Eduardo/reduced_dataset/train/in_mul"
    ref_dir = "/home/efernand/InnovR_Eduardo/reduced_dataset/train/ref_mul"
    aug_dir = "/home/efernand/InnovR_Eduardo/reduced_dataset/train/augmented_samples"

def correct_names(aug_dir):
    """
        This function correct the names of augmented files
    """
    aug_files = [f for f in os.listdir(aug_dir) if os.path.isfile(os.path.join(aug_dir, f))]
    tmp = []
    cnst = 0
    for file in aug_files:
        corrected_name = file.split('.png')[0].split('in_mul_original_')[1]
        if corrected_name in tmp:
            os.rename(aug_dir + r'/' + file, aug_dir + r'/aug_' + corrected_name + str(cnst) + '.png')
            cnst += 1
        else:
            tmp.append(corrected_name)
            os.rename(aug_dir + r'/' + file, aug_dir + r'/aug_' + corrected_name + '.png')

def erase_folder(aug_dir):
    """
        This function erase all contents of aug_dir
    """
    aug_files = [f for f in os.listdir(aug_dir) if os.path.isfile(os.path.join(aug_dir, f))]
    for file in aug_files:
        os.remove(aug_dir + r'/' + file)


def filter_filenames(filenames, filter_set):
    for filename in filenames:
        sub_filename = str.split(os.path.splitext(filename)[0], '_')
        key = sub_filename[0] + '_' + sub_filename[1]
        if key in filter_set:
            yield filename


def create_dataset(full_dataset_path="/home/efernand/InnovR_Eduardo/dataset/100M_alpha_num/",
                   reduced_dataset_path="/home/efernand/InnovR_Eduardo/reduced_dataset/train/"):
    """
        This function takes the directories for the original, and reduced dataset, and produces
        the arrays for training/evaluating the neural network models
    """
    start = time.time();
    print("starting creation")
    test_path = full_dataset_path + "test/"
    valid_path = full_dataset_path + "valid/"

    test_path_files = os.listdir(test_path + "in/")
    valid_path_files = os.listdir(valid_path + "in/")
    train_path_files = os.listdir(reduced_dataset_path + "in_mul/")

    csv_path = os.path.join(full_dataset_path, "content.csv")
    csv_data = pd.read_csv(csv_path)
    sub_datas = csv_data[csv_data.solid_bg == True]  # filtering sample on solid_bg
    filter_set = set(sub_datas.apply(lambda row: str(row['timestamp']) + '_' + str(row['sample_index']), axis=1))

    train_filenames = list(filter_filenames(train_path_files, filter_set))
    valid_filenames = list(filter_filenames(valid_path_files, filter_set))
    test_filenames = list(filter_filenames(test_path_files, filter_set))
    augm_filenames = os.listdir(reduced_dataset_path + "augmented_samples/")
    # augm_filenames  = list(filter_filenames(augmented_path_files, filter_set))

    # valid_filenames = valid_filenames[:batch_size]

    data_train_input = np.expand_dims(np.stack(
        skimage.img_as_ubyte(skimage.io.imread(os.path.join(reduced_dataset_path, "in_mul/", filename), as_grey=True))
        for filename in train_filenames), axis=3)
    data_train_output = np.expand_dims(np.stack(
        skimage.img_as_ubyte(skimage.io.imread(os.path.join(reduced_dataset_path, "in_mul/", filename), as_grey=True))
        for filename in train_filenames), axis=3)
    data_aug_input = np.expand_dims(np.stack(skimage.img_as_ubyte(
        skimage.io.imread(os.path.join(reduced_dataset_path, "augmented_samples/", filename), as_grey=True)) for
                                             filename in augm_filenames), axis=3)
    data_aug_output = np.expand_dims(np.stack(skimage.img_as_ubyte(
        skimage.io.imread(os.path.join(reduced_dataset_path, "ref_mul/", filename.split("aug_")[1]), as_grey=True)) for
                                              filename in augm_filenames), axis=3)
    data_valid_input = np.expand_dims(np.stack(
        skimage.img_as_ubyte(skimage.io.imread(os.path.join(valid_path, "in", filename), as_grey=True)) for filename in
        valid_filenames), axis=3)
    data_valid_output = np.expand_dims(np.stack(
        skimage.img_as_ubyte(skimage.io.imread(os.path.join(valid_path, "ref", filename), as_grey=True)) for filename in
        valid_filenames), axis=3)
    data_test_input = np.expand_dims(np.stack(
        skimage.img_as_ubyte(skimage.io.imread(os.path.join(test_path, "in", filename), as_grey=True)) for filename in
        test_filenames), axis=3)
    data_test_output = np.expand_dims(np.stack(
        skimage.img_as_ubyte(skimage.io.imread(os.path.join(test_path, "ref", filename), as_grey=True)) for filename in
        test_filenames), axis=3)

    train_data = {
        'input': np.concatenate((data_train_input, data_aug_input), axis=0),
        'output': np.concatenate((data_train_output, data_aug_output), axis=0)
    }

    test_data = {
        'input': data_test_input,
        'output': data_test_output
    }

    valid_data = {
        'input': data_valid_input,
        'output': data_valid_output
    }

    finish = time.time();
    print("Processing took: {} sec".format(finish - start))
    return train_data, test_data, valid_data



