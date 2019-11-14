import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy import ndimage
import pydicom
import os
from tqdm import tqdm
from time import time
ImageFile.LOAD_TRUNCATED_IMAGES = True


data_path = "/mnt/storage_dimm2/kaggle_data/rsna-intracranial-hemorrhage-detection/"


def get_metadata(image_dir):

    labels = [
        'BitsAllocated', 'BitsStored', 'Columns', 'HighBit',
        'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
        'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
        'ImagePositionPatient_0', 'ImagePositionPatient_1', 'ImagePositionPatient_2',
        'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation',
        'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope', 'Rows', 'SOPInstanceUID',
        'SamplesPerPixel', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID',
        'WindowCenter', 'WindowWidth', 'Image',
    ]

    data = {l: [] for l in labels}

    for image in tqdm(os.listdir(image_dir)):
        data["Image"].append(image[:-4])

        ds = pydicom.dcmread(os.path.join(image_dir, image))

        for metadata in ds.dir():
            if metadata != "PixelData":
                metadata_values = getattr(ds, metadata)
                if type(metadata_values) == pydicom.multival.MultiValue and metadata not in ["WindowCenter", "WindowWidth"]:
                    for i, v in enumerate(metadata_values):
                        data[f"{metadata}_{i}"].append(v)
                else:
                    if type(metadata_values) == pydicom.multival.MultiValue and metadata in ["WindowCenter", "WindowWidth"]:
                        data[metadata].append(metadata_values[0])
                    else:
                        data[metadata].append(metadata_values)

    return pd.DataFrame(data).set_index("Image")


def build_triplets(metadata):
    metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False)
    studies = metadata.groupby("StudyInstanceUID")
    triplets = []

    for study_name, study_df in tqdm(studies):
        padded_names = np.pad(study_df.index, (1, 1), 'edge')

        for i, img in enumerate(padded_names[1:-1]):
            t = [padded_names[i], img, padded_names[i + 2]]
            triplets.append(t)

    return pd.DataFrame(triplets, columns=["red", "green", "blue"])


class CropHead(object):
    def __init__(self, offset=10):
        """
        Crops the head by labelling the objects in an image and keeping the second largest object (the largest object
        is the background). This method removes most of the headrest

        Originally made as a image transform for use with PyTorch, but too slow to run on the fly :(
        :param offset: Pixel offset to apply to the crop so that it isn't too tight
        """
        self.offset = offset

    def crop_extents(self, img):
        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            labeled_blobs, number_of_blobs = ndimage.label(img_array)
            blob_sizes = np.bincount(labeled_blobs.flatten())
            head_blob = labeled_blobs == np.argmax(blob_sizes[1:]) + 1  # The number of the head blob
            head_blob = np.max(head_blob, axis=-1)

            mask = head_blob == 0
            rows = np.flatnonzero((~mask).sum(axis=1))
            cols = np.flatnonzero((~mask).sum(axis=0))

            x_min = max([rows.min() - self.offset, 0])
            x_max = min([rows.max() + self.offset + 1, img_array.shape[0]])
            y_min = max([cols.min() - self.offset, 0])
            y_max = min([cols.max() + self.offset + 1, img_array.shape[1]])

            return x_min, x_max, y_min, y_max
        except ValueError:
            return 0, 0, -1, -1

    def __call__(self, img):
        """
        Crops a CT image to so that as much black area is removed as possible
        :param img: PIL image
        :return: Cropped image
        """

        x_min, x_max, y_min, y_max = self.crop_extents(img)

        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            return Image.fromarray(np.uint8(img_array[x_min:x_max, y_min:y_max]))
        except ValueError:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(offset={})'.format(self.offset)


def prepare_dicom(dcm, default_window=False):
    """
    Converts a DICOM object to a 16-bit Numpy array (in Housnfield units) or a uint8 image if the default window is used
    :param dcm: DICOM Object
    :param default_window: Flag to use the window settings specified in the metadata
    :return: Numpy array in either int16 or uint8
    """

    try:
        # https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        if dcm.BitsStored == 12 and dcm.PixelRepresentation == 0 and dcm.RescaleIntercept > -100:
            x = dcm.pixel_array + 1000
            px_mode = 4096
            x[x >= px_mode] = x[x >= px_mode] - px_mode
            dcm.PixelData = x.tobytes()
            dcm.RescaleIntercept = -1000

        pixels = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope + dcm.RescaleIntercept
    except ValueError as e:
        print("ValueError with", dcm.SOPInstanceUID, e)
        return np.zeros((512, 512))

    # Pad the image if it isn't square
    if pixels.shape[0] != pixels.shape[1]:
        (a, b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)

    # Return image windows as per the metadata parameters
    if default_window:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]

        level = dcm.WindowCenter
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]

        img_windowed = linear_windowing(pixels, level, width)
        return img_windowed
    # Return array Hounsfield units only
    else:
        return pixels.astype(np.int16)


def prepare_png(dataset, folder_name, channels=(0, 1, 2), crop=False):
    """
    Create PNG images using 3 specified window settings
    :param dataset: One of "train", "test_stage_1" or "test_stage_2"
    :param folder_name: Name of the output folder
    :param channels: Tuple to specifiy what windows to use for RGB channels
    :param crop: Flag to crop image to only the head
    :return:
    """

    start = time()

    image_dirs = {
        "train": os.path.join(data_path, "stage_1_train_images"),
        "test_stage_1": os.path.join(data_path, "stage_1_test_images"),
        "test_stage_2": os.path.join(data_path, "stage_2_test_images")
    }

    windows = [
        (None, None),  # No windowing
        (80, 40),  # Brain
        (200, 80),  # Subdural
        (40, 40),  # Stroke
        (2800, 600),  # Temporal bone
        (380, 40),  # Soft tissue
        (2000, 600),  # Bone
    ]

    output_path = os.path.join(data_path, "png", dataset, f"{folder_name}")
    crop_head = CropHead()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image_name in tqdm(os.listdir(image_dirs[dataset])):
        ds = pydicom.dcmread(os.path.join(image_dirs[dataset], image_name))

        rgb = []
        for c in channels:
            if c == 0:
                ch = prepare_dicom(ds, default_window=True)
            else:
                ch = prepare_dicom(ds)
                ch = linear_windowing(ch, windows[c][0], windows[c][1])
            rgb.append(ch)

        img = np.stack(rgb, -1)

        if crop:
            x_min, x_max, y_min, y_max = crop_head.crop_extents(img > 0)
            img = img[x_min:x_max, y_min:y_max]

            if img.shape[0] == 0 or img.shape[1] == 0:
                img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

        im = Image.fromarray(img.astype(np.uint8))
        im.save(os.path.join(output_path, image_name[:-4] + ".png"))

    print("Done in", (time() - start) // 60, "minutes")


def prepare_png_adjacent(dataset, folder_name, crop=True):
    """
    Prepare 3 channel adjacent images in Hounsfield units clipped between 0-255 HU
    The target image is the green channel. The reg and blue channels are spatially adjacent slices
    :param dataset: One of "train", "test_stage_1" or "test_stage_2"
    :param folder_name: Name of the output folder
    :param crop: Flag to crop image to only the head
    """
    start = time()

    triplet_dfs = {
        "train": os.path.join(data_path, "train_triplets.csv"),
        "test_stage_1": os.path.join(data_path, "stage_1_test_triplets.csv"),
        "test_stage_2": os.path.join(data_path, "stage_2_test_triplets.csv")
    }

    image_dirs = {
        "train": os.path.join(data_path, "stage_1_train_images"),
        "test_stage_1": os.path.join(data_path, "stage_1_test_images"),
        "test_stage_2": os.path.join(data_path, "stage_2_test_images")
    }

    output_path = os.path.join(data_path, "png", dataset, f"{folder_name}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    triplets = pd.read_csv(triplet_dfs[dataset])
    crop_head = CropHead()

    for _, row in tqdm(triplets.iterrows(), total=len(triplets), desc=dataset):

        rgb = []
        for ch in ["red", "green", "blue"]:
            dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row[ch] + ".dcm"))
            rgb.append(prepare_dicom(dcm))

        img = np.stack(rgb, -1)
        img = np.clip(img, 0, 255).astype(np.uint8)

        if crop:
            x_min, x_max, y_min, y_max = crop_head.crop_extents(img > 0)
            img = img[x_min:x_max, y_min:y_max]

            if img.shape[0] == 0 or img.shape[1] == 0:
                img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

        im = Image.fromarray(img)
        im.save(os.path.join(output_path, row["green"] + ".png"))

    print("Done in", (time() - start) // 60, "minutes")


def dicom_to_npy(dataset, folder_name):
    """
    Saves DICOM images as 16-bit Numpy arrays
    :param dataset: One of "train", "test_stage_1" or "test_stage_2"
    :param folder_name: Name of the output folder
    """

    image_dirs = {
        "train": os.path.join(data_path, "stage_1_train_images"),
        "test_stage_1": os.path.join(data_path, "stage_1_test_images"),
        "test_stage_2": os.path.join(data_path, "stage_2_test_images")
    }

    output_path = os.path.join(data_path, "npy", dataset, f"{folder_name}")
    print("Saving slices to", output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image_name in tqdm(os.listdir(image_dirs[dataset])):
        dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], image_name))
        np.save(os.path.join(output_path, image_name[:-4]), prepare_dicom(dcm))


def prepare_npy_adjacent(dataset, folder_name, crop=True):
    """
    Prepare 3 channel adjacent images in Hounsfield units (unclipped)
    :param dataset: One of "train", "test_stage_1" or "test_stage_2"
    :param folder_name: Name of the output folder
    :param crop: Flag to crop image to only the head
    """
    start = time()

    triplet_dfs = {
        "train": os.path.join(data_path, "train_triplets.csv"),
        "test_stage_1": os.path.join(data_path, "stage_1_test_triplets.csv"),
        "test_stage_2": os.path.join(data_path, "stage_2_test_triplets.csv")
    }

    image_dirs = {
        "train": os.path.join(data_path, "npy", dataset, "single_hu_slices"),
        "test_stage_1": os.path.join(data_path, "npy", dataset, "single_hu_slices"),
        "test_stage_2": os.path.join(data_path, "npy", dataset, "single_hu_slices")
    }

    output_path = os.path.join(data_path, "npy", dataset, f"{folder_name}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    triplets = pd.read_csv(triplet_dfs[dataset])
    crop_head = CropHead()

    for _, row in tqdm(triplets.iterrows(), total=len(triplets), desc=dataset):
        r = np.load(os.path.join(image_dirs[dataset], row["red"] + ".npy"))
        g = np.load(os.path.join(image_dirs[dataset], row["green"] + ".npy"))
        b = np.load(os.path.join(image_dirs[dataset], row["blue"] + ".npy"))
        img = np.stack([r, g, b], -1)

        if crop:
            x_min, x_max, y_min, y_max = crop_head.crop_extents(img > 0)
            img = img[x_min:x_max, y_min:y_max]

            if img.shape[0] == 0 or img.shape[1] == 0:
                img = np.zeros(shape=(512, 512, 3), dtype=np.int16)

        np.save(os.path.join(output_path, row["green"]), img)

    print("Done in", (time() - start) // 60, "minutes")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear_windowing(img, window_width, window_length):
    """
    Applies a linear window on an array
    :param img: Image array (in Hounsfield units)
    :param window_width:
    :param window_length:
    :return:
    """
    if window_width and window_length:
        lower = window_length - (window_width / 2)
        upper = window_length + (window_width / 2)
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
        return (img*255).astype(np.uint8)
    else:
        return img


def sigmoid_windowing(img, window_width, window_length, u=255, epsilon=255):
    """
    Applies a sigmoid window on an array
    From Practical Window Setting Optimization for Medical Image Deep Learning https://arxiv.org/pdf/1812.00572.pdf
    :param img: Image array (in Hounsfield units)
    :param window_width:
    :param window_length:
    :param u:
    :param epsilon:
    :return:
    """
    if window_width and window_length:
        weight = (2 / window_width) * np.log((u / epsilon) - 1)
        bias = (-2 * window_length / window_width) * np.log((u / epsilon) - 1)
        img = u * sigmoid(weight * img + bias)
        return img.astype(np.uint8)
    else:
        return img


if __name__ == '__main__':
    # Generate metadata dataframes
    train_metadata = get_metadata(os.path.join(data_path, "stage_1_train_images"))
    test_metadata = get_metadata(os.path.join(data_path, "stage_1_test_images"))
    train_metadata.to_parquet(f'{data_path}/train_metadata.parquet.gzip', compression='gzip')
    test_metadata.to_parquet(f'{data_path}/stage_1_test_metadata.parquet.gzip', compression='gzip')

    # Build triplets of adjacent images
    train_triplets = build_triplets(train_metadata)
    test_triplets = build_triplets(test_metadata)
    train_triplets.to_csv(os.path.join(data_path, "train_triplets.csv"))
    test_triplets.to_csv(os.path.join(data_path, "stage_1_test_triplets.csv"))

    # Prepare adjacent images
    prepare_png_adjacent("train", "adjacent_hu_cropped")
    prepare_png_adjacent("test_stage_1", "adjacent_hu_cropped")

    # Prepare 3 window images (brain-subdural-bone)
    prepare_png("train", "brain-subdural-bone", channels=(1, 2, 6), crop=True)
    prepare_png("test_stage_1", "brain-subdural-bone", channels=(1, 2, 6), crop=True)

    # Stage 2 preparations
    test_metadata = get_metadata(os.path.join(data_path, "stage_2_test_images"))
    test_metadata.to_parquet(f'{data_path}/stage_2_test_metadata.parquet.gzip', compression='gzip')
    test_triplets = build_triplets(test_metadata)
    test_triplets.to_csv(os.path.join(data_path, "stage_2_test_triplets.csv"))
    prepare_png_adjacent("test_stage_2", "adjacent_hu_cropped")
    prepare_png("test_stage_2", "brain-subdural-bone", channels=(1, 2, 6), crop=True)
