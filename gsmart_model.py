import os
import sys
import skimage.io
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from time import time 
import matplotlib.pyplot as plt
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import get_size
from mrcnn.config import Config
import matplotlib
from tensorflow.python.keras.backend import set_session
import scipy.io
import scipy.ndimage
matplotlib.use('Agg')



# Global PATH
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "gsmart_zy_ver02.h5")
IMAGE_sav_DIR = os.path.join(ROOT_DIR, "test results")

# Global parameters
dpi_to_mm = 25.4/300*3
# Session config
sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)
image = fname = None


def load_image(image_path):
    global image,fname
    image = skimage.io.imread(image_path)
    print('image to be detected: ', image_path)
    fname = os.path.join(IMAGE_sav_DIR, image_path)


def create_model(weight_path=COCO_MODEL_PATH):
    class InferenceConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "shapes"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 3  # background + 3 shapes

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 500

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 100

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 50

#        IMAGE_MIN_DIM = (1800//64)*64
#       IMAGE_MAX_DIM = (1800//64)*64

    # model config
    model_config = InferenceConfig()

    # Create model object in inference mode.
    global model
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=model_config)

    print('loading model weight...')
    # Load weights trained on MS-COCO
    model.load_weights(weight_path, by_name=True)




def run_model():
    print('running model...')
    start_time = time()
    with graph.as_default():
        set_session(sess)
        results = model.detect([image], verbose=1)
    print("Time for detection: ", (time() - start_time))
    return results[0]


def get_kernel_numbers(result):
    Kernel_class_id = result['class_ids']
    #print('class ID:', Kernel_class_id)

    kernel_no=Kernel_class_id.shape[0]                                  #1   -table
    print('Kernel_number: ', kernel_no)

    kernel_husk_no=np.sum(np.where(Kernel_class_id == 1, 1, 0))             #2 -table
    print('Kernel_husk_number: ', kernel_husk_no)

    kernel_bran_no=np.sum(np.where(Kernel_class_id == 2, 1, 0))                #3 -table
    print('Kernel_bran_number: ', kernel_bran_no)

    kernel_milled_no=np.sum(np.where(Kernel_class_id == 3, 1, 0))             #4 -table
    print('Kernel_milled_number: ', kernel_milled_no)
    return np.int64(kernel_no), kernel_husk_no, kernel_bran_no, kernel_milled_no


def get_chalky_mask(result):
    final_mask = result['masks']
    if fname is None or image is None:
        print('image not loaded')
    mask_chky = visualize.abnormal_mask(fname, image, final_mask)   # chalky mask   4         $$$$$$conv
    print('chalky mask shape:', mask_chky.shape)
    return mask_chky


def get_kernel_ratio(result, kernel_no, mask_chky):
    kernel_length = []      #array of kernel length           $$$$$$$$$$$$$$$$$            #5
    kernel_width = []       #array of kernel width                                         #6
    kernel_length_to_width = []       #array of kernel length/width                        #7
    final_mask = result['masks']
    
    mask_chky = scipy.ndimage.interpolation.zoom(input=mask_chky, zoom=[1/3, 1/3], order = 0)
    final_mask = scipy.ndimage.interpolation.zoom(input=final_mask, zoom=[1/3, 1/3, 1], order = 0)

    
    for i in range(kernel_no):
        kernel_size = get_size.get_min_max_feret_from_mask(final_mask[:, :, i])

        kernel_width.append(round(kernel_size[0] * dpi_to_mm, 2))
        kernel_length.append(round(kernel_size[1] * dpi_to_mm, 2))
        kernel_length_to_width.append(round(kernel_size[1]/kernel_size[0], 2))

    kwargs = dict(alpha=0.5, bins=50, density=True, stacked=True)                              #8   3 - plots
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), dpi=100, sharex=True, sharey=True)
    axes[0].hist(kernel_length, **kwargs, color='g', label='Length')
    axes[0].set_title('Kernel length distribution')
    axes[0].set_xlabel('Length(mm)')
    axes[0].set_ylabel('Kernel number')

    axes[1].hist(kernel_width, **kwargs, color='b', label='Width')
    axes[1].set_title('Kernel width distribution')
    axes[1].set_xlabel('Width(mm)')
    axes[1].set_ylabel('Kernel number')

    axes[2].hist(kernel_length_to_width, **kwargs, color='r', label='Length/Width')
    axes[2].set_title('Length/width ratio')
    axes[2].set_xlabel('L/W ratio')
    axes[2].set_ylabel('Kernel number')

    # axes.set_xlim(0, 10); ax.set_ylim(0, 1);
    plt.tight_layout()
    plt.show()    #---show image
    plt.savefig('static/img/output/distribution.jpg')

    kernel_chalky_ratio = np.sum(np.transpose(final_mask, (2, 0, 1)) * mask_chky, axis=(1, 2)) / np.sum(final_mask, axis=(0, 1))

    final_chalky_ratio = round(np.sum(np.where(np.array(kernel_chalky_ratio) >= 0.5, 1, 0)) / kernel_no, 6)
    print('final chalky ratio: ', final_chalky_ratio)  # 9  -table

    kernel_length_nm = np.array(kernel_length) / np.array(kernel_length).max()

    whole_kernel_ratio = round(np.sum(np.where(kernel_length_nm >= 0.99, 1, 0)) / kernel_no, 6)
    print('whole kernel ratio: ', whole_kernel_ratio)  # 10  -table

    head_kernel_ratio = round(np.sum(np.where((0.99 > kernel_length_nm) & (kernel_length_nm >= 0.8), 1, 0)) / kernel_no,
                              6)
    print('head rice ratio: ', head_kernel_ratio)  # 11  -table

    bb_kernel_ratio = round(np.sum(np.where((0.8 > kernel_length_nm) & (kernel_length_nm >= 0.5), 1, 0)) / kernel_no, 6)
    print('big broken rice ratio: ', bb_kernel_ratio)  # 12  -table

    broken_kernel_ratio = round(np.sum(np.where(kernel_length_nm < 0.5, 1, 0)) / kernel_no, 6)
    print('Broken kernel ratio: ', broken_kernel_ratio)  # 13  -table

    return axes, kernel_length, kernel_width, kernel_length_to_width, [final_chalky_ratio, whole_kernel_ratio, head_kernel_ratio, bb_kernel_ratio, broken_kernel_ratio]


create_model()

# # for model debugging -Zhiyong
# IMAGE_file = os.path.join(ROOT_DIR, "images/300dpi_test.jpg")
# load_image(IMAGE_file)
# result = run_model()
# mask_chky = get_chalky_mask(result)
# kernel_no, kernel_husk_no, kernel_bran_no, kernel_milled_no = get_kernel_numbers(result)
# axes, kernel_length, kernel_width, kernel_length_to_width, [final_chalky_ratio, whole_kernel_ratio, head_kernel_ratio, bb_kernel_ratio, broken_kernel_ratio] = get_kernel_ratio(result, kernel_no, mask_chky)
