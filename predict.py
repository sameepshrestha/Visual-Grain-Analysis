### LOCAL USE ONLY ###
# This file is used to simulate the CrowdML Pipeline on your local machine when
# using the our CLI. E.g. unearthed predict

import logging
import argparse
from os import getenv
import os
import numpy as np
# from train import model_fn
# from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.preprocessing import image
import pandas as pd
import zipfile
import glob
import shutil
IMG_SIZE=224
LBL = dict(zip(['B_BSMUT1', 'B_CLEV5B', 'B_DISTO', 'B_GRMEND', 'B_HDBARL',
       'B_PICKLD', 'B_SKINED', 'B_SOUND', 'B_SPRTED', 'B_SPTMLD',
       'O_GROAT', 'O_HDOATS', 'O_SEPAFF', 'O_SOUND', 'O_SPOTMA',
       'WD_RADPODS', 'WD_RYEGRASS', 'WD_SPEARGRASS', 'WD_WILDOATS',
       'W_DISTO', 'W_FLDFUN', 'W_INSDA2', 'W_PICKLE', 'W_SEVERE',
       'W_SOUND', 'W_SPROUT', 'W_STAIND', 'W_WHITEG'], range(28)))
cls_map = dict(zip(LBL.values(),LBL.keys()))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def init_dir(pth):
    if os.path.exists(pth):
        shutil.rmtree(pth)
    os.mkdir(pth)


if __name__ == "__main__":
    """Prediction.

    The main function is only used by the Unearthed CLI.

    When a submission is made online AWS SageMaker Processing Jobs are used to perform
    preprocessing and Batch Transform Jobs are used to pass the result of preprocessing
    to the trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    args, _ = parser.parse_known_args()

    # load the model
    logger.info("loading the model")
    model = model_fn(args.model_dir)
    logger.info("Reading the test set")

    ## Mock the sagemaker api
    val_zip_file = os.path.join(args.data_dir,'val.zip')
    data_dir = "/tmp/tmp"
    init_dir(data_dir)
    with zipfile.ZipFile(val_zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    res = []
    for fn in glob.iglob(data_dir + '/**/*.png', recursive=True):
        file_name = os.path.basename(fn)
        path = os.path.abspath(fn)
        folder = os.path.split(os.path.dirname(path))[1]
        if len(file_name.split("-")) > 2:  # ignore master image with may grains, raw image names are in guid format
            im = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(im)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            pred = model.predict(img_preprocessed)
            top3 = (-pred[0]).argsort()[:3]
            res.append({'file_name': file_name, 'path': path, 'cls': folder, 'prediction':top3[0],  'proba_1':pred[0][top3[0]], 'prediction2':top3[1], 'proba_2':pred[0][top3[1]],  'prediction3':top3[2], 'proba_3':pred[0][top3[2]]})
    df = pd.DataFrame(res)
    df['prediction'] = df.prediction.map(cls_map)
    df['prediction2'] = df.prediction2.map(cls_map)
    df['prediction3'] = df.prediction3.map(cls_map)

    logger.info("creating predictions")
    df.to_csv("/opt/ml/output/public.csv.out", index=False, header=False)
