### TRAINING ###
# This is where the majority of your code should live for training the model. #

import argparse
import logging
import sys
import os
from os import getenv
from os.path import abspath, basename, split,dirname
import tarfile
from shutil import copyfile
import numpy as np
# from tensorflow.python.ops.numpy_ops.np_config
# np_config.enable_numpy_behavior()
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# tf.experimental.numpy.experimental_enable_numpy_behavior(
#     prefer_float32=True
# )
from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.applications.densenet import preprocess_input
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
IMG_SIZE = 224
LBL = dict(zip(['B_BSMUT1', 'B_CLEV5B', 'B_DISTO', 'B_GRMEND', 'B_HDBARL',
       'B_PICKLD', 'B_SKINED', 'B_SOUND', 'B_SPRTED', 'B_SPTMLD',
       'O_GROAT', 'O_HDOATS', 'O_SEPAFF', 'O_SOUND', 'O_SPOTMA',
       'WD_RADPODS', 'WD_RYEGRASS', 'WD_SPEARGRASS', 'WD_WILDOATS',
       'W_DISTO', 'W_FLDFUN', 'W_INSDA2', 'W_PICKLE', 'W_SEVERE',
       'W_SOUND', 'W_SPROUT', 'W_STAIND', 'W_WHITEG'], range(28)))
cls_map = dict(zip(LBL.values(),LBL.keys()))

model_version = '001'

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))

def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

img_augmentation = Sequential(
                                        [layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                                         layers.experimental.preprocessing.RandomRotation(0.3),
                                         layers.experimental.preprocessing.RandomContrast(0.5),
                                         layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1,interpolation='bilinear')],  
    name="img_augmentation")

def train(args):
    global NUM_CLASSES
    """Train
    """
    logger.info("calling training function")

    ### Build the tensorflow dataset from downloaded files
    ds_build_cmd = "tfds build grains --manual_dir=" + args.data_dir
    logger.info(ds_build_cmd)
    os.system(ds_build_cmd)

    logger.info("Training Model")
    strategy = tf.distribute.MirroredStrategy()
    batch_size = 32

    dataset_name = "grains"
    (ds_train, ds_test), ds_info = tfds.load(dataset_name, split=["train", "val"], with_info=True, as_supervised=True)
    NUM_CLASSES = ds_info.features["label"].num_classes

    size = (IMG_SIZE, IMG_SIZE)
    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size,method='bicubic'), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size,method='bicubic'), label))
    logger.info("number of classes is " + str(NUM_CLASSES))

    ds_train = ds_train.map( input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
    ds_train = ds_train.map(lambda image,labels :(img_augmentation(image,training=True),labels),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.map(lambda image,label:(tf.keras.applications.efficientnet.preprocess_input(image),label))


    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

    with strategy.scope():
        model = build_model(num_classes=NUM_CLASSES)

    # This comes from an environment variable so we can set it to 1 for our
    # development pipeline. Feel free to set this to any value.
    epochs = int(os.getenv('UNEARTHED_EPOCHS',28))
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2,callbacks=[CustomLearningRateScheduler(lr_scheduler)])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=.9)#change in the optimizer
    model.compile(
        optimizer=optimizer, loss=custom_loss, metrics=["accuracy"]
    )
    epochs = int(os.getenv('UNEARTHED_EPOCHS',6))
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)

    save_model(model,args.model_dir)

from tensorflow.keras.callbacks import LearningRateScheduler,Callback
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nbatch %05d: Learning rate is %8f." % (epoch, scheduled_lr))
def lr_scheduler(epoch,lr):
    
    lr = lr/1.25
    return lr


loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
NUM_CLASSES =28
def custom_loss(y_true, y_pred):
    def weights(x):
        y= [1,1,1,1,1,1,1,8.94,1,1,1,1,1,8.94,1,1,1,1,1,1,1,1,1,1,8.94,1,1,1]
        x = tf.cast(x,dtype=tf.float32)
        y= tf.math.multiply(y,x)
        y= tf.reduce_sum(y)
        return y
    x=[]
    for i in range(y_true.shape[0]):
        y = weights (y_true[i,:])
        x.append(y)
    losses = loss(y_true, y_pred)
    return tf.nn.compute_average_loss(losses, global_batch_size=32,sample_weight=x)

def build_model(num_classes):

    input = layers.Input((IMG_SIZE,IMG_SIZE,3))
    model2 = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False, weights='imagenet', input_tensor=input
        )
    model2.training =False
    x = layers.GlobalAveragePooling2D()(model2.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(28, activation="softmax", name="pred")(x)
    model = tf.keras.Model(input, outputs, name="EfficientNet")
    print(model.summary())
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.003,epsilon=.1)#change in the optimizer
    model.compile(
        optimizer=optimizer, loss=custom_loss, metrics=["accuracy"]
    )
    return model




def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    sm_model_dir = os.path.join(getenv('SM_MODEL_DIR'), model_version)
    logger.info(f" model dir is {model_dir}")
    model.save(sm_model_dir)
    
    modelPath = os.path.join(sm_model_dir, 'output')
    if (not os.path.isdir(modelPath)):
        os.makedirs(modelPath)
    if (not os.path.isdir(getenv('SM_MODEL_DIR') + '/code')):
        os.makedirs(getenv('SM_MODEL_DIR') + '/code')

    # Move inference.py so it gets picked up in the archive
    copyfile(os.path.dirname(os.path.realpath(__file__)) + '/inference.py', getenv('SM_MODEL_DIR') + '/code/inference.py')
    copyfile(os.path.dirname(os.path.realpath(__file__)) + '/inference-requirements.txt', getenv('SM_MODEL_DIR') + '/code/requirements.txt')

    with tarfile.open(os.path.join(modelPath, 'model.tar.gz'), mode='x:gz') as archive:
        archive.add(sm_model_dir, recursive=True)

def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    model_filepath = os.path.join(model_dir,  model_version)
    logger.info("loading model from " + model_filepath)
    model = keras.models.load_model(model_filepath)
    return model

if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.
    
    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
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
    train(parser.parse_args())
