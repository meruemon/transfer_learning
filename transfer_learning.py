import os

from absl import app
from absl import flags

from datetime import datetime

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD


IMG_SIZE = (299, 299)
IMG_SHAPE = IMG_SIZE + (3,)
CLASS_NUM = 16

FLAGS = flags.FLAGS
flags.DEFINE_string('images_dir', '/data/e-kikai/data/samples',
                    'Directory where dataset images are located, all in .jpg format.')
flags.DEFINE_integer('batch_size', 128,
                     'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epochs', 100,
                     'Epoch num.')
flags.DEFINE_string('output_model', None,
                    'Path where the trained model will be stored.')

def augumentation(data_root, img_size, batch_size, subset):
    gen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True,
                             width_shift_range=4.0 / 32.0, height_shift_range=4.0 / 32.0,
                             zoom_range=[0.75, 1.25], channel_shift_range=50.0,
                             rotation_range=10)
    return gen.flow_from_directory(
        str(data_root),
        target_size=img_size,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="sparse",
        shuffle=True,
        seed=42,
        subset=subset
    ), gen

def main(argv=None):

    output_model = FLAGS.output_model
    data_root = FLAGS.images_dir
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs

    train_generator_augmented, train_gen = augumentation(data_root, IMG_SIZE, batch_size, 'training')

    # create the base pre-trained model
    base_model = InceptionV3(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(train_generator_augmented.num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy",)


    # define callback
    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=1
    )

    # TensorBoard用コールバック
    log_dir_name = datetime.now().strftime('./logs/finetune_%Y%m%d%I%M%S')
    tb_callback = TensorBoard(log_dir="./logs/finetune-augmented", histogram_freq=1, write_graph=True)

    history = model.fit_generator(
        train_generator_augmented,
        epochs=epochs,
        verbose=1,
        shuffle=True,
        callbacks=[
            cp_callback,
            tb_callback,
        ]
    )

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss="sparse_categorical_crossentropy",)

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history = model.fit_generator(
        train_generator_augmented,
        epochs=epochs,
        verbose=1,
        shuffle=True,
        callbacks=[
            cp_callback,
            tb_callback,
        ]
    )

    # save trained model
    model.save(output_model)

if __name__ == '__main__':
    app.run(main)
