# -*- coding: utf-8 -*-

'''
Calculate image feature vectors

Usage:
  python process_images.py --image_files="image_dir"

Inception-v3特徴を算出する．

使い方：

--image_filesで指定したフォルダに存在する画像（jpg）を読み込んで
特徴量を算出する．算出した特徴量は「（画像名）.npy」で保存される．

次で指定したディレクトリ（例では'static'）に「image_vectors」が自動で
作成されて，その中に特徴量が保存される．

flags.DEFINE_string('output_folder', 'static', 'The folder where output files will be stored')

実行例：
画像ファイルを「static/img」に保存したとする．

--image_files="static/img"

process_images.py
- static - images_vectors - (image name).npy
         - img - (image name).jpg
'''

from os.path import join
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import sys
import subprocess

# configure command line interface arguments
flags = tf.app.flags
flags.DEFINE_string('model_dir', './saved_models/', 'The location of downloaded imagenet model')
flags.DEFINE_string('image_files', '/data/e-kikai/data/valid_img/', 'A glob path of images to process')
flags.DEFINE_boolean('validate_images', False, 'Whether to validate images before processing')
flags.DEFINE_string('output_folder', '/data/e-kikai/CBIR/utils/static/', 'The folder where output files will be stored')
flags.DEFINE_string('model_name', 'freezed_model_50.pb', 'The name of weight file')
FLAGS = flags.FLAGS

IMG_HEIGHT_V3 = 299
IMG_WIDTH_V3 = 299
IMG_CHANNEL_V3 = 3

types = ['jpg', 'JPG', 'jpeg', 'JPEG']

class FeatureExtractor:
    def __init__(self, image_glob):
        print(' * writing outputs for ' + str(len(image_glob)) +
              ' images to folder ' + FLAGS.output_folder)

        self.image_files = image_glob
        self.output_dir = FLAGS.output_folder
        self.model_name = FLAGS.model_name
        self.errored_images = set()
        self.vector_files = []
        self.image_vectors = []
        self.rewrite_image_vectors = False
        self.validate_inputs(FLAGS.validate_images)
        self.create_image_vectors()
        print('Processed output for ' + \
              str(len(self.image_files) - len(self.errored_images)) + ' images')

    def validate_inputs(self, validate_files):
        '''
        Make sure the inputs are valid, and warn users if they're not
        '''
        if not validate_files:
            print(' * skipping image validation')
            return

        # test whether each input image can be processed
        print(' * validating input files')
        invalid_files = []
        for i in self.image_files:
            try:
                cmd = get_magick_command('identify') + ' "' + i + '"'
                response = subprocess.check_output(cmd, shell=True)
            except Exception as exc:
                invalid_files.append(i)
        if invalid_files:
            message = '\n\nThe following files could not be processed:'
            message += '\n  ! ' + '\n  ! '.join(invalid_files) + '\n'
            message += 'Please remove these files and reprocess your images.'
            print(message)
            sys.exit()

    def create_image_vectors(self):
        '''
        Create one image vector for each input file
        '''
        graph = self.create_tf_graph()

        print(' * creating image vectors')
        with tf.compat.v1.Session(graph=graph) as sess:
            input_tensor = sess.graph.get_tensor_by_name('input_1:0')
            feature_tensor = sess.graph.get_tensor_by_name('global_average_pooling2d/Mean:0')
            holder_image = tf.placeholder(tf.string)
            processed_image = tf.image.decode_jpeg(holder_image, channels=IMG_CHANNEL_V3)
            processed_image = preprocess_for_eval(processed_image, IMG_HEIGHT_V3, IMG_WIDTH_V3)
            for image_index, image in enumerate(tqdm(self.image_files)):
                outfile_name = os.path.basename(image) + '.npy'
                out_path = join(self.output_dir, outfile_name)
                # if os.path.exists(out_path):
                #    continue
                try:
                    with tf.gfile.FastGFile(image, 'rb') as f:
                        # cmd = get_magick_command('identify') + ' ' + image
                        # response = subprocess.check_output(cmd, shell=True)
                        input_image = sess.run(processed_image, {holder_image: f.read()})
                    data = {input_tensor: input_image}
                    feature_vector = np.squeeze(sess.run(feature_tensor, data))
                    np.save(out_path, feature_vector)
                except Exception as exc:
                    self.errored_images.add(get_filename(image))
                    print(' * image', get_ascii_chars(image), 'is collapsed', exc)
                    f.close()

    def create_tf_graph(self, verbose=False):
        '''
        Create a graph from the saved graph_def.pb
        '''
        print(' * creating tf graph')
        graph_path = join(FLAGS.model_dir, self.model_name)

        with tf.io.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )

            [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
            if verbose:
                # [print(n.name) for n in tf.compat.v1.get_default_graph().as_graph_def().node]
                [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        return graph


def get_magick_command(cmd):
    '''
    Return the specified imagemagick command prefaced with magick if
    the user is on Windows
    '''
    if os.name == 'nt':
        return 'magick ' + cmd
    return cmd


def get_ascii_chars(s):
    '''
    Return a string that contains the ascii characters from string `s`
    '''
    return ''.join(i for i in s if ord(i) < 128)


def get_filename(path):
    '''
    Return the root filename of `path` without file extension
    '''
    return os.path.splitext(os.path.basename(path))[0]


def ensure_dir_exists(directory):
    '''
    Create the input directory if it doesn't exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocess_for_eval(image,
                        height,
                        width,
                        central_fraction=0.875,
                        scope=None,
                        central_crop=False,
                        use_grayscale=False):

    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if use_grayscale:
          image = tf.image.rgb_to_grayscale(image)
        if central_crop and central_fraction:
          image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
          # Resize the image to the specified height and width.
          image = tf.expand_dims(image, 0)
          image = tf.image.resize_bilinear(image, [height, width],
                                           align_corners=False)
          image = tf.squeeze(image, [0])
        # image = tf.subtract(image, 0.5)
        # image = tf.multiply(image, 2.0)
        image = tf.expand_dims(image, 0)

        return image

def main(*args, **kwargs):
    '''
    The main function to run
    '''

    if FLAGS.image_files:
        image_glob = []
        for ext in types:
            img_dir = FLAGS.image_files + '/*.%s' % ext
            image_glob.extend(glob(img_dir))

    elif len(sys.argv) == 2:
        image_glob = glob(sys.argv[1])

    elif len(sys.argv) > 2:
        image_glob = sys.argv[1:]

    else:
        print('Please specify a glob path of images to process\n' +
              'e.g. python3.6 process_images.py "folder/*.jpg"')

    print('%d images can be found!' % len(image_glob))
    FeatureExtractor(image_glob)


if __name__ == '__main__':
    tf.app.run()
