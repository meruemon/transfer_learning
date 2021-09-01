# -*- coding: utf-8 -*-

'''
Calculate image feature vectors

Usage:
  python process_images.py --image_files="data/*/*.jpg"

Inception-v3特徴を算出する．

使い方：

--image_filesで指定したフォルダに存在する画像（jpg）を読み込んで
特徴量を算出する．算出した特徴量は「（画像名）.npy」で保存される．

次で指定したディレクトリ（例では'static'）に「vectors」が自動で
作成されて，その中に特徴量が保存される．

flags.DEFINE_string('output_folder', 'static', 'The folder where output files will be stored')

実行例：
画像ファイルを「static/img」に保存したとする．

python process_images.py --image_files="static/img/*.jpg"

'''

from six.moves import urllib
from os.path import join
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import sys
import tarfile
import subprocess

# configure command line interface arguments
flags = tf.app.flags
flags.DEFINE_string('model_dir', './saved_models/', 'The location of downloaded imagenet model')
flags.DEFINE_string('image_files', '/home/student/Programs/transfer_learning/CBIR/utils/static/img', 'A glob path of images to process')
flags.DEFINE_boolean('validate_images', False, 'Whether to validate images before processing')
flags.DEFINE_string('output_folder', '/home/student/Programs/transfer_learning/CBIR/utils/static/', 'The folder where output files will be stored')
FLAGS = flags.FLAGS

types = ['jpg', 'JPG', 'jpeg', 'JPEG']

_MODEL_NAME = 'classify_image_graph_def.pb'

class FeatureExtractor:
    def __init__(self, image_glob):
        print(' * writing outputs for ' + str(len(image_glob)) +
              ' images to folder ' + FLAGS.output_folder)

        self.image_files = image_glob
        self.output_dir = FLAGS.output_folder
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
        self.download_inception()
        graph = self.create_tf_graph()

        print(' * creating image vectors')

        with tf.Session(graph=graph) as sess:
            feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            for image_index, image in enumerate(tqdm(self.image_files)):
                try:
                    outfile_name = os.path.basename(image) + '.npy'
                    out_path = join(self.output_dir, outfile_name)
                    if os.path.exists(out_path) and not self.rewrite_image_vectors:
                        continue
                    with tf.gfile.FastGFile(image, 'rb') as f:
                        data = {'DecodeJpeg/contents:0': f.read()}
                    feature_vector = np.squeeze(sess.run(feature_tensor, data))
                    np.save(out_path, feature_vector)
                except Exception as exc:
                    self.errored_images.add(get_filename(image))
                    print(' * image', get_ascii_chars(image), 'is collapsed', exc)
                    input_image = None
                    del input_image
                    f.close()

    def download_inception(self):
        '''
        Download the inception model to FLAGS.model_dir
        '''
        print(' * verifying inception model availability')
        ## inception-v3
        inception_path = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        dest_directory = FLAGS.model_dir
        ensure_dir_exists(dest_directory)
        filename = inception_path.split('/')[-1]
        filepath = join(dest_directory, filename)
        if not os.path.exists(filepath):
            def progress(count, block_size, total_size):
                percent = float(count * block_size) / float(total_size) * 100.0
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, percent))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(inception_path, filepath, progress)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def create_tf_graph(self):
        '''
        Create a graph from the saved graph_def.pb
        '''
        print(' * creating tf graph')
        graph_path = join(FLAGS.model_dir, _MODEL_NAME)
        with tf.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.GraphDef()
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

    FeatureExtractor(image_glob)


if __name__ == '__main__':
    tf.app.run()