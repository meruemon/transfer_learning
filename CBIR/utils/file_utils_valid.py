# -*- coding: utf-8 -*-
import os
import argparse

from tqdm import tqdm

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Visual Reranking')
parser.add_argument('--input_csv', default='utils/static/csv/tools.csv', type=str, help='input csv file')
parser.add_argument('--query', default='エンドミル', type=str, help='query keyword to search from db')
parser.add_argument('--static_folder', default='utils/static', type=str, help='output folder')

args = parser.parse_args()

def validate_inputs(image_files, vectors_dir):
    '''
    Validate whether image features corresponding to input image files
    has been already extracted. If not, extract features corresponding to them.
    :param image_files: list
    :return: None
    '''
    print(' * validating input files')
    invalid_files = []
    for i in tqdm(image_files):
        if not os.path.exists('%s/%s/%s.npy' % (args.static_folder, vectors_dir, i)):
            invalid_files.append(i)
    if invalid_files:
        message = '\n\nThe following files could not be processed:'
        message += '\n  ! ' + '\n  ! '.join(invalid_files) + '\n'
        message += 'Please extract image features from these files.'
        print(message)

    return invalid_files

def load_image_vectors(image_files, vectors_dir):
    '''
    Return all image vectors.
    :param image_files: list
    :return image_vectors: list
    '''
    print(' * loading image vectors')
    image_vectors = []
    import os.path
    for i in tqdm(image_files):
        name, ext = os.path.splitext(i)
        if(ext == '.'):
            i = name
        vector_file = '%s/%s/%s.npy' % (args.static_folder, vectors_dir, i)
        image_vectors.append(np.load(vector_file))

    return image_vectors


class Dataloader():
    def __init__(self, input_csv, vectors_dir):
        self.table = self.load_csv(input_csv)
        self.vectors_dir = vectors_dir
        self.names = []
        self.image_files = []
        self.image_vectors = []
        self.idx = []
        self.sorted_idx = []
        self.argsort_ids = []
        self.sorted_dists = []
        self.total_num = 0
        self.keyword = ''

    def load_csv(self, csv_file):

        with open(csv_file) as f:
            table = pd.read_table(f, sep=',', index_col='id',
                                  usecols=['id', 'no', 'name', 'genre_id', 'maker', 'top_img', 'model', 'spec'],
                                  lineterminator='\n', encoding='utf8')

        table = table.dropna(subset=['name', 'genre_id', 'top_img'])

        return table

    def all_search(self, keyword):
        self.keyword = keyword

        ret = self.table

        # Error check
        self.image_files = ret['top_img'].values.tolist()
        invalid_files = validate_inputs(self.image_files, self.vectors_dir)

        ret = ret[~ret['top_img'].isin(invalid_files)]
        self.image_files = ret['top_img'].values.tolist()
        self.names = ret['name'].values.tolist()

        self.total_num = ret.count()

        self.image_vectors = load_image_vectors(self.image_files, self.vectors_dir)

        self.idx = list(ret.index.values)

        return self.table[self.table['name'].str.contains(keyword)]


    def keyword_search(self, keyword):
        self.keyword = keyword
        # Keyword search with partial match
        ret = self.table[self.table['name'].str.contains(keyword)]

        # Error check
        self.image_files = ret['top_img'].values.tolist()
        invalid_files = validate_inputs(self.image_files)

        ret = ret[~ret['top_img'].isin(invalid_files)]
        self.image_files = ret['top_img'].values.tolist()

        self.total_num = ret.count()

        self.image_vectors = load_image_vectors(self.image_files)

        self.idx = list(ret.index.values)

        return ret


    def genre_search(self, keyword):
        self.keyword = keyword
        # Keyword search with partial match
        ret = self.table[self.table['name'].str.contains(keyword)]
        self.idx = list(ret.index.values)

        genre = ret['genre_id'].values.tolist()
        genre = list(set(genre))

        ret_genre = self.table[self.table['genre_id'].isin(genre)]
        ret_genre = ret_genre.sample(frac=1)

        # Error check
        self.image_files = ret_genre['top_img'].values.tolist()
        invalid_files = validate_inputs(self.image_files)

        self.image_files = [e for e in self.image_files if e not in invalid_files]
        ret_genre = ret_genre[~ret_genre['top_img'].isin(invalid_files)]
        self.image_files = ret_genre['top_img'].values.tolist()

        self.total_num = ret_genre.count()

        self.image_vectors = load_image_vectors(self.image_files)

        self.idx = list(ret_genre.index.values)

        return ret

    def similar_image_retrieval(self, query_ids):

        all_ids = np.array(list(map(int, self.idx)))
        ind = [np.where(all_ids == i)[0][0] for i in query_ids]

        # Calculate cosine similarities between images and query images
        image_vectors = np.array(self.image_vectors)
        query_vectors = image_vectors[ind]

        # Calculate mean query vectors
        query_vectors = np.mean(query_vectors, axis=0)

        # Euclidean distance
        dists = np.linalg.norm(image_vectors - query_vectors, axis=1)  # Do search
        ids = np.argsort(dists)  # Top results
        self.argsort_ids = ids
        self.sorted_idx = [all_ids[id] for id in ids]
        self.sorted_dists = [dists[id] for id in ids]

        image_files = list(map(lambda x: 'static/img/' + x, self.image_files))
        scores = [{"image": str(image_files[id]), "score": str(self.names[id])}
                  for id in ids]

        return scores




