from flask import request, render_template, jsonify
from utils import app

import numpy as np

from utils.file_utils_valid import Dataloader

dataloader = Dataloader(app.config['VALID_CSV'], app.config['VECTORS_DIR'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        algorithm = request.json["algorithm"]

        try:
            keyword = request.json['keyword']
            if keyword:
                if algorithm == 'keyword':
                    ret = dataloader.all_search(keyword)
                    if keyword == 'all':
                        ret = dataloader.all_search()
                else:
                    ret = dataloader.genre_search(keyword)
                ret_json = ret.to_json(orient='index')
                return jsonify(results=(ret_json))

        except Exception as e:
            print(e)
            pass

    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    print('Similar Images Retrieval ', dataloader.keyword, request.json["ids"])
    # Transform the type of query id from "string" into "int"
    query_ids = np.array(list(map(int, request.json['ids'])))
    scores = dataloader.similar_image_retrieval(query_ids)

    return jsonify(results=(scores))

