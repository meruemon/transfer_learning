from flask import Flask

app = Flask(__name__)
app.config.from_object('utils.config')

import utils.views