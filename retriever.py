import io

import numpy as np
import requests


class Retriever():

    def __init__(self):
        pass

    def retrieve_test(self):
        params = {'k': 1}
        embedding = np.random.rand(1,768).astype('float32')
        resp = requests.post("http://0.0.0.0:8000/retrieve", params=params, files={'embedding': io.BytesIO(embedding.tobytes())})
        return resp.json()

