import io

import numpy as np
import requests
from transformers import BertModel, BertTokenizer
from datasets import load_dataset

def main():
    embedder = Embedder()
    embedder.update_index()

class Embedder():
    def __init__(self):
        pass

    def load_new_checkpoint(self):
        # check if a new checkpoint exists and replace current one
        pass

    def dummy_encode(self, num_embeddings: int = 50_000, position: int = 0, max_size: int = 120_000):
        if position + num_embeddings >= max_size:
            return  np.random.rand(max_size - position, 768).astype('float32')
        return np.random.rand(num_embeddings, 768).astype('float32')

    def embed_shard(self):
        # 1. get start_id and length from embeddingindex server
        # 2. load new checkpoint
        # 3. encode
        # 4. push to embedding server
        pass

    def update_index(self):
        resp = requests.get("http://0.0.0.0:8000/next_update_ids")
        resp = resp.json()
        shard_size = resp["shard_size"]
        position = resp["position_idx"]

        # get data from dataset + maybe tokenize if not already done
        # encode
        dummy_embeddings = self.dummy_encode(shard_size, position=position)
        params = {'position': position}
        resp = requests.post("http://0.0.0.0:8000/replace_embeddings", params=params,
                             files={'embeddings': io.BytesIO(dummy_embeddings.tobytes())})


if __name__ == "__main__":
    main()