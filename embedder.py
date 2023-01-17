import io
import os

import numpy as np
import requests
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from datasets import load_dataset
import torch
import torch.utils.data
from transformers import DataCollatorWithPadding


def main():
    embedder = Embedder()
    embedder.update_index()

class Embedder():
    def __init__(self,
                 initial_checkpoint="bert-base-uncased",
                 tokenizer_name="bert-base-uncased",
                 dataset_path = ""):
        self.model = AutoModel.from_pretrained(initial_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if dataset_path != "":
            self.dataset = load_dataset(dataset_path)
            # apply tokenizer or use already tokenized dataset

    def dummy_encode(self, num_embeddings: int = 50_000, position: int = 0, max_size: int = 120_000):
        if position + num_embeddings >= max_size:
            return  np.random.rand(max_size - position, 768).astype('float32')
        return np.random.rand(num_embeddings, 768).astype('float32')

    def embed_shard(self, position: int, shard_size: int):
        collator = DataCollatorWithPadding(self.tokenizer, max_length=int(os.getenv("MAX_SEQ_LEN", "512")),pad_to_multiple_of=64)
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=int(os.getenv("BATCHSIZE", "512")),
                                                 shuffle=False,
                                                 num_workers=int(os.getenv("NUM_WORKERS", "0")),
                                                 collate_fn=collator)
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for batch in dataloader:

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