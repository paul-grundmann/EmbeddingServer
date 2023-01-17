import faiss
import numpy as np
from tqdm import tqdm
def get_index():
    INDEX_SIZE = 23_975_497
    initial_embeddings = np.random.rand(INDEX_SIZE, 768).astype('float32')
    index = faiss.IndexFlatL2(768)
    index.add(initial_embeddings)
    return index
def main():
    index = get_index()
    for i in tqdm(range(1000)):
        a = index.search(np.random.rand(32,768),k=5)

if __name__ == "__main__":
    main()