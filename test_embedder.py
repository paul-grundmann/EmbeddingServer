import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process

from embedder import Embedder
from retriever import Retriever
import random
import time

def f_embed(i):
    embedder = Embedder()
    print("embed: ", i)
    for i in range(100):
        embedder.update_index()
        time.sleep(random.uniform(0.1,0.2))

def f_retrieve(i):
    retriever = Retriever()
    print("retrieve: ",i)
    for i in range(10000):
        retriever.retrieve_test()
        time.sleep(random.uniform(0.1,0.2))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(mp_context=ctx,max_workers=10) as pool:
        for i in range(10):
            pool.submit(f_retrieve, i)
            pool.submit(f_embed, i)
    pool.shutdown()