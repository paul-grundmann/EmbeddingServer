import os

import faiss
import numpy as np
import uvicorn
from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI, File

INDEX_SIZE = 1200_000
initial_embeddings = np.random.rand(INDEX_SIZE, 768).astype('float32')
index = faiss.IndexFlatL2(768)
# build the index
index.add(initial_embeddings)
shard_size = int(os.getenv("SHARDSIZE", "50000"))

app = FastAPI()
app.state.index = index
app.state.position_id = 0

@app.on_event("startup")
def startup():
    print("start")
    RunVar("_default_thread_limiter").set(CapacityLimiter(1))

@app.post("/retrieve")
def retrieve(embedding: bytes = File(), k: int = 1):
    print("Retrieve")
    embedding = np.frombuffer(embedding, 'float32').reshape(-1,768)
    _, I = app.state.index.search(embedding, k)  # return k nearest ids
    return I.tolist()


@app.get("/next_update_ids")
def next_update_ids():
    print("Old index: ", app.state.position_id)
    res = {"position_idx": app.state.position_id, "shard_size": shard_size}
    app.state.position_id += shard_size
    if app.state.position_id >= app.state.index.ntotal:
        app.state.position_id = 0
    print("New index: ", app.state.position_id)
    return res


@app.post("/replace_embeddings")
def replace_embeddings(embeddings: bytes = File(), position: int = 0):
    print("Replace embeddings at ", position)
    embeddings = np.frombuffer(embeddings, 'float32').reshape(-1,768)
    initial_embeddings[position:position+len(embeddings)] = embeddings
    app.state.index = faiss.IndexFlatL2(768)
    app.state.index.add(initial_embeddings)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,workers=1)
