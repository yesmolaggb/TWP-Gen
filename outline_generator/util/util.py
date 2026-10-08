import pickle as pk
import os

def savePk(path: str, data: object) -> None:
    if os.path.exists(path):
        os.remove(path)
    with open(path, "wb") as f:
        pk.dump(data, f)
