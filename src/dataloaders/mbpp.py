from datasets import load_from_disk

def load_mbpp(split: str):
    return load_from_disk("data/mbpp_train")