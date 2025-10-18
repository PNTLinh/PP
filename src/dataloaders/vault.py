from datasets import load_from_disk

def load_vault(split: str):
    return load_from_disk("data/vault_test")