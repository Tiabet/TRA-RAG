import numpy as np
import json

def inspect_paths():
    print("Loading embeddings...")
    data = np.load('HotpotQA/path_embeddings.npz', allow_pickle=True)
    titles = data['titles']
    key_paths = data['key_paths']
    values = data['values']
    
    target = "Julian McMahon"
    print(f"\nSearching for paths for: {target}")
    
    found = False
    for i, title in enumerate(titles):
        if title == target:
            found = True
            print(f"\nIndex: {i}")
            print(f"Key Path: {key_paths[i]}")
            print(f"Value: {values[i]}")
            
    if not found:
        print("No paths found for this title.")

if __name__ == "__main__":
    inspect_paths()
