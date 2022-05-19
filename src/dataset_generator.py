import pandas as pd
import os
from tqdm import tqdm
from imutils.paths import list_images


def build_dataset(directory: str, output_dir):

    all_images = list(list_images(directory))

    data = []
    for path in tqdm(all_images):

        if "normal" in path:
            label = 0
        elif "colitis" in path:
            label = 1
        else:
            label = 2

        data.append({"path": path, "label": label})

    data = pd.DataFrame(data)
    data.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)
    print(data.label.value_counts())

if __name__ == "__main__":
    directory = "/home/leonardo/Codes/ViT-bowel-disease-diagnosis/data"
    build_dataset(directory, directory)