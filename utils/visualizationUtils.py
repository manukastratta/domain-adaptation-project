import matplotlib.pyplot as plt
import random
from PIL import Image
import fire
import pandas as pd

def visualize_image_samples(metadata_file="data/CelebA/splits/debug_setting1_val.csv"):
    df = pd.read_csv(metadata_file)
    df = df.sample(n=20)
    image_paths = list(df["img_filename"])
    for i in range(len(image_paths)):
        image_paths[i] = "data/CelebA/img_align_celeba/" + image_paths[i]

    # Create a figure with 4 rows and 5 columns for visualization
    fig, axes = plt.subplots(4, 5, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        # Read the image using PIL
        image = Image.open(image_paths[i])

        # Display the image on the corresponding subplot
        ax.imshow(image)
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

if __name__ == "__main__":
    fire.Fire()