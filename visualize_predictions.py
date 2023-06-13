import json
import pandas as pd
import fire
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_10_images(df, incorrect, exp_dir):
    image_names = list(df["img_filename"])
    #img_dir = Path("data/CelebA/img_align_celeba")
    img_dir = Path("celebAdata/img_align_celeba")

    # Create a grid of subplots
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))

    # Iterate over the image paths and display each image in a subplot
    for i, path in enumerate(image_names):
        path = img_dir / path
        # Load the image
        img = mpimg.imread(path)

        # Get the corresponding subplot index
        row = i // 5
        col = i % 5

        # Display the image in the subplot
        axs[row, col].imshow(img)
        axs[row, col].axis('off')

        # Add the textbox below the image
        conf = round(df["conf"].iloc[i], 2)
        pred = df["prediction"].iloc[i]
        typ = "Wrong" if incorrect else "Correct"
        axs[row, col].text(0.5, -0.1, f"{typ} pred: {pred}, conf: {conf}", transform=axs[row, col].transAxes, fontsize=10, ha='center')

    plt.subplots_adjust(hspace=0.3)  # Adjust the vertical spacing between the rows

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the grid of images
    plt.savefig(exp_dir / f"visualize_{typ}_pred.png")

def visualize_predictions(  exp_dir,
                            pred_filename,
                            vis_incorrect=True,
                            vis_confident=True):
    exp_dir = Path(exp_dir)
    pred_path = exp_dir / pred_filename
    with open(pred_path) as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    if "conf" in df.columns:
        df = df[['img_filename', 'prediction', 'conf', 'ground_truth']]
    else:
        df = df[['img_filename', 'prediction', 'sigmoid_output', 'ground_truth']]
    
    def create_correct(row):
        return row["prediction"] == row["ground_truth"]
    df["correct"] = df.apply(create_correct, axis=1)

    if "conf" not in df.columns:
        def create_conf_scaled(row):
            if row["sigmoid_output"] > 0.5:
                # scale from 0.5-1 to 0-1
                return (row["sigmoid_output"] - 0.5) * 2
            else:
                # scale from 0-0.5 to 0-1
                return row["sigmoid_output"] * 2
            
        df["conf"] = df.apply(create_conf_scaled, axis=1)
    
    df = df.sort_values(by="conf", ascending=False)
    
    if vis_incorrect:
        incorrect = df.loc[df["correct"] == False]
        # Save bad predictions to file 
        (incorrect.iloc[:10]).to_csv(exp_dir / "bad_confident_predictions.csv", index=False)
        display_10_images(incorrect.iloc[:10], vis_incorrect, exp_dir)
    else:
        correct = df.loc[df["correct"] == True]
        display_10_images(correct.iloc[:10], vis_incorrect, exp_dir)

        

if __name__ == "__main__":
    fire.Fire()