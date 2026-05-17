import textwrap
from typing import List, Dict

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_test(
    wrong_predictions: List[Dict],
    tokenizer,
    k: int = 10,
    MEAN=[0.485, 0.456, 0.406],
    STD=[0.229, 0.224, 0.225],
):
    """
    Visualize k randomly selected predictions, showing each query, predicted images with PIDs, and correct image.

    Args:
        wrong_predictions: List of dictionaries with structure:
        {'query', 'img_1': {'image': tensor, 'pid': str}, ..., 'img_10': {...}, 'correct_img': {...}}
        tokenizer: Tokenizer for decoding the query
        k: Number of predictions to visualize
    """
    REVERSE_MEAN = (-MEAN / STD).tolist()
    REVERSE_STD = (1.0 / STD).tolist()
    denormalize = T.Normalize(mean=REVERSE_MEAN, std=REVERSE_STD)
    toPIL = T.ToPILImage()
    transform = T.Compose([denormalize, toPIL])

    # Randomly select k predictions
    k_selected_for_visualization = np.random.choice(wrong_predictions, k, replace=False)
    total_columns = 12  # query + 10 predictions + correct_img

    # Create a figure with k rows (one for each query)
    fig = plt.figure(figsize=(24, 4 * k))
    fig.suptitle("Query-Image Predictions Visualization", fontsize=16, y=0.98)

    def wrap_text(text, width=30):
        """Wrap text to specified width"""
        lines = textwrap.wrap(text, width=width, break_long_words=True)
        return "\n".join(lines)

    for idx, pred_dict in enumerate(k_selected_for_visualization):
        base_pos = idx * total_columns + 1

        # 1. Display Query
        ax = plt.subplot(k, total_columns, base_pos)
        if isinstance(pred_dict["query"], torch.Tensor):
            query_text = tokenizer.decode(pred_dict["query"], skip_special_tokens=True)
        else:
            query_text = pred_dict["query"]

        ax.set_facecolor("#f0f0f0")
        wrapped_text = wrap_text(query_text)
        bbox_props = dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray", lw=1)
        plt.text(
            0.5,
            0.5,
            f"Query:\n{wrapped_text}",
            horizontalalignment="center",
            verticalalignment="center",
            wrap=True,
            fontsize=10,
            bbox=bbox_props,
            transform=ax.transAxes,
        )
        ax.set_aspect("equal", adjustable="box")
        plt.axis("off")

        # 2. Display Predicted Images
        for img_idx in range(1, 11):  # Display img_1 through img_10
            ax = plt.subplot(k, total_columns, base_pos + img_idx)

            img_key = f"img_{img_idx}"
            img_data = pred_dict[img_key]
            img_tensor = img_data["image"]
            pid = img_data["pid"]

            if isinstance(img_tensor, torch.Tensor):
                if len(img_tensor.shape) == 4:
                    img_tensor = img_tensor.squeeze(0)
                img = transform(img_tensor.cpu())
            else:
                img = img_tensor

            plt.imshow(img)
            plt.title(f"Pred {img_idx}\nPID: {pid}", fontsize=10, pad=5)
            plt.axis("off")

        # 3. Display Correct Image (at the end)
        ax = plt.subplot(k, total_columns, base_pos + 11)
        correct_img_data = pred_dict["correct_img"]
        correct_img_tensor = correct_img_data["image"]
        correct_pid = correct_img_data["pid"]

        if isinstance(correct_img_tensor, torch.Tensor):
            if len(correct_img_tensor.shape) == 4:
                correct_img_tensor = correct_img_tensor.squeeze(0)
            correct_img = transform(correct_img_tensor.cpu())
        else:
            correct_img = correct_img_tensor

        plt.imshow(correct_img)
        plt.title(f"Ground Truth\nPID: {correct_pid}", fontsize=10, pad=5)
        plt.axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def add_text_header(
    image,
    text,
    font_size=40,
    padding=20,
    text_color=(0, 0, 0),
    bg_color=(255, 255, 255),
):
    """
    Add centered text above an image by extending the canvas.

    Args:
        image (PIL.Image): Input image
        text (str): Text to write above the image (short text expected)
        font_size (int): Font size for the text
        padding (int): Padding around the text in pixels
        text_color (tuple): RGB color tuple for the text
        bg_color (tuple): RGB color tuple for the background

    Returns:
        PIL.Image: New image with centered text header
    """
    # Create a font object (using default font)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font if arial is not available
        font = ImageFont.load_default()

    # Calculate text dimensions
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Create new image with extra space for text
    header_height = text_height + (2 * padding)
    new_height = image.height + header_height
    new_image = Image.new("RGB", (image.width, new_height), bg_color)

    # Paste original image at bottom
    new_image.paste(image, (0, header_height))

    # Add text centered both horizontally and vertically in the header space
    draw = ImageDraw.Draw(new_image)
    x = (image.width - text_width) // 2
    y = (header_height - text_height) // 2
    draw.text((x, y), text, font=font, fill=text_color)

    return new_image


def prepare_prediction_for_wandb_table(
    wrong_predictions: List[Dict],
    tokenizer,
    k: int = 10,
    MEAN: List[float] = [0.485, 0.456, 0.406],
    STD: List[float] = [0.229, 0.224, 0.225],
):
    """
    Prepare the wrong predictions for logging in a Weights & Biases table.

    Args:
        wrong_predictions: List of dictionaries with structure:
        {'query', 'img_1': {'image': tensor, 'pid': str}, ..., 'img_10': {...}, 'correct_img': {...}}
    Returns:
        Dict[[columns, data], Any]: Dictionary containing columns and data for logging in a W&B table
    """
    REVERSE_MEAN = (-MEAN / STD).tolist()
    REVERSE_STD = (1.0 / STD).tolist()
    denormalize = T.Normalize(mean=REVERSE_MEAN, std=REVERSE_STD)
    toPIL = T.ToPILImage()
    transform = T.Compose([denormalize, toPIL])

    # Randomly select k predictions
    k_selected_for_visualization = np.random.choice(wrong_predictions, k, replace=False)

    columns = ["Query"] + [f"Pred {i}" for i in range(1, 11)] + ["Ground Truth"]
    data = []

    for pred_dict in k_selected_for_visualization:
        row = []

        # 1. Query
        if isinstance(pred_dict["query"], torch.Tensor):
            query_text = tokenizer.decode(pred_dict["query"], skip_special_tokens=True)
        else:
            query_text = pred_dict["query"]
        row.append(query_text)

        # 2. Predicted Images
        for img_data in pred_dict["predictions"]:
            pid = img_data["pid"]

            if isinstance(img_data["image"], torch.Tensor):
                img = transform(img_data["image"].cpu())
            else:
                img = img_data["image"]

            # Add PID text to the image
            new_img = add_text_header(img, f"PID: {pid}", font_size=20, padding=5)
            row.append(new_img)

        # 3. Correct Image
        correct_img_data = pred_dict["correct_img"]
        correct_pid = correct_img_data["pid"]
        if isinstance(correct_img_data["image"], torch.Tensor):
            correct_img = transform(correct_img_data["image"].cpu())
        else:
            correct_img = correct_img_data["image"]

        # Add PID text to the image
        new_correct_img = add_text_header(
            correct_img, f"PID: {correct_pid}", font_size=20, padding=5
        )
        row.append(new_correct_img)

        data.append(row)

    return {"columns": columns, "data": data}
