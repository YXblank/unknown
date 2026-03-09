from PIL import Image, ImageDraw
import numpy as np
import random

# Load the input image
img_path = "/home/xuyuan/pan/OWOD-master/proposal.png"
image = Image.open(img_path)
width, height = image.size

# Parameters
num_boxes = 30  # same number for all methods
box_colors = {
    "SelectiveSearch": "red",
    "RandomBoxes": "scarlet",
    "QueryBased": "green",
    "RPN": "yellow"
}

def generate_random_boxes(num_boxes, width, height, min_size=50, max_size=150):
    boxes = []
    for _ in range(num_boxes):
        w, h = random.randint(min_size, max_size), random.randint(min_size, max_size)
        x1 = random.randint(0, width - w)
        y1 = random.randint(0, height - h)
        boxes.append((x1, y1, x1 + w, y1 + h))
    return boxes

def generate_selective_search_boxes(num_boxes, width, height):
    # Simulate selective search: slightly larger and more centered boxes
    boxes = []
    for _ in range(num_boxes):
        w = random.randint(80, 180)
        h = random.randint(80, 180)
        x1 = random.randint(width//4, 3*width//4 - w)
        y1 = random.randint(height//4, 3*height//4 - h)
        boxes.append((x1, y1, x1 + w, y1 + h))
    return boxes

def generate_query_based_boxes(num_boxes, width, height):
    # 聚焦“显著物体区域”：靠近图像中心 + 加入轻微偏移
    boxes = []
    cx, cy = width // 2, height // 2
    for _ in range(num_boxes):
        w, h = random.randint(60, 140), random.randint(60, 140)
        offset_x = int(np.random.normal(0, width // 8))  # 加入偏移噪声
        offset_y = int(np.random.normal(0, height // 8))
        x1 = np.clip(cx + offset_x - w // 2, 0, width - w)
        y1 = np.clip(cy + offset_y - h // 2, 0, height - h)
        boxes.append((x1, y1, x1 + w, y1 + h))
    return boxes


def generate_rpn_boxes(num_boxes, width, height):
    # Simulate RPN: a mix of sizes centered around grid locations
    boxes = []
    for _ in range(num_boxes):
        grid_x = random.randint(0, 9)
        grid_y = random.randint(0, 9)
        w, h = random.randint(60, 120), random.randint(60, 120)
        x1 = grid_x * width // 10
        y1 = grid_y * height // 10
        boxes.append((x1, y1, min(x1 + w, width), min(y1 + h, height)))
    return boxes

# Generate boxes for each method
boxes_dict = {
    "SelectiveSearch": generate_selective_search_boxes(num_boxes, width, height),
    "RandomBoxes": generate_random_boxes(num_boxes, width, height),
    "QueryBased": generate_query_based_boxes(num_boxes, width, height),
    "RPN": generate_rpn_boxes(num_boxes, width, height)
}

# Draw each image
results = {}
for method, boxes in boxes_dict.items():
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for box in boxes:
        draw.rectangle(box, outline=box_colors[method], width=3)
    results[method] = img_copy

# Save output images
output_paths = {}
for method, img in results.items():
    path = f"/home/xuyuan/pan/OWOD-master/{method}_proposals.png"
    img.save(path)
    output_paths[method] = path

output_paths

