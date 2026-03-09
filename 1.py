import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的模拟图像（白色背景）
image = np.ones((500, 500, 3), dtype=np.uint8) * 255

# 定义每种方法生成的候选框数量和位置（以 [x, y, w, h] 形式表示）
num_boxes = 20

# 1. Selective Search: 模拟 Selective Search 框选（随机）
selective_search_boxes = [tuple(np.random.randint(0, 500, size=4)) for _ in range(num_boxes)]

# 2. Random Boxes: 随机生成框
random_boxes = [tuple(np.random.randint(0, 500, size=4)) for _ in range(num_boxes)]

# 3. Query-based: 模拟基于查询的框选（使用类似的随机位置）
query_based_boxes = [tuple(np.random.randint(0, 500, size=4)) for _ in range(num_boxes)]

# 4. RPN: 模拟 RPN 框选（假设框与目标物体更紧密）
rpn_boxes = [tuple(np.random.randint(100, 400, size=4)) for _ in range(num_boxes)]

# 绘制每个方法的框并显示
def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')

# 设置绘图
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# 选择不同颜色来表示每种方法
colors = {'Selective Search': (255, 0, 0),  # Blue
          'Random Boxes': (0, 255, 0),  # Green
          'Query-based': (0, 0, 255),  # Red
          'RPN': (0, 255, 255)}  # Yellow

# 1. Selective Search 框选
img_ss = image.copy()
draw_boxes(img_ss, selective_search_boxes, colors['Selective Search'], 'Selective Search')

# 2. Random Boxes 框选
img_rb = image.copy()
draw_boxes(img_rb, random_boxes, colors['Random Boxes'], 'Random Boxes')

# 3. Query-based 框选
img_qb = image.copy()
draw_boxes(img_qb, query_based_boxes, colors['Query-based'], 'Query-based')

# 4. RPN 框选
img_rpn = image.copy()
draw_boxes(img_rpn, rpn_boxes, colors['RPN'], 'RPN')

# 显示结果
axs[0, 0].imshow(img_ss)
axs[0, 0].set_title('Selective Search')
axs[0, 0].axis('off')

axs[0, 1].imshow(img_rb)
axs[0, 1].set_title('Random Boxes')
axs[0, 1].axis('off')

axs[1, 0].imshow(img_qb)
axs[1, 0].set_title('Query-based')
axs[1, 0].axis('off')

axs[1, 1].imshow(img_rpn)
axs[1, 1].set_title('RPN')
axs[1, 1].axis('off')

# 调整布局并展示
plt.tight_layout()
plt.show()

