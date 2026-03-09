import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取上传的图片
img_path = "/home/xuyuan/pan/OWOD-master/proposal.png"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape
num_boxes = 40  # 每种方法生成的候选框数量

# 定义四个proposal方法的框生成逻辑
def selective_search_like(h, w, n):
    boxes = []
    for _ in range(n):
        # 主要集中在图像中部，有部分重叠
        x = np.random.randint(w//4, 3*w//4)
        y = np.random.randint(h//3, 2*h//3)
        bw = np.random.randint(40, 100)
        bh = np.random.randint(60, 120)
        boxes.append((x, y, bw, bh))
    return boxes

def random_boxes(h, w, n):
    boxes = []
    for _ in range(n):
        x = np.random.randint(0, w-100)
        y = np.random.randint(0, h-100)
        bw = np.random.randint(40, 120)
        bh = np.random.randint(40, 120)
        boxes.append((x, y, bw, bh))
    return boxes

def query_based_like(h, w, n):
    boxes = []
    for _ in range(n):
        # 模拟：查询到的是“远景人群与车辆”区域（y 偏下）
        x = np.random.randint(w//4, 3*w//4)
        y = np.random.randint(int(0.55*h), int(0.85*h))
        # 框更宽、更矮，模拟注意力横向分布
        bw = np.random.randint(100, 180)
        bh = np.random.randint(50, 80)
        boxes.append((x, y, bw, bh))
    return boxes


def rpn_like(h, w, n):
    boxes = []
    for _ in range(n):
        # 更紧密围绕图像中心区域（近景人物）
        x = np.random.randint(w//2 - 100, w//2 + 100)
        y = np.random.randint(h//2 - 60, h//2 + 60)
        bw = np.random.randint(50, 90)
        bh = np.random.randint(80, 140)
        boxes.append((x, y, bw, bh))
    return boxes

# 生成四种框
ss_boxes = selective_search_like(h, w, num_boxes)
rb_boxes = random_boxes(h, w, num_boxes)
qb_boxes = query_based_like(h, w, num_boxes)
rpn_boxes = rpn_like(h, w, num_boxes)

# 绘制函数
def draw_boxes(image, boxes, color):
    img = image.copy()
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
    return img

# 绘制四种方法结果
img_ss = draw_boxes(image, ss_boxes, (255, 0, 0))     # 蓝色
img_rb = draw_boxes(image, rb_boxes, (229, 76, 94))     # 绿色
img_qb = draw_boxes(image, qb_boxes, (255, 0, 255))   # 品红
img_rpn = draw_boxes(image, rpn_boxes, (255, 255, 0)) # 黄色

# 显示结果
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].imshow(img_ss)
axs[0, 0].set_title('Selective Search-like')
axs[0, 0].axis('off')

axs[0, 1].imshow(img_rb)
axs[0, 1].set_title('Random Boxes')
axs[0, 1].axis('off')

axs[1, 0].imshow(img_qb)
axs[1, 0].set_title('Query-based-like')
axs[1, 0].axis('off')

axs[1, 1].imshow(img_rpn)
axs[1, 1].set_title('RPN-like')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()


