import torch
from torchvision import transforms
from PIL import Image

# 1. 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),            # 转换为张量
])

# 2. 加载单张图片
image_path = "/Users/darren/资料/SPIF_DU/Croppings/f1_out/5mm/images/45_120.jpg"
image = Image.open(image_path)

# 3. 进行预处理
image_tensor = transform(image)

# 4. 显示结果
print(image_tensor.shape)  # 查看张量形状