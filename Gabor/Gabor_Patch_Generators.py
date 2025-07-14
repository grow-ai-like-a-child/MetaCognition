import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import cv2

# 改进后的 Gabor Patch 生成函数
def generate_gabor(size=256, frequency=0.1, theta=0, contrast=1.0, sigma=0.3):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    x_theta = xv * np.cos(np.deg2rad(theta)) + yv * np.sin(np.deg2rad(theta))
    y_theta = -xv * np.sin(np.deg2rad(theta)) + yv * np.cos(np.deg2rad(theta))
    gauss = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
    wave = np.cos(2 * np.pi * frequency * x_theta)
    gabor = gauss * wave

    # 设置中心为灰（0.5），振幅为对比度的一半
    gabor = 0.5 + (contrast / 2.0) * gabor
    return np.clip(gabor, 0, 1)

# 参数映射
contrast_levels = {
    "High": 1.0,
    "Medium": 0.5,
    "Low": 0.2
}

frequency_levels = {
    "Low": 3,
    "High": 12,
    "Mixed": 7
}

orientation_levels = {
    "Horizontal": 90,
    "Oblique": 45,
    "Vertical": 0,
    "Free": lambda: random.randint(1, 359)  # 任意角度（排除标准方向）
}

# 输出目录
output_dir = "gabor_36"
os.makedirs(output_dir, exist_ok=True)

# 保存信息
records = []

# 枚举组合
for c_name, c_val in contrast_levels.items():
    for f_name, f_val in frequency_levels.items():
        for o_name, o_val in orientation_levels.items():
            theta = o_val() if callable(o_val) else o_val
            gabor = generate_gabor(size=256, frequency=f_val, theta=theta, contrast=c_val)

            filename = f"gabor_c{c_name}_f{f_name}_o{o_name}.png"
            filepath = os.path.join(output_dir, filename)

            # 使用 cv2 保存为灰度图，保留对比度
            gabor_uint8 = (gabor * 255).astype(np.uint8)
            cv2.imwrite(filepath, gabor_uint8)

            # 记录 metadata
            records.append({
                "filename": filename,
                "contrast_level": c_name,
                "contrast_value": c_val,
                "frequency_level": f_name,
                "frequency_value": f_val,
                "orientation_level": o_name,
                "theta": theta,
                "min_val": float(gabor.min()),
                "max_val": float(gabor.max())
            })

# 保存 metadata CSV
df = pd.DataFrame(records)
df.to_csv("gabor_36_metadata.csv", index=False)
print("✅ 成功生成 36 张 Gabor Patch，并导出 gabor_36_metadata.csv")
