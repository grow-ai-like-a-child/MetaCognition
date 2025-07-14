import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

# 设置参数
grid_sizes = [10, 15, 20, 25]
xo_ratios = [
    ("100%O", 1.0, 0.0), ("90%O", 0.9, 0.1), ("70%O", 0.7, 0.3),
    ("50/50", 0.5, 0.5), ("70%X", 0.3, 0.7), ("90%X", 0.1, 0.9), ("100%X", 0.0, 1.0)
]
distributions = ["uniform", "random", "clustered", "symmetric"]

output_dir = "xo_grids_strict"
os.makedirs(output_dir, exist_ok=True)

# 绘图函数
def draw_xo_grid(grid, size, filepath):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for i in range(size):
        for j in range(size):
            symbol = grid[i, j]
            if symbol:
                ax.text(j + 0.5, size - i - 0.5, symbol, fontsize=12, ha='center', va='center')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

# 生成函数
def generate_xo_grid_fixed(size, p_o, p_x, distribution):
    total_cells = size * size
    num_o = int(round(p_o / (p_o + p_x) * total_cells))
    num_x = total_cells - num_o
    num_blank = 0


    # 动态修正数量误差
    while num_o + num_x + num_blank > total_cells:
        if num_blank > 0:
            num_blank -= 1
        elif num_o > num_x:
            num_o -= 1
        else:
            num_x -= 1
    while num_o + num_x + num_blank < total_cells:
        num_blank += 1

    full_symbols = ['O'] * num_o + ['X'] * num_x + [''] * num_blank
    symbols = []

    if distribution == "uniform":
        # O/X交错，不打乱
        remaining_o, remaining_x = num_o, num_x
        while remaining_o > 0 or remaining_x > 0:
            if remaining_o > 0:
                symbols.append('O')
                remaining_o -= 1
            if remaining_x > 0:
                symbols.append('X')
                remaining_x -= 1
        symbols += [''] * (total_cells - len(symbols))

    elif distribution == "random":
        symbols = full_symbols[:]
        random.shuffle(symbols)

    elif distribution == "clustered":
        # 成块插入，不打乱
        counts = {'O': num_o, 'X': num_x}
        for ch in ['O', 'X']:
            while counts[ch] > 0:
                chunk_size = random.randint(2, 5)
                chunk = [ch] * min(chunk_size, counts[ch])
                symbols.extend(chunk)
                counts[ch] -= len(chunk)
        symbols += [''] * (total_cells - len(symbols))

    elif distribution == "symmetric":
        random.shuffle(full_symbols)
        half_len = total_cells // 2
        base = full_symbols[:half_len]
        reflected = base[::-1]
        middle = full_symbols[half_len:half_len+1] if total_cells % 2 == 1 else []
        symbols = base + middle + reflected

    # 统一补齐
    while len(symbols) < total_cells:
        symbols.append('')
    symbols = symbols[:total_cells]

    assert len(symbols) == total_cells
    return np.array(symbols).reshape((size, size))

# 批量生成图像和 CSV 记录
records = []
for size in grid_sizes:
    for ratio_label, p_o, p_x in xo_ratios:
        for dist in distributions:
            grid = generate_xo_grid_fixed(size, p_o, p_x, dist)
            filename = f"xo_{size}x{size}_{ratio_label}_{dist}.png".replace('%', 'pct').replace('/', '_')
            filepath = os.path.join(output_dir, filename)
            draw_xo_grid(grid, size, filepath)
            records.append({
                "filename": filename,
                "grid_size": size,
                "xo_ratio": ratio_label,
                "distribution": dist,
                "num_O": np.count_nonzero(grid == 'O'),
                "num_X": np.count_nonzero(grid == 'X')
            })

# 保存 CSV 文件
df = pd.DataFrame(records)
df.to_csv("xo_grid_strict_metadata.csv", index=False)
print("✅ 所有图像已生成完毕，结构一致性已修复！")
