import os, random, numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 参数设置
grid_sizes = [10, 15, 20, 25]
xo_ratios = [
    ("100%O", 1.0, 0.0), ("90%O", 0.9, 0.1),
    ("70%O", 0.7, 0.3), ("50%O", 0.5, 0.5),
    ("70%X", 0.3, 0.7), ("90%X", 0.1, 0.9),
    ("100%X", 0.0, 1.0)
]
distributions = ["uniform", "random", "clustered", "symmetric"]
output_dir = "./xo_grids"
os.makedirs(output_dir, exist_ok=True)

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
                ax.text(j + 0.5, size - i - 0.5, symbol,
                        fontsize=12, ha='center', va='center')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def generate_xo_grid(size, p_o, p_x, distribution):
    total_cells = size * size
    num_o = int(round(p_o * total_cells))
    num_x = int(round(p_x * total_cells))
    while num_o + num_x > total_cells:
        if num_o > num_x:
            num_o -= 1
        else:
            num_x -= 1
    while num_o + num_x < total_cells:
        if p_o >= p_x:
            num_o += 1
        else:
            num_x += 1
    num_blank = total_cells - num_o - num_x
    symbols = ['O'] * num_o + ['X'] * num_x + [''] * num_blank

    if distribution == "uniform":
        random.shuffle(symbols)
    elif distribution == "random":
        np.random.shuffle(symbols)
    elif distribution == "clustered":
        symbols = sorted(symbols, key=lambda x: (x == 'X') * random.random())
    elif distribution == "symmetric":
        half = symbols[:total_cells // 2]
        reflected = half[::-1]
        symbols = (half + reflected)[:total_cells]
        if len(symbols) < total_cells:
            symbols += [''] * (total_cells - len(symbols))

    assert len(symbols) == total_cells
    return np.array(symbols).reshape((size, size))

# 批量生成
records = []
for size in grid_sizes:
    for ratio_label, p_o, p_x in xo_ratios:
        for dist in distributions:
            grid = generate_xo_grid(size, p_o, p_x, dist)
            filename = f"xo_{size}x{size}_{ratio_label}_{dist}.png".replace('%', 'pct')
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

# 输出CSV
pd.DataFrame(records).to_csv("xo_grid_metadata.csv", index=False)
