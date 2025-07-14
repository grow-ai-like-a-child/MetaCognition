import os
import random
import matplotlib.pyplot as plt
import pandas as pd

# 输出目录
output_dir = "color_shading_groundtruth"
os.makedirs(output_dir, exist_ok=True)

# 词对类型及示例
word_bank = {
    "Same Word": [("Dog", "Dog")],
    "Different Words": [("Dog", "Cat"), ("Chair", "Table")],
    "Semantic Conflict": [("Good", "Bad"), ("Happy", "Sad")],
    "Emotional vs Neutral": [("Love", "Chair"), ("Fear", "Box")]
}

# 改进后的颜色映射
contrast_colors = {
    "High": ("black", "white"),
    "Low": ("dimgray", "gray"),
    "Color": ("darkred", "lightcoral")
}

# 布局类型
position_modes = ["Left-Right", "Top-Bottom", "Random", "Symmetric"]

# 用于判断“左侧是否为高对比”的辅助映射
contrast_rank = {"black": 3, "white": 3, "darkred": 2, "dimgray": 2,
                 "gray": 1, "lightcoral": 1}

# 绘图函数（浅灰背景 + 小字体）
def draw_words(word1, word2, color1, color2, position, filename):
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.patch.set_facecolor('#D3D3D3')
    ax.set_facecolor('#D3D3D3')
    ax.axis("off")

    fontsize = 30
    if position == "Left-Right":
        ax.text(0.3, 0.5, word1, fontsize=fontsize, ha='center', va='center', color=color1)
        ax.text(0.7, 0.5, word2, fontsize=fontsize, ha='center', va='center', color=color2)
    elif position == "Top-Bottom":
        ax.text(0.5, 0.7, word1, fontsize=fontsize, ha='center', va='center', color=color1)
        ax.text(0.5, 0.3, word2, fontsize=fontsize, ha='center', va='center', color=color2)
    elif position == "Random":
        if random.random() < 0.5:
            ax.text(0.3, 0.5, word1, fontsize=fontsize, ha='center', va='center', color=color1)
            ax.text(0.7, 0.5, word2, fontsize=fontsize, ha='center', va='center', color=color2)
        else:
            ax.text(0.7, 0.5, word1, fontsize=fontsize, ha='center', va='center', color=color1)
            ax.text(0.3, 0.5, word2, fontsize=fontsize, ha='center', va='center', color=color2)
            # 交换左右标签
            word1, word2 = word2, word1
            color1, color2 = color2, color1
    elif position == "Symmetric":
        ax.text(0.2, 0.8, word1, fontsize=fontsize, ha='center', va='center', color=color1)
        ax.text(0.8, 0.2, word2, fontsize=fontsize, ha='center', va='center', color=color2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return word1, word2, color1, color2  # 返回最终顺序（考虑 Random 情况）

# 保存 metadata
metadata = []
img_id = 0

for pair_type, wordlist in word_bank.items():
    word_pair = random.choice(wordlist)
    for contrast_type, (color1, color2) in contrast_colors.items():
        for position_mode in position_modes:
            w1, w2 = word_pair
            filename = f"img_{img_id:03d}_{pair_type.replace(' ', '')}_{contrast_type}_{position_mode}.png"

            # 绘图并获取最终顺序（考虑 random 时的左右变换）
            final_w1, final_w2, final_c1, final_c2 = draw_words(w1, w2, color1, color2, position_mode, filename)

            metadata.append({
                "filename": filename,
                "pair_type": pair_type,
                "word_left_or_top": final_w1,
                "word_right_or_bottom": final_w2,
                "contrast_type": contrast_type,
                "color_left_or_top": final_c1,
                "color_right_or_bottom": final_c2,
                "position_mode": position_mode,
                "is_semantic_conflict": int(pair_type == "Semantic Conflict"),
                "is_high_contrast_on_left": int(contrast_rank[final_c1] > contrast_rank[final_c2])
            })
            img_id += 1

# 保存 CSV
df = pd.DataFrame(metadata)
df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

print(f"✅ 已生成 {img_id} 张图像（含 ground truth 标签），保存至 '{output_dir}/'")
