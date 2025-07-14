# 🧠 MetaCognition

A codebase for generating experimental data across **Color**, **Gabor**, and **XO\_Grid** settings for the MetaCognition project.

---

## 🚀 Quick Start

### 1. Create Environment

```bash
conda create -n metacognition python=3.10
conda activate metacognition
```

### 2. Install Requirements

```bash
python -m pip install -r requirements.txt
```

---

## 🧪 Generate Data

All generated data will be saved into the `data/` folder.

### ▶️ Color Setting

```bash
python Color/Color_Gene.py
```

### ▶️ Gabor Setting

```bash
python Gabor/Gabor_Patch_Generators.py
```

### ▶️ XO\_Grid Setting

```bash
python XO_Grid/90pcto.py
```

---

## 📁 Repository Structure

```
MetaCognition/
│
├── Color/
│   ├── color_shading_groundtruth/
│   └── Color_Gene.py  ← Color experiment generator
│
├── Gabor/
│   ├── gabor_36/
│   └── Gabor_Patch_Generators.py  ← Gabor experiment generator
│
├── XO_Grid/
│   ├── xo_grids_strict/
│   └── 90pcto.py  ← Grid experiment generator
│
├── deprecated/  ← Archive for outdated or unused files
└── requirements.txt
```

For full documentation on experiment logic and task design, see our [Project Guide](https://docs.google.com/document/d/1yqHH4spfVSg0gSNLIoSmdScTMH12f9tfs9h7wAQ2tlI/edit?tab=t.0#heading=h.38tt23dbxxgn).

---

## 📝 Citation & License

If you use this codebase, please cite the corresponding paper (link coming soon).
Licensed under [MIT](LICENSE).
