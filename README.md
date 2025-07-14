## This is the codebase for MetaCognition

# How to use?

First create a new conda env - 

```
conda create -n metacognition python=3.10
```

Install all neccesary packages with 
```
python -m pip install -r requirments.txt
```

To generate all data in Color setting,
```python Color/Color_Gene.py```

To generate all data in Gabor setting
```python Gabor/Gabor_Patch_Generators.py```

To generate all data in XO_Grid setting
```python XO_Grid/90pcto.py```

1) 
Repository Explanation
-Color
--color_shading_groundtruth
--Color_Gene.py
# This is the code to generate the experiment data for Color Experiment. Please refer to [https://docs.google.com/document/d/1yqHH4spfVSg0gSNLIoSmdScTMH12f9tfs9h7wAQ2tlI/edit?tab=t.0#heading=h.38tt23dbxxgn]
-Gabor
--gabor_36
--Gabor_Patch_Generators.py
# This is the code to generate the experiment data for Gabor Experiment. Please refer to [https://docs.google.com/document/d/1yqHH4spfVSg0gSNLIoSmdScTMH12f9tfs9h7wAQ2tlI/edit?tab=t.0#heading=h.38tt23dbxxgn]
-XO_Grid
--xo_grids_strict
--90pcto.py
# This is the code to generate the experiment data for Grid Experiment. Please refer to [https://docs.google.com/document/d/1yqHH4spfVSg0gSNLIoSmdScTMH12f9tfs9h7wAQ2tlI/edit?tab=t.0#heading=h.38tt23dbxxgn]
-deprecated
# Here is where we put the deprecated files.

2)