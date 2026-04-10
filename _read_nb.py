import json

nb = json.load(open('d:/webapp/Node_level_DiffusionNet/Node_level_DiffusionNet.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']

# Print cell with the NodeLevel class - it should be around cell 15-16
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if 'NodeLevel_DFM_DiffusionNet' in src and 'class ' in src:
        print(f"Cell {i} — FULL SOURCE ({len(src)} chars):")
        print(src)
        break
