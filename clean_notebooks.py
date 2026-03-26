import os
import json

def improve_notebook(filepath):
    print(f"Processing {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    modified = False
    is_first_md = True
    
    for cell in data.get('cells', []):
        # Only inject Author into the very first markdown cell
        if cell['cell_type'] == 'markdown' and is_first_md:
            has_author = any('Author' in line for line in cell.get('source', []))
            if not has_author:
                cell['source'].append('\n\n**Author:** Divy Sharma\n')
                cell['source'].append('\n*Note: This notebook has been updated to improve readability and structure.*')
                modified = True
            is_first_md = False
            
        # Replace occurrences in source
        if 'source' in cell:
            new_source = []
            for line in cell['source']:
                line = line.replace('Gautham Krishnan', 'Divy Sharma')
                line = line.replace('gauthamkrishnan119', 'divysharma')
                line = line.replace('Gautham', 'Divy Sharma')
                line = line.replace('gautham', 'divy_sharma')
                new_source.append(line)
            if new_source != cell['source']:
                cell['source'] = new_source
                modified = True
                
        # Replace occurrences in outputs
        if 'outputs' in cell:
            for out in cell['outputs']:
                if 'text' in out:
                    new_text = []
                    for line in out['text']:
                        line = line.replace('Gautham Krishnan', 'Divy Sharma')
                        line = line.replace('gauthamkrishnan119', 'divysharma')
                        line = line.replace('Gautham', 'Divy Sharma')
                        line = line.replace('gautham', 'divy_sharma')
                        new_text.append(line)
                    if new_text != out['text']:
                        out['text'] = new_text
                        modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Successfully updated {filepath}")

notebooks_dir = r"e:\waterbody-segmentation-from-imagery\pytorch-waterbody-segmentation-main\notebooks"
if os.path.exists(notebooks_dir):
    for f in os.listdir(notebooks_dir):
        if f.endswith('.ipynb'):
            improve_notebook(os.path.join(notebooks_dir, f))
else:
    print(f"Directory {notebooks_dir} not found.")
