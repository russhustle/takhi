Colab & Jupyter toolkit
===

1. Mount google drive in Colab
2. Create a soft link of the working directory
3. Show an image in Colab
4. Upload kaggle.json onto Colab
5. GPU information
6. Delete all Markdown or Code cells
7. Delete empty lines

Mount google drive in Colab
---

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create a soft link of the working directory
---

```python
import os
DIR_PATH = ""
DST_PATH = ""
full_dir_path = os.path.join("/content/drive/MyDrive", DIR_PATH)
full_dst_path = os.path.join("/content", DST_PATH)
os.symlink(src=full_dir_path, dst=full_dst_path, target_is_directory=True)
os.chdir('/content')
```

Show an image in Colab
---

```python
from IPython.display import Image
PATH = ""
Image(PATH)
```

Upload kaggle.json onto Colab
---

```python
from google.colab import files
import os
os.chdir("/content")
files.upload()
os.mkdir("/root/.kaggle")
os.rename("/content/kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 600)
```

GPU information
---

```bash
!nvidia-smi
```

Delete all Markdown cells [[link-1](https://stackoverflow.com/questions/57113816/how-to-delete-all-markdown-cells-in-jupyter-notebook), [link-2](https://discourse.jupyter.org/t/delete-all-code-cells-except-markdown-text/3072?u=fomightez)]
---

- [nbformat](https://nbformat.readthedocs.io/en/latest/api.html): Python API for working with notebook files

```python
import nbformat as nbf
ntbk = nbf.read("old_notebook.ipynb", nbf.NO_CONVERT)
new_ntbk = ntbk
new_ntbk.cells = [cell for cell in ntbk.cells if cell.cell_type != "markdown"] # here
nbf.write(new_ntbk, "no_markdown_notebook.ipynb", version=nbf.NO_CONVERT)
```

- Delete markdown: `!= "markdown"` @ line 4
- Delete code: `== "markdown"` @ line 4

Delete empty lines [[link](https://www.youtube.com/watch?v=jQrET5HYyAE)]
---

1. In VS code, install extension `Remove empty lines`.
2. Add shortcut to remove all empty lines in document; mine is `cmd+h`.
3. In the cell, type the shortcut.

