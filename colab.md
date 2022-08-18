Colab toolkit
===

小实验

[TOC]

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

