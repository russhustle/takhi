# 1. Mount google drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Create a soft link of direcctory in content
import os
DIR_PATH = "/content/drive/MyDrive/02-udemy/udemy-ds-ml-bootcamp/17-K-Means-Clustering"
DST_PATH = "/content/KMEANS"
os.symlink(src=DIR_PATH, dst=DST_PATH, target_is_directory=True)
os.chdir(DST_PATH)
