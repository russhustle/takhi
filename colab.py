# 1. Mount google drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Create a soft link of direcctory in content
import os
CHAPTER_PATH = "/content/drive/MyDrive/02-udemy/udemy-ds-ml-bootcamp/17-K-Means-Clustering"
CONTENT_PATH = "/content/KMEANS"
os.symlink(src=CHAPTER_PATH, dst=CONTENT_PATH, target_is_directory=True)
os.chdir("/content")



