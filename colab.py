# 1. Mount google drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Create a soft link of direcctory in content
import os
DIR_PATH = "/content/drive/MyDrive/02-udemy/udemy-ds-ml-bootcamp/17-K-Means-Clustering"
DST_PATH = "/content/KMEANS"
os.symlink(src=DIR_PATH, dst=DST_PATH, target_is_directory=True)
os.chdir('/content')

# 3. Show an image in colab
from IPython.display import Image
PCA_PATH = os.path.join(DST_PATH, 'images/PCA.png')
Image(PCA_PATH)

# 4. Upload kaggle.json to colab

from google.colab import files
import os

def upload_kaggle_json():
    """ Upload kaggle.json to colab environment
    Usage:
        !git clone https://github.com/Sihan-A/sihan_utils.git
        from sihan_utils.colab import upload_kaggle_json
        upload_kaggle_json()
        !kaggle datasets download -d berkeleyearth/climate-change-earth-surface-temperature-data
    """
    os.chdir("/content")
    files.upload()
    os.mkdir("/root/.kaggle")
    os.rename("/content/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 600)

