
from IPython.display import Image

# 1. Mount google drive
def mount_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')

# 2. Create a soft link of direcctory in content
import os
def dir_soft_link(DIR_PATH, DST_PATH):
    full_dir_path = os.path.join("/content/drive/MyDrive", DIR_PATH)
    full_dst_path = os.path.join("/content", DST_PATH)
    os.symlink(src=full_dir_path, dst=full_dst_path, target_is_directory=True)
    os.chdir('/content')

# 3. Show an image in colab
def colab_show_image(PATH):
    from IPython.display import Image
    Image(PATH)

# 4. Upload kaggle.json to colab
def upload_kaggle_json():
    """ Upload kaggle.json to colab environment
    Usage:
        !git clone https://github.com/Sihan-A/sihan_utils.git
        from sihan_utils.colab import upload_kaggle_json
        upload_kaggle_json()
        !kaggle datasets download -d berkeleyearth/climate-change-earth-surface-temperature-data
    """
    from google.colab import files
    import os
    os.chdir("/content")
    files.upload()
    os.mkdir("/root/.kaggle")
    os.rename("/content/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 600)

