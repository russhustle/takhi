from google.colab import files
import os


def kaggle_json_upload():
    """To upload the kaggle.json to use the kaggle dataset in colab
    Usage:
        from takhi.colab import kaggle_json_upload
        kaggle_json_upload()
    """
    os.chdir("/content")
    files.upload()
    os.mkdir("/root/.kaggle")
    os.rename("/content/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 600)
