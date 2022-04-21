def upload_kaggle_json():
    """
    Upload kaggle.json to colab environment

    Usage:
        !git clone https://github.com/Sihan-A/sihan_utils.git
        from sihan_utils.preprocess.upload_kaggle_json import upload_kaggle_json
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
