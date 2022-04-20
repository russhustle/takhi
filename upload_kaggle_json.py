def upload_kaggle_json():
    """
    Upload kaggle.json to colab environment
    """
    from google.colab import files
    import os
    os.chdir("/content")
    files.upload()
    os.mkdir("/.kaggle")
    os.rename("/content/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 600)
