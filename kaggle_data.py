from google.colab import files
uploaded = files.upload() # choose kaggle.json
!mv /content/kaggle.json /root/.kaggle
# !chmod 600 /root/.kaggle/kaggle.json

# kaggle API
!kaggle datasets download -d ajayrana/hymenoptera-data
# then unzip


## version 2
#!pip install -q kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d tongpython/cat-and-dog
!unzip cat-and-dog.zip