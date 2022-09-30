Data Version Control (DVC)
===

- Blogs

    - [Introduction to DVC: Data Version Control Tool for Machine Learning Projects - Khuyen Tran](https://medium.com/towards-data-science/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-7cb49c229fe0)

- Supports VS Code [[link](https://dvc.org/doc/vs-code-extension)]

- Links

    - [DVC HP](https://dvc.org/)

    - [Common references](https://dvc.org/doc/command-reference)


Installation
---

```shell
pip install dvc
conda install -c conda-forge dvc
```

Basic usage
---

1. `dvc init` [[link](https://dvc.org/doc/command-reference/init)]
2. `dvc add` [[link](https://dvc.org/doc/command-reference/add)]

```shell
dvc init
```

- Initialization; create `.dvc/.gitignore` and `.dvc/config`

```shell
dvc add data
```

- Start tracking a file or directory `data`; create `data.dvc` to store changes

Remote storage
---

The [remote add](https://dvc.org/doc/command-reference/remote/add) supports S3, GCP, Azure, Google Drive, SSH, and more.

```shell
dvc remote add -d remote gdrive://{Google Drive Folder ID}
```

- `-d`: default setting
- Changes saved in `.dvc/config`

```shell
dvc push
```

- Push data to Google Drive

Pull data
---

```
dvc pull
```

