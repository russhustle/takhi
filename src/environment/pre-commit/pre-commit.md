Automation with `pre-commit`
===

[TOC]

Content

1. `pre-commit`
2. `black`: code formatter
3. `flake8`: code style and quality checker
4. `shellcheck`: shell script analysis
5. `isort`: automatically sorts imported libraries
6. `interrogate`: missing docstring checker

Reference

- [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
- [How to setup your project with pre-commit, black, and flake8](https://dev.to/m1yag1/how-to-setup-your-project-with-pre-commit-black-and-flake8-183k)
- [4 pre-commit Plugins to Automate Code Reviewing and Formatting in Python](https://towardsdatascience.com/4-pre-commit-plugins-to-automate-code-reviewing-and-formatting-in-python-c80c6d2e9f5)

![pre-commit pipeline](https://ljvmiranda921.github.io/assets/png/tuts/precommit_pipeline.png)

pre-commit
---

We need to have a `.pre-commit-config.yaml`

```shell
pre-commit autoupdate
```

black
---

- Check and reformat the current directory

    ```shell
    python -m black .
    ```

- [`jupyter-black`](https://pypi.org/project/jupyter-black/): To beautify python code in Jupyter Notebook

flake8
---

Shellcheck
---

isort
---

interrogate
---
