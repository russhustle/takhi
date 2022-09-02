Hydra
===

Compose API in Jupyter Notebook
---

- Reference: [[Hydra Compose API](https://hydra.cc/docs/advanced/compose_api/)]

```python
from hydra import compose, initialize
with initialize(version_base=None, config_path="./configs"):
    cfg = compose(config_name="default")
```