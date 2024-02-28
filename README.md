# sd_hcp

HCP-based stable diffusion image generator

## Installation

```shell
git clone https://github.com/HansBug/sd_hcp.git
cd sd_hcp
pip install -r requirements.txt
make init
```

## T2I

```python
import matplotlib.pyplot as plt
from ditk import logging

from sd_hcp.infer import infer_images

logging.try_init_root(level=logging.INFO)

if __name__ == '__main__':
    images = infer_images(
        prompts=[
            'masterpiece, best quality, 1girl, solo, tohsaka rin, portrait',
        ],
        neg_prompts=[
            'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry',
        ],
        seeds=[42],
    )

    plt.imshow(images[0])
    plt.show()

```

