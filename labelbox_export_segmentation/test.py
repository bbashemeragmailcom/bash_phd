from PIL import Image
import numpy as np

mask = np.array(
    Image.open(
        "labelbox_export/composite_masks/slice_0039_mask.png"
    ).convert("RGB")
)

colors, counts = np.unique(
    mask.reshape(-1,3),
    axis=0,
    return_counts=True
)


for c,n in zip(colors, counts):
    print(tuple(c), n)

# print(img.mode)

import numpy as np

data = np.load(
    "stroke_dataset/npz/slice_0039.npz"
)

print(data["image"].shape)
print(data["mask"].shape)

print(
    np.unique(data["mask"])
)

from PIL import Image

img = Image.open(
    "labelbox_export/composite_masks/slice_0039_mask.png"
)

print(img.mode)
print(img.getpixel((100,100)))
print(np.unique(mask.reshape(-1,3),axis=0))