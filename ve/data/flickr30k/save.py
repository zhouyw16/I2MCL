import os
import numpy as np
from PIL import Image


for dir in ['flickr30k-images']:
    images = []
    indices = {}
    for i, name in enumerate(os.listdir(dir)):
        path = os.path.join(dir, name)
        image = np.array(Image.open(path).convert('RGB'))
        images.append(image)
        indices[name] = i
        if (i + 1) % 10000 == 0:
            print('process %d' % (i + 1))
        # if i == 10:
        #     break
    assert len(images) == len(indices)
    images = np.array(images, dtype=object)
    np.save('img.npy', images, allow_pickle=True)
    np.save('img2id.npy', indices, allow_pickle=True)
    # np.save('img_example.npy', images, allow_pickle=True)
    # np.save('img2id_example.npy', indices, allow_pickle=True)


# indices = np.load('img2id_example.npy', allow_pickle=True)
# print(indices)

# images = np.load('img_example.npy', allow_pickle=True)
# for image in images:
#     image = Image.fromarray(image.astype('uint8')).convert('RGB')
#     image.save('example.png')
#     input()
