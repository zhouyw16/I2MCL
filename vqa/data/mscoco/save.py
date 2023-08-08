import os
import numpy as np
from PIL import Image


for dir in ['train2014', 'val2014', 'test2015']:
    images = []
    indices = {}
    for i, name in enumerate(os.listdir(dir)):
        path = os.path.join(dir, name)
        image = np.array(Image.open(path).convert('RGB'))
        images.append(image)
        indices[path] = i
        if (i + 1) % 10000 == 0:
            print('process %d' % (i + 1))
        # if i == 10:
        #     break
    assert len(images) == len(indices)
    images = np.array(images, dtype=object)
    np.save('%s.npy' % (dir), images, allow_pickle=True)
    np.save('%s_img2id.npy' % (dir), indices, allow_pickle=True)
    # np.save('%s_example.npy' % (dir), images, allow_pickle=True)
    # np.save('%s_img2id_example.npy' % (dir), indices, allow_pickle=True)


# indices = np.load('train2014_img2id_example.npy', allow_pickle=True)
# print(indices)

# images = np.load('train2014_example.npy', allow_pickle=True)
# for image in images:
#     image = Image.fromarray(image.astype('uint8')).convert('RGB')
#     image.save('example.png')
#     input()
