import scipy
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, img_res=(256,256)):        
        self.img_res = img_res

    def load_data(self):
        path = glob('./datasets/saree/new_handloom_saree/img_339968683.jpg')
        batch=path
        imgs_A, imgs_B = [], []
        for img in batch:
            img = self.imread(img)
            h, w, _ = img.shape
            half_w = int(w/2)
            img_A = img[:, :half_w, :]
            img_B = img[:, half_w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
