import cv2
import torch
import numpy as np
from skimage import transform

# reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))
        mask = transform.resize(mask, (new_h, new_w))

        sample['image'], sample['mask'] = image, mask
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        mask = mask[top: top + new_h,
                      left: left + new_w]

        sample['image'], sample['mask'] = image, mask
        return sample

class CenterCrop(object):
    """Gives a center crop of the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top: top + new_h,
                      left: left + new_w]
        mask = mask[top: top + new_h,
                      left: left + new_w]

        sample['image'], sample['mask'] = image, mask
        return sample

class RandomFlip(object):
    """
    Flip the image horizontally (axis=1) or vertically (axis=0).
    """

    def __init__(self, prob=0.5):
        assert isinstance(prob, (float, tuple))
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.prob:

            axis = 0
            if np.random.random() > 0.5:                                        # sometimes horizontal, othertimes vertical
                axis = 1

            image = cv2.flip(image, axis)
            mask = cv2.flip(mask, axis)

        sample['image'], sample['mask'] = image, mask
        return sample

class RandomRotate(object):
    """
    Rotate the image randomly.
    """

    def __init__(self, prob=0.5):
        assert isinstance(prob, (int, float, tuple))
        self.prob = prob

    def rotate(self, image, angle):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        shape = (image.shape[1], image.shape[0])
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotate = cv2.warpAffine(image, matrix, shape, flags=cv2.INTER_LINEAR)
        rotate = rotate[10:-15, 10:-15]
        resize = cv2.resize(rotate, shape, interpolation=cv2.INTER_AREA)
        return resize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.prob:
            angle = np.random.randint(0, 360)
            image = self.rotate(image, angle)
            mask = self.rotate(mask, angle)
        
        if len(mask.shape) != 3:
            mask = np.expand_dims(mask, axis=-1)
        
        sample['image'], sample['mask'] = image, mask
        return sample

class RandomErase(object):
    """
    Adds to blank patch randomly to an image-mask pair
    """

    def __init__(self, prob=0.5):
        assert isinstance(prob, (int, float, tuple))
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.prob:
            img_h, img_w, _ = image.shape

            h = np.random.randint(0.1*img_h, 0.4*img_h)
            w = np.random.randint(0.1*img_w, 0.4*img_w)

            x = np.random.randint(0, img_h - h)
            y = np.random.randint(0, img_w - w)

            image[x:x+h, y:y+w] = np.random.random((h, w, 3))
            mask[x:x+h, y:y+w] = 0
        
        sample['image'], sample['mask'] = image, mask
        return sample

class RandomShear(object):
    """
    Adds sheer to the image-mask pair randomly
    """

    def __init__(self, prob=0.2):
        assert isinstance(prob, (int, float, tuple))
        self.prob = prob

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.random() < self.prob:
            img_h, img_w, _ = image.shape

            xs, ys = np.random.rand()-0.5, np.random.rand()-0.5
            M = np.float32([[1, xs, 0],
                            [ys, 1  , 0],
                            [0, 0  , 1]])               
            image = cv2.warpPerspective(image, M, (img_h, img_w))
            mask = cv2.warpPerspective(mask, M, (img_h, img_w))
        
        if len(mask.shape) != 3:
            mask = np.expand_dims(mask, axis=-1)

        sample['image'], sample['mask'] = image, mask
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        
        sample['image'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(mask)
        return sample
