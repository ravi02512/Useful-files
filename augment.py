import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance


class augment_techniques(params):

    def rotate_bound(self, image, angle):

        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image

        resized = cv2.warpAffine(image, M, (nW, nH))
        resized_img = cv2.resize(resized, (h, w))
        return resized_img

    def increase_brightness(self, img, value=30):

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img

    def create_random_black_patches(self, img):

        center_x, center_y = int(img.shape[0] / 2), int(img.shape[1] / 2)

        try:
            x = np.random.randint(1, img.shape[0] // 2)
            y = np.random.randint(1, img.shape[1] // 2)
            crop_height = img.shape[0] // 2
            crop_width = img.shape[1] // 2

            plt_im = img.copy()

            plt_im[y : y + crop_height, x : x + crop_width] = np.zeros_like(
                plt_im[y : y + crop_height, x : x + crop_width]
            )
        except Exception as e:
            print(e)

        return plt_im

    def self_cutmix(self, image, grid):

        X = list(range(0, self.image_size[1], int(self.image_size[1] / grid)))
        Y = list(range(0, self.image_size[0], int(self.image_size[0] / grid)))
        sliced_list = []
        for x in X:
            for y in Y:
                sliced_list.append(
                    image[
                        y : y + int(self.image_size[0] / grid),
                        x : x + int(self.image_size[1] / grid),
                    ]
                )
        img = np.zeros(self.image_size)
        np.random.shuffle(sliced_list)
        a = 0
        for i in range(len(X)):
            x = X[i]
            for j in range(len(Y)):
                img[
                    Y[j] : Y[j] + int(self.image_size[0] / grid),
                    x : x + int(self.image_size[1] / grid),
                ] = sliced_list[a]
                a = a + 1
        return img




    def random_aug(self,img):

        img=Image.fromarray(np.uint8(img)).convert('RGB')

        choice=np.random.choice([1,2,3,4])
        rand=np.random.uniform(0.2,0.7)

        if choice==1:
            en1=ImageEnhance.Color(img)
            img1=en1.enhance(rand)
            return np.asarray(img1)

        elif choice==2:
            en2=ImageEnhance.Brightness(img)
            img2=en2.enhance(rand)
            return np.asarray(img2)

        elif choice==3:
            en3=ImageEnhance.Contrast(img)
            img3=en3.enhance(rand)
            return np.asarray(img3)

        elif choice==4:
            return np.asarray(img)

