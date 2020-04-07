import cv2
import numpy as np
import glob
import os

PATH = 'directory path'
NOISE_FOLDER = 'output folder'
dataset_path = os.path.join(PATH, 'dataset folder')
IMAGES = glob.glob(dataset_path + os.sep + '*.jpg')
#IMAGES = os.listdir(dataset_path)

def posterize(im, levels=3):
    n = levels  # Number of levels of quantization


    indices = np.arange(0, 256)  # List of all colors
    divider = np.linspace(0, 255, n + 1)[1]  # we get a divider
    quantiz = np.int0(np.linspace(0, 255, n))  # we get quantization colors

    color_levels = np.clip(np.int0(indices / divider), 0, n - 1)  # color levels 0,1,2..

    palette = quantiz[color_levels]  # Creating the palette

    im2 = palette[im]  # Applying palette on image
    im2 = cv2.convertScaleAbs(im2)  # Converting image back to uint8

    return im2


# opencv tutorial
def proceed_2_distort(filename, sensible=25, save=None, display=True):


    img = cv2.imread(filename)
    if display:
        cv2.imshow('Image', img)
    # if not working correctly change to cv2.COLOR_BGR2HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # by changing this value, you decide the range
    # to pick different green colors..
    sensitivity = sensible
    lower = np.array([60 - sensitivity, 100, 50])
    upper = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    background = np.full(img.shape, 255, dtype=np.uint8)
    green2black = cv2.bitwise_not(background, img, mask=mask)

    blurImg = cv2.blur(green2black, (30, 30))
    blurry_poste = posterize(blurImg, 8)
    gausBlur = cv2.GaussianBlur(blurry_poste, (7, 7), 0)
    # cv2.imshow('Posterization1 + Gaussian', gausBlur)
    gausBlur_post = posterize(gausBlur, 4)
    final_result = cv2.GaussianBlur(gausBlur_post, (7, 7), 0)
    # cv2.imshow('Posterization2 + Gaussian', final_result)

    if display:
        cv2.imshow('Result', final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save:
        out_dir = os.path.join(PATH, NOISE_FOLDER)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = filename.rsplit(os.sep, 1)[1]

    cv2.imwrite(os.path.join(out_dir, filename), final_result)

if __name__ == '__main__':
    idx = 5
    for i, img in enumerate(IMAGES):
        proceed_2_distort(img, save=True)

        if i % 1000 == 0:
            print(f"Already {i} images(s).")

'References:'
'https://stackoverflow.com/questions/11064454/adobe-photoshop-style-posterization-and-opencv '
'https://stackoverflow.com/questions/52107379/intensify-or-increase-saturation-of-an-image'
'https://medium.com/@gastonace1/detecci%C3%B3n-de-objetos-por-colores-en-im%C3%A1genes-con-python-y-opencv-c8d9b6768ff'
'https://justpaste.it/distorsionar_imagenes'