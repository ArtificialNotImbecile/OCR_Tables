from skimage import filters, segmentation, io, feature
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
from matplotlib.pyplot import imread
from scipy.misc import imsave
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from aip import AipOcr





class Image2Csv_CL:
    """
    Convert image of table with CL(clear line) to csv file
    """
    APP_ID = '11655489'
    API_KEY = 'UWRvLnevXSiXVUZA52wic0FN'
    SECRET_KEY = 'R37MCzrwZgWNV8u9ghz53mctuTO5tIF2'
    baiduclient = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    options = {}
    options["language_type"] = "CHN_ENG"
    #options["detect_direction"] = "true"
    #options["detect_language"] = "true"
    #options["probability"] = "true"

    def __init__(self, image_file, hight_resolution=True):
        im_ = imread(image_file)
        im = rgb2gray(im_)
        self.image = im
        self.image_dir = image_file
        val  = filters.gaussian(im)
        mask = im < val
        clean_border = segmentation.clear_border(mask)
        if hight_resolution:
            self.row_vals = list([sum(r) for r in 1-im  ])
            self.col_vals = list([sum(r) for r in 1-im.T])
        else:
            self.row_vals = list([sum(r) for r in clean_border  ])
            self.col_vals = list([sum(c) for c in clean_border.T])

    def determine_spike_position(self, values):
        MAX = np.mean(sorted(values)[-2:])
        # This should be more flexible, consider 1-sigma, two-sigma and three-sigma. Then decide which one is the best.
        # For example, 2-sigma we have 2 position and 3-sigma we hvae 4 position, and 3-sigma have 100 position. 
        # Actually, if we want, say 
        MAX_95_CI_L = MAX - 4*np.sqrt(np.var(values))
        rough_position = np.where((values > MAX_95_CI_L)==True)[0]
        fine_position = [rough_position[0]]
        thd = np.mean(rough_position[1:]-rough_position[:-1])
        for first, second in zip(rough_position[:-1],rough_position[1:]):
            if  second - first > 15:
                fine_position.append(second)
        return fine_position

    def determine_spike_position_col(self, values):
        MAX = np.mean(sorted(values)[-2:])
        sigma_choosen = 2 #3.5 or 2? smaller one?
        for sigma in range(1,5):
            MAX_95_CI_L_ = MAX - sigma*np.sqrt(np.var(values))
            rough_position_ = np.where((values > MAX_95_CI_L_)==True)[0]
            if 4 < len(rough_position_) < 20 and sorted([values[i] for i in rough_position_])[0] > MAX_95_CI_L_+50:
                sigma_choosen = sigma
        print(sigma_choosen)
        MAX_95_CI_L = MAX - sigma_choosen*np.sqrt(np.var(values))
        rough_position = np.where((values > MAX_95_CI_L)==True)[0]
        fine_position = [rough_position[0]]
        thd = np.mean(rough_position[1:]-rough_position[:-1])
        for first, second in zip(rough_position[:-1],rough_position[1:]):
            if  second - first > 30:
                fine_position.append(second)
        return fine_position

    def determine_spike_position_row(self, values):
        MAX = np.mean(sorted(values)[-2:])
        sigma_choosen = 2# smaller one?
        for sigma in range(1,5):
            MAX_95_CI_L_ = MAX - sigma*np.sqrt(np.var(values))
            rough_position_ = np.where((values > MAX_95_CI_L_)==True)[0]
            if len(rough_position_) < self.image.shape[0]/20 and sorted([values[i] for i in rough_position_])[0] > MAX_95_CI_L_+50:
                sigma_choosen = sigma
        #print(sigma_choosen)
        MAX_95_CI_L = MAX - sigma_choosen*np.sqrt(np.var(values))
        rough_position = np.where((values > MAX_95_CI_L)==True)[0]
        fine_position = [rough_position[0]]
        thd = np.mean(rough_position[1:]-rough_position[:-1])
        for first, second in zip(rough_position[:-1],rough_position[1:]):
            if  second - first > 30: # what's the proper value?
                fine_position.append(second)
        return fine_position

    def crop_image(self):
        # Crop images along col first
        row_position, col_position = self.determine_spike_position_row(self.row_vals), self.determine_spike_position_col(self.col_vals)
        #row_position.append() 0 position and end position
        if row_position[0] > 80 or row_position[-1] < self.image.shape[0]-120:
            row_position.append(self.image.shape[0])
            row_position.insert(0,0)
        if col_position[0] > 120:
            col_position.append(self.image.shape[1])
            col_position.insert(0,0)
        self.table_shape = len(row_position)-1,len(col_position)-1,
        vertival_slices = []
        for first, second in zip(col_position[:-1],col_position[1:]):
            vertival_slices.append(self.image[:,first:second])
        # Crop vertical slices into chunks along row
        cropped = []
        for vertival_slice in vertival_slices:
            for first, second in zip(row_position[:-1],row_position[1:]):
                cropped.append(vertival_slice[first:second,:])
        return cropped
        #return np.array(cropped).reshape(len(col_position)-1,len(row_position)-1).T
        
    def image2words(self, image_array_2d):
        s, encoded_image = cv2.imencode('.png', np.uint8(image_array_2d*255))
        content = encoded_image.tobytes(order='C')
        result = self.baiduclient.basicGeneral(content, self.options)
        try:
            return "".join([i['words'] for i in result['words_result']])
        except:
            return ''

    def write_to_csv(self):
        cropped = self.crop_image()
        container = np.zeros(self.table_shape, dtype=object)
        for col in range(self.table_shape[1]):
            for row in range(self.table_shape[0]):
                image = cropped[col*self.table_shape[0]+row]
                words = self.image2words(image)
                container[row, col] = words

        df = pd.DataFrame(data=container)
        #df.drop()
        affix = self.image_dir.split('.')[-1]
        csv_dir = self.image_dir.replace("."+affix, '.csv')
        df = df.replace('',np.nan).dropna(axis=1,how='all').dropna(axis=0,how='all')
        df.to_csv(csv_dir, index=None, header=False)
        print(f'Success, write to file:{csv_dir}')

    def plot(self, fig_size = (15,20)):
        g = plt.figure(figsize=fig_size)
        plt.imshow(self.image, cmap='gray')
        row_position, col_position = self.determine_spike_position_row(self.row_vals), self.determine_spike_position_col(self.col_vals)
        if row_position[0] > 80 or row_position[-1] < self.image.shape[0]-120:
            row_position.append(self.image.shape[0]-3)
            row_position.insert(0,3)
        if col_position[0] > 120:
            col_position.append(self.image.shape[1]-3)
            col_position.insert(0,3)
        for row in row_position:
            plt.plot([0, self.image.shape[1]],[row, row], color='red', lw=2)
        for col in col_position:
            plt.plot([col, col],[0, self.image.shape[0]],color='red', lw=2)
        plt.show()


def rgb2gray(rgb):
    return np.round(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]),3)

if __name__ == '__main__':
    pass
    #img2csv = Image2Csv_CL('/Users/sfi/Desktop/ocr/test4.png')
    #img2csv.write_to_csv()
    #img2csv = Image2Csv_CL('/Users/sfi/Desktop/ocr/test5.png')
    #img2csv.write_to_csv()
    #img2csv = Image2Csv_CL('/Users/sfi/Desktop/ocr/test6.png')
    #img2csv.write_to_csv()




















