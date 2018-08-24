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
import urllib3
import textract
from pdf2image import convert_from_path
urllib3.disable_warnings()

#This part should be done by the user
#im_ = imread(image_file)
#im = rgb2gray(im_)


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

    def __init__(self, im, hight_resolution=True):
        self.image = im
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
        #print(sigma_choosen)
        MAX_95_CI_L = MAX - sigma_choosen*np.sqrt(np.var(values))
        rough_position = np.where((values > MAX_95_CI_L)==True)[0]
        fine_position = [rough_position[0]]
        thd = np.mean(rough_position[1:]-rough_position[:-1])
        for first, second in zip(rough_position[:-1],rough_position[1:]):
            if  second - first > 30:
                fine_position.append(second)
        return fine_position

    def determine_spike_position_row(self, values):
        try:
            MAX = np.mean(sorted(values)[-2:])
            sigma_choosen = 2# smaller one?
            for sigma in range(1,5):
                MAX_95_CI_L_ = MAX - sigma*np.sqrt(np.var(values))
                rough_position_ = np.where((values > MAX_95_CI_L_)==True)[0]
                #print(rough_position_)
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
        except Exception as e:
            #print(e)
            return [0, self.image.shape[0]]

    def crop_image(self):
        # Crop images along col first
        col_position = self.determine_spike_position_col(self.col_vals)
        # col_position.append() 0 position and end position
        if col_position[0] > 120:
            col_position.insert(0,3)
        if col_position[-1] < self.image.shape[1]-120:
            col_position.append(self.image.shape[1]-3)
        vertical_slices = []
        for first, second in zip(col_position[:-1],col_position[1:]):
            vertical_slices.append(self.image[:,first:second])
        # Crop vertical slices into chunks along row--different vetical_slice may contain diff rows
        row_position_list = []
        for vertical_slice in vertical_slices:
            row_vals = list([sum(r) for r in 1-vertical_slice])
            row_position = self.determine_spike_position_row(row_vals)
            if row_position[0] > 80:
                row_position.insert(0,3)
            if row_position[-1] < self.image.shape[0]-80:
                row_position.append(self.image.shape[0]-3)
            row_position_list.append(row_position)
        cropped = []
        row_position_length = [len(rp) for rp in row_position_list]
        self.table_shape = max(row_position_length)-1, len(col_position)-1
        # If all columns have the same number of rows
        if max(row_position_length)-min(row_position_length) == 0:
            for vertical_slice, row_position in zip(vertical_slices, row_position_list):
                for first, second in zip(row_position[:-1],row_position[1:]):
                    cropped.append(vertical_slice[first:second,:])
        else:
            max_length_p = row_position_list[row_position_length.index(max(row_position_length))]
            repeat_number_list = []
            for row_position in row_position_list:
                closest_index = [closest_value_index(i, max_length_p) for i in row_position]
                repeat_number = [i-j for i,j in zip(closest_index[1:], closest_index[:-1])]
                repeat_number_list.append(repeat_number)
            for vertical_slice, row_position, repeat_number in zip(vertical_slices, row_position_list, repeat_number_list):
                for first, second, repeat in zip(row_position[:-1], row_position[1:], repeat_number):
                    for _ in range(repeat):
                        cropped.append(vertical_slice[first:second,:])
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

    def write_to_csv(self, output_dir):
        df = self.image2df()
        df.to_csv(output_dir, index=None, header=False)
        print(f'Success, write to file:{output_dir}')
    
    def image2df(self):
        cropped = self.crop_image()
        container = np.zeros(self.table_shape, dtype=object)
        for col in range(self.table_shape[1]):
            for row in range(self.table_shape[0]):
                image = cropped[col*self.table_shape[0]+row]
                words = self.image2words(image)
                container[row, col] = words
        df = pd.DataFrame(data=container)
        #df.drop()
        df = df.replace('',np.nan).dropna(axis=1,how='all').dropna(axis=0,how='all')
        return df

    def plot(self, fig_size = (15,20)):
        g = plt.figure(figsize=fig_size)
        plt.imshow(self.image, cmap='gray')
        row_position, col_position = self.determine_spike_position_row(self.row_vals), self.determine_spike_position_col(self.col_vals)
        if row_position[0] > 80:
            row_position.insert(0,3)
        if row_position[-1] < self.image.shape[0]-120:
            row_position.append(self.image.shape[0]-3)
        if col_position[0] > 120:
            col_position.insert(0,3)
        if col_position[-1] < self.image.shape[1]-120:
            col_position.append(self.image.shape[1]-3)

        for row in row_position:
            plt.plot([0, self.image.shape[1]],[row, row], color='red', lw=2)
        for col in col_position:
            plt.plot([col, col],[0, self.image.shape[0]],color='red', lw=2)
        plt.show()

    def is_col_position_valid(self, col_position, img):
        allPoints = [[col, row] for col in list([col_position]) for row in range(img.shape[0])]
        allPoints = np.array([point for point in allPoints if img[point[1], point[0]]<0.8])
        if allPoints.shape[0] > img.shape[0]*0.8:
            return True
        else:
            return False

    def is_area_valid_table(self, row1_position, row2_position):
        img_between_rows = self.image[row1_position:row2_position, :]
        # looking for vertical line between two row positions
        rough_position = []
        for col in range(img_between_rows.shape[1]):
            if self.is_col_position_valid(col, img_between_rows):
                rough_position.append(col)
        try:
            fine_position = [rough_position[0]]
        except:
            return False
        if len(rough_position) > self.image.shape[1]*0.7:
            return True

        for first, second in zip(rough_position[:-1], rough_position[1:]):
            if second - first > 40:
                fine_position.append(second)
        if len(fine_position)>=3 or\
            (len(fine_position)==2 and 200<fine_position[0]<self.image.shape[1]-200 and 200<fine_position[1]<self.image.shape[1]-200) or\
            (len(fine_position)==1 and 200<fine_position[0]<self.image.shape[1]-200):
            return True
        else:
            return False

    def get_tables_position(self, row_positions):
        row1_positions, row2_positions = row_positions[:-1], row_positions[1:]
        # Get boolean values of wheter an area is table or not, [T,F] means (position1,position2) is valid while (position2, position3) invalid
        b = []
        for i, i_next in zip(row1_positions, row2_positions):
            is_table = self.is_area_valid_table(i, i_next)
            b.append(is_table)
        # Split into several lists and store in dictionary
        a = {0:[]}
        i = 0
        for idx, b in enumerate(b):
            if b:
                a[i].append(idx)
            else:
                i = i + 1
                a[i] = []
        # Remove empty lists 
        for key, value in a.copy().items():
            if value == []:
                del a[key]
        # Complete positions by append 
        for key, value in a.items():
            value.append(value[-1]+1)
        # Index to actual position
        for key, value in a.items():
            value = [row_positions[i] for i in value]
            a[key] = value
        return a

    def plot_tables(self, fig_size = (15,20)):
        try:
            self.tables_p_dict
        except:
            self.set_tables_dict()
        if self.tables_p_dict == {}:
            print('Empty table in this page')
            return None 
        for key, value in self.tables_p_dict.items():
            g = plt.figure(figsize = fig_size)
            plt.imshow(self.image[ value[0]:value[-1] ], cmap='gray')
            plt.show()
    
    def set_tables_dict(self):
        row_positions = self.determine_spike_position_row(self.row_vals)
        try:
            self.tables_p_dict
        except:
            self.tables_p_dict = self.get_tables_position(row_positions)
    




def rgb2gray(rgb):
    return np.round(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]),3)

def closest_value_index(a, b_list):
    abs_ab = [abs(a-b) for b in b_list]
    min_index = abs_ab.index(min(abs_ab))
    return min_index

def getPDFpages(path, keyword='接待时间'):
    text = textract.process(path).decode('utf-8')
    text_all_pages = text.split('\x0c')[:-1]
    pages = []
    for page, text in enumerate(text_all_pages):
        if keyword in text:
            pages.append(page)
    return pages

def pdf2image(path, pages, gray_scale=True):
    imgs = convert_from_path(path, 300, fmt='png', first_page=pages[0]+1, last_page=pages[-1]+1)
    im_list = [np.array(imgs[i])/255 for i in range(len(imgs))]
    if not gray_scale:
    	return im_list
    else:
	    im_gray_list = [rgb2gray(im) for im in im_list]
	    return im_gray_list

if __name__ == '__main__':
    pass
    #img2csv = Image2Csv_CL('/Users/sfi/Desktop/ocr/test4.png')
    #img2csv.write_to_csv()
    #img2csv = Image2Csv_CL('/Users/sfi/Desktop/ocr/test5.png')
    #img2csv.write_to_csv()
    #img2csv = Image2Csv_CL('/Users/sfi/Desktop/ocr/test6.png')
    #img2csv.write_to_csv()




















