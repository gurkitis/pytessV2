import math
import cv2
import Helper
import xml.etree.ElementTree as ElementTree


class TextBlock:
    def __init__(self, lpp, text_block, page):
        self._lpp = lpp
        self._text_block = text_block
        self._page = page
        self._offset = 3
        self._file_operator = Helper.FileOperator()
        self._original_img = cv2.imread(self._file_operator.get_abby_jpg_path(self._lpp))
        self._img = self.extract_img(text_block)
        self._text = self._set_text()
        self._config_operator = Helper.ConfigOperator()
        self._confidence_threshold = float(self._config_operator.get_config('TESSERACT_CONFIG', 'ConfidenceThreshold'))

    def extract_img(self, block, img=None):
        if img is None:
            img = self._original_img
        h_coefficient = img.shape[0] / int(self._page.get('HEIGHT'))
        v_coefficient = img.shape[1] / int(self._page.get('WIDTH'))
        h_point, v_point = self.get_page_coordinates(int(block.get('HPOS')), int(block.get('VPOS')))
        return img[math.floor(v_point): math.ceil(v_point + int(block.get('HEIGHT')) * v_coefficient) + self._offset,
                   math.floor(h_point): math.ceil(h_point + int(block.get('WIDTH')) * h_coefficient) + self._offset]

    def get_page_coordinates(self, x, y):
        y_coefficient = self._original_img.shape[0] / int(self._page.get('HEIGHT'))
        x_coefficient = self._original_img.shape[1] / int(self._page.get('WIDTH'))
        return x * x_coefficient, y * y_coefficient

    def get_img(self):
        return self._img

    def get_orig_img(self):
        return self._original_img

    def get_img_size(self):
        return int(self._text_block.get('WIDTH')) * int(self._text_block.get('HEIGHT'))

    def get_text(self):
        return self._text

    def get_text_block(self):
        return self._text_block

    def draw_below_threshold_heatmap(self, prefix='', color=(0, 0, 255), thickness=2):
        img = self._original_img
        for string in self._text_block.iter('String'):
            if float(string.get('WC')) < self._confidence_threshold / 100:
                x, y = self.get_page_coordinates(int(string.get('HPOS')), int(string.get('VPOS')))
                width, height = self.get_page_coordinates(int(string.get('WIDTH')), int(string.get('HEIGHT')))
                start_point = (math.floor(x), math.floor(y))
                end_point = (math.ceil(x + width), math.ceil(y + height))
                img = cv2.rectangle(img, start_point, end_point, color, thickness)

        img = self.extract_img(self._text_block)
        self._file_operator.save_image(img, self._lpp, f'{prefix}abby_below_threshold')

    def _set_text(self):
        text = ''
        for text_line in self._text_block.iter('TextLine'):
            for string in text_line.iter('String'):
                text += string.get('CONTENT') + ' '
            text = text[:-1]
            text += '\n'
        text = text[:-1]
        return text