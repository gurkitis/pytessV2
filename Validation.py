import fastwer
import Helper
import xml.etree.ElementTree as ElementTree
import Alto
import Ocr
import cv2
import math

class WordErrorRate:
    def __init__(self):
        self._file_operator = Helper.FileOperator()
        self._alto_parser = Alto.Parser()
        self._segmentator = Alto.Parser()

    def tess_test(self, lpp, prefix=''):
        hypo_text = self._file_operator.get_tess_result_text(lpp, prefix).split('\n')
        ref_text = self._file_operator.get_base_data(lpp).split('\n')
        return fastwer.score(hypo_text, ref_text, char_level=False)

    def abby_test(self, lpp):
        hypo_text = self._alto_parser.get_largest_text_block(lpp).get_text().split('\n')
        ref_text = self._file_operator.get_base_data(lpp).split('\n')
        return fastwer.score(hypo_text, ref_text, char_level=False)

    def draw_tess_error_heatmap(self, lpp, prefix='', color=(0, 0, 255), thickness=2):
        ocr = Ocr.Processor()
        data = ocr.run(lpp)
        data = data[data['text'].notnull()]

        img = self._segmentator.get_largest_text_block(lpp).get_img()

        ref_text = self._file_operator.get_base_data(lpp)
        ref_lines = ref_text.split('\n')

        data_line_nr = int(data.iloc[0]['line_num'])
        data_par_nr = int(data.iloc[0]['par_num'])

        ref_line = 0
        ref_word = 0
        for index, row in data.iterrows():
            data_curr_line = int(row['line_num'])
            data_curr_par = int(row['par_num'])
            if data_curr_line > data_line_nr or data_curr_par > data_par_nr:
                if data_curr_par > data_par_nr:
                    data_par_nr = data_curr_par
                data_line_nr = data_curr_line
                ref_line += 1
                ref_word = 0

            if row['text'] != ref_lines[ref_line].split(' ')[ref_word]:
                img = cv2.rectangle(img, (row['left'], row['top']),
                                    (row['left'] + row['width'], row['top'] + row['height']),
                                    color, thickness)
            ref_word += 1
        self._file_operator.save_image(img, lpp, f'{prefix}tess_word_error_map')

    def draw_abby_error_heatmap(self, lpp, prefix='', color=(0, 0, 255), thickness=2):
        text_block = self._segmentator.get_largest_text_block(lpp)
        ref_text = self._file_operator.get_base_data(lpp)
        ref_lines = ref_text.split('\n')

        img = text_block.get_orig_img()

        text_lines = []
        for text_line in text_block.get_text_block().iter('TextLine'):
            text_lines.append(text_line)

        if len(text_lines) != len(ref_lines):
            raise Exception('ref_lines not the same as text_lines')

        for text_line_index in range(len(text_lines)):
            ref_words = ref_lines[text_line_index].split(' ')
            text_words = []
            for word in text_lines[text_line_index].iter('String'):
                text_words.append(word)

            if len(ref_words) != len(text_words):
                print(f'line {text_line_index} was skipped due to word misalignment')
                continue
                raise Exception('ref_words not the same as text_words')

            for text_word_index in range(len(text_words)):
                text_word = text_words[text_word_index]
                if text_word.get('CONTENT') != ref_words[text_word_index]:
                    #print(f'line {text_line_index}\tword {text_word.get("CONTENT")}')
                    y, x = text_block.get_page_coordinates(int(text_word.get('VPOS')), int(text_word.get('HPOS')))
                    width, height = text_block.get_page_coordinates(int(text_word.get('WIDTH')), int(text_word.get('HEIGHT')))
                    img = cv2.rectangle(img, (math.floor(x), math.floor(y)),
                                        (math.ceil(x + width), math.ceil(y + height)), color, thickness)

        img = text_block.extract_img(text_block.get_text_block(), img)
        self._file_operator.save_image(img, lpp, f'{prefix}abby_word_error_map')


class CharacterErrorRate:
    def __init__(self):
        self._file_operator = Helper.FileOperator()
        self._alto_parser = Alto.Parser()

    def tess_test(self, lpp, prefix=''):
        hypo_text = self._file_operator.get_tess_result_text(lpp, prefix).split('\n')
        ref_text = self._file_operator.get_base_data(lpp).split('\n')
        return fastwer.score(hypo_text, ref_text, char_level=True)

    def abby_test(self, lpp):
        hypo_text = self._alto_parser.get_largest_text_block(lpp).get_text().split('\n')
        ref_text = self._file_operator.get_base_data(lpp).split('\n')
        return fastwer.score(hypo_text, ref_text, char_level=True)
