import pytesseract
import Helper
import Alto
import cv2
import re
import numpy


class Processor:
    def __init__(self):
        self._file_operator = Helper.FileOperator()
        self._segmentator = Alto.Parser()
        self._config_operator = Helper.ConfigOperator()
        self._primary_language = self._config_operator.get_config('TESSERACT_CONFIG', 'PrimaryLanguage')
        self._confidence_threshold = float(self._config_operator.get_config('TESSERACT_CONFIG', 'ConfidenceThreshold'))

    def set_primary_language(self, language):
        if language in self._config_operator.get_config('TESSERACT_CONFIG'):
            self._primary_language = language
        else:
            raise Exception(f'Language code {language} is not defined in language book')

    def set_confidence_threshold(self, threshold):
        if 0 <= threshold <= 100:
            self._confidence_threshold = threshold
        else:
            raise Exception(f'Threshold must be >=0 and <=100, passed: {threshold}')

    def run(self, lpp, prefix=''):
        print('pytesseract version: ' + pytesseract.__version__)
        block = self._segmentator.get_largest_text_block(lpp)
        img = block.get_img()
        # block.draw_below_threshold_heatmap()
        self._file_operator.save_image(img, prefix='page')
        data = pytesseract.image_to_data(img,
                                         self._primary_language,
                                         output_type='data.frame',
                                         config=self._get_tesseract_config())
        data.to_csv('data/results/data.csv')
        # data = self._spell_process(data, img, lpp)
        # data = self._language_process(data, img, lpp)
        data = self._reference_process(data, img, lpp)
        text = self._extract_text(data)
        self._file_operator.save_tess_result_text(lpp, text, prefix)
        data.to_csv('data/results/data_after.csv')
        return data

    def _draw_below_threshold_heatmap(self, data, img, lpp, prefix='', color=(0, 0, 255), thickness=2):
        for index, row in data.iterrows():
            if 0 < float(row['conf']) < self._confidence_threshold:
                img = cv2.rectangle(img, (row['left'], row['top']),
                                    (row['left'] + row['width'], row['top'] + row['height']),
                                    color, thickness)
        self._file_operator.save_image(img, lpp, f'{prefix}below_threshold')

    def _language_process(self, data, img, lpp=None, verbose=False):
        book_dict = self._config_operator.get_inverse_dict(self._config_operator.get_config('TESS_TO_BOOK_LANGUAGES'))
        _img = img
        for index, row in data.iterrows():
            if row['conf'] > 0:
                for lang in book_dict:
                    if lang == '':
                        continue
                    pattern = lang[:-1]
                    pattern += '\.$'
                    if re.search(pattern, row['text']):
                        start_cord = (row['left'] - 2, row['top'] - 2)
                        update_data = []
                        for indx, rw in data[index + 1:].iterrows():
                            if rw['conf'] > 0:
                                end_cord = (rw['left'] + rw['width'] + 2, rw['top'] + rw['height'] + 2)
                                if row['line_num'] == rw['line_num'] and row['par_num'] == rw['par_num']:
                                    _img = cv2.rectangle(_img, start_cord, end_cord, (0, 0, 255), 2)
                                else:
                                    _img = cv2.rectangle(_img, start_cord,
                                                         (row['left'] + row['width'], row['top'] + row['height']),
                                                         (0, 0, 255), 2)
                                    _img = cv2.rectangle(_img, (rw['left'], rw['top']),
                                                         end_cord,
                                                         (0, 0, 255), 2)

                                for lng in book_dict[lang]:
                                    update = pytesseract.image_to_data(img[rw['top'] - 2:  rw['top'] + rw['height'] + 2,
                                                                       rw['left'] - 2: rw['left'] + rw['width'] + 2],
                                                                       lng,
                                                                       output_type='data.frame',
                                                                       config=self._get_tesseract_config(psm=8))
                                    update = update[update['text'].notnull()]
                                    update_data.append({'lang': lang,
                                                        'conf': update.iloc[0]['conf'],
                                                        'text': update.iloc[0]['text']})

                                for update in update_data:
                                    if rw['conf'] < update['conf']:
                                        if verbose:
                                            print(f'{rw["text"]} ({rw["conf"]}) => {update["text"]} ({update["conf"]})')
                                        data.at[indx, 'conf'] = update['conf']
                                        data.at[indx, 'text'] = update['text']
                                break
        if lpp:
            self._file_operator.save_image(_img, lpp, 'language_process')
        return data

    def _spell_process(self, data, img, lpp=None):
        book_dict = self._config_operator.get_config('TESS_TO_BOOK_LANGUAGES')
        clean_data = data[data['text'].notnull()]
        _img = img
        for index, row in clean_data.iterrows():
            word = None
            if re.search('\[.*]', row['text']):
                word = re.sub('[\[\]]', '', row['text'])

            elif re.search('\[', row['text']):
                for indx, next_word in data[index:].iterrows():
                    if next_word['conf'] <= 0:
                        continue

                    if re.search(']', next_word['text']):
                        word = re.sub('\[', '', row['text']) + re.sub(']', '', next_word['text'])
                        data.at[index, 'width'] = next_word['left'] - row['left'] + next_word['width']
                        if row['height'] < next_word['height']:
                            data.at[index, 'height'] = next_word['height']
                        data.drop(labels=range(index + 1, indx))

                    break
            if word is not None:
                _img = cv2.rectangle(img, (data.at[index, 'left'] - 2, data.at[index, 'top'] - 2),
                                     (data.at[index, 'left'] + data.at[index, 'width'] + 2,
                                      data.at[index, 'top'] + data.at[index, 'height'] + 2),
                                     (0, 0, 255), 2)
                max_score = {'conf': row['conf'], 'text': row['text']}

                for lang in book_dict:
                    if book_dict[lang] == '':
                        continue

                    score = pytesseract.image_to_data(
                        img[data.at[index, 'top'] - 2:  data.at[index, 'top'] + data.at[index, 'height'] + 2,
                        data.at[index, 'left'] - 2: data.at[index, 'left'] + data.at[index, 'width'] + 2],
                        lang,
                        output_type='data.frame',
                        config=self._get_tesseract_config(psm=8))
                    score = score[score['text'].notnull()]

                    if len(score) and score.iloc[0]['conf'] > max_score['conf']:
                        max_score['conf'] = score.iloc[0]['conf']
                        max_score['text'] = score.iloc[0]['text']

                data.at[index, 'conf'] = max_score['conf']
                data.at[index, 'text'] = max_score['text']
        if lpp:
            self._file_operator.save_image(img, lpp, f'spell_process')
        return data

    def _reference_process(self, data, img, lpp):
        threshold = img.shape[1] * 0.03
        most_left = None
        _img = img
        clean_data = data[data['text'].notnull()]
        for index, row in clean_data.iterrows():
            if most_left is None:
                most_left = row
                continue
            if row['left'] < most_left['left']:
                most_left = row

        base_words = []
        for index, row in clean_data.iterrows():
            if row['left'] - most_left['left'] < threshold:
                base_words.append(index)

        info_blocks = []
        info_block = []
        for index, row in clean_data.iterrows():
            if index in base_words:
                info_blocks.append(info_block)
                info_block = []
            info_block.append(row)
        info_blocks.append(info_block)

        for index in range(len(info_blocks)):
            if len(info_blocks) <= index:
                break
            info_block = info_blocks[index]
            if info_block[0]['par_num'] == info_block[-1]['par_num']:
                break
                del info_blocks[index]

        paragraphs = []
        for info_block in info_blocks[:-1]:
            par_num = info_block[0]['par_num']
            last_paragraph = []
            for word in info_block:
                if word['par_num'] > par_num:
                    par_num = word['par_num']
                    last_paragraph = []
                last_paragraph.append(word)

            if last_paragraph[0].name not in base_words:
                paragraphs.append(last_paragraph)

        for paragraph in paragraphs:
            top = paragraph[0]
            left = paragraph[0]
            bottom = paragraph[0]
            right = paragraph[0]
            for word in paragraph:
                if word['top'] < top['top']:
                    top = word
                if word['left'] < left['left']:
                    left = word
                if word['top'] + word['height'] > bottom['top'] + bottom['height']:
                    bottom = word
                if word['left'] + word['width'] > right['left'] + right['width']:
                    right = word
            _img = cv2.rectangle(_img, (left['left'] + 2, top['top'] + 2),
                                 (right['left'] + right['width'] - 2, bottom['top'] + bottom['height'] - 2),
                                 (0, 0, 255), 2)
        if lpp:
            self._file_operator.save_image(_img, lpp, 'reference_process')

        lang_dict = self._config_operator.get_config('TESS_TO_BOOK_LANGUAGES')
        #lang_dict = ['lav', 'rus']
        for paragraph in paragraphs:
            for word in paragraph:
                max_score = {'conf': word['conf'], 'text': word['text']}
                for lang in lang_dict:
                    if lang_dict[lang] == '':
                        continue

                    if word['top'] + word['height'] + 2 > img.shape[0]:
                        word['height'] += 2
                    if word['left'] + word['width'] + 2 > img.shape[1]:
                        word['width'] += 2
                    if word['top'] >= 2:
                        word['top'] -= 2
                    if word['left'] >= 2:
                        word['left'] -= 2

                    score = pytesseract.image_to_data(
                        img[word['top']:  word['top'] + word['height'],
                        word['left']: word['left'] + word['width']],
                        lang,
                        output_type='data.frame',
                        config=self._get_tesseract_config(psm=8))
                    score = score[score['text'].notnull()]

                    if len(score) and score.iloc[0]['conf'] > max_score['conf']:
                        if type(score.iloc[0]['text']) == numpy.float64:
                            score.at[score.iloc[0].name, 'text'] = str(int(score.iloc[0]['text']))
                        print(f'update {lang}:\n\tbefore {max_score}')
                        max_score['conf'] = score.iloc[0]['conf']
                        max_score['text'] = score.iloc[0]['text']
                        print(f'\tafter {max_score}')

                data.at[word.name, 'conf'] = max_score['conf']
                data.at[word.name, 'text'] = max_score['text']

        return data

    def _extract_text(self, data):
        data = data[data['text'].notnull()]
        line_nr = int(data.iloc[0]['line_num'])
        par_nr = int(data.iloc[0]['par_num'])
        text = ''
        for index, row in data.iterrows():
            curr_line = int(row['line_num'])
            curr_par = int(row['par_num'])
            if curr_line > line_nr or curr_par > par_nr:
                if curr_par > par_nr:
                    par_nr = curr_par
                line_nr = curr_line
                text = text[:-1]
                text += '\n'
            text += row['text'] + ' '
        return text

    def _get_tesseract_config(self, psm=None):
        if psm is None:
            psm = '--psm ' + self._config_operator.get_config('TESSERACT_CONFIG', 'psm')
        else:
            psm = f'--psm {psm}'
        tessdata_dir = '--tessdata-dir ' + self._config_operator.get_config('TESSERACT_CONFIG', 'TessdataDir')
        return psm + '\n' + tessdata_dir
