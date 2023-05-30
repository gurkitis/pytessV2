import configparser
import cv2


class FileOperator:
    def __init__(self):
        self._config_operator = ConfigOperator()
        self._abby_data_path = self._config_operator.get_config('DATA', 'AbbyDataPath') + '/' + self._config_operator.get_config('DATA', 'AbbyDataPrefix')
        self._abby_page_min_number_count = self._config_operator.get_config('DATA', 'AbbyPageMinNumberCount')
        self._tess_result_path = self._config_operator.get_config('DATA', 'TessResultPath')
        self._abby_result_path = self._config_operator.get_config('DATA', 'AbbyResultPath')
        self._base_data_path = self._config_operator.get_config('DATA', 'BaseDataPath')
        self._image_save_path = self._config_operator.get_config('DATA', 'ImageSavePath')

    def get_abby_alto_path(self, lpp):
        return self._abby_data_path + self._lpp_parser(lpp) + '_alto.xml'

    def get_abby_jpg_path(self, lpp):
        return self._abby_data_path + self._lpp_parser(lpp) + '.jpg'

    def save_tess_result_text(self, lpp, data, prefix=''):
        with open(f'{self._tess_result_path}/{prefix}_{str(lpp)}.txt', 'w') as file:
            file.write(data)

    def get_tess_result_text(self, lpp, prefix=''):
        with open(f'{self._tess_result_path}/{prefix}_{str(lpp)}.txt', 'r') as file:
            return file.read()

    def save_abby_result_text(self, lpp, data):
        with open(self._abby_result_path + '/abby_' + str(lpp) + '.txt', 'w') as file:
            file.write(data)

    def get_abby_result_text(self, lpp):
        with open(self._abby_result_path + '/tess_' + str(lpp) + '.txt', 'r') as file:
            return file.read()

    def get_base_data(self, lpp):
        with open(self._base_data_path + '/base_' + str(lpp) + '.txt', 'r') as file:
            return file.read()

    def save_image(self, img, lpp='', prefix=''):
        if prefix != '' and lpp != '':
            prefix += '_'
        cv2.imwrite(f'{self._image_save_path}/{prefix}{lpp}.jpg', img)

    def _lpp_parser(self, lpp):
        lpp = str(lpp)
        if len(lpp) < int(self._abby_page_min_number_count):
            diff = int(self._abby_page_min_number_count) - len(lpp)

            for tmp in range(diff):
                lpp = '0' + lpp

        return lpp


class ConfigOperator:
    def __init__(self):
        self._config_parser = configparser.ConfigParser()
        self._config_parser.read('config.ini')

    def get_config(self, group, name=None):
        if name is not None:
            return self._config_parser[group][name]

        return dict(self._config_parser.items(group))

    @staticmethod
    def get_inverse_dict(data):
        inv_dict = dict()
        for tess in data:
            book = data[tess].split(' ')
            for lang in book:
                if lang in inv_dict:
                    inv_dict[lang].append(tess)
                else:
                    inv_dict[lang] = [tess]

        return inv_dict
