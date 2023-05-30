import Alto
import cv2
import Ocr
import Validation
import alive_progress
import pandas
import Helper
import time


def avg_conf(data):
    count = 0
    conf = 0
    for index, row in data.iterrows():
        if row['conf'] > 0:
            count += 1
            conf += row['conf']
    return conf / count


if __name__ == '__main__':
    config_operator = Helper.ConfigOperator()
    lpps = [18, 155, 401, 669, 1000]
    test_name = 'reference'
    ocr = Ocr.Processor()
    wer = Validation.WordErrorRate()
    cer = Validation.CharacterErrorRate()
    cer_data = []
    wer_data = []
    time_data = []
    conf_data = []

    for lpp in lpps:
        print(f'processing {lpp} lpp')
        start_time = time.time()
        data_frame = ocr.run(lpp, test_name)
        end_time = time.time()
        wer_data.append(wer.tess_test(lpp, test_name))
        cer_data.append(cer.tess_test(lpp, test_name))
        conf_data.append(avg_conf(data_frame))
        time_data.append(end_time - start_time)
        print(f'spent time: {end_time - start_time}')

    data = {
        f'wer_{test_name}': wer_data,
        f'cer_{test_name}': cer_data,
        f'time_{test_name}': time_data,
        f'conf_{test_name}': conf_data
    }

    pandas.DataFrame(data, lpps).T.to_csv(
        f'/home/roberts/PycharmProjects/pytessV2/data/results/tests/{test_name}.csv'
    )