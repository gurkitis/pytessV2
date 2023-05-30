import Block
import cv2
import xml.etree.ElementTree as ElementTree
import Helper


class Parser:
    def __init__(self):
        self._file_operator = Helper.FileOperator()

    def get_largest_text_block(self, lpp):
        text_blocks = self.get_text_blocks(lpp)
        largest_text_block = text_blocks[0]
        if len(text_blocks) > 1:
            for text_block in text_blocks[1:]:
                if text_block.get_img_size() > largest_text_block.get_img_size():
                    largest_text_block = text_block

        self._file_operator.save_abby_result_text(lpp, largest_text_block.get_text())
        return largest_text_block

    def get_text_blocks(self, lpp):
        return self._get_segments(lpp)

    def _get_segments(self, lpp):
        tree = ElementTree.parse(self._file_operator.get_abby_alto_path(lpp))
        blocks = []
        for page in tree.getroot().iter('Page'):
            for print_space in page.iter('PrintSpace'):
                for text_block in print_space.iter('TextBlock'):
                    block = Block.TextBlock(lpp, text_block, page)
                    blocks.append(block)

        return blocks
