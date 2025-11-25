import os
from utils.c_docx_parser import DocxParser
from utils.c_file_generate import FileGenerate

class GeneratePipeline:
    def __init__(self, config):
        file_name = (config.FILE_CONFIG.docx_file_path).split('/')[-1].split('.')[0]
        save_path = f'{config.FILE_CONFIG.save_path}/{file_name}'
        os.makedirs(save_path, exist_ok=True)
        config.FILE_CONFIG.save_path = save_path

        self.dp = DocxParser(config=config)
        self.fg = FileGenerate(config=config)

    def process(self):
        parser_docx_res = self.dp.debug_main_process()
        final_res = self.fg.debug_main_process(input_dict=parser_docx_res)
        return final_res