# -*- coding: utf-8 -*-
import re
import json
import copy
import time
import random
from copy import deepcopy
from typing import Dict, Any
import numpy as np
from utils.my_util import *
from concurrent.futures import ThreadPoolExecutor, as_completed

class FileGenerate:
    def __init__(self, config):
        self.config = config
        self.score_table_prompt_path = config.PROMPT_CONFIG.score_table_prompt_path   # 从表格list中提取评分表prompt
        self.fuse_table_tec_prompt_path = config.PROMPT_CONFIG.fuse_table_tec_prompt_path     # 融合表格和技术建议书prompt
        self.expand_dir_prompt_path = config.PROMPT_CONFIG.expand_dir_prompt_path     # 扩充目录prompt
        self.fill_content_prompt_path = config.PROMPT_CONFIG.fill_content_prompt_path     # 填充内容prompt
        self.polish_content_prompt_path = config.PROMPT_CONFIG.polish_content_prompt_path     # 内容润色prompt
        self.overview_project_prompt_path = config.PROMPT_CONFIG.overview_project_prompt_path     # 项目总览prompt
        self.save_path = config.FILE_CONFIG.save_path
        self.num_workers = config.num_workers
        self.single_spec_data = config.single_spec_data     # 技术规范是单个文件
        self.single_score_table = config.single_score_table # 评分表是单个文件

        self.step4_score_table_save_path = f'{self.save_path}/step4_score_table.json'
        self.step4_spec_save_path = f'{self.save_path}/step4_spec.json'
        self.step4_st_tech_save_path = f'{self.save_path}/step4_st_tech.json'
        self.step5_st_tech_spec_save_path = f'{self.save_path}/step5_st_tech_spec.json'
        self.base_info_save_path = f'{self.save_path}/base_info.json'
        self.expand_dir_save_path = f'{self.save_path}/step6_expand_dir.json'
        self.final_res_save_path = f'{self.save_path}/step6_final_res.json'
        self.polish_content_save_path = f'{self.save_path}/step6_polish_content.json'

        self.progress_file = self.config.progress_file
        self.normal_log_file = self.config.normal_log_file

        self.dotx_template_path = config.dotx_template_path
        self.real_examples_template_path = config.real_examples_template_path

        self.model_name = config.model_name
        self.api_setting = open_json(config.api_setting_path)

        self.completed_steps = 0

        self.style_setting_id = config.style_setting_id
        self.stlye_json_data = open_json(config.style_json_path)['personas'][self.style_setting_id]
        # 页面相关配置
        self.max_pages = config.get('max_pages', 300)  # 最大页数，默认300页
        self.chars_per_page = config.get('chars_per_page', 1000)  # 每页字数，默认850字
        # 计算总字数范围
        self.total_chars_max = self.max_pages * self.chars_per_page

    def _write_progress(self):
        self.completed_steps += 1
        with open(self.progress_file, "w", encoding="utf-8") as f:
            f.write(f"{self.completed_steps}/{self.config.total_steps}\n")

    # 提取技术规范书
    def extract_spec(self, json_data):
        print('提取技术规范书...')
        final_res = []
        for item in json_data:
            if '技术规范' in item['section']:
                if item['layers'] != [] or item['content'] != []:
                    final_res = item
                    break
        return final_res

    # 提取评分表
    def extract_score_table(self, tables_data):
        if tables_data == None or tables_data == '':
            return None
        extract_table_prompt = open_prompt(self.score_table_prompt_path)
        new_tables_data = []
        # 给表格数据编号
        for idx, item in enumerate(tables_data):
            new_tables_data.append(
                {
                    'id': idx,
                    'table_data': item['table_data']
                }
            )
        new_prompt = update_prompt(
            [
                ['[input_tables]', json.dumps(new_tables_data, ensure_ascii=False)]
            ],
            extract_table_prompt
        )
        result = use_llm_models(
            new_prompt, 
            model_name=self.model_name, 
            base_url=self.api_setting[self.model_name]['base_url'], 
            api_key=self.api_setting[self.model_name]['api_key']
        )
        result = json.loads(result)
        result = tables_data[result['id'][0]]
        return result

    # 总结项目概况
    def overview_project(self, spec_data):
        raw_prompt = open_prompt(self.overview_project_prompt_path)
        new_prompt = update_prompt(
            [
                ['[input_spec]', json.dumps(spec_data, ensure_ascii=False)]
            ],
            raw_prompt
        )
        result = use_llm_models(
            new_prompt, 
            model_name=self.model_name, 
            base_url=self.api_setting[self.model_name]['base_url'], 
            api_key=self.api_setting[self.model_name]['api_key']
        )
        return result

    # 融合评分表和技术建议书
    def fuse_table_tec(self, table_data, tech_data=''):
        if table_data == None or table_data == '':
            return ''
        prompt = open_prompt(self.fuse_table_tec_prompt_path)
        json_str = json.dumps(table_data, ensure_ascii=False)
        new_prompt = update_prompt(
            [
                ['[ref_table_json]', json_str],
                ['[ref_tec_advice_content]', tech_data]
            ],
            prompt
        )
        result = use_llm_models(
            new_prompt, 
            model_name=self.model_name, 
            base_url=self.api_setting[self.model_name]['base_url'], 
            api_key=self.api_setting[self.model_name]['api_key']
        )
        result = json.loads(result)
        return result

    # 融合评分表和技术建议书和技术规范
    def fuse_table_tec_spec(self, fuse_res1, spec):
        temp_res = copy.deepcopy(fuse_res1)
        if temp_res == None or temp_res == '':
            temp_res = []
            next_num = 1
        else:
            prev_section = temp_res[-1]['section']
            m = re.match(r'(\d+)\.(.+)', prev_section)
            if m:
                next_num = int(m.group(1)) + 1
            else:
                next_num = 1

        def renumber_layers(layers, parent_number):
            new_layers = []
            for idx, item in enumerate(layers, 1):
                # 提取原始标题（去掉原编号）
                section = item.get('section', '').strip()
                m = re.match(r'^(?:\d+(?:\.\d+)*[)\.\．]?\s*)+([\s\S]*)', section)
                if m:
                    title = m.group(1).strip()
                else:
                    title = section
                new_number = f"{parent_number}.{idx}"
                new_section = f"{new_number} {title}" if title else new_number
                new_item = copy.deepcopy(item)
                new_item['section'] = new_section
                if 'layers' in new_item and new_item['layers']:
                    new_item['layers'] = renumber_layers(new_item['layers'], new_number)
                new_layers.append(new_item)
            return new_layers

        # 直接将spec的layers数组中的每一项添加到结果中
        spec_layers = spec.get("layers", [])
        for idx, item in enumerate(spec_layers, 1):
            # 提取原始标题（去掉原编号）
            section = item.get('section', '').strip()
            m = re.match(r'^(?:\d+(?:\.\d+)*[)\.\．]?\s*)+([\s\S]*)', section)
            if m:
                title = m.group(1).strip()
            else:
                title = section
            new_number = f"{next_num}.{idx}"
            new_section = f"{new_number} {title}" if title else new_number
            new_item = copy.deepcopy(item)
            new_item['section'] = new_section
            if 'layers' in new_item and new_item['layers']:
                new_item['layers'] = renumber_layers(new_item['layers'], new_number)
            temp_res.append(new_item)

        return temp_res

    def process_tree_with_leaf_nodes(self, data, base_info='',
                                     is_leaf_node_fn=None,
                                     process_leaf_node_fn=None,
                                     reconstruct_tree_fn=None):
        """
        通用的树叶节点批处理方法。
        - is_leaf_node_fn: 判断是否为叶子节点的函数(item) -> bool
        - process_leaf_node_fn: 处理叶子节点的函数(leaf_info) -> (leaf_info, processed_item)
        - reconstruct_tree_fn: 重建树结构的函数(item, processed_leafs) -> item
        """
        leaf_nodes = []
        def collect_leaf_nodes(item, parent_path=''):
            if isinstance(item, list):
                for sub_item in item:
                    collect_leaf_nodes(sub_item, parent_path)
                return
            if not isinstance(item, dict):
                return
            current_title = item.get('section', '').strip()
            new_path = f"{parent_path}->{current_title}" if parent_path else current_title
            if item.get('layers'):
                for child in item['layers']:
                    collect_leaf_nodes(child, new_path)
                return
            if is_leaf_node_fn and is_leaf_node_fn(item):
                leaf_nodes.append({
                    'item': item,
                    'parent_path': new_path,
                    'original_item': deepcopy(item)
                })
        collect_leaf_nodes(data)

        # 叶子数量
        leaf_count = len(leaf_nodes)

        max_workers = min(self.num_workers, len(leaf_nodes))
        processed_leafs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_leaf = {
                executor.submit(process_leaf_node_fn, leaf_info): leaf_info
                for leaf_info in leaf_nodes
            }
            completed_count = 0
            for future in as_completed(future_to_leaf):
                leaf_info, processed_item = future.result()
                processed_leafs.append((leaf_info, processed_item))
                completed_count += 1
                print(f"已完成: {completed_count}/{len(leaf_nodes)}")

        if reconstruct_tree_fn is None:
            def reconstruct_tree(item, processed_leafs):
                if isinstance(item, list):
                    return [reconstruct_tree(sub_item, processed_leafs) for sub_item in item]
                if not isinstance(item, dict):
                    return item
                if item.get('layers'):
                    item['layers'] = [
                        reconstruct_tree(child, processed_leafs)
                        for child in item['layers']
                    ]
                    if 'content' in item:
                        item['content'] = []
                    return item
                if is_leaf_node_fn and is_leaf_node_fn(item):
                    for leaf_info, processed_item in processed_leafs:
                        if leaf_info['item'] == item:
                            return processed_item
                return item
        else:
            reconstruct_tree = reconstruct_tree_fn
        return reconstruct_tree(data, processed_leafs)

    # 重构expand_dir，使用通用框架
    def expand_dir(self, data, base_info=''):
        raw_prompt = open_prompt(self.expand_dir_prompt_path)
        def is_leaf_node(item):
            return bool(item.get('content'))
        def process_leaf_node(leaf_info):
            try:
                item = leaf_info['item']
                parent_path = leaf_info['parent_path']
                temp_json_str = json.dumps(item, ensure_ascii=False)
                temp_prompt = deepcopy(raw_prompt)
                temp_prompt = update_prompt(
                    [
                        ['[input_json]', temp_json_str],
                        ['[parent_path]', parent_path],
                        ['[base_info]', base_info]
                    ],
                    temp_prompt
                )
                temp_res = use_llm_models(
                    temp_prompt,
                    model_name=self.model_name,
                    base_url=self.api_setting[self.model_name]['base_url'],
                    api_key=self.api_setting[self.model_name]['api_key']
                )
                return leaf_info, json.loads(temp_res)
            except Exception as e:
                print(f"处理叶子节点时出错: {e}")
                return leaf_info, leaf_info['original_item']
        return self.process_tree_with_leaf_nodes(
            data,
            base_info=base_info,
            is_leaf_node_fn=is_leaf_node,
            process_leaf_node_fn=process_leaf_node
        )

    # 重构fill_content，使用通用框架
    def fill_content(self, input_data, base_info=''):
        raw_prompt = open_prompt(self.fill_content_prompt_path)
        
        def is_leaf_node(item: Dict[str, Any]) -> bool:
            return item.get("content") is not None
        
        # 计算所有叶子节点
        leaf_nodes = []
        def collect_leaves(node):
            if isinstance(node, list):
                for item in node:
                    collect_leaves(item)
            elif isinstance(node, dict):
                if node.get('content') is not None:
                    leaf_nodes.append(node)
                if node.get('layers'):
                    for child in node['layers']:
                        collect_leaves(child)
        
        collect_leaves(input_data)
        total_leaves = len(leaf_nodes)
        
        # 计算每个叶子节点的平均字数范围
        avg_chars_max = self.total_chars_max // total_leaves
        
        def is_leaf_node(item):
            return item.get('content') is not None

        # 根据功能性需求和非功能性需求设置不同字数和段落数
        def paragraph_character_num_set(category):
            if category == '功能性需求':
                # 功能性需求需要更详细的描述
                par_num_min = 4  # 至少4段
                par_num_max = 6  # 最多6段
                cha_number = str(avg_chars_max)
            else:  # 非功能性需求
                # 非功能性需求相对简洁
                par_num_min = 2  # 至少2段
                par_num_max = 4  # 最多4段
                cha_number = str(avg_chars_max * 3 // 5)

            return par_num_min, par_num_max, cha_number

        def process_leaf_node(leaf_info):
            try:
                item = leaf_info['item']
                parent_path = leaf_info['parent_path']
                temp_json_str = json.dumps(item, ensure_ascii=False)
                temp_prompt = deepcopy(raw_prompt)

                par_num_min, par_num_max, cha_number = paragraph_character_num_set(item['category'])

                idx = random.randint(par_num_min, par_num_max)
                # cha_number = ''
                # for i in range(idx):
                #     cha_number += f'第{i+1}段的字数至少是{cha_num}个字;'

                format_types = ['A', 'B', 'C']
                format_weights = [0.5, 0.3, 0.2]
                selected_format = random.choices(format_types, weights=format_weights)[0]
                if selected_format == 'A':
                    format_instruction = "\n【格式要求】请使用格式A（普通段落）：直接分多段进行描述，每段都是完整的自然段。"
                elif selected_format == 'B':
                    format_instruction = "\n【格式要求】请使用格式B（序号段落）：使用(1)(2)(3)...等序号开头的小点进行描述，每个小点是一个自然段，内容要详细具体。"
                else:
                    format_instruction = "\n【格式要求】请使用格式C（混合格式）：既有普通段落，也有序号段落，形成丰富的内容结构。"
                templte_data = open_json(self.real_examples_template_path)
                templte_data = templte_data[random.randint(0, len(templte_data)-1)]
                temp_prompt += format_instruction
                # temp_prompt += "\n如果你认为当前生成的子标题的某些段落适合插入图片（如架构图、流程图、结构图、示意图、功能截图等），请在合适位置插入占位符（如[图片占位：架构图]、[图片占位：流程图]等）。占位符标记单独作为一段，不占用原本需要生成的段落数量。"
                temp_prompt = update_prompt(
                    [
                        ['[input_json]', temp_json_str],
                        ['[parent_path]', parent_path],
                        ['[idx]', str(idx)],
                        ['[cha_number]', cha_number],
                        ['[base_info]', base_info],
                        ['[prompt_snippet]', self.stlye_json_data['prompt_snippet']],
                        ['[write_style]', self.stlye_json_data['instructions']]
                        # ['[real_examples]', templte_data]
                    ],
                    temp_prompt
                )
                temp_res = use_llm_models(
                    temp_prompt,
                    model_name=self.model_name,
                    base_url=self.api_setting[self.model_name]['base_url'],
                    api_key=self.api_setting[self.model_name]['api_key']
                )
                try:
                    if temp_res.strip().startswith('```json'):
                        temp_res = temp_res.strip()[7:]
                    if temp_res.strip().endswith('```'):
                        temp_res = temp_res.strip()[:-3]
                    result_json = json.loads(temp_res)
                    item['content'] = result_json.get('content', item['content'])
                except Exception as e:
                    print(f'解析LLM返回内容失败: {e}, 原始返回: {temp_res}')
                return leaf_info, item
            except Exception as e:
                print(f'处理叶子节点时出错: {e}')
                return leaf_info, leaf_info['original_item']

        return self.process_tree_with_leaf_nodes(
            input_data,
            base_info=base_info,
            is_leaf_node_fn=is_leaf_node,
            process_leaf_node_fn=process_leaf_node
        )

    # 二次润色内容，使用通用框架
    def polish_content(self, input_data, base_info=''):
        raw_prompt = open_prompt(self.polish_content_prompt_path)
        def is_leaf_node(item):
            return item.get('content') is not None and len(item.get('content', [])) > 0
        def process_leaf_node(leaf_info):
            try:
                item = leaf_info['item']
                parent_path = leaf_info['parent_path']
                temp_json_str = json.dumps(item, ensure_ascii=False)
                temp_prompt = deepcopy(raw_prompt)
                
                # # 随机抽取人工书写的案例
                # templte_data = open_json(self.real_examples_template_path)
                # templte_data = templte_data[random.randint(0, len(templte_data)-1)]
                
                temp_prompt = update_prompt(
                    [
                        ['[input_json]', temp_json_str],
                        ['[parent_path]', parent_path],
                        ['[base_info]', base_info],
                        # ['[real_examples]', templte_data]
                    ],
                    temp_prompt
                )
                temp_res = use_llm_models(
                    temp_prompt,
                    model_name=self.model_name,
                    base_url=self.api_setting[self.model_name]['base_url'],
                    api_key=self.api_setting[self.model_name]['api_key']
                )
                try:
                    if temp_res.strip().startswith('```json'):
                        temp_res = temp_res.strip()[7:]
                    if temp_res.strip().endswith('```'):
                        temp_res = temp_res.strip()[:-3]
                    result_json = json.loads(temp_res)
                    item['content'] = result_json.get('content', item['content'])
                except Exception as e:
                    print(f'解析LLM返回内容失败: {e}, 原始返回: {temp_res}')
                return leaf_info, item
            except Exception as e:
                print(f'处理叶子节点时出错: {e}')
                return leaf_info, leaf_info['original_item']
        return self.process_tree_with_leaf_nodes(
            input_data,
            base_info=base_info,
            is_leaf_node_fn=is_leaf_node,
            process_leaf_node_fn=process_leaf_node
        )

    # 生成标书主流程
    def debug_main_process(self, input_dict):
        tables_data = input_dict['tables_data']
        post_process_data = input_dict['post_process_data']
        tech_data = ''

        print('step4.1: 提取评分表...')
        if os.path.exists(self.step4_score_table_save_path):
            score_table = open_json(self.step4_score_table_save_path)
        else:
            if self.single_score_table:
                score_table = input_dict['score_table']
            else:
                score_table = self.extract_score_table(tables_data)
            write_json(score_table, self.step4_score_table_save_path)

        print('step4.2: 提取技术规范书...')
        if os.path.exists(self.step4_spec_save_path):
            spec_data = open_json(self.step4_spec_save_path)
        else:
            if self.single_spec_data:
                spec_data = input_dict['spec_data']
            else:
                spec_data = self.extract_spec(post_process_data)
            new_sepc_data = {
                'section': '技术规范书',
                'content': [],
                'layers': []
            }
            if 'layers' in spec_data:
                for layer in spec_data['layers']:
                    if '目录' not in layer['section']:
                        new_sepc_data['layers'].append(layer)
            spec_data = new_sepc_data
            write_json(spec_data, self.step4_spec_save_path)

        print('step4.3: 合并评分表和技术建议书...')
        if os.path.exists(self.step4_st_tech_save_path):
            step4_st_tech = open_json(self.step4_st_tech_save_path)
        else:
            step4_st_tech = self.fuse_table_tec(score_table, tech_data)
            write_json(step4_st_tech, self.step4_st_tech_save_path)

        print('step5: 融合评分表、技术建议书和技术规范...')
        if os.path.exists(self.step5_st_tech_spec_save_path):
            step5_st_tech_spec = open_json(self.step5_st_tech_spec_save_path)
        else:
            step5_st_tech_spec = self.fuse_table_tec_spec(step4_st_tech, spec_data)
            write_json(step5_st_tech_spec, self.step5_st_tech_spec_save_path)

        start = time.time()

        print('step6.1: 生成完整的目录以及提示词...')
        if os.path.exists(self.base_info_save_path):
            base_info = open_json(self.base_info_save_path)
        else:
            base_info = self.overview_project(spec_data)
            write_json(base_info, self.base_info_save_path)

        if os.path.exists(self.expand_dir_save_path):
            expand_dir = open_json(self.expand_dir_save_path)
        else:
            expand_dir = self.expand_dir(step5_st_tech_spec, base_info=base_info)
            write_json(expand_dir, self.expand_dir_save_path)
        markdown_text = json_to_markdown(expand_dir)
        with open(f'{self.config.FILE_CONFIG.save_path}/step6_expand_dir.md', "w", encoding="utf-8") as f:
            f.write(markdown_text)
        md_to_word(
            f'{self.config.FILE_CONFIG.save_path}/step6_expand_dir.md', 
            f'{self.config.FILE_CONFIG.save_path}/step6_expand_dir.docx',
            template_path=self.dotx_template_path
        )

        print('step6.2: 填充目录内容...')
        if os.path.exists(self.final_res_save_path):
            final_res = open_json(self.final_res_save_path)
        else:
            final_res = self.fill_content(expand_dir, base_info=base_info)
            write_json(final_res, self.final_res_save_path)
        
        markdown_text = json_to_markdown(final_res)
        with open(f'{self.config.FILE_CONFIG.save_path}/step6_final_res.md', "w", encoding="utf-8") as f:
            f.write(markdown_text)
        md_to_word(
            f'{self.config.FILE_CONFIG.save_path}/step6_final_res.md', 
            f'{self.config.FILE_CONFIG.save_path}/step6_final_res.docx',
            template_path=self.config.dotx_template_path
        )

        print('step6.3: 二次润色内容...')
        if os.path.exists(self.polish_content_save_path):
            polished_res = open_json(self.polish_content_save_path)
        else:
            polished_res = self.polish_content(final_res, base_info=base_info)
            write_json(polished_res, self.polish_content_save_path)
        
        polished_markdown_text = json_to_markdown(polished_res)
        with open(f'{self.config.FILE_CONFIG.save_path}/step6_polished_res.md', "w", encoding="utf-8") as f:
            f.write(polished_markdown_text)
        md_to_word(
            f'{self.config.FILE_CONFIG.save_path}/step6_polished_res.md', 
            f'{self.config.FILE_CONFIG.save_path}/step6_polished_res.docx',
            template_path=self.config.dotx_template_path
        )

        end = time.time()
        elapsed_minutes = (end - start) / 60
        print(f"🔥 step6 耗时: {elapsed_minutes:.2f} 分钟")
        
        return {
            'final_res': final_res,
            'polished_res': polished_res
        }

    # 写日志
    def write_normal_log(self, message):
        with open(self.normal_log_file, "a+", encoding="utf-8") as f:
            f.write(message)
            f.write("\n")

    # 生成标书主流程
    def main_process(self, input_dict):
        tables_data = input_dict['tables_data']
        post_process_data = input_dict['post_process_data']
        tech_data = ''

        print('    step4.1: 提取评分表...')
        self.write_normal_log('    step4.1: 提取评分表...')
        if self.single_score_table:
            score_table = input_dict['score_table']
        else:
            score_table = self.extract_score_table(tables_data)

        print('    step4.2: 提取技术规范书...')
        self.write_normal_log('    step4.2: 提取技术规范书...')
        if self.single_spec_data:
            spec_data = input_dict['spec_data']
        else:
            spec_data = self.extract_spec(post_process_data)
        new_sepc_data = {
            'section': '技术规范书',
            'content': [],
            'layers': []
        }
        if 'layers' in spec_data:
            for layer in spec_data['layers']:
                if '目录' not in layer['section']:
                    new_sepc_data['layers'].append(layer)
        spec_data = new_sepc_data
        
        print('    step4.3: 合并评分表和技术建议书...')
        self.write_normal_log('    step4.3: 合并评分表和技术建议书...')
        step4_st_tech = self.fuse_table_tec(score_table, tech_data)
        
        print('  step5: 融合评分表、技术建议书和技术规范...')
        self.write_normal_log('  step5: 融合评分表、技术建议书和技术规范...')
        step5_st_tech_spec = self.fuse_table_tec_spec(step4_st_tech, spec_data)
        
        self._write_progress()

        print('  step6: 生成...')
        print('    step6.1: 生成完整的目录以及提示词...')
        self.write_normal_log('  step6: 生成...\n    step6.1: 生成完整的目录以及提示词...')
        base_info = self.overview_project(spec_data)
        expand_dir = self.expand_dir(step5_st_tech_spec, base_info=base_info)

        self._write_progress()

        print('    step6.2: 填充目录内容...')
        self.write_normal_log('    step6.2: 填充目录内容...')
        final_res = self.fill_content(expand_dir, base_info=base_info)
        
        self._write_progress()

        print('    step6.3: 二次润色内容...')
        self.write_normal_log('    step6.3: 二次润色内容...')
        polished_res = self.polish_content(final_res, base_info=base_info)
        
        self._write_progress()

        return {
            'final_res': final_res,
            'polished_res': polished_res
        }