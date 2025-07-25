import pandas as pd
import json
import csv
import sys

def log_interaction(log_data, log_filepath, lock):
    """(线程安全)将单次交互记录追加到JSONL文件。"""
    with lock:
        with open(log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

def load_text_file(filepath):
    """加载文本文件，如果文件不存在则返回空字符串。"""
    if not filepath:
        return ""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def load_data(filepath, load_options=None):
    """从CSV文件加载数据。"""
    if load_options is None:
        load_options = {}
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', **load_options)
        print(f"成功从 {filepath} 加载 {len(df)} 条数据。")
        return df
    except FileNotFoundError:
        print(f"数据错误: 数据文件 {filepath} 未找到。")
        sys.exit(1)
    except Exception as e:
        print(f"数据错误: 读取 {filepath} 时出错: {e}。")
        sys.exit(1)

def initialize_output_files(csv_path, txt_path, jsonl_path, header_info):
    """创建并初始化所有输出文件及其表头/标题。"""
    try:
        # 写入CSV表头
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(["Input", "Model_Output"])
        
        # 写入TXT报告标题
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(header_info)
        
        # 清空或创建JSONL文件
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            pass
    except IOError as e:
        print(f"文件错误: 无法初始化输出文件。请检查权限和路径。错误: {e}")
        sys.exit(1)

def append_to_files(result, paths, lock):
    """(线程安全)将单条结果追加到CSV和TXT文件。"""
    with lock:
        # 追加到CSV
        with open(paths['csv'], 'a', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow([result['input_value'], result['parsed_model_response']])
        
        # 追加到TXT
        with open(paths['txt'], 'a', encoding='utf-8') as f:
            f.write(f"--- 序号 {result['index']} ---\n")
            f.write(f"输入: {result['input_value']}\n")
            f.write(f"模型输出 (完整):\n{result['full_model_response']}\n\n")