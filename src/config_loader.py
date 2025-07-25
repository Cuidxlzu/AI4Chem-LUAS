import json
import sys

def load_json_config(filepath):
    """从指定路径加载一个JSON配置文件。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置错误: 找不到配置文件 {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"配置错误: 文件 {filepath} 不是有效的JSON格式。")
        sys.exit(1)

def get_prompts():
    """加载并返回所有Prompt模板。"""
    return load_json_config('configs/prompts.json')

def get_task_configs():
    """加载并返回所有任务的配置。"""
    return load_json_config('configs/tasks.json')