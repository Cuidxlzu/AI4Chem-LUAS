import argparse
import sys
from src.config_loader import get_prompts, get_task_configs
from src.experiment_runner import run_experiment
from src.utils.file_io import load_data
from src.llm_api import set_model_config

# 定义所有可用的模型及其配置
# key是用户在命令行中使用的模型名称
MODEL_MAPPING = {
    "qwen-max-latest": {
        "model_name": "qwen-max-latest",
        "api_key_name": "DASHSCOPE_API_KEY",
        "base_url_name": "DASHSCOPE_BASE_URL",
    },
    "deepseek-reasoner": {
        "model_name": "deepseek-reasoner",
        "api_key_name": "DEEPSEEK_API_KEY",
        "base_url_name": "DEEPSEEK_BASE_URL",
    },
    "o4-mini": {
        "model_name": "o4-mini",
        "api_key_name": "ZHIZENGZENG_API_KEY",
        "base_url_name": "ZHIZENGZENG_BASE_URL",
    },
    "doubao-thinker": {
        "model_name": "doubao-seed-1-6-thinking-250615",
        "api_key_name": "ARK_API_KEY",
        "base_url_name": "ARK_BASE_URL",
    },
    # 在这里可以继续添加其他模型
}

def main():
    """主函数，负责解析命令行参数、加载配置并启动实验。"""
    # 1. 设置命令行解析器
    parser = argparse.ArgumentParser(
        description="LLM_Chem: A Framework for Chemical Experiment Automation.",
        formatter_class=argparse.RawTextHelpFormatter  # Keep formatting of help text
    )
    parser.add_argument(
        "--model", 
        required=True, 
        choices=MODEL_MAPPING.keys(), 
        help=f"Specify the model to use for the experiment.\nAvailable models:\n" + "\n".join([f"  - {name}" for name in MODEL_MAPPING.keys()])
    )
    parser.add_argument("--experiments", required=True, nargs='+', help="Specify one or more experiment IDs to run.")

    args = parser.parse_args()

    # 2. 获取并设置模型配置
    model_key = args.model
    model_config = MODEL_MAPPING[model_key]
    set_model_config(model_config)

    # 3. 加载所有实验和任务配置
    prompts_config = get_prompts()
    tasks_config = get_task_configs()

    # 4. 验证并运行指定的实验
    experiments_to_run = args.experiments
    print(f"模型 '{model_config['model_name']}' 已激活。")
    print(f"准备批量运行以下实验: {experiments_to_run}")

    for exp_id in experiments_to_run:
        print(f"\n{'='*25}\n[ 准备实验: {exp_id} ]\n{'='*25}")

        if exp_id not in prompts_config:
            print(f"错误: 实验 {exp_id} 在 prompts.json 中未定义，已跳过。")
            continue
            
        task_prefix = exp_id.split('-')[0]
        task_config = tasks_config.get(task_prefix)
        if not task_config:
            print(f"错误: 实验 {exp_id} 在 tasks.json 中没有找到对应配置，已跳过。")
            continue
            
        # 加载对应的数据
        data = load_data(task_config['data_filepath'], task_config.get('load_options'))
        
        # 启动实验
        run_experiment(exp_id, model_config, data, prompts_config, task_config)

    print(f"\n{'='*25}\n[ 所有指定实验均已成功运行 ]\n{'='*25}")

if __name__ == "__main__":
    main()