import os
import json
import pandas as pd
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.llm_api import get_llm_response, APIError
from src.utils.file_io import log_interaction, initialize_output_files, append_to_files, load_text_file

def _process_single_molecule(system_prompt, prompt_template, index, row, input_column):
    """Internal helper function for handling API requests for a single molecule."""
    input_value = str(row[input_column])
    if pd.isna(row[input_column]):
        return None

    try:
        if isinstance(prompt_template, dict):
            user_prompt = json.dumps(prompt_template, ensure_ascii=False).replace("{molecule}", json.dumps(input_value)[1:-1])
        else:
            user_prompt = prompt_template.format(molecule=input_value)
        
        full_response = get_llm_response(system_prompt, user_prompt)
        parsed_response = full_response.split("[ANSWER]")[-1].strip() if "[ANSWER]" in full_response else full_response
        
        return {
            "success": True,
            "data": {
                "timestamp": datetime.now().isoformat(),
                "index": index + 1,
                "final_user_prompt": user_prompt,
                "full_model_response": full_response,
                "parsed_model_response": parsed_response,
                "input_value": input_value
            }
        }
    except APIError as e:
        return {
            "success": False,
            "error_message": str(e),
            "input_value": input_value,
            "index": index + 1
        }

def run_experiment(exp_id, model_config, data, prompts_config, task_config):
    """Core function for executing a single experiment, including error handling and summary reporting."""
    prompt_info = prompts_config[exp_id]
    task_type = prompt_info["type"]
    column_map = task_config['column_map']
    
    if task_type not in column_map:
        print(f"ERROR:Task TYPE '{task_type}' does not have a corresponding column mapping in its task configuration."); return
    
    input_column, _ = column_map[task_type]

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_desc_safe = prompt_info['description'].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    
    subdirectory_name = f"{exp_id}_{exp_desc_safe}"
    output_dir = os.path.join("results", subdirectory_name)
    log_dir = os.path.join("logs", subdirectory_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    base_filename = f"{exp_id}-{model_config['model_name']}-{run_timestamp}"
    paths = {
        'csv': os.path.join(output_dir, f"{base_filename}.csv"),
        'txt': os.path.join(output_dir, f"{base_filename}.txt"),
        'jsonl': os.path.join(log_dir, f"{base_filename}.jsonl")
    }

    # --- Fine-grained background data loading logic. ---
    final_system_prompt = ""
    print("Determining background data...")
    if prompt_info.get("requires_learning_material", False):
        material_path = f"learning_materials/task_{exp_id}_specific_material.md"
        print(f"  > Loading specific learning material: {material_path}")
        specific_material = load_text_file(material_path)
        if specific_material:
            final_system_prompt = specific_material
            print("    > Specific learning material loaded successfully.")
        else:
            print(f"    > Warning: Specific learning material not found. {material_path}。")
    else:
        print("  > This experiment does not use specific learning material.")

    txt_header = (
        f"# Exp: {exp_id} | Model: {model_config['model_name']} | Time: {run_timestamp}\n"
        f"# System Prompt (Context):\n"
        f"{final_system_prompt if final_system_prompt else 'None'}\n{'='*50}\n\n"
    )
    initialize_output_files(paths['csv'], paths['txt'], paths['jsonl'], txt_header)

    print(f"\n--- Starting parallel experiments: {exp_id} | Model: {model_config['model_name']} (并发数: 100) ---")
    print(f"Input column '{input_column}'")
    print(f"Results will be written in real time.: {output_dir}/ 和 {log_dir}/")

    successful_results = []
    failed_requests = []
    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {
            executor.submit(_process_single_molecule, final_system_prompt, prompt_info["template"], index, row, input_column): index
            for index, row in data.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(data), desc=f"Processing data... {exp_id}"):
            result = future.result()
            if not result:
                continue

            if result["success"]:
                successful_results.append(result["data"])
                log_entry = {
                    "timestamp": result["data"]['timestamp'], "run_id": run_timestamp, "experiment_id": exp_id,
                    "model_key": model_config['model_name'], "data_index": result["data"]['index'],
                    "system_prompt": final_system_prompt, "final_user_prompt": result["data"]['final_user_prompt'],
                    "model_output": result["data"]['full_model_response'], "input_value": result["data"]['input_value']
                }
                log_interaction(log_entry, paths['jsonl'], file_lock)
            else:
                failed_requests.append(result)

    successful_results.sort(key=lambda r: r['index'])
    for res in successful_results:
        append_to_files(res, paths, file_lock)

    if failed_requests:
        error_filepath = os.path.join(output_dir, f"{base_filename}_errors.json")
        failed_requests.sort(key=lambda r: r['index'])
        with open(error_filepath, 'w', encoding='utf-8') as f:
            json.dump(failed_requests, f, ensure_ascii=False, indent=4)
        print(f"\n Warning: {len(failed_requests)} requests failed. Details have been saved to: {error_filepath}")

    total_requests = len(data)
    print(f"\nExperiment {exp_id} completed.")
    print(f"  Total: {total_requests} | Successful: {len(successful_results)} | Failed: {len(failed_requests)}")