import time
from openai import OpenAI
import config

class APIError(Exception):
    """A custom API exception to handle API call failures."""
    pass

# 全局的模型配置，由主程序在启动时设置
MODEL_CONFIG = {}

def set_model_config(config_dict):
    """Set the current model configuration."""
    global MODEL_CONFIG
    MODEL_CONFIG = config_dict

def get_llm_response(system_prompt, user_prompt):
    """Calls the LLM API based on the global model configuration, throwing an APIError if the call fails."""
    if not MODEL_CONFIG:
        raise ValueError("Model configuration has not been set. Please call set_model_config first.")

    api_key = getattr(config, MODEL_CONFIG["api_key_name"])
    base_url = getattr(config, MODEL_CONFIG.get("base_url_name"))
    
    for attempt in range(3):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            params = {
                "model": MODEL_CONFIG["model_name"],
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": 0.0,
            }
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API调用失败 (TRY {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                raise APIError(f"API call failed after 3 attempts: {e}")
