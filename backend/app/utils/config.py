import os

from app.utils.config_loader import ConfigLoader
from app.utils.path_tool import get_abstract_path


def _load_chroma_config() -> dict:
    """加载 Chroma 配置，并允许部署环境覆盖持久化位置。"""
    config = ConfigLoader.load_yaml(config_path=get_abstract_path('app/config/chroma.yaml'))
    environment_overrides = {
        "persist_directory": os.getenv("CHROMA_PERSIST_DIRECTORY"),
        "collection_name": os.getenv("CHROMA_COLLECTION_NAME"),
        "md5_hex_store": os.getenv("CHROMA_MD5_STORE"),
    }
    config.update({key: value for key, value in environment_overrides.items() if value})
    return config


chroma_config = _load_chroma_config()
prompt_config = ConfigLoader.load_yaml(config_path=get_abstract_path('app/config/prompt.yaml'))
agent_config = ConfigLoader.load_yaml(config_path=get_abstract_path('app/config/agent.yaml'))

if __name__ == '__main__':
    print(chroma_config)
    print(prompt_config)
    print(agent_config)
