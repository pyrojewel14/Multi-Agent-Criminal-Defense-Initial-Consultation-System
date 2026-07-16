import importlib


def test_chroma_config_accepts_environment_overrides(monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/tmp/phase7/chroma")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "phase7_collection")
    monkeypatch.setenv("CHROMA_MD5_STORE", "/tmp/phase7/md5/store.txt")

    import app.utils.config as config_module

    reloaded = importlib.reload(config_module)

    assert reloaded.chroma_config["persist_directory"] == "/tmp/phase7/chroma"
    assert reloaded.chroma_config["collection_name"] == "phase7_collection"
    assert reloaded.chroma_config["md5_hex_store"] == "/tmp/phase7/md5/store.txt"


def test_chroma_config_keeps_yaml_defaults_without_environment(monkeypatch):
    monkeypatch.delenv("CHROMA_PERSIST_DIRECTORY", raising=False)
    monkeypatch.delenv("CHROMA_COLLECTION_NAME", raising=False)
    monkeypatch.delenv("CHROMA_MD5_STORE", raising=False)

    import app.utils.config as config_module

    reloaded = importlib.reload(config_module)

    assert reloaded.chroma_config["persist_directory"] == "data/chromadb"
    assert reloaded.chroma_config["collection_name"] == "rag_collection"
    assert reloaded.chroma_config["md5_hex_store"] == "data/md5_hex_store/md5_hex_store.txt"
