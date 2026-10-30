import pytest

from docgraph.config import ConfigError, EndpointConfig, load_config, load_dotenv


def write(tmp_path, text):
    p = tmp_path / "docgraph.toml"
    p.write_text(text)
    return p


def test_defaults_use_local_ollama_without_a_key(tmp_path):
    cfg = load_config(env={})
    assert [e.base_url for e in cfg.chat.endpoints] == ["http://localhost:11434/v1"]
    assert cfg.chat.endpoints[0].options["reasoning_effort"] == "none"
    assert cfg.chat.endpoints[0].available({})
    assert cfg.embeddings.provider == "local"


def test_endpoint_order_is_preserved(tmp_path):
    p = write(tmp_path, """
[[chat.endpoints]]
name = "nim"
base_url = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"
model = "a"

[[chat.endpoints]]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "b"

[[chat.endpoints]]
name = "ollama"
base_url = "http://localhost:11434/v1"
model = "c"
options = { reasoning_effort = "none", temperature = 0.7 }
""")
    cfg = load_config(p, env={})
    assert [e.name for e in cfg.chat.endpoints] == ["nim", "openrouter", "ollama"]
    assert cfg.chat.endpoints[2].options["temperature"] == 0.7


def test_literal_api_key_in_the_file_is_rejected(tmp_path):
    p = write(tmp_path, """
[[chat.endpoints]]
base_url = "http://x/v1"
model = "m"
api_key = "sk-should-not-be-here"
""")
    with pytest.raises(ConfigError) as e:
        load_config(p, env={})
    assert "environment variables only" in str(e.value)
    assert "sk-should-not-be-here" not in str(e.value)


def test_unknown_keys_are_errors(tmp_path):
    p = write(tmp_path, "[retrieval]\nvectro = true\n")
    with pytest.raises(ConfigError):
        load_config(p, env={})


def test_keys_come_from_the_environment_only():
    ep = EndpointConfig(base_url="http://x/v1", model="m", api_key_env="MY_KEY")
    assert not ep.available({})
    assert ep.api_key({"MY_KEY": "secret"}) == "secret"
    assert ep.available({"MY_KEY": "secret"})
    assert "secret" not in ep.model_dump_json()


def test_env_shortcut_replaces_endpoints_and_sets_data_dir(tmp_path):
    env = {"DOCGRAPH_BASE_URL": "http://h:1/v1", "DOCGRAPH_MODEL": "m1", "DOCGRAPH_API_KEY_ENV": "K",
           "DOCGRAPH_DATA_DIR": str(tmp_path / "d")}
    cfg = load_config(env=env)
    assert len(cfg.chat.endpoints) == 1
    assert cfg.chat.endpoints[0].model == "m1" and cfg.chat.endpoints[0].api_key_env == "K"
    assert cfg.data_dir == tmp_path / "d"


def test_embedding_env_overrides():
    cfg = load_config(env={"DOCGRAPH_EMBED_BASE_URL": "http://e/v1", "DOCGRAPH_EMBED_MODEL": "nomic-embed-text"})
    assert cfg.embeddings.provider == "openai" and cfg.embeddings.model == "nomic-embed-text"


def test_openai_embeddings_need_a_base_url(tmp_path):
    p = write(tmp_path, '[embeddings]\nprovider = "openai"\nmodel = "x"\n')
    with pytest.raises(ConfigError):
        load_config(p, env={})


def test_overlap_must_be_smaller_than_chunk_size(tmp_path):
    p = write(tmp_path, "[ingest]\nchunk_size = 100\nchunk_overlap = 100\n")
    with pytest.raises(ConfigError):
        load_config(p, env={})


def test_dotenv_does_not_override_existing(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# comment\nA=1\nexport B="two"\nC=\'3\'\nEXISTING=new\n')
    env = {"EXISTING": "old"}
    assert load_dotenv(f, env)
    assert env == {"EXISTING": "old", "A": "1", "B": "two", "C": "3"}
