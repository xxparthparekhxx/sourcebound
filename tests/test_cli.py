import pytest

from docgraph.cli import main
from fakes import Reply

DOC = "# Cats\n\n## Sleep\n\nCats sleep for up to sixteen hours every day, mostly in sunny spots.\n"


@pytest.fixture
def conf(tmp_path, llm_servers, monkeypatch):
    srv = llm_servers(Reply(content="Cats sleep sixteen hours [1]. Not given [9]."))
    srv.embed_dim = 64
    cfg = tmp_path / "docgraph.toml"
    cfg.write_text(f'''
data_dir = "{tmp_path / 'idx'}"
[[chat.endpoints]]
name = "fake"
base_url = "{srv.base_url}"
model = "m"
[embeddings]
provider = "openai"
base_url = "{srv.base_url}"
model = "hash-embed"
[retrieval]
min_vector_score = 0.2
''')
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cats.md").write_text(DOC)
    (docs / "notes.xyz").write_text("ignored")
    monkeypatch.chdir(tmp_path)
    return str(cfg), docs, srv


def test_ingest_ask_docs_remove_reset_end_to_end(conf, capsys):
    cfg, docs, srv = conf
    assert main(["--config", cfg, "ingest", str(docs)]) == 0
    out = capsys.readouterr().out
    assert "ingested" in out and "cats.md" in out and "1 chunks" in out and "notes.xyz" not in out

    assert main(["--config", cfg, "ingest", str(docs)]) == 0
    assert "1 unchanged" in capsys.readouterr().out

    assert main(["--config", cfg, "ask", "How many hours do cats sleep every day?"]) == 0
    cap = capsys.readouterr()
    assert "Cats sleep sixteen hours [1]." in cap.out and "[9]" not in cap.out
    assert "Sources:" in cap.out and "[1] cats.md, Sleep" in cap.out
    assert "removed citation(s) [9]" in cap.err

    assert main(["--config", cfg, "docs"]) == 0
    d = capsys.readouterr().out
    assert "Cats" in d and "hash-embed (dimension 64)" in d

    assert main(["--config", cfg, "remove", str(docs / "cats.md")]) == 0
    assert main(["--config", cfg, "ask", "anything"]) == 1
    assert "index is empty" in capsys.readouterr().err

    assert main(["--config", cfg, "reset"]) == 1                    # needs --yes
    assert main(["--config", cfg, "reset", "--yes"]) == 0


def test_ask_json_output_has_the_citation_details(conf, capsys):
    import json
    cfg, docs, _ = conf
    main(["--config", cfg, "ingest", str(docs)])
    capsys.readouterr()
    assert main(["--config", cfg, "ask", "--json", "How many hours do cats sleep every day?"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["text"] == "Cats sleep sixteen hours [1]. Not given." and data["flagged"] == [9]
    assert data["citations"][0]["document"] == "Cats" and "sixteen hours" in data["citations"][0]["text"]


def test_switching_the_embedding_model_on_a_populated_index_is_refused(conf, capsys, tmp_path):
    cfg, docs, _ = conf
    main(["--config", cfg, "ingest", str(docs)])
    capsys.readouterr()
    other = tmp_path / "other.toml"
    other.write_text(open(cfg).read().replace("hash-embed", "another-model"))
    assert main(["--config", str(other), "ask", "cats"]) == 1
    err = capsys.readouterr().err
    assert "hash-embed" in err and "another-model" in err and "cannot be mixed" in err


def test_bad_config_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('[[chat.endpoints]]\nbase_url = "http://x"\nmodel = "m"\napi_key = "sk-1"\n')
    assert main(["--config", str(bad), "docs"]) == 2
    assert "environment variables only" in capsys.readouterr().err
