"""The onnx backend of openjev-flash.

The unit tests run on a tiny ONNX graph built here (a masked sum of per-token logit vectors) so they need no
download; they check the plumbing that has to match the torch path: label order from `id2label`, padding that
does not change a result, and truncation of the premise only. `test_parity_with_torch` compares the real int8
export with the torch model and runs when SOURCEBOUND_ONNX_DIR points at a folder holding model_quantized.onnx,
config.json and tokenizer.json (and torch and transformers are installed).
"""

import json
import os

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
tokenizers = pytest.importorskip("tokenizers")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402

from sourcebound.config import RerankConfig  # noqa: E402
from sourcebound.rerank import ENT, OnnxFlashScorer, make_scorer  # noqa: E402
from sourcebound.runtime import LazyReranker  # noqa: E402

VOCAB = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "good": 4, "bad": 5, "maybe": 6, "cats": 7, "sleep": 8, "a": 9, "b": 10}
# One logit vector per token id, in the model's own column order.
W = np.array(
    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [0.5, 0, 0], [0, 0.5, 0], [0.1, 0.1, 0], [0, 0.1, 0.1]],
    dtype=np.float32,
)


def build_model(path):
    """logits = sum over non-pad tokens of W[token]; takes input_ids and attention_mask like the real export."""
    nodes = [
        helper.make_node("Gather", ["W", "input_ids"], ["emb"]),
        helper.make_node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["maskf", "axes2"], ["mask3"]),
        helper.make_node("Mul", ["emb", "mask3"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axes1"], ["logits"], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 3])],
        initializer=[numpy_helper.from_array(W, "W"),
                     numpy_helper.from_array(np.array([2], dtype=np.int64), "axes2"),
                     numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


def build_tokenizer(path):
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tok.save(str(path))


@pytest.fixture
def export(tmp_path):
    """A model folder like the one published under onnx/. The model's columns are [neutral, contradiction, entailment]."""
    build_model(tmp_path / "model_quantized.onnx")
    build_tokenizer(tmp_path / "tokenizer.json")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"pad_token": "[PAD]"}))
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {"0": "neutral", "1": "contradiction", "2": "entailment"}}))
    return tmp_path


def scorer(export, **kw):
    return OnnxFlashScorer(RerankConfig(backend="onnx", onnx_dir=export, **kw))


def reference(premise_ids, hyp_ids):
    """Expected [contradiction, entailment, neutral] for the tiny model: softmax of the summed columns, reordered."""
    ids = [1, *premise_ids, 2, *hyp_ids, 2]
    logits = W[ids].sum(axis=0)[[1, 2, 0]]  # model columns -> [contradiction, entailment, neutral]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def test_probabilities_follow_id2label_not_column_position(export):
    s = scorer(export)
    p = s.predict([("good", "a"), ("bad", "a"), ("maybe", "a")])  # column 0 = neutral, 1 = contradiction, 2 = entailment
    assert p.shape == (3, 3) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    # "good" is wired to column 0 (neutral), "bad" to column 1 (contradiction), "maybe" to column 2 (entailment)
    assert p[0].argmax() == 2 and p[1].argmax() == 0 and p[2].argmax() == 1
    assert np.allclose(p[0], reference([4], [9]), atol=1e-5)


def test_padding_does_not_change_a_result(export):
    s = scorer(export, batch_size=8)
    alone = s.predict([("cats sleep", "a")])[0]
    batched = s.predict([("good bad maybe cats sleep a b good", "b"), ("cats sleep", "a"), ("bad", "a b a b")])
    assert np.allclose(batched[1], alone, atol=1e-6)
    assert np.allclose(alone, reference([7, 8], [9]), atol=1e-5)


def test_only_the_premise_is_truncated(export):
    # max_length 6: [CLS] premise.. [SEP] hypothesis [SEP] leaves room for 2 premise tokens when the hypothesis is one token
    s = scorer(export, max_length=6)
    p = s.predict([("good bad maybe cats sleep", "a")])[0]
    assert np.allclose(p, reference([4, 5], [9]), atol=1e-5)
    # a two-token hypothesis takes a premise token's place; the hypothesis is never cut
    p2 = s.predict([("good bad maybe cats sleep", "a b")])[0]
    assert np.allclose(p2, reference([4], [9, 10]), atol=1e-5)


def test_score_is_entailment_probability_and_strips_whitespace(export):
    s = scorer(export)
    sc = s.score("a", ["  maybe  ", "good"])
    assert sc[0] > 0.7 > 0.2 > sc[1]
    assert s.entailment("maybe", "a") == pytest.approx(float(s.predict([("maybe", "a")])[0, ENT]))


def test_make_scorer_and_lazy_reranker_pick_the_backend(export):
    cfg = RerankConfig(backend="onnx", onnx_dir=export)
    assert isinstance(make_scorer(cfg), OnnxFlashScorer)
    lazy = LazyReranker(cfg)
    assert lazy._scorer is None  # nothing is loaded until the first use
    assert len(lazy.score("a", ["maybe", "good"])) == 2 and isinstance(lazy.scorer, OnnxFlashScorer)


def test_config_defaults_keep_torch_and_onnx_is_selectable_from_toml(tmp_path):
    from sourcebound.config import load_config

    assert RerankConfig().backend == "torch" and RerankConfig().onnx_file == "onnx/model_quantized.onnx"
    f = tmp_path / "c.toml"
    f.write_text('[rerank]\nbackend = "onnx"\nrevision = "abc123"\nthreads = 2\n')
    cfg = load_config(f, env={})
    assert cfg.rerank.backend == "onnx" and cfg.rerank.revision == "abc123" and cfg.rerank.threads == 2


PAIRS = [
    ("A man is playing a guitar on stage.", "Someone is making music."),
    ("A man is playing a guitar on stage.", "Nobody is making music."),
    ("The meeting was moved to Thursday at three in the afternoon.", "The meeting is on Monday."),
    ("The model was trained for 3 epochs with the Adam optimizer, a batch size of 32 and a learning rate of 2e-5.", "How many epochs was the model trained for?"),
    ("The dataset contains 1,200 annotated sentences from news articles.", "What is the size of the dataset?"),
    ("We evaluate on two benchmarks and report F1.", "Which benchmark has the highest score?"),
]


@pytest.mark.skipif(not os.environ.get("SOURCEBOUND_ONNX_DIR"), reason="set SOURCEBOUND_ONNX_DIR to a folder with the real int8 export")
def test_parity_with_torch():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from sourcebound.rerank import FlashScorer

    ref = FlashScorer(RerankConfig(model=os.environ.get("SOURCEBOUND_TORCH_MODEL", "xxparthparekhxx/openjev-flash"))).predict(PAIRS)
    got = OnnxFlashScorer(RerankConfig(backend="onnx", onnx_dir=os.environ["SOURCEBOUND_ONNX_DIR"])).predict(PAIRS)
    print("max |dp| =", float(np.abs(ref - got).max()))
    assert (ref.argmax(1) == got.argmax(1)).all()
    assert np.abs(ref - got).max() < 0.15
