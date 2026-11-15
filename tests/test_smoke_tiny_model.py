"""End-to-end smoke test on a tiny random model.

Skipped when torch/peft are not installed or the tiny model cannot be
fetched (offline, or CI without a model cache). CI runs offline on purpose.
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny():
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip("offline")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(TINY)
        model = AutoModelForCausalLM.from_pretrained(TINY)
    except Exception as e:  # network or cache miss
        pytest.skip(f"tiny model unavailable: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    return tok, model


def test_lora_step_and_generate(tiny):
    from peft import LoraConfig, get_peft_model

    from lorafin.data import generate_splits
    from lorafin.modeling import count_params, generate_batch
    from lorafin.train import collate, encode_example

    tok, model = tiny
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    trainable, total = count_params(model)
    assert 0 < trainable < total
    rows = generate_splits(0, 4, 1, 1, 1)["train"]
    enc = [encode_example(tok, r, 256) for r in rows]
    assert all(-100 in e["labels"] and e["labels"][-1] != -100 for e in enc)
    with pytest.raises(ValueError):
        encode_example(tok, rows[0], 8)  # never silently truncate the answer
    batch = collate(enc, tok.pad_token_id)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss = model(**batch).loss
    loss.backward()
    opt.step()
    after = [p for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(after, before))
    tok.padding_side = "left"
    texts, npt, ngt, secs = generate_batch(model, tok, [[{"role": "user", "content": "hi"}]], 4)
    assert len(texts) == 1 and ngt[0] <= 4
