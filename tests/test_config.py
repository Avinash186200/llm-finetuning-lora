from pathlib import Path

import pytest

from lorafin.config import ConfigError, load_config, run_name

ROOT = Path(__file__).resolve().parents[1]


def test_base_yaml_loads():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    assert cfg["lora"]["r"] == 8 and cfg["lora"]["alpha"] == 16
    assert cfg["model"]["name"].startswith("HuggingFaceTB/")


def test_overrides_parse_types():
    cfg = load_config(ROOT / "configs" / "base.yaml", ["lora.r=16", "train.lr=1e-4", "lora.target_modules=[q_proj]"])
    assert cfg["lora"]["r"] == 16 and cfg["train"]["lr"] == 1e-4 and cfg["lora"]["target_modules"] == ["q_proj"]
    assert run_name(cfg) == "lora_r16_s0"


@pytest.mark.parametrize("ov", ["lora.rank=4", "nope=1", "lora.r"])
def test_bad_override(ov):
    with pytest.raises(ConfigError):
        load_config(None, [ov])


def test_validation():
    with pytest.raises(ConfigError):
        load_config(None, ["lora.r=0"])
    with pytest.raises(ConfigError):
        load_config(None, ["lora.target_modules=[]"])


def test_unknown_yaml_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lora:\n  rnk: 4\n")
    with pytest.raises(ConfigError):
        load_config(p)
