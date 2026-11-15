import json

from lorafin.scoring import aggregate, extract_json, schema_errors, score_one

T = {"entity": "Halvorn Labs", "metric": "revenue", "value": 4.2, "unit": "billion", "period": "Q3 2024", "direction": "up"}


def test_extract_plain_and_fenced():
    assert extract_json(json.dumps(T)) == T
    assert extract_json("```json\n" + json.dumps(T) + "\n```") == T
    assert extract_json("Sure! " + json.dumps(T) + " Hope that helps {") == T


def test_extract_braces_inside_strings():
    assert extract_json('{"a": "x}y", "b": 1}') == {"a": "x}y", "b": 1}


def test_extract_invalid():
    assert extract_json("no json here") is None
    assert extract_json('{"a": 1,}') is None
    assert extract_json('{"a": ') is None


def test_schema_errors():
    assert schema_errors(T) == []
    assert schema_errors(None)
    assert any("missing" in e for e in schema_errors({"entity": "x"}))
    assert any("value" in e for e in schema_errors({**T, "value": "4.2"}))
    assert any("value" in e for e in schema_errors({**T, "value": True}))
    assert any("direction" in e for e in schema_errors({**T, "direction": "rose"}))
    assert any("unexpected" in e for e in schema_errors({**T, "extra": 1}))


def test_score_exact_and_partial():
    s = score_one(json.dumps(T), T)
    assert s["valid"] and s["exact"]
    s = score_one(json.dumps({**T, "entity": " halvorn  labs ", "value": 4.20}), T)
    assert s["exact"]  # case/whitespace and 4.2 == 4.20
    s = score_one(json.dumps({**T, "period": "Q3 2023"}), T)
    assert s["valid"] and not s["exact"] and not s["fields"]["period"] and s["fields"]["value"]


def test_score_garbage():
    s = score_one("I cannot help", T)
    assert not s["valid"] and not s["exact"] and not any(s["fields"].values())


def test_aggregate():
    scores = [score_one(json.dumps(T), T), score_one("nope", T)]
    m = aggregate(scores)
    assert m["n"] == 2 and m["json_valid"] == 0.5 and m["exact_match"] == 0.5 and m["acc_entity"] == 0.5
    assert aggregate([]) == {"n": 0}
