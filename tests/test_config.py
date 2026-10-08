import json

from pl_flow.config import read_config


def test_json_scientific_notation_remains_numeric(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"norm_eps": 1e-5}))
    assert read_config(path)["norm_eps"] == 1e-5
