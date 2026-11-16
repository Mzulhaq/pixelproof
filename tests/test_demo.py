from scripts.demo import run_demo


def test_real_http_demo():
    result = run_demo()
    assert result["http_checks"] == "passed"
    assert len(result["holdout_examples"]) == 3
