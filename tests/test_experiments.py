import pandas as pd

from health_qa.experiments import suggest_next_config


def test_suggest_next_config_increases_beams_on_plateau():
    config = {"inference": {"num_beams": 4, "length_penalty": 1.0}, "training": {"epochs": 3}}
    history = pd.DataFrame(
        {
            "experiment_id": ["e1", "e2", "e3"],
            "local_score": [0.501, 0.503, 0.504],
        }
    )

    suggested = suggest_next_config(config, history)

    assert suggested["inference"]["num_beams"] == 5
    assert suggested["inference"]["length_penalty"] == 1.05
    assert config["inference"]["num_beams"] == 4


def test_suggest_next_config_extends_training_before_strong_score():
    config = {"inference": {"num_beams": 4}, "training": {"epochs": 3}}
    history = pd.DataFrame({"experiment_id": ["e1"], "local_score": [0.42]})

    suggested = suggest_next_config(config, history)

    assert suggested["training"]["epochs"] == 4
