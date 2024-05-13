import pydantic
import pytest

from asapdiscovery.ml.config import LossFunctionConfig
from asapdiscovery.ml.schema import TrainingPrediction, TrainingPredictionTracker


@pytest.fixture()
def identifiers():
    # Identifying values for a training example
    return {
        "compound_id": "test_compound",
        "xtal_id": "test_xtal",
        "target_prop": "pIC50",
        "target_val": 5.0,
        "in_range": 0,
        "uncertainty": 0.2,
    }


@pytest.fixture()
def loss_configs():
    # Just get a couple different loss function configs of different types
    return [
        LossFunctionConfig(loss_type="mse_step"),
        LossFunctionConfig(loss_type="range", range_lower_lim=0, range_upper_lim=10),
    ]


def test_training_pred_constructor(identifiers, loss_configs):
    _ = TrainingPrediction(**identifiers, loss_config=loss_configs[0])
    _ = TrainingPrediction(**identifiers, loss_config=loss_configs[1])


def test_training_pred_tracker_constructor_no_dict():
    tp_tracker = TrainingPredictionTracker()

    assert tp_tracker.split_dict.keys() == {"train", "val", "test"}
    assert all([v == [] for v in tp_tracker.split_dict.values()])


def test_training_pred_tracker_constructor_ok_dict(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers, loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers, loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    assert tp_tracker.split_dict.keys() == {"train", "val", "test"}
    assert len(tp_tracker.split_dict["train"]) == 1
    assert len(tp_tracker.split_dict["val"]) == 1
    assert len(tp_tracker.split_dict["test"]) == 0


def test_training_pred_tracker_constructor_bad_dict(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers, loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers, loss_config=loss_configs[1])

    with pytest.raises(pydantic.error_wrappers.ValidationError):
        _ = TrainingPredictionTracker(split_dict={"train": [tp1], "val": [tp2]})


def test_find_value_idxs():
    pass


def test_get_values_split():
    pass


def test_get_values_no_split():
    pass


def test_update_values_existing():
    pass


def test_update_values_new():
    pass
