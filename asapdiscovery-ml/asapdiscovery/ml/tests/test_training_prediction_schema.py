import json

import pytest
from asapdiscovery.ml.config import LossFunctionConfig
from asapdiscovery.ml.schema import TrainingPrediction, TrainingPredictionTracker
from pydantic.v1 import ValidationError


@pytest.fixture()
def identifiers():
    # Identifying values for a training example
    return (
        {
            "compound_id": "test_compound",
            "xtal_id": "test_xtal",
            "target_prop": "pIC50",
            "target_val": 5.0,
            "in_range": 0,
            "uncertainty": 0.2,
            "loss_weight": 0.8,
        },
        {
            "compound_id": "test_compound2",
            "xtal_id": "test_xtal",
            "target_prop": "pIC50",
            "target_val": 5.0,
            "in_range": 0,
            "uncertainty": 0.2,
            "loss_weight": 0.2,
        },
        {
            "compound_id": "test_compound",
            "xtal_id": "test_xtal",
            "target_prop": "pIC50",
            "target_val": 5.0,
            "in_range": 0,
            "uncertainty": 0.2,
            "loss_weight": 0.2,
        },
        {
            "compound_id": "test_compound2",
            "xtal_id": "test_xtal",
            "target_prop": "pIC50",
            "target_val": 5.0,
            "in_range": 0,
            "uncertainty": 0.2,
            "loss_weight": 0.8,
        },
    )


@pytest.fixture()
def loss_configs():
    # Just get a couple different loss function configs of different types
    return [
        LossFunctionConfig(loss_type="mse_step"),
        LossFunctionConfig(loss_type="range", range_lower_lim=0, range_upper_lim=10),
    ]


def test_training_pred_constructor(identifiers, loss_configs):
    _ = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    _ = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])


def test_training_pred_json_roundtrip(identifiers, loss_configs):
    tp = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])

    json_str = tp.json()
    tp_roundtrip = TrainingPrediction(**json.loads(json_str))

    for k, v in tp.dict().items():
        assert getattr(tp_roundtrip, k) == v


def test_training_pred_tracker_constructor_no_dict():
    tp_tracker = TrainingPredictionTracker()

    assert tp_tracker.split_dict.keys() == {"train", "val", "test"}
    assert all([v == [] for v in tp_tracker.split_dict.values()])


def test_training_pred_tracker_constructor_ok_dict(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    assert tp_tracker.split_dict.keys() == {"train", "val", "test"}
    assert len(tp_tracker.split_dict["train"]) == 1
    assert len(tp_tracker.split_dict["val"]) == 1
    assert len(tp_tracker.split_dict["test"]) == 0


def test_training_pred_tracker_constructor_bad_dict(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    with pytest.raises(ValidationError):
        _ = TrainingPredictionTracker(split_dict={"train": [tp1], "val": [tp2]})


def test_training_pred_tracker_json_roundtrip(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    json_str = tp_tracker.json()
    tp_roundtrip = TrainingPredictionTracker(**json.loads(json_str))

    for k, v in tp_tracker.dict().items():
        assert getattr(tp_roundtrip, k) == v


def test_training_pred_tracker_len(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    assert len(tp_tracker) == 2


def test_training_pred_tracker_iter(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    it = iter(tp_tracker)

    sp, tp = next(it)
    assert sp == "train"
    assert tp == tp1

    sp, tp = next(it)
    assert sp == "val"
    assert tp == tp2

    with pytest.raises(StopIteration):
        sp, tp = next(it)


def test_find_value_idxs(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    idxs = tp_tracker._find_value_idxs(loss_config=loss_configs[0])
    assert idxs == {"train": [0], "val": [], "test": []}

    idxs = tp_tracker._find_value_idxs(split="train")
    assert idxs == {"train": [0], "val": [], "test": []}

    idxs = tp_tracker._find_value_idxs(
        xtal_id=identifiers[0]["xtal_id"],
        target_prop=identifiers[0]["target_prop"],
    )
    assert idxs == {"train": [0], "val": [0], "test": []}

    idxs = tp_tracker._find_value_idxs()
    assert idxs == {"train": [0], "val": [0], "test": []}

    idxs = tp_tracker._find_value_idxs(compound_id="bad_id")
    assert idxs == {"train": [], "val": [], "test": []}


def test_get_values_split(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    vals = tp_tracker.get_values(split="train")
    assert vals == [tp1]

    vals = tp_tracker.get_values(split="train", compound_id="bad_id")
    assert vals == []


def test_get_values_no_split(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    vals = tp_tracker.get_values()
    assert vals == {"train": [tp1], "val": [tp2], "test": []}

    vals = tp_tracker.get_values(loss_config=loss_configs[0])
    assert vals == {"train": [tp1], "val": [], "test": []}

    vals = tp_tracker.get_values(
        xtal_id=identifiers[0]["xtal_id"],
        target_prop=identifiers[0]["target_prop"],
    )
    assert vals == {"train": [tp1], "val": [tp2], "test": []}

    vals = tp_tracker.get_values(compound_id="bad_id")
    assert vals == {"train": [], "val": [], "test": []}


def test_update_values_existing(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=0.0,
        **identifiers[0],
        loss_config=loss_configs[0],
    )

    tp = tp_tracker.get_values(
        split="train",
        compound_id=identifiers[0]["compound_id"],
        xtal_id=identifiers[0]["xtal_id"],
        target_prop=identifiers[0]["target_prop"],
        loss_config=loss_configs[0],
    )[0]

    assert len(tp.predictions) == 1
    assert len(tp.pose_predictions) == 1
    assert len(tp.loss_vals) == 1


def test_update_values_existing_multi_bad(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    with pytest.raises(ValueError):
        tp_tracker.update_values(
            prediction=1.0,
            pose_predictions=[1.0, 2.0, 3.0],
            loss_val=0.0,
            xtal_id=identifiers[0]["xtal_id"],
            target_prop=identifiers[0]["target_prop"],
        )


def test_update_values_existing_multi_ok(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [tp2], "test": []}
    )

    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=0.0,
        xtal_id=identifiers[0]["xtal_id"],
        target_prop=identifiers[0]["target_prop"],
        allow_multiple=True,
    )

    tp = tp_tracker.get_values(
        split="train",
        compound_id=identifiers[0]["compound_id"],
        xtal_id=identifiers[0]["xtal_id"],
        target_prop=identifiers[0]["target_prop"],
        loss_config=loss_configs[0],
    )[0]

    assert len(tp.predictions) == 1
    assert len(tp.pose_predictions) == 1
    assert len(tp.loss_vals) == 1

    tp = tp_tracker.get_values(
        split="val",
        compound_id=identifiers[1]["compound_id"],
        xtal_id=identifiers[1]["xtal_id"],
        target_prop=identifiers[1]["target_prop"],
        loss_config=loss_configs[1],
    )[0]

    assert len(tp.predictions) == 1
    assert len(tp.pose_predictions) == 1
    assert len(tp.loss_vals) == 1


def test_update_values_new(identifiers, loss_configs):
    tp_tracker = TrainingPredictionTracker()

    tp_tracker.update_values(
        split="train",
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=0.0,
        **identifiers[0],
        loss_config=loss_configs[0],
    )

    tp = tp_tracker.get_values(
        split="train",
        compound_id=identifiers[0]["compound_id"],
        xtal_id=identifiers[0]["xtal_id"],
        target_prop=identifiers[0]["target_prop"],
        loss_config=loss_configs[0],
    )[0]

    assert len(tp.predictions) == 1
    assert len(tp.pose_predictions) == 1
    assert len(tp.loss_vals) == 1


def test_update_values_new_no_split(identifiers, loss_configs):
    tp_tracker = TrainingPredictionTracker()

    with pytest.raises(ValueError):
        tp_tracker.update_values(
            prediction=1.0,
            pose_predictions=[1.0, 2.0, 3.0],
            loss_val=0.0,
            **identifiers[0],
            loss_config=loss_configs[0],
        )


def test_training_pred_tracker_compounds(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [], "test": []}
    )

    cpds = tp_tracker.get_compounds()
    assert cpds == {
        "train": {(tp1.xtal_id, tp1.compound_id)},
        "val": set(),
        "test": set(),
    }

    tp_tracker.update_values(
        split="val",
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=0.0,
        **identifiers[1],
        loss_config=loss_configs[1],
    )

    cpds = tp_tracker.get_compounds()
    assert cpds == {
        "train": {(tp1.xtal_id, tp1.compound_id)},
        "val": {(tp2.xtal_id, tp2.compound_id)},
        "test": set(),
    }


def test_training_pred_tracker_compound_ids(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1], "val": [], "test": []}
    )

    cpds = tp_tracker.get_compound_ids()
    assert cpds == {
        "train": {tp1.compound_id},
        "val": set(),
        "test": set(),
    }

    tp_tracker.update_values(
        split="val",
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=0.0,
        **identifiers[1],
        loss_config=loss_configs[1],
    )

    cpds = tp_tracker.get_compound_ids()
    assert cpds == {
        "train": {tp1.compound_id},
        "val": {tp2.compound_id},
        "test": set(),
    }


def test_training_pred_tracker_get_losses_no_agg(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp1_1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[1])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1, tp1_1], "val": [tp2], "test": []}
    )

    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=5.0,
        **identifiers[0],
        loss_config=loss_configs[0],
    )
    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=20.0,
        **identifiers[2],
        loss_config=loss_configs[1],
    )
    tp_tracker.update_values(
        prediction=2.0,
        pose_predictions=[2.0, 4.0, 6.0],
        loss_val=10.0,
        **identifiers[1],
        loss_config=loss_configs[1],
    )

    loss_dict = tp_tracker.get_losses()

    assert set(loss_dict.keys()) == {"train", "val"}

    assert set(loss_dict["train"].keys()) == {tp1.compound_id}
    assert set(loss_dict["val"].keys()) == {tp2.compound_id}

    assert set(loss_dict["train"][tp1.compound_id].keys()) == {
        loss_configs[0].json(),
        loss_configs[1].json(),
    }
    assert set(loss_dict["val"][tp2.compound_id].keys()) == {loss_configs[1].json()}

    assert (loss_dict["train"][tp1.compound_id][loss_configs[0].json()] == [5.0]).all()
    assert (loss_dict["train"][tp1.compound_id][loss_configs[1].json()] == [20.0]).all()
    assert (loss_dict["val"][tp2.compound_id][loss_configs[1].json()] == [10.0]).all()


def test_training_pred_tracker_get_losses_agg_losses(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp1_1 = TrainingPrediction(**identifiers[2], loss_config=loss_configs[1])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1, tp1_1], "val": [tp2], "test": []}
    )

    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=5.0,
        **identifiers[0],
        loss_config=loss_configs[0],
    )
    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=20.0,
        **identifiers[2],
        loss_config=loss_configs[1],
    )
    tp_tracker.update_values(
        prediction=2.0,
        pose_predictions=[2.0, 4.0, 6.0],
        loss_val=10.0,
        **identifiers[1],
        loss_config=loss_configs[1],
    )

    loss_dict = tp_tracker.get_losses(agg_losses=True)

    assert set(loss_dict.keys()) == {"train", "val"}

    assert set(loss_dict["train"].keys()) == {tp1.compound_id}
    assert set(loss_dict["val"].keys()) == {tp2.compound_id}

    assert (
        loss_dict["train"][tp1.compound_id]
        == [5 * identifiers[0]["loss_weight"] + 20 * identifiers[2]["loss_weight"]]
    ).all()
    # Loss vals were weighted bc aggregating over losses
    assert (loss_dict["val"][tp2.compound_id] == [2.0]).all()


def test_training_pred_tracker_get_losses_agg_compounds(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp1_1 = TrainingPrediction(**identifiers[2], loss_config=loss_configs[1])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])
    tp2_1 = TrainingPrediction(**identifiers[3], loss_config=loss_configs[0])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1, tp1_1, tp2, tp2_1], "val": [], "test": []}
    )

    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=5.0,
        **identifiers[0],
        loss_config=loss_configs[0],
    )
    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=20.0,
        **identifiers[2],
        loss_config=loss_configs[1],
    )
    tp_tracker.update_values(
        prediction=2.0,
        pose_predictions=[2.0, 4.0, 6.0],
        loss_val=10.0,
        **identifiers[1],
        loss_config=loss_configs[1],
    )
    tp_tracker.update_values(
        prediction=2.0,
        pose_predictions=[2.0, 4.0, 6.0],
        loss_val=30.0,
        **identifiers[3],
        loss_config=loss_configs[0],
    )

    loss_dict = tp_tracker.get_losses(agg_compounds=True)

    assert set(loss_dict.keys()) == {"train"}

    assert set(loss_dict["train"].keys()) == {
        loss_configs[0].json(),
        loss_configs[1].json(),
    }

    # Dividing by 2 now bc we're taking mean across multiple compounds
    assert (loss_dict["train"][loss_configs[0].json()] == [5 / 2 + 30 / 2]).all()
    assert (loss_dict["train"][loss_configs[1].json()] == [20 / 2 + 10 / 2]).all()


def test_training_pred_tracker_get_losses_agg_both(identifiers, loss_configs):
    tp1 = TrainingPrediction(**identifiers[0], loss_config=loss_configs[0])
    tp1_1 = TrainingPrediction(**identifiers[2], loss_config=loss_configs[1])
    tp2 = TrainingPrediction(**identifiers[1], loss_config=loss_configs[1])
    tp2_1 = TrainingPrediction(**identifiers[3], loss_config=loss_configs[0])

    tp_tracker = TrainingPredictionTracker(
        split_dict={"train": [tp1, tp1_1, tp2, tp2_1], "val": [], "test": []}
    )

    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=5.0,
        **identifiers[0],
        loss_config=loss_configs[0],
    )
    tp_tracker.update_values(
        prediction=1.0,
        pose_predictions=[1.0, 2.0, 3.0],
        loss_val=20.0,
        **identifiers[2],
        loss_config=loss_configs[1],
    )
    tp_tracker.update_values(
        prediction=2.0,
        pose_predictions=[2.0, 4.0, 6.0],
        loss_val=10.0,
        **identifiers[1],
        loss_config=loss_configs[1],
    )
    tp_tracker.update_values(
        prediction=2.0,
        pose_predictions=[2.0, 4.0, 6.0],
        loss_val=30.0,
        **identifiers[3],
        loss_config=loss_configs[0],
    )

    loss_dict = tp_tracker.get_losses(agg_compounds=True, agg_losses=True)

    assert set(loss_dict.keys()) == {"train"}

    # Weighted mean across loss configs and regular mean across compounds
    loss_val = (
        5 * identifiers[0]["loss_weight"] + 20 * identifiers[2]["loss_weight"]
    ) / 2 + (
        10 * identifiers[1]["loss_weight"] + 30 * identifiers[3]["loss_weight"]
    ) / 2
    assert (loss_dict["train"] == [loss_val]).all()
