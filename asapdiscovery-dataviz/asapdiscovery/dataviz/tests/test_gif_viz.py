import logging

import pytest
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
from asapdiscovery.dataviz.viz_targets import VizTargets


@pytest.fixture(scope="session")
def top():
    top = fetch_test_file("example_traj_top.pdb")
    return top


@pytest.fixture(scope="session")
def traj():
    traj = fetch_test_file("example_traj.xtc")
    return traj

@pytest.mark.parametrize(
    "logger",
    [
        None,
        FileLogger(
            "gif_to_viz", path="./", stdout=True, level=logging.DEBUG
        ).getLogger(),
    ],
)
def test_gif_viz(traj, top, logger, tmp_path):
    gif_visualiser = GIFVisualizer(
        [traj],
        [top],
        [tmp_path / "gif_viz.gif"],
        "SARS-CoV-2-Mpro", # just do a fast test with one target
        frames_per_ns=200,
        smooth=5,
        start=0,
        logger=logger,
        pse=False,
        pse_share=False,
    )
    gif_visualiser.write_traj_visualizations()
