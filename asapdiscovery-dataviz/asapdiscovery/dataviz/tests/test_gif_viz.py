import logging

import pytest
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.dataviz.gif_viz import GIFVisualizer
from asapdiscovery.dataviz.viz_targets import VizTargets


@pytest.fixture(scope="session")
def top():
    pose = fetch_test_file("example_traj_top.pdb")
    return pose


@pytest.fixture(scope="session")
def traj():
    protein = fetch_test_file("example_traj.xtc")
    return protein


# enumerate over the allowed targets, these will produce rubbish GIFS but
# that's fine for testing, mostly just testing that they will run
@pytest.mark.parametrize("target", VizTargets.get_allowed_targets())
@pytest.mark.parametrize(
    "logger",
    [
        None,
        FileLogger(
            "gif_to_viz", path="./", stdout=True, level=logging.DEBUG
        ).getLogger(),
    ],
)
def test_html_viz(traj, top, logger, target, tmp_path):
    gif_visualiser = GIFVisualizer(
        [traj],
        [top],
        [tmp_path / "gif_viz.gif"],
        target,
        frames_per_ns=200,
        smooth=5,
        start=0,
        logger=logger,
        pse=False,
        pse_share=False,
    )
    gif_visualiser.write_traj_visualizations()
