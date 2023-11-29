import pytest
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from asapdiscovery.dataviz.viz_targets import VizTargets


# enumerate over the allowed targets, these will produce rubbish poses but
# that's fine for testing, mostly just testing that they will run
@pytest.mark.parametrize("target", VizTargets.get_allowed_targets())
def test_html_viz_subpockets(pose, protein, target, tmp_path):
    html_visualizer = HTMLVisualizer(
        poses=[pose],
        output_paths=[tmp_path / "html_viz.html"],
        target=target,
        protein=protein,
    )
    html_visualizer.write_pose_visualizations()


# No fitness data for MERS-CoV-Mpro
@pytest.mark.parametrize("target", ["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1"])
def test_html_viz_fitness(pose, protein, target, tmp_path):
    html_visualizer = HTMLVisualizer(
        poses=[pose],
        output_paths=[tmp_path / "html_viz_fitness.html"],
        target=target,
        protein=protein,
        color_method="fitness",
    )
    html_visualizer.write_pose_visualizations()
