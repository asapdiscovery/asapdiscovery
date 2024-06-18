import pytest
from asapdiscovery.dataviz.gif_viz import GIFVisualizer


@pytest.mark.parametrize("use_dask", [True, False])
def test_gif_viz(tmp_path, simulation_results, use_dask):
    gv = GIFVisualizer(target="SARS-CoV-2-Mpro", output_dir=tmp_path)
    vizs = gv.visualize(inputs=simulation_results, use_dask=use_dask)
    assert len(vizs) == 1


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("outpaths", [["my_sub_path/viz.gif"], None])
def test_gif_viz_paths(tmp_path, traj, top, use_dask, outpaths):
    gv = GIFVisualizer(target="SARS-CoV-2-Mpro", output_dir=tmp_path)
    vizs = gv.visualize(inputs=[(traj, top)], use_dask=use_dask, outpaths=outpaths)
    assert len(vizs) == 1
