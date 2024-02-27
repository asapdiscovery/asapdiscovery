import pytest
from asapdiscovery.dataviz.viz_v2.gif_viz import GIFVisualizerV2


@pytest.mark.parametrize("use_dask", [True, False])
def test_gif_viz_v2(tmp_path, simulation_results, use_dask):
    gv = GIFVisualizerV2(target="SARS-CoV-2-Mpro", output_dir=tmp_path)
    vizs = gv.visualize(inputs=simulation_results, use_dask=use_dask)
    assert len(vizs) == 1
