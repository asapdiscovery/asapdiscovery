from pathlib import Path

import pytest
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.dataviz.viz_v2.html_viz import HTMLVisualizerV2
from asapdiscovery.docking.openeye import POSITDockingResults


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("target", sorted(TargetTags.get_values()))
def test_html_viz_subpockets_disk(target, docking_results_file, use_dask, tmp_path):
    html_viz = HTMLVisualizerV2(
        target=target, output_dir=tmp_path, colour_method="subpockets"
    )
    vizs = html_viz.visualize(
        inputs=docking_results_file,
        backend="disk",
        use_dask=use_dask,
        reconstruct_cls=POSITDockingResults,
    )
    assert len(vizs) == 1


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("target", sorted(TargetTags.get_values()))
def test_html_viz_subpockets_in_mem(
    target, docking_results_in_memory, use_dask, tmp_path
):
    html_viz = HTMLVisualizerV2(
        target=target, output_dir=tmp_path, colour_method="subpockets"
    )
    vizs = html_viz.visualize(
        inputs=docking_results_in_memory, use_dask=use_dask, backend="in-memory"
    )
    assert len(vizs) == 1
