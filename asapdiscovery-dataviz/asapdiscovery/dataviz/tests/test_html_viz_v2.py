import pytest
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.dataviz.html_viz import HTMLVisualizer
from asapdiscovery.docking.openeye import POSITDockingResults
from asapdiscovery.data.schema.complex import Complex

@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("target", sorted(TargetTags.get_values()))
def test_html_viz_subpockets_disk(target, docking_results_file, use_dask, tmp_path):
    html_viz = HTMLVisualizer(
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
    html_viz = HTMLVisualizer(
        target=target, output_dir=tmp_path, colour_method="subpockets"
    )
    vizs = html_viz.visualize(
        inputs=docking_results_in_memory, use_dask=use_dask, backend="in-memory"
    )
    assert len(vizs) == 1

@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("write_to_disk", [True, False])

def test_html_viz_fitness_in_mem(
    docking_results_in_memory, use_dask, align, write_to_disk, tmp_path
):
    html_viz = HTMLVisualizer(
        target="SARS-CoV-2-Mpro", output_dir=tmp_path, colour_method="fitness", align=align, write_to_disk=write_to_disk
    )
    vizs = html_viz.visualize(
        inputs=docking_results_in_memory, use_dask=use_dask, backend="in-memory"
    )
    assert len(vizs) == 1


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("write_to_disk", [True, False])
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("color_method", ["fitness", "subpockets"])
def test_html_viz_from_pdb_file(
    use_dask, tmp_path, protein, write_to_disk, align, color_method
):
    html_viz = HTMLVisualizer(
        target="SARS-CoV-2-Mpro", output_dir=tmp_path, color_method=color_method, align=align, write_to_disk=write_to_disk
    )
    vizs = html_viz.visualize(inputs=[protein], use_dask=use_dask)
    assert len(vizs) == 1


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("write_to_disk", [True, False])
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("color_method", ["fitness", "subpockets"])
def test_html_viz_from_complex(
    use_dask, tmp_path, protein, write_to_disk, align, color_method
):
    html_viz = HTMLVisualizer(
        target="SARS-CoV-2-Mpro", output_dir=tmp_path, color_method=color_method, align=align, write_to_disk=write_to_disk
    )
    vizs = html_viz.visualize(inputs=[Complex.from_pdb(protein,target_kwargs={"target_name": "unknown_target"},
                ligand_kwargs={"compound_name": f"unknown_compound"})], use_dask=use_dask)
    assert len(vizs) == 1


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("write_to_disk", [True, False])
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("color_method", ["fitness", "subpockets"])
def test_html_viz_from_multisdf(
    tmp_path, protein, multisdf, color_method, align, use_dask, write_to_disk
):
    html_viz = HTMLVisualizer(
            target="SARS-CoV-2-Mpro", output_dir=tmp_path, color_method=color_method, align=align, write_to_disk=write_to_disk
        )
    ligs = MolFileFactory(filename=multisdf.load()
    vizs = html_viz.visualize(inputs=[(protein, ligs)], use_dask=use_dask)
    assert len(vizs) == 1





