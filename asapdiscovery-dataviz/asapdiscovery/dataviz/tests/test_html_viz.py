from pathlib import Path

import pytest
from asapdiscovery.data.backend.openeye import load_openeye_pdb, load_openeye_sdf
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.dataviz.html_viz import HTMLVisualizer


# enumerate over the allowed targets, these will produce rubbish poses but
# that's fine for testing, mostly just testing that they will run
@pytest.mark.parametrize("target", sorted(TargetTags.get_values()))
def test_html_viz_subpockets(pose, protein, target, tmp_path):
    html_visualizer = HTMLVisualizer(
        poses=[pose],
        output_paths=[tmp_path / "html_viz.html"],
        target=target,
        protein=protein,
    )
    html_visualizer.write_pose_visualizations()
    assert Path(tmp_path / "html_viz.html").exists()


@pytest.mark.parametrize("align", [True, False])
def test_html_viz_subpockets_align(pose, protein, align, tmp_path):
    html_visualizer = HTMLVisualizer(
        poses=[pose],
        output_paths=[tmp_path / "html_viz.html"],
        target="SARS-CoV-2-Mpro",
        protein=protein,
        align=align,
    )
    html_visualizer.write_pose_visualizations()
    assert Path(tmp_path / "html_viz.html").exists()


# test the OEMol code path
@pytest.mark.parametrize("align", [True, False])
def test_html_viz_subpockets_oemol_align(pose, protein, align, tmp_path):
    pose_mol = load_openeye_sdf(pose)
    protein_mol = load_openeye_pdb(protein)
    html_visualizer = HTMLVisualizer(
        poses=[pose_mol],
        output_paths=[tmp_path / "html_viz.html"],
        target="SARS-CoV-2-Mpro",
        protein=protein_mol,
        align=align,
    )
    html_visualizer.write_pose_visualizations()
    assert Path(tmp_path / "html_viz.html").exists()


# No fitness data for MERS-CoV-Mpro
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("target", sorted(["SARS-CoV-2-Mpro", "SARS-CoV-2-Mac1"]))
def test_html_viz_fitness(pose, protein, target, align, tmp_path):
    html_visualizer = HTMLVisualizer(
        poses=[pose],
        output_paths=[tmp_path / "html_viz_fitness.html"],
        target=target,
        protein=protein,
        color_method="fitness",
        align=align,
    )
    html_visualizer.write_pose_visualizations()
    assert Path(tmp_path / "html_viz_fitness.html").exists()
