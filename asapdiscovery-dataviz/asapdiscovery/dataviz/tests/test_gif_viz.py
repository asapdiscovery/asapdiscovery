from asapdiscovery.dataviz.gif_viz import GIFVisualizer


def test_gif_viz(traj, top, tmp_path):
    gif_visualiser = GIFVisualizer(
        [traj],
        [top],
        [tmp_path / "gif_viz.gif"],
        "SARS-CoV-2-Mpro",  # just do a fast test with one target
        frames_per_ns=200,
        smooth=5,
        start=0,
        pse=False,
        pse_share=False,
    )
    gif_visualiser.write_traj_visualizations()
