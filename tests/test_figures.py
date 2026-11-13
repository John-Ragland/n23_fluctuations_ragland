"""Lightweight tests for the publication figures config and driver (no data needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest
import yaml

mpl.use("Agg")

import matplotlib.pyplot as plt

from n23_fluctuations_ragland import figures


def test_template_lists_every_figure_in_order():
    data = yaml.safe_load(figures.template_path().read_text())
    assert data["which"] == list(figures.FIGURES)


def test_template_is_the_defaults(tmp_path):
    cfg = figures.load_config(figures.write_config_template(tmp_path))
    defaults = figures.PublicationFiguresConfig(output_dir=tmp_path / "figures")
    assert cfg == defaults


def test_write_template_refuses_to_clobber(tmp_path):
    dest = figures.write_config_template(tmp_path / "config.yaml")
    with pytest.raises(FileExistsError):
        figures.write_config_template(dest)
    figures.write_config_template(dest, overwrite=True)


def test_output_dir_is_relative_to_the_config(tmp_path):
    path = tmp_path / "sub" / "config.yaml"
    path.parent.mkdir()
    path.write_text("output_dir: out\n")
    assert figures.load_config(path).output_dir == path.parent / "out"


@pytest.mark.parametrize(
    "text",
    [
        "not_a_key: 1\n",
        "which: [NotAFigure]\n",
        "formats: [jpg]\n",
        "formats: []\n",
        "usetex: maybe\n",
        "remove_array_tilt: maybe\n",
        "dpi: high\n",
        "map: {fontt: x}\n",
        "map: {ssh_dates: [2023-04-30]}\n",
    ],
)
def test_invalid_config_is_an_error(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(figures.ConfigError):
        figures.load_config(path)


def test_remove_array_tilt_defaults_on(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    assert figures.load_config(path).remove_array_tilt is True
    path.write_text("remove_array_tilt: false\n")
    assert figures.load_config(path).remove_array_tilt is False


def test_dilation_is_not_a_config_key(tmp_path):
    """The dilation option was removed, so an old config that still sets it is rejected."""
    path = tmp_path / "config.yaml"
    path.write_text("dilation: true\n")
    with pytest.raises(figures.ConfigError):
        figures.load_config(path)


def test_wavefront_values_follow_remove_array_tilt(monkeypatch, tmp_path):
    monkeypatch.setattr(figures.tools, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(figures, "_wavefront_values", lambda name: name)
    cfg = figures.PublicationFiguresConfig(remove_pattern="none")
    assert figures._wavefront_values_for(cfg) == "wavefront_values_tilt_removed.pkl"
    cfg = figures.PublicationFiguresConfig(
        remove_array_tilt=False, remove_pattern="none"
    )
    assert figures._wavefront_values_for(cfg) == "wavefront_values.pkl"
    cfg = figures.PublicationFiguresConfig()
    assert (
        figures._wavefront_values_for(cfg)
        == "wavefront_values_tilt_removed_phase_magnitude_removed.pkl"
    )


def test_remove_pattern_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("remove_pattern: phase\n")
    assert figures.load_config(path).remove_pattern == "phase"
    assert figures.PublicationFiguresConfig().remove_pattern == "both"
    path.write_text("remove_pattern: gain\n")
    with pytest.raises(figures.ConfigError, match="remove_pattern"):
        figures.load_config(path)


def test_spectral_method_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    cfg = figures.load_config(path)
    assert cfg.window == "hann"
    assert cfg.first_difference is True
    for window in ("tukey", "rectangular"):
        path.write_text(f"window: {window}\nfirst_difference: true\n")
        cfg = figures.load_config(path)
        assert cfg.window == window
        assert cfg.first_difference is True
    for bad in ("window: hamming\n", "hann: false\n", "first_difference: yes please\n"):
        path.write_text(bad)
        with pytest.raises(figures.ConfigError, match=r"window|hann|first_difference"):
            figures.load_config(path)


def test_missing_config_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="create publication figures config"):
        figures.load_config(tmp_path / "config.yaml")


def _dummy(cfg):
    fig = mpl.figure.Figure()
    fig.add_subplot().plot([0, 1], [0, cfg.seed])
    return fig


def _broken(cfg):
    msg = "boom"
    raise RuntimeError(msg)


def test_one_failing_figure_does_not_stop_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(
        figures,
        "FIGURES",
        {
            "a": figures.FigureSpec(_dummy),
            "b": figures.FigureSpec(_broken),
            "c": figures.FigureSpec(_dummy, "png"),
        },
    )
    cfg = figures.PublicationFiguresConfig(output_dir=tmp_path / "out", usetex=False)
    build = figures.generate_publication_figures(cfg)

    assert build.written == [tmp_path / "out" / "a.pdf", tmp_path / "out" / "c.png"]
    assert all(Path(p).exists() for p in build.written)
    assert list(build.failed) == ["b"]
    assert "boom" in build.failed["b"]


def test_formats_override_each_figure_s_default(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURES", {"a": figures.FigureSpec(_dummy, "png")})
    cfg = figures.PublicationFiguresConfig(
        output_dir=tmp_path, formats=("pdf", "svg"), usetex=False
    )
    build = figures.generate_publication_figures(cfg)
    assert build.written == [tmp_path / "a.pdf", tmp_path / "a.svg"]


def test_only_rejects_an_unknown_figure(tmp_path):
    cfg = figures.PublicationFiguresConfig(output_dir=tmp_path)
    with pytest.raises(figures.ConfigError):
        figures.generate_publication_figures(cfg, only=["nope"])


def test_label_panels_letters_and_box_style():
    fig, axes = plt.subplots(1, 3)
    figures.label_panels(
        axes, ["upper right", "upper left", (0.5, 0.5, "center", "top")]
    )
    labels = [ax.texts[0] for ax in axes]
    assert [t.get_text() for t in labels] == ["(a)", "(b)", "(c)"]
    for t in labels:
        assert t.get_fontsize() == 7
        patch = t.get_bbox_patch()
        assert patch.get_linewidth() == 0.5
        assert patch.get_edgecolor()[3] == 1
        assert patch.get_facecolor()[3] == pytest.approx(0.8)
    plt.close(fig)


def test_label_panels_rejects_bad_locations():
    fig, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="2 axes"):
        figures.label_panels(axes, ["upper right"] * 3)
    with pytest.raises(ValueError, match="unknown panel label location"):
        figures.label_panels(axes, "top middle")
    plt.close(fig)


def test_spectra_model_defaults_to_gm_levels(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    assert figures.load_config(path).spectra_model == "gm_levels"
    path.write_text("spectra_model: fit\n")
    assert figures.load_config(path).spectra_model == "fit"
    path.write_text("spectra_model: 3xGM\n")
    with pytest.raises(figures.ConfigError):
        figures.load_config(path)


def test_gm_levels_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}\n")
    assert figures.load_config(path).gm_levels == (1.0,)
    path.write_text("gm_levels: [1, 2]\n")
    assert figures.load_config(path).gm_levels == (1.0, 2.0)
    path.write_text("gm_levels: [0.5, 3]\n")
    assert figures.load_config(path).gm_levels == (0.5, 3.0)
    for bad in ["[]", "[1, 2, 3]", "[0]", "[-1]", "1", "[true]", "[one]"]:
        path.write_text(f"gm_levels: {bad}\n")
        with pytest.raises(figures.ConfigError):
            figures.load_config(path)


def test_defaults_are_the_paper_settings(tmp_path):
    """The default config (and so the template and 'reproduce paper') builds the figures as in the paper."""
    cfg = figures.load_config(figures.write_config_template(tmp_path))
    for key, value in figures.tools.PAPER_SETTINGS.items():
        assert getattr(cfg, key) == value, key
    assert cfg.spectra_model == "gm_levels"
    assert cfg.gm_levels == (1.0,)


def test_map_input_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(figures, "get_data_dir", lambda: tmp_path)
    paths = figures.map_input_paths(figures.PublicationFiguresConfig())
    assert paths["bathymetry"] == tmp_path / "environment_data" / "map_bathy.nc"
    assert paths["ssh"] == tmp_path / "environment_data" / figures.SSH_FILENAME
    cfg = figures.PublicationFiguresConfig(map=figures.MapConfig(ssh="/x/ssh.nc"))
    assert figures.map_input_paths(cfg)["ssh"] == Path("/x/ssh.nc")
