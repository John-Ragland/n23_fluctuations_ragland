"""Lightweight tests for the publication figures config and driver (no data needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest
import yaml

mpl.use("Agg")

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
