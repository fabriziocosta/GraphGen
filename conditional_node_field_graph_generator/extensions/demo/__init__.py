"""Demo-oriented helpers for notebook workflows."""

from ...persistence import list_saved_graph_generators, load_graph_generator, save_graph_generator
from .pipeline import build_dataset, build_graph_generator, fit_graph_generator, prepare_experiment
from .storage import describe_resume_checkpoint, find_latest_checkpoint, list_training_checkpoints
from .visualization import (
    compare_real_vs_generated,
    infer_display_mode,
    offset_neg_graphs,
    plot_label_histogram_alignment,
    plot_networkx_graphs,
    plot_sample,
    plot_similarity_distribution_with_iqr,
    run_label_histogram_analysis,
    sample_positive_endpoint_pair,
    select_pos_neg,
    show_samples,
    summarize_label_histogram_alignment,
)

__all__ = [
    "build_dataset",
    "build_graph_generator",
    "compare_real_vs_generated",
    "describe_resume_checkpoint",
    "find_latest_checkpoint",
    "fit_graph_generator",
    "infer_display_mode",
    "list_training_checkpoints",
    "list_saved_graph_generators",
    "load_graph_generator",
    "offset_neg_graphs",
    "plot_label_histogram_alignment",
    "plot_networkx_graphs",
    "plot_sample",
    "plot_similarity_distribution_with_iqr",
    "prepare_experiment",
    "run_label_histogram_analysis",
    "sample_positive_endpoint_pair",
    "save_graph_generator",
    "select_pos_neg",
    "show_samples",
    "summarize_label_histogram_alignment",
]
