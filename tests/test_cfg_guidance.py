import types

import networkx as nx
import numpy as np
import pytest
import torch

from equilibrium_matching_decompositional_graph_generator.graph_engine import (
    EquilibriumMatchingDecompositionalGraphGenerator,
)
from equilibrium_matching_decompositional_graph_generator.node_engine import (
    EquilibriumMatchingDecompositionalNodeGenerator,
    EquilibriumMatchingDecompositionalNodeGeneratorModule,
)
from equilibrium_matching_decompositional_graph_generator.node_engine import (
    GeneratedNodeBatch,
    GraphConditioningBatch,
)


class _GraphVectorizer:
    def fit(self, graphs):
        return self

    def transform(self, graphs):
        return np.asarray(
            [[graph.number_of_nodes(), graph.number_of_edges()] for graph in graphs],
            dtype=float,
        )


class _NodeVectorizer:
    def fit(self, graphs):
        return self

    def transform(self, graphs):
        out = []
        for graph in graphs:
            out.append(np.asarray([[float(graph.degree(node))] for node in graph.nodes()], dtype=float))
        return out


class _DecoderStub:
    def __init__(self):
        self.verbose = False

    def compute_edge_supervision(self, *args, **kwargs):
        return None, None

    def decode(self, *args, **kwargs):
        return []


class _ConditionalStub:
    def __init__(self):
        self.verbose = False
        self.setup_targets = None
        self.fit_targets = None
        self.last_predict_kwargs = None

    def setup(self, node_batch, graph_conditioning, targets=None):
        self.setup_targets = targets

    def fit(self, node_batch, graph_conditioning, targets=None):
        self.fit_targets = targets

    def predict(self, graph_conditioning, desired_target=None, guidance_scale=1.0, desired_class=None):
        self.last_predict_kwargs = {
            "desired_target": desired_target,
            "guidance_scale": guidance_scale,
            "desired_class": desired_class,
        }
        n = len(graph_conditioning)
        return GeneratedNodeBatch(
            node_embeddings_list=[np.zeros((2, 1), dtype=float) for _ in range(n)],
            node_presence_mask=np.ones((n, 2), dtype=bool),
            node_degree_predictions=np.ones((n, 2), dtype=np.int64),
        )


def _graph():
    graph = nx.Graph()
    graph.add_node(0, label="C")
    graph.add_node(1, label="O")
    graph.add_edge(0, 1, label="-")
    return graph


def test_graph_generator_fit_forwards_targets_to_conditional_model():
    graphs = [_graph(), _graph()]
    targets = [0, 1]
    cond = _ConditionalStub()
    generator = EquilibriumMatchingDecompositionalGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=cond,
        graph_decoder=_DecoderStub(),
        verbose=False,
    )

    generator.fit(graphs, targets=targets)

    assert cond.setup_targets == targets
    assert cond.fit_targets == targets


def test_graph_generator_decode_passes_cfg_args_to_predict():
    cond = _ConditionalStub()
    generator = EquilibriumMatchingDecompositionalGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=cond,
        graph_decoder=_DecoderStub(),
        verbose=False,
    )
    generator.is_fitted_ = True
    conditioning = generator.graph_encode([_graph()])
    generator.decode(conditioning, desired_target=1, guidance_scale=2.5)
    assert cond.last_predict_kwargs["desired_target"] == 1
    assert cond.last_predict_kwargs["guidance_scale"] == 2.5


def test_graph_generator_decode_requires_fit():
    generator = EquilibriumMatchingDecompositionalGraphGenerator(
        graph_vectorizer=_GraphVectorizer(),
        node_graph_vectorizer=_NodeVectorizer(),
        conditional_node_generator_model=_ConditionalStub(),
        graph_decoder=_DecoderStub(),
        verbose=False,
    )

    with pytest.raises(RuntimeError, match="is not fitted"):
        generator.decode(
            GraphConditioningBatch(
                graph_embeddings=np.zeros((1, 2), dtype=float),
                node_counts=np.ones((1,), dtype=np.int64),
                edge_counts=np.zeros((1,), dtype=np.int64),
            )
        )


def test_node_generator_target_mode_inference_classification_vs_regression():
    generator = EquilibriumMatchingDecompositionalNodeGenerator(target_classification_max_distinct=3)
    generator._fit_target_encoder([0, 1, 1, 0])
    assert generator.target_mode_ == "classification"
    assert generator.target_condition_dim_ == 2

    generator._fit_target_encoder(np.arange(10))
    assert generator.target_mode_ == "regression"
    assert generator.target_condition_dim_ == 1


def test_cfg_dropout_zeros_target_slice_when_probability_is_one():
    module = EquilibriumMatchingDecompositionalNodeGeneratorModule(
        number_of_rows_per_example=2,
        input_feature_dimension=1,
        condition_feature_dimension=5,
        latent_embedding_dimension=8,
        number_of_transformer_layers=1,
        transformer_attention_head_count=1,
        max_degree=2,
        guidance_enabled=True,
        target_condition_start_index=3,
        target_condition_feature_count=2,
        cfg_condition_dropout_prob=1.0,
    )
    cond = torch.tensor([[1.0, 2.0, 3.0, 9.0, 8.0]], dtype=torch.float32)
    dropped = module._apply_cfg_dropout(cond)
    assert dropped[0, :3].tolist() == [1.0, 2.0, 3.0]
    assert dropped[0, 3:].tolist() == [0.0, 0.0]


def test_cfg_score_mixing_in_generate_uses_uncond_plus_scaled_delta():
    module = EquilibriumMatchingDecompositionalNodeGeneratorModule(
        number_of_rows_per_example=2,
        input_feature_dimension=1,
        condition_feature_dimension=4,
        latent_embedding_dimension=8,
        number_of_transformer_layers=1,
        transformer_attention_head_count=1,
        max_degree=2,
        guidance_enabled=True,
        target_condition_start_index=2,
        target_condition_feature_count=2,
    )

    def _fake_score(self, noisy_input, global_condition, node_mask=None, create_graph=False):
        del node_mask, create_graph
        scale = global_condition.sum(dim=1, keepdim=True).unsqueeze(-1)
        score = torch.ones_like(noisy_input) * scale
        latent = torch.zeros(
            noisy_input.shape[0],
            noisy_input.shape[1],
            self.latent_embedding_dimension,
            device=noisy_input.device,
        )
        phi = torch.zeros(noisy_input.shape[0], device=noisy_input.device)
        return score, phi, latent

    module._compute_score_field = types.MethodType(_fake_score, module)
    cond = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    uncond = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

    torch.manual_seed(0)
    out_s0 = module.generate(
        cond,
        total_steps=1,
        guidance_scale=0.0,
        global_condition_unconditional=uncond,
        use_heads_projection=False,
    )
    torch.manual_seed(0)
    out_s1 = module.generate(
        cond,
        total_steps=1,
        guidance_scale=1.0,
        global_condition_unconditional=uncond,
        use_heads_projection=False,
    )
    torch.manual_seed(0)
    out_s2 = module.generate(
        cond,
        total_steps=1,
        guidance_scale=2.0,
        global_condition_unconditional=uncond,
        use_heads_projection=False,
    )

    # Same initial x for each run; deltas should scale linearly with guidance_scale.
    delta_10 = out_s1 - out_s0
    delta_21 = out_s2 - out_s1
    assert torch.allclose(delta_10, delta_21, atol=1e-6)


def test_predict_rejects_negative_guidance_scale():
    generator = EquilibriumMatchingDecompositionalNodeGenerator()
    generator.model = types.SimpleNamespace(parameters=lambda: iter([torch.tensor(0.0)]))
    with pytest.raises(ValueError, match="guidance_scale must be >= 0"):
        generator.predict(
            graph_conditioning=GraphConditioningBatch(
                graph_embeddings=np.zeros((1, 2), dtype=float),
                node_counts=np.ones((1,), dtype=np.int64),
                edge_counts=np.zeros((1,), dtype=np.int64),
            ),
            guidance_scale=-1.0,
        )


def test_node_generator_predict_requires_setup_or_fit():
    generator = EquilibriumMatchingDecompositionalNodeGenerator()

    with pytest.raises(RuntimeError, match="Call setup\\(\\) or fit\\(\\) before predict\\(\\)"):
        generator.predict(
            graph_conditioning=GraphConditioningBatch(
                graph_embeddings=np.zeros((1, 2), dtype=float),
                node_counts=np.ones((1,), dtype=np.int64),
                edge_counts=np.zeros((1,), dtype=np.int64),
            ),
        )
