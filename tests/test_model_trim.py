from typing import ClassVar

import pytest

from trimbed.config import EmbeddingTrimConfig
from trimbed.model_trim import _remap_config_token_ids, trim_model
from trimbed.remap import IdRemap


pytestmark = pytest.mark.torch

VOCAB_SIZE = 40
KEPT = [0, 1, 2, 5, 9, 17, 30]


@pytest.fixture
def remap():
    return IdRemap.from_kept(KEPT)


def test_kept_rows_keep_their_trained_weights(tiny_model_factory, remap):
    import torch

    model = tiny_model_factory(VOCAB_SIZE)
    before = model.get_input_embeddings().weight.data.clone()

    trim_model(model, remap)

    after = model.get_input_embeddings().weight.data
    for new_id, old_id in enumerate(remap.new_to_old):
        assert torch.allclose(after[new_id], before[old_id]), f"row {new_id} lost its weights"


def test_the_vocabulary_shrinks_and_the_config_follows(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    stats = trim_model(model, remap)

    assert stats.old_embedding_rows == VOCAB_SIZE
    assert stats.new_embedding_rows == len(KEPT)
    assert model.config.vocab_size == len(KEPT)
    assert stats.parameters_removed > 0


def test_untied_output_head_and_bias_are_gathered_too(tiny_model_factory, remap):
    import torch

    model = tiny_model_factory(VOCAB_SIZE, tied=False)
    head = model.get_output_embeddings()
    before_weight = head.weight.data.clone()
    before_bias = head.bias.data.clone() if head.bias is not None else None

    stats = trim_model(model, remap)

    assert stats.tied_embeddings is False
    assert stats.has_output_head is True
    new_head = model.get_output_embeddings()
    for new_id, old_id in enumerate(remap.new_to_old):
        assert torch.allclose(new_head.weight.data[new_id], before_weight[old_id])
        if before_bias is not None:
            assert torch.allclose(new_head.bias.data[new_id], before_bias[old_id])


def test_an_untied_head_without_a_bias_is_supported(tiny_model_factory, remap):
    import torch

    model = tiny_model_factory(VOCAB_SIZE, tied=False)
    head = model.get_output_embeddings()
    # Most decoder-only checkpoints declare `lm_head` as nn.Linear(..., bias=False).
    head.bias = None
    before_weight = head.weight.data.clone()

    trim_model(model, remap)

    new_head = model.get_output_embeddings()
    assert new_head.bias is None
    for new_id, old_id in enumerate(remap.new_to_old):
        assert torch.allclose(new_head.weight.data[new_id], before_weight[old_id])


def test_tied_embeddings_stay_tied(tiny_model_factory, remap):
    import torch

    model = tiny_model_factory(VOCAB_SIZE, tied=True)
    stats = trim_model(model, remap)

    assert stats.tied_embeddings is True
    head = model.get_output_embeddings()
    if head is not None:
        assert torch.allclose(head.weight.data, model.get_input_embeddings().weight.data)


def test_a_tied_head_still_gathers_its_bias(tiny_model_factory, remap):
    import torch

    model = tiny_model_factory(VOCAB_SIZE, tied=True)
    head = model.get_output_embeddings()
    # A masked-LM head's bias starts out at zero, which would hide a wrong gather.
    torch.nn.init.normal_(head.bias)
    before_bias = head.bias.data.clone()

    stats = trim_model(model, remap)

    # Tying carries the weights but not the bias, so the bias has to be gathered too.
    assert stats.tied_embeddings is True
    new_head = model.get_output_embeddings()
    for new_id, old_id in enumerate(remap.new_to_old):
        assert torch.allclose(new_head.bias.data[new_id], before_bias[old_id])


def test_token_ids_on_a_sub_config_are_remapped_too(remap):
    from transformers import PretrainedConfig

    class TextConfig(PretrainedConfig):
        pass

    class CompositeConfig(PretrainedConfig):
        sub_configs: ClassVar = {"vision_config": TextConfig, "text_config": TextConfig}

    # A multimodal checkpoint keeps its text ids down here, with nothing at the top level
    # shadowing them, so a top-level-only pass leaves them pointing at the old vocabulary.
    config = CompositeConfig()
    config.vision_config = None
    config.text_config = TextConfig(eos_token_id=17, pad_token_id=3)

    _remap_config_token_ids(config, remap, "config")

    assert config.text_config.eos_token_id == remap.old_to_new[17]
    assert config.text_config.pad_token_id is None


def test_an_encoder_without_an_output_head_is_supported(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE, with_head=False)
    assert model.get_output_embeddings() is None

    stats = trim_model(model, remap)

    assert stats.has_output_head is False
    assert stats.new_embedding_rows == len(KEPT)


def test_padding_aligns_the_embedding_matrix(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    stats = trim_model(model, remap, EmbeddingTrimConfig(pad_to_multiple_of=8))

    assert stats.new_embedding_rows % 8 == 0
    assert stats.new_embedding_rows >= len(KEPT)


def test_alignment_padding_rows_are_zeroed(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    trim_model(model, remap, EmbeddingTrimConfig(pad_to_multiple_of=8))

    # transformers fills new rows from the mean of the existing embeddings, which leaves
    # `generate` able to emit an id past the end of the trimmed vocabulary.
    padding = model.get_input_embeddings().weight.data[len(KEPT) :]
    assert padding.numel() > 0
    assert bool((padding == 0).all())


def test_alignment_padding_is_zeroed_on_an_untied_head_and_its_bias(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE, tied=False)
    head = model.get_output_embeddings()
    assert head.bias is not None

    trim_model(model, remap, EmbeddingTrimConfig(pad_to_multiple_of=8))

    # Untied, the head keeps its own storage, so zeroing the input embedding does not
    # reach it and the padded logits stay whatever the resize put there.
    assert bool((model.get_output_embeddings().weight.data[len(KEPT) :] == 0).all())
    assert bool((model.get_output_embeddings().bias.data[len(KEPT) :] == 0).all())


def test_padding_idx_follows_the_remap(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    model.get_input_embeddings().padding_idx = 5

    trim_model(model, remap)

    # `resize_token_embeddings` builds a fresh nn.Embedding without it, and a pad row
    # whose gradient is no longer zeroed drifts the moment the trimmed model is trained.
    assert model.get_input_embeddings().padding_idx == remap.to_new(5)


def test_an_embedding_without_a_padding_idx_stays_that_way(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    model.get_input_embeddings().padding_idx = None

    trim_model(model, remap)

    assert model.get_input_embeddings().padding_idx is None


def test_a_dropped_padding_idx_is_cleared(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    model.get_input_embeddings().padding_idx = 33  # not in KEPT

    trim_model(model, remap)

    assert model.get_input_embeddings().padding_idx is None


def test_surviving_special_token_ids_are_remapped(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    model.config.bos_token_id = 5
    model.config.eos_token_id = 9

    trim_model(model, remap)

    assert model.config.bos_token_id == remap.to_new(5)
    assert model.config.eos_token_id == remap.to_new(9)


def test_dropped_special_token_ids_are_cleared_not_left_dangling(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    model.config.pad_token_id = 33  # not in KEPT

    trim_model(model, remap)

    # A stale id would silently index whatever token now sits at row 33.
    assert model.config.pad_token_id is None


def test_only_genuine_token_ids_are_remapped(remap):
    from types import SimpleNamespace

    from trimbed.model_trim import _remap_config_token_ids

    # Configs in the wild put odd things in these fields: the token string instead of
    # its id, or a boolean (which is an int in Python). Neither is a token id.
    config = SimpleNamespace(pad_token_id="[PAD]", bos_token_id=True, eos_token_id=5)

    _remap_config_token_ids(config, remap, "config")

    assert config.pad_token_id == "[PAD]"
    assert config.bos_token_id is True
    assert config.eos_token_id == remap.to_new(5)


def test_ids_outside_the_embedding_matrix_are_refused(tiny_model_factory):
    model = tiny_model_factory(VOCAB_SIZE)
    with pytest.raises(ValueError, match="outside the embedding matrix"):
        trim_model(model, IdRemap.from_kept([0, 1, VOCAB_SIZE + 5]))


def test_a_padded_vocab_size_does_not_confuse_the_trim(tiny_model_factory, remap):
    model = tiny_model_factory(VOCAB_SIZE)
    # Qwen-style: config claims more tokens than the matrix has rows.
    model.config.vocab_size = VOCAB_SIZE + 100

    stats = trim_model(model, remap)

    assert stats.old_embedding_rows == VOCAB_SIZE
    assert stats.new_embedding_rows == len(KEPT)


def test_the_trimmed_model_still_runs(tiny_model_factory, remap):
    import torch

    model = tiny_model_factory(VOCAB_SIZE, with_head=False)
    trim_model(model, remap)

    ids = torch.tensor([[0, 1, 2, 3]])
    output = model(input_ids=ids)
    assert output.last_hidden_state.shape[:2] == ids.shape


def test_outputs_are_unchanged_for_text_made_of_kept_tokens(tiny_model_factory, remap):
    import torch

    model = tiny_model_factory(VOCAB_SIZE, with_head=False)
    old_ids = torch.tensor([[KEPT[1], KEPT[3], KEPT[5]]])
    with torch.no_grad():
        before = model(input_ids=old_ids).last_hidden_state

    trim_model(model, remap)

    new_ids = torch.tensor([[remap.to_new(i) for i in old_ids[0].tolist()]])
    with torch.no_grad():
        after = model(input_ids=new_ids).last_hidden_state

    # The strongest available proof that the right rows were gathered.
    assert torch.allclose(before, after, atol=1e-5)
