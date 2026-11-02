"""Reuse exact pinned upstream generation/realignment and prompt functions.

No import of upstream vLLM entry points and no loading through its broken 8B
CLI choices. KV0 inheritance is explicit in the four-role caller.
"""
import ast
import copy
from pathlib import Path
import runpy
from types import MethodType, SimpleNamespace

METHODS = ('_build_latent_realign_matrix', '_ensure_latent_realign_matrix', '_apply_latent_realignment',
           'render_chat', 'prepare_chat_batch', 'generate_latent_batch', 'generate_text_batch')


def build_core(model, tokenizer, trace, realign=False):
    import torch
    path = Path(__file__).resolve().parent/'vendor/LatentMAS/models.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelWrapper')
    methods = [copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)) for name in METHODS]
    helper = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_past_length'))
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), helper,
        ast.ClassDef(name='PinnedCore', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]))
    ns = {'torch': torch}
    exec(compile(module, str(path), 'exec'), ns)
    core = ns['PinnedCore']()
    core.model = core.HF_model = model
    core.tokenizer = tokenizer
    core.device = model.get_input_embeddings().weight.device
    core.args = SimpleNamespace(latent_space_realign=bool(realign))
    core._latent_realign_matrices = {}
    trace('alignment_build_started', compute='CPU embedding algebra; no CPU model inference')
    with torch.inference_mode():
        matrix, target = core._build_latent_realign_matrix(model, torch.device('cpu'), core.args)
    assert torch.isfinite(matrix).all() and torch.isfinite(target).all() and target.item() > 0
    core._latent_realign_matrices[id(model)] = (matrix.to(core.device), target.to(core.device))
    trace('alignment_build_finished', shape=list(matrix.shape), target_norm=target.item(), upstream_methods_changed=False)
    return core


def prompt_builders():
    return runpy.run_path(str(Path(__file__).resolve().parent/'vendor/LatentMAS/prompts.py'))


def install_position_fix(model):
    """Keep cache storage slots separate from RoPE position IDs.

    The normal path is unchanged. A diagnostic caller may set
    model._latentmas_position_offset while generate runs; each prepared step
    then uses position_ids = cache_position + offset.
    """
    import torch
    original = model._get_initial_cache_position
    def initial(this, seq_length, device, model_kwargs):
        position = model_kwargs.get('cache_position')
        if position is not None:
            past = model_kwargs.get('past_key_values')
            start = 0 if past is None else past.get_seq_length()
            assert torch.equal(position, torch.arange(start, start+seq_length, device=position.device))
            return model_kwargs
        return original(seq_length, device, model_kwargs)
    model._get_initial_cache_position = MethodType(initial, model)
    original_prepare = model.prepare_inputs_for_generation
    def prepare(this, *args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        offset = getattr(this, '_latentmas_position_offset', None)
        if offset is None:
            # A full-coordinate cache with holes is the zero-fill sparse path.
            # Detect it from the mask over existing cache slots, then keep new
            # RoPE positions aligned with their original cache coordinates.
            attention_mask = kwargs.get('attention_mask')
            past = kwargs.get('past_key_values')
            if (torch.is_tensor(attention_mask) and attention_mask.ndim == 2
                    and past is not None):
                past_length = int(past.get_seq_length())
                if past_length and attention_mask.shape[-1] >= past_length:
                    if torch.any(attention_mask[:, :past_length] == 0):
                        offset = 0
        if offset is not None:
            cache_position = prepared.get('cache_position')
            if cache_position is None:
                raise RuntimeError('absolute-position patch requires cache_position')
            prepared['position_ids'] = (cache_position.to(dtype=torch.long) + int(offset)).unsqueeze(0)
        return prepared
    model.prepare_inputs_for_generation = MethodType(prepare, model)
