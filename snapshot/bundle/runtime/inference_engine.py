"""Meter every native model forward; use unchanged upstream generation loops."""
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

from runtime_common import ROOT, Ledger, append, sha
from pinned_core import build_core, install_position_fix


class InvalidGeneration(RuntimeError):
    pass


class Engine:
    def __init__(self, config, trace, output, model=None, tokenizer=None):
        import torch
        self.config, self.trace, self.output = config, trace, Path(output)
        self.active = None
        self.phase = 'mechanical_compatibility'
        if model is None:
            assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one admitted GPU required; no CPU inference fallback'
            # Claim visible memory while hashing existing weights, so another
            # admission poll does not mistake this loading worker for free GPU.
            gpu_claim = torch.empty(256*1024*1024, dtype=torch.float32, device='cuda:0')
            self.trace('loading_gpu_memory_claimed', bytes=gpu_claim.numel()*gpu_claim.element_size())
            from transformers import AutoModelForCausalLM, AutoTokenizer
            lock = json.loads((ROOT/'model_lock.json').read_text())
            self.trace('model_sha_verification_started', revision=lock['revision'])
            for name, expected in lock['files'].items():
                path = Path(lock['path'])/name
                assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
                self.trace('model_file_verified', file=name)
            tokenizer = AutoTokenizer.from_pretrained(lock['path'], local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(lock['path'], local_files_only=True,
                torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
            assert all(p.device.type == 'cuda' and p.dtype == torch.bfloat16 for p in model.parameters())
            self.trace('model_resident', parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                       parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()), all_CUDA=True)
            del gpu_claim
            torch.cuda.empty_cache()
        self.model, self.tokenizer = model.eval(), tokenizer
        install_position_fix(self.model)
        self.core = build_core(self.model, tokenizer, trace)
        self.device = self.core.device
        self.ledger = Ledger(config, trace, self.phase)
        self.pre_handle = self.model.register_forward_pre_hook(self.before, with_kwargs=True)
        self.post_handle = self.model.register_forward_hook(self.after, with_kwargs=True)
        original = self.model.generate
        engine = self
        def generate(this, *args, **kwargs):
            assert engine.active is not None
            kwargs['top_k'] = config['top_k']
            output = original(*args, **kwargs)
            size = kwargs['input_ids'].shape[-1]
            engine.active['generated_ids'] = output.sequences[0, size:].detach().cpu().tolist()
            return output
        self.model.generate = MethodType(generate, self.model)
        self.last_receipt = None

    def before(self, module, args, kwargs):
        a = self.active
        assert a is not None, 'unmetered production forward'
        self.ledger.check()
        if time.monotonic()-a['started'] > self.config['request_timeout_seconds']:
            raise RuntimeError('request timeout')
        ids, embeds = kwargs.get('input_ids'), kwargs.get('inputs_embeds')
        length = int(ids.shape[-1] if ids is not None else embeds.shape[-2])
        past = kwargs.get('past_key_values')
        before = 0 if past is None else int(past.get_seq_length())
        if not a['forward_attempts']:
            a['initial_past_positions'] = before
            if a['expected_past'] is not None:
                assert past is a['expected_past'], 'receiver bypassed decoded cache'
                a['decoded_cache_consumed'] = True
        assert before+length <= self.config['max_total_context_positions'], 'context cap in native forward'
        a['inflight_length'] = length
        a['expected_cache_length'] = before+length
        a['forward_attempts'] += 1

    def after(self, module, args, kwargs, output):
        a = self.active
        assert output.past_key_values.get_seq_length() == a['expected_cache_length'], 'cache-position drift'
        a['positions'] += a['inflight_length']
        a['forwards'] += 1
        now = time.monotonic()
        if now-a['last_progress'] >= 1:
            self.trace('native_forward_progress', reservation_id=a['rid'], completed_forwards=a['forwards'],
                       completed_positions=a['positions'], cache_positions=a['expected_cache_length'])
            a['last_progress'] = now

    def invoke(self, bound, fn, context, raw_input=None, expected_past=None):
        import torch
        assert self.active is None
        self.ledger.phase = self.phase
        rid = self.ledger.begin(bound, context)
        started = time.monotonic()
        self.active = dict(rid=rid, started=started, last_progress=started, positions=0, forwards=0,
                           forward_attempts=0, initial_past_positions=0, expected_past=expected_past,
                           decoded_cache_consumed=False, generated_ids=[])
        append(self.output/'requests_raw.jsonl', dict(event='request_input', reservation_id=rid,
               phase=self.phase, context=context, bound=bound, input=raw_input))
        failed = False
        try:
            with torch.inference_mode():
                result = fn()
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            return result
        except BaseException:
            failed = True
            raise
        finally:
            a = self.active
            receipt = dict(reservation_id=rid, **context, phase=self.phase, elapsed_seconds=time.monotonic()-started,
                           actual_positions=a['positions'], native_forward_calls=a['forwards'],
                           initial_past_positions=a['initial_past_positions'],
                           decoded_cache_consumed=a['decoded_cache_consumed'], generated_ids=a['generated_ids'], failed=failed)
            self.active = None
            self.last_receipt = receipt
            self.ledger.settle(rid, a['positions'], a['forwards'], failed=failed)
            append(self.output/'requests_raw.jsonl', dict(event='request_output', **receipt))

    def prepare(self, messages):
        prompts, ids, mask, tokens = self.core.prepare_chat_batch([messages], add_generation_prompt=True)
        return prompts[0], ids, mask

    def text(self, messages, max_new_tokens, context, past=None):
        prompt, ids, mask = self.prepare(messages)
        start = 0 if past is None else past.get_seq_length()
        if start+ids.shape[-1]+max_new_tokens > self.config['max_total_context_positions']:
            raise InvalidGeneration('context_cap_before_request')
        value = self.invoke(int(ids.shape[-1])+max_new_tokens,
            lambda: self.core.generate_text_batch(ids, mask, max_new_tokens=max_new_tokens,
                temperature=self.config['temperature'], top_p=self.config['top_p'], past_key_values=past),
            context, dict(prompt=prompt, input_ids=ids[0].detach().cpu().tolist(),
                          past_positions=start, max_new_tokens=max_new_tokens), expected_past=past)
        text = value[0][0]
        generated = self.last_receipt['generated_ids']
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        truncated = len(generated) >= max_new_tokens and (not generated or generated[-1] not in eos)
        append(self.output/'agent_outputs_raw.jsonl', dict(**context, phase=self.phase, text=text,
               full_decode=self.tokenizer.decode(generated, skip_special_tokens=False), truncated=truncated,
               request=self.last_receipt))
        return text, truncated, dict(self.last_receipt), prompt

    def latent(self, messages, steps, context, past=None):
        prompt, ids, mask = self.prepare(messages)
        start = 0 if past is None else past.get_seq_length()
        if start+ids.shape[-1]+steps > self.config['max_total_context_positions']:
            raise InvalidGeneration('context_cap_before_request')
        value = self.invoke(int(ids.shape[-1])+steps,
            lambda: self.core.generate_latent_batch(ids, mask, latent_steps=steps, past_key_values=past),
            context, dict(prompt=prompt, input_ids=ids[0].detach().cpu().tolist(),
                          past_positions=start, latent_steps=steps), expected_past=past)
        assert value.get_seq_length() == start+ids.shape[-1]+steps
        assert self.last_receipt['native_forward_calls'] == 1+steps
        return value, dict(self.last_receipt), prompt
