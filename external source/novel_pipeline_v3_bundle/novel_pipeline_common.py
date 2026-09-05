#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

SUPPORTED_EXTENSIONS = {'.txt', '.md', '.markdown', '.pdf'}
THINKING_ENABLED = False
CHAT_BACKEND = 'auto'
QWEN35_MAX_OUTPUT_TOKENS = 6000
QWEN35_LENGTH_RETRIES = 2
_RUN_PROGRESS: '_RunProgress | None' = None


def alpha_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)


def slug(text: str, max_len: int = 96) -> str:
    value = re.sub(r'[^\w.-]+', '_', text.strip(), flags=re.UNICODE).strip('._')
    return (value[:max_len] or 'chapter').rstrip('._')


def discover_inputs(items: list[Path], extensions: set[str] | None = None) -> list[Path]:
    extensions = extensions or SUPPORTED_EXTENSIONS
    found: list[Path] = []
    for raw in items:
        text = str(raw)
        if '*' in text or '?' in text:
            matches = [Path(x) for x in glob.glob(text)]
            if not matches:
                print(f'WARNING: wildcard matched nothing: {text}', file=sys.stderr)
            for item in matches:
                if item.is_file() and item.suffix.lower() in extensions:
                    found.append(item)
                elif item.is_dir():
                    found.extend(p for p in item.iterdir() if p.is_file() and p.suffix.lower() in extensions)
            continue
        item = Path(text)
        if item.is_file() and item.suffix.lower() in extensions:
            found.append(item)
        elif item.is_dir():
            found.extend(p for p in item.iterdir() if p.is_file() and p.suffix.lower() in extensions)
        else:
            print(f'WARNING: ignoring unsupported/missing input: {item}', file=sys.stderr)
    unique = {str(p.resolve()): p for p in found}
    return sorted(unique.values(), key=lambda p: alpha_key(p.name))


def read_text_document(path: Path) -> str:
    if path.suffix.lower() in {'.txt', '.md', '.markdown'}:
        text = path.read_text(encoding='utf-8-sig', errors='replace')
    elif path.suffix.lower() == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError('PDF input requires: pip install pypdf') from exc
        text = '\n\n'.join((page.extract_text() or '') for page in PdfReader(str(path)).pages)
    else:
        raise ValueError(f'Unsupported file type: {path.suffix}')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text).strip()
    if len(text) < 100:
        raise ValueError('Input is empty or too short after extraction.')
    return text


def split_chunks(text: str, max_chars: int, overlap_paragraphs: int = 2) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return [text]
    out: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        add = len(para) + (2 if current else 0)
        if current and current_len + add > max_chars:
            out.append('\n\n'.join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(x) for x in current) + max(0, len(current) - 1) * 2
        current.append(para)
        current_len += add
    if current:
        out.append('\n\n'.join(current))
    return out


def dedupe(values: Iterable[str], max_items: int = 1000) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = re.sub(r'\s+', ' ', str(raw)).strip()
        key = value.casefold()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def norm_name(value: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]', ' ', value.casefold(), flags=re.UNICODE).split())


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip('/'), api_key=api_key, timeout=300.0, max_retries=2)


def select_model(client: OpenAI, requested: str | None) -> str:
    if requested:
        return requested
    models = list(client.models.list().data)
    if not models:
        raise RuntimeError('LM Studio exposes no models. Load one first.')
    return next((m.id for m in models if 'qwen' in m.id.casefold()), models[0].id)


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I).strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _is_qwen35_model(model: str) -> bool:
    normalized = model.casefold().replace('_', '').replace('-', '').replace('.', '')
    return 'qwen35' in normalized


def _use_qwen35_chatml(model: str) -> bool:
    if CHAT_BACKEND == 'qwen35-chatml':
        return True
    if CHAT_BACKEND == 'openai-chat':
        return False
    return _is_qwen35_model(model)


def configure_llm(*, thinking: bool, chat_backend: str, qwen35_max_output_tokens: int, qwen35_length_retries: int) -> None:
    global THINKING_ENABLED, CHAT_BACKEND, QWEN35_MAX_OUTPUT_TOKENS, QWEN35_LENGTH_RETRIES
    THINKING_ENABLED = bool(thinking)
    CHAT_BACKEND = chat_backend
    QWEN35_MAX_OUTPUT_TOKENS = max(256, int(qwen35_max_output_tokens))
    QWEN35_LENGTH_RETRIES = max(0, int(qwen35_length_retries))


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds > 365 * 86400:
        return 'estimating...'
    seconds = int(round(seconds))
    if seconds < 60:
        return f'~{seconds:02d}s'
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f'~{minutes:02d}:{sec:02d}'
    hours, minutes = divmod(minutes, 60)
    return f'~{hours:d}:{minutes:02d}:{sec:02d}'


class _RunProgress:
    def __init__(self, total_items: int) -> None:
        self.total_items = max(1, int(total_items))
        self.run_started = time.perf_counter()
        self.input_started = self.run_started
        self.current_index = 1
        self.current_progress = 0.0
        self.op_base = 0.0
        self.op_span = 0.0
        self.op_peak = 0.0
        self.call_samples: list[tuple[float, int]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _clamp(v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    def start_item(self, index: int) -> None:
        with self._lock:
            self.current_index = min(self.total_items, max(1, int(index)))
            self.input_started = time.perf_counter()
            self.current_progress = self.op_base = self.op_span = self.op_peak = 0.0

    def start_operation(self, base: float, span: float) -> None:
        with self._lock:
            self.op_base = self._clamp(base)
            self.op_span = max(0.0, min(1.0 - self.op_base, float(span)))
            self.current_progress = max(self.current_progress, self.op_base)
            self.op_peak = 0.0

    def advance(self, progress: float) -> None:
        with self._lock:
            self.current_progress = max(self.current_progress, self._clamp(progress))
            self.op_base = self.current_progress
            self.op_span = 0.0
            self.op_peak = 0.0

    def finish_operation(self) -> None:
        self.advance(self.op_base + self.op_span)

    def record_call(self, elapsed: float, token_events: int) -> None:
        if elapsed <= 0:
            return
        with self._lock:
            self.call_samples.append((elapsed, max(0, token_events)))
            self.call_samples[:] = self.call_samples[-24:]

    @staticmethod
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    def _active_fraction(self, token_events: int, elapsed: float) -> float:
        durations = [d for d, _ in self.call_samples if d > 0]
        counts = [float(t) for _, t in self.call_samples if t > 0]
        d = self._median(durations)
        t = self._median(counts)
        if d is None and t is None:
            return 0.0
        tf = elapsed / d if d else 0.0
        kf = token_events / t if t else 0.0
        if token_events <= 0:
            est = 0.25 * tf
        elif d and t:
            est = 0.45 * tf + 0.55 * kf
        else:
            est = kf or tf
        return min(0.97, max(0.0, est))

    def snapshot(self, token_events: int = 0, active_elapsed: float = 0.0) -> tuple[float, float | None, float, float | None]:
        now = time.perf_counter()
        with self._lock:
            self.op_peak = max(self.op_peak, self._active_fraction(token_events, active_elapsed))
            current = max(self.current_progress, self._clamp(self.op_base + self.op_span * self.op_peak))
            total = self._clamp(((self.current_index - 1) + current) / self.total_items)
            ie = max(0.0, now - self.input_started)
            re = max(0.0, now - self.run_started)
        input_eta = ie * (1 - current) / current if current >= 0.01 else None
        total_eta = re * (1 - total) / total if total >= 0.005 else None
        return current, input_eta, total, total_eta


def init_progress(total_items: int) -> None:
    global _RUN_PROGRESS
    _RUN_PROGRESS = _RunProgress(total_items)


def progress_start_item(index: int) -> None:
    if _RUN_PROGRESS:
        _RUN_PROGRESS.start_item(index)


def progress_start_operation(base: float, span: float) -> None:
    if _RUN_PROGRESS:
        _RUN_PROGRESS.start_operation(base, span)


def progress_advance(progress: float) -> None:
    if _RUN_PROGRESS:
        _RUN_PROGRESS.advance(progress)


def progress_finish_operation() -> None:
    if _RUN_PROGRESS:
        _RUN_PROGRESS.finish_operation()


def _enable_windows_ansi() -> bool:
    if sys.platform != 'win32' or not sys.stdout.isatty():
        return sys.stdout.isatty()
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if handle in (0, -1) or not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class _LiveTokenRate:
    GREEN='\x1b[92m'; CYAN='\x1b[96m'; YELLOW='\x1b[93m'; DIM='\x1b[2m'; RESET='\x1b[0m'; CLEAR='\x1b[2K'
    def __init__(self, refresh_interval: float = 0.25) -> None:
        self.started = time.perf_counter(); self.first_token_at: float | None = None; self.token_events = 0
        self.refresh_interval = refresh_interval; self.is_tty = sys.stdout.isatty(); self.use_ansi = _enable_windows_ansi() if self.is_tty else False
        self.lock = threading.Lock(); self.stop = threading.Event(); self.thread = None
        if self.is_tty:
            self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
    def _loop(self) -> None:
        while not self.stop.wait(self.refresh_interval): self._render()
    def update(self, piece: str) -> None:
        if not piece: return
        with self.lock:
            if self.first_token_at is None: self.first_token_at = time.perf_counter()
            self.token_events += 1
    def _render(self, now: float | None = None) -> None:
        if not self.is_tty: return
        now = now or time.perf_counter()
        with self.lock: tokens, first = self.token_events, self.first_token_at
        elapsed = max(now - self.started, 1e-9); gen = max(now - first, .10) if first else 0.0; rate = tokens / gen if gen else 0.0
        extra = ''
        if _RUN_PROGRESS:
            cp, ce, tp, te = _RUN_PROGRESS.snapshot(tokens, elapsed)
            if self.use_ansi:
                extra = f' | Current: {self.CYAN}{cp*100:5.1f}%{self.RESET} ETA {format_eta(ce)} | Total: {self.YELLOW}{tp*100:5.1f}%{self.RESET} ETA {format_eta(te)}'
                msg = f'    LLM: {self.GREEN}{rate:6.1f} tok/s{self.RESET} | {self.DIM}{tokens:5d} tok{self.RESET}{extra}'
                sys.stdout.write('\r'+self.CLEAR+msg)
            else:
                extra = f' | Current: {cp*100:5.1f}% ETA {format_eta(ce)} | Total: {tp*100:5.1f}% ETA {format_eta(te)}'
                sys.stdout.write('\r'+f'    LLM: {rate:6.1f} tok/s | {tokens:5d} tok{extra}')
        else:
            sys.stdout.write('\r'+f'    LLM: {rate:6.1f} tok/s | {tokens:5d} tok')
        sys.stdout.flush()
    def finish(self) -> tuple[int,float,float]:
        now=time.perf_counter(); self.stop.set()
        if self.thread: self.thread.join(timeout=1.0)
        if self.is_tty:
            self._render(now); sys.stdout.write(self.RESET+'\n' if self.use_ansi else '\n'); sys.stdout.flush()
        with self.lock: tokens, first = self.token_events, self.first_token_at
        elapsed=max(now-self.started,1e-9); gen=max(now-first,.10) if first else elapsed; rate=tokens/gen
        if _RUN_PROGRESS: _RUN_PROGRESS.record_call(elapsed,tokens)
        return tokens,elapsed,rate


def _complete_json_prefix(text: str) -> str | None:
    start=text.find('{')
    if start<0: return None
    depth=0; in_string=False; escaped=False
    for i,ch in enumerate(text[start:],start):
        if in_string:
            if escaped: escaped=False
            elif ch=='\\': escaped=True
            elif ch=='"': in_string=False
            continue
        if ch=='"': in_string=True
        elif ch=='{': depth+=1
        elif ch=='}':
            depth-=1
            if depth==0: return text[start:i+1]
    return None


def _qwen35_prompt(system: str, user: str, schema: dict[str,Any]) -> str:
    assistant_prefix = '<think>\n' if THINKING_ENABLED else '<think>\n\n</think>\n\n'
    return ('<|im_start|>system\n'+system+'\n\nReturn ONLY valid JSON. No Markdown.<|im_end|>\n'
            '<|im_start|>user\n'+user+'\n\nRequired JSON schema:\n'+json.dumps(schema['schema'],ensure_ascii=False)+'<|im_end|>\n'
            '<|im_start|>assistant\n'+assistant_prefix)


def _manual_stream(client: OpenAI, model: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> tuple[str,float,int,str]:
    meter=_LiveTokenRate(); chunks=[]; complete=None; finish='json_complete'
    stream=client.completions.create(model=model,prompt=prompt,temperature=temperature,top_p=top_p,max_tokens=max_tokens,stop=['<|im_end|>','<END_JSON>'],stream=True)
    try:
        for event in stream:
            if not event.choices: continue
            c=event.choices[0]; piece=c.text or ''
            if piece:
                chunks.append(piece); meter.update(piece); complete=_complete_json_prefix(''.join(chunks))
                if complete is not None: break
            if getattr(c,'finish_reason',None): finish=str(c.finish_reason)
    finally:
        close=getattr(stream,'close',None)
        if callable(close):
            try: close()
            except Exception: pass
    tok,elapsed,_=meter.finish(); return complete if complete is not None else ''.join(chunks),elapsed,tok,finish


def _chat_stream(client: OpenAI, **kwargs: Any) -> tuple[str,float,int,str,bool]:
    meter=_LiveTokenRate(); out=[]; finish='unknown'; reasoning=False
    stream=client.chat.completions.create(stream=True,**kwargs)
    try:
        for event in stream:
            if not event.choices: continue
            c=event.choices[0]; d=c.delta
            content=getattr(d,'content',None) or ''; r=getattr(d,'reasoning_content',None) or ''
            if content: out.append(content)
            if r: reasoning=True
            meter.update(content or r)
            if getattr(c,'finish_reason',None): finish=str(c.finish_reason)
    finally:
        close=getattr(stream,'close',None)
        if callable(close):
            try: close()
            except Exception: pass
    tok,elapsed,_=meter.finish(); return ''.join(out),elapsed,tok,finish,reasoning


def chat_json(client: OpenAI, model: str, system: str, user: str, schema: dict[str,Any], temperature: float, max_tokens: int) -> dict[str,Any]:
    if _use_qwen35_chatml(model):
        cap=min(max_tokens,QWEN35_MAX_OUTPUT_TOKENS); last_error=None; last_raw=''
        for attempt in range(QWEN35_LENGTH_RETRIES+1):
            note='' if attempt==0 else '\n\nCRITICAL RETRY: previous JSON was invalid/truncated. Return a smaller complete JSON object.'
            try:
                raw,elapsed,tokens,finish=_manual_stream(client,model,_qwen35_prompt(system,user+note,schema),min(temperature,.15) if attempt else temperature,.8 if attempt else .9,cap)
                last_raw=raw
                print(f'    LLM done: {tokens/max(elapsed,1e-9):.1f} stream-events/s, {elapsed:.1f}s, stop={finish}, attempt={attempt+1}')
                return parse_json(raw)
            except Exception as exc:
                last_error=exc
        raise RuntimeError(f'Qwen3.5 JSON generation failed: {last_error}\nTail:\n{last_raw[-1200:]}')

    directive='/think' if THINKING_ENABLED else '/no_think'
    messages=[{'role':'system','content':directive+'\n\n'+system},{'role':'user','content':user}]
    extra={'enableThinking':bool(THINKING_ENABLED),'chat_template_kwargs':{'enable_thinking':bool(THINKING_ENABLED)}}
    first_error=None
    for structured in (True,False):
        try:
            kwargs=dict(model=model,messages=messages,temperature=temperature,top_p=.9,max_tokens=max_tokens,extra_body=extra)
            if structured: kwargs['response_format']={'type':'json_schema','json_schema':schema}
            raw,elapsed,tokens,finish,reasoning=_chat_stream(client,**kwargs)
            if not raw.strip() and reasoning:
                raise RuntimeError('Only reasoning_content was returned; try the model-specific backend or disable thinking.')
            print(f'    LLM done: {tokens/max(elapsed,1e-9):.1f} stream-events/s, {elapsed:.1f}s, stop={finish}')
            return parse_json(raw)
        except Exception as exc:
            if first_error is None: first_error=exc
            messages=[{'role':'system','content':directive+'\n\n'+system+'\nReturn ONLY valid JSON with no Markdown.'},{'role':'user','content':user+'\n\nRequired JSON schema:\n'+json.dumps(schema['schema'],ensure_ascii=False)}]
    raise RuntimeError(f'JSON generation failed. Structured error: {first_error}')


def add_common_llm_args(p: argparse.ArgumentParser, *, max_tokens: int = 6000) -> None:
    p.add_argument('--base-url',default='http://127.0.0.1:1234/v1')
    p.add_argument('--api-key',default='lm-studio')
    p.add_argument('--model',default=None)
    g=p.add_mutually_exclusive_group(); g.add_argument('--thinking',dest='thinking',action='store_true'); g.add_argument('--no-thinking',dest='thinking',action='store_false'); p.set_defaults(thinking=False)
    p.add_argument('--chat-backend',choices=['auto','openai-chat','qwen35-chatml'],default='auto')
    p.add_argument('--temperature',type=float,default=.16)
    p.add_argument('--max-tokens',type=int,default=max_tokens)
    p.add_argument('--qwen35-max-output-tokens','--max-output-tokens',dest='qwen35_max_output_tokens',type=int,default=max_tokens)
    p.add_argument('--qwen35-length-retries',type=int,default=2)
    p.add_argument('--delay',type=float,default=0.0)
    p.add_argument('--force',action='store_true')
