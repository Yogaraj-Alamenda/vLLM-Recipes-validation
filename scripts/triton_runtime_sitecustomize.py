"""Best-effort per-process Triton dispatch tracing for Xeon recipe inspection.

Records lower-level CompiledKernel.run launches when observable, and JIT
invocations separately. The report counts lower-level launches first and uses
JIT-only events only when no lower-level interception was possible.
"""
import functools
import json
import os
import sys
import threading
import time
import uuid

_OUTPUT = os.environ.get('TRITON_RUNTIME_TRACE_OUTPUT', '')
_ENABLED = os.environ.get('TRITON_RUNTIME_TRACE') == '1' and bool(_OUTPUT)
_LOCAL = threading.local()


def _emit(data):
    try:
        data.update(pid=os.getpid(), timestamp_ns=time.time_ns())
        line = (json.dumps(data, default=str, separators=(',', ':')) + '\n').encode()
        fd = os.open(_OUTPUT, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:
        pass  # Diagnostic tracing must not break model serving.


def _grid(v):
    if callable(v):
        return '<callable-grid>'
    if isinstance(v, (tuple, list)):
        return list(v)
    return v


def _name(obj):
    return getattr(obj, '__name__', getattr(obj, 'name', type(obj).__name__))


def _install():
    import triton
    from triton.runtime.jit import JITFunction

    # CompiledKernel often installs the actual launcher on each instance.
    # A class-level replacement of CompiledKernel.run will miss this path.
    try:
        from triton.compiler.compiler import CompiledKernel
        original_getattribute = CompiledKernel.__getattribute__
        if not getattr(original_getattribute, '_xeon_trace', False):
            def traced_getattribute(self, attr):
                value = original_getattribute(self, attr)
                if attr != 'run' or not callable(value):
                    return value
                @functools.wraps(value)
                def traced_compiled_launch(*args, **kwargs):
                    context = getattr(_LOCAL, 'context', None)
                    result = value(*args, **kwargs)
                    if not kwargs.get('warmup', False):
                        if context is not None:
                            context['lower_seen'] = True
                        _emit({
                            'event': 'triton_compiled_launch',
                            'dispatch_id': context['id'] if context else None,
                            'jit_name': context['name'] if context else None,
                            'compiled_name': getattr(self, 'name', 'unknown'),
                            'kernel_hash': getattr(self, 'hash', None),
                            'grid': context.get('grid') if context else (list(args[:3]) if len(args) >= 3 and all(isinstance(x, int) for x in args[:3]) else None),
                            'entry': 'CompiledKernel.run',
                            'caller': context.get('caller') if context else None,
                            'backend': str(getattr(getattr(self, 'metadata', None), 'target', 'unknown')),
                        })
                    return result
                return traced_compiled_launch
            traced_getattribute._xeon_trace = True
            CompiledKernel.__getattribute__ = traced_getattribute
            _emit({'event': 'trace_hook', 'hook': 'CompiledKernel.run (instance launcher)'})
    except Exception as exc:
        _emit({'event': 'trace_error', 'hook': 'CompiledKernel.run', 'reason': repr(exc)})

    original_run = JITFunction.run
    if not getattr(original_run, '_xeon_trace', False):
        @functools.wraps(original_run)
        def traced_jit_run(self, *args, **kwargs):
            parent = getattr(_LOCAL, 'context', None)
            context = {
                'id': uuid.uuid4().hex,
                'name': _name(self),
                'caller': f'{sys._getframe(1).f_code.co_filename}:{sys._getframe(1).f_lineno}',
                'grid': _grid(kwargs.get('grid')),
                'lower_seen': False,
            }
            _LOCAL.context = context
            try:
                kernel = original_run(self, *args, **kwargs)
                if not kwargs.get('warmup', False) and kernel is not None:
                    _emit({
                        'event': 'triton_jit_dispatch',
                        'dispatch_id': context['id'],
                        'jit_name': context['name'],
                        'compiled_name': getattr(kernel, 'name', 'unknown'),
                        'kernel_hash': getattr(kernel, 'hash', None),
                        'grid': context['grid'],
                        'caller': context['caller'],
                        'lower_seen': context['lower_seen'],
                    })
                return kernel
            finally:
                _LOCAL.context = parent
        traced_jit_run._xeon_trace = True
        JITFunction.run = traced_jit_run
        _emit({'event': 'trace_hook', 'hook': 'JITFunction.run'})
    _emit({'event': 'trace_ready', 'triton_version': getattr(triton, '__version__', 'unknown')})


if _ENABLED:
    try:
        _install()
    except ImportError as exc:
        _emit({'event': 'trace_unavailable', 'reason': repr(exc)})
    except Exception as exc:
        _emit({'event': 'trace_error', 'reason': repr(exc)})
