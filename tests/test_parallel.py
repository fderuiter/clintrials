import pytest
from unittest.mock import MagicMock, patch
import sys
import time

from clintrials.core.parallel import WorkerPool, _run_batch
from clintrials.core.protocol import Protocol

def dummy_func(x):
    return x * 2

class DummyClass:
    def __init__(self, x):
        self.x = x
    def run(self, b_size, method, seed):
        outer_x = self.x
        class Res:
            def to_list(self):
                return [outer_x] * b_size
        return Res()

def test_run_batch_func():
    payload = {
        "module": "tests.test_parallel",
        "func": "dummy_func",
        "kwargs": {"x": 5}
    }
    res = _run_batch(payload, 3, 0)
    assert res == [10, 10, 10]

def test_run_batch_class():
    payload = {
        "module": "tests.test_parallel",
        "class": "DummyClass",
        "kwargs": {"x": 42}
    }
    res = _run_batch(payload, 2, 0)
    assert res == [42, 42]

def test_run_batch_invalid():
    payload = {
        "module": "tests.test_parallel"
    }
    with pytest.raises(ValueError):
        _run_batch(payload, 2, 0)

def test_worker_pool_native():
    pool = WorkerPool(pool_size=2)
    assert not pool.is_pyodide
    
    payload = {
        "module": "tests.test_parallel",
        "func": "dummy_func",
        "kwargs": {"x": 7}
    }
    
    progress_called = []
    def on_progress(p):
        progress_called.append(p)
        
    res = pool.execute(payload, total_sims=5, batch_size=2, on_progress=on_progress)
    assert len(res) == 5
    assert all(x == 14 for x in res)
    assert len(progress_called) > 0
    
    pool._cancelled = True
    res2 = pool._execute_native(payload, total_sims=5, batch_size=2)
    pool.cancel()
    assert pool._cancelled

def test_worker_pool_pyodide(monkeypatch):
    mock_pyodide = MagicMock()
    mock_pyodide.ffi.to_js.side_effect = lambda x: x
    mock_js = MagicMock()
    monkeypatch.setitem(sys.modules, "pyodide", mock_pyodide)
    monkeypatch.setitem(sys.modules, "js", mock_js)
    
    pool = WorkerPool(pool_size=1)
    
    on_message_cb = mock_pyodide.ffi.create_proxy.call_args[0][0]
    
    event1 = MagicMock()
    event1.data.to_py.return_value = {"type": "init_done", "worker_id": 0}
    on_message_cb(event1)
    
    event2 = MagicMock()
    import json
    event2.data.to_py.return_value = {"type": "result", "task_id": "test_t1", "result": json.dumps({"status": "success", "data": [99, 99]}), "worker_id": 0}
    on_message_cb(event2)
    
    event3 = MagicMock()
    event3.data.to_py.return_value = {"type": "error", "task_id": "test_t2", "error": "test error", "worker_id": 0}
    on_message_cb(event3)
    
    pool._results.clear()
    
    payload = {
        "module": "tests.test_parallel",
        "func": "dummy_func",
        "kwargs": {"x": 7}
    }
    
    def mock_postMessage(msg):
        task_id = msg["task_id"]
        pool._results[task_id] = ({"status": "success", "data": [14, 14]}, 0)
        
    pool.workers[0]["worker"].postMessage = mock_postMessage
    pool.workers[0]["ready"] = False
    
    import threading
    def set_ready():
        time.sleep(0.15)
        pool.workers[0]["ready"] = True
    t = threading.Thread(target=set_ready)
    t.start()
    
    progress = []
    res = pool.execute(payload, total_sims=3, batch_size=2, on_progress=lambda p: progress.append(p))
    t.join()
    
    def mock_postMessage_error(msg):
        task_id = msg["task_id"]
        pool._results[task_id] = ({"status": "error", "error": "Mocked exception"}, 0)
    pool.workers[0]["worker"].postMessage = mock_postMessage_error
    res_err = pool.execute(payload, total_sims=1, batch_size=1)
    
    pool._cancelled = True
    res_canc = pool.execute(payload, total_sims=1, batch_size=1)
    
    pool.cancel()

def test_protocol_rng():
    class DummyProtocol(Protocol):
        def generate_data(self): pass
        def analyze(self): pass
        def next_action(self): pass
        def get_data(self): pass
        def update_data(self): pass
        def has_more(self): pass
        def report(self): pass
        def reset(self): pass
        def update(self): pass
    p = DummyProtocol()
    assert p.rng is not None
    p.set_rng(None)
    assert p.rng is not None

from clintrials.core.viz_interface import get_visualization_provider, set_visualization_provider
def test_viz_interface_import_error(monkeypatch):
    import clintrials.core.viz_interface as vi
    vi._provider = None
    
    # Mock to raise ImportError
    def mock_get_default():
        raise ImportError()
        
    monkeypatch.setattr("clintrials.visualization.provider.get_default_provider", mock_get_default, raising=False)
    # Actually wait, we just need to patch sys.modules to prevent loading it?
    # Better to patch importlib or something.
    
    monkeypatch.setitem(sys.modules, "clintrials.visualization.provider", None)
    
    with pytest.raises(ImportError):
        get_visualization_provider()
        
    set_visualization_provider("test")
    assert get_visualization_provider() == "test"
    vi._provider = None
