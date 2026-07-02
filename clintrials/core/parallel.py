import logging
import json
import uuid
import sys
import time

logger = logging.getLogger(__name__)

def _run_batch(payload, b_size, b_idx):
    import importlib
    mod = importlib.import_module(payload["module"])
    
    if "class" in payload:
        cls = getattr(mod, payload["class"])
        trial = cls(**payload["kwargs"])
        res = trial.run(b_size, method="iterative", seed=42 + b_idx)
        return res.to_list()
    elif "func" in payload:
        func = getattr(mod, payload["func"])
        import numpy as np
        np.random.seed((42 + b_idx) % (2**32))
        return [func(**payload["kwargs"]) for _ in range(b_size)]
    else:
        raise ValueError("Payload must contain 'class' or 'func'")

class WorkerPool:
    def __init__(self, pool_size=4):
        self.pool_size = pool_size
        self.is_pyodide = "pyodide" in sys.modules
        
        self.workers = []
        self._cancelled = False
        self._results = {}

        if self.is_pyodide:
            self._init_pyodide_workers()
        else:
            self._init_native_workers()

    def _init_native_workers(self):
        import concurrent.futures
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.pool_size)

    def _init_pyodide_workers(self):
        import pyodide
        import js
        from js import Blob, Worker, URL

        worker_code = """
        importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

        let pyodide_ready = false;

        self.onmessage = async function(e) {
            if (e.data.type === 'init') {
                try {
                    self.pyodide = await loadPyodide();
                    await self.pyodide.loadPackage(['numpy', 'scipy', 'pandas']);
                    pyodide_ready = true;
                    self.postMessage({type: 'init_done', worker_id: e.data.worker_id});
                } catch (err) {
                    self.postMessage({type: 'error', error: err.toString(), worker_id: e.data.worker_id});
                }
            } else if (e.data.type === 'run') {
                if (!pyodide_ready) {
                    self.postMessage({type: 'error', task_id: e.data.task_id, error: "Pyodide not ready"});
                    return;
                }
                try {
                    self.pyodide.globals.set('worker_seed', e.data.seed);
                    self.pyodide.globals.set('worker_batch_size', e.data.batch_size);
                    self.pyodide.globals.set('worker_payload', e.data.payload);
                    
                    let result = self.pyodide.runPython(`
import json
import importlib
try:
    payload = json.loads(worker_payload)
    mod = importlib.import_module(payload["module"])
    
    if "class" in payload:
        cls = getattr(mod, payload["class"])
        trial = cls(**payload["kwargs"])
        res = trial.run(worker_batch_size, method="iterative", seed=worker_seed)
        output = res.to_list()
    elif "func" in payload:
        func = getattr(mod, payload["func"])
        # ensure reproducible random if needed
        import numpy as np
        np.random.seed(worker_seed % (2**32))
        
        output = [func(**payload["kwargs"]) for _ in range(worker_batch_size)]
    else:
        raise ValueError("Payload must contain 'class' or 'func'")
        
    json.dumps({'status': 'success', 'data': output})
except Exception as ex:
    import traceback
    json.dumps({'status': 'error', 'error': str(ex), 'traceback': traceback.format_exc()})
                    `);
                    self.postMessage({type: 'result', task_id: e.data.task_id, result: result, worker_id: e.data.worker_id});
                } catch (err) {
                    self.postMessage({type: 'error', task_id: e.data.task_id, error: err.toString(), worker_id: e.data.worker_id});
                }
            }
        };
        """
        for i in range(self.pool_size):
            blob = Blob.new([worker_code], {"type": "application/javascript"})
            url = URL.createObjectURL(blob)
            worker = Worker.new(url)
            
            def on_message(event, w_id=i):
                data = event.data.to_py()
                msg_type = data.get("type")
                if msg_type == "init_done":
                    self.workers[w_id]["ready"] = True
                elif msg_type == "result":
                    task_id = data.get("task_id")
                    res = json.loads(data.get("result"))
                    self._results[task_id] = (res, w_id)
                elif msg_type == "error":
                    task_id = data.get("task_id")
                    self._results[task_id] = ({"status": "error", "error": data.get("error")}, w_id)

            worker.onmessage = pyodide.ffi.create_proxy(on_message)
            self.workers.append({"worker": worker, "busy": False, "ready": False})
            msg = pyodide.ffi.to_js({"type": "init", "worker_id": i})
            worker.postMessage(msg)

    def execute(self, payload, total_sims, batch_size=100, on_progress=None):
        if not self.is_pyodide:
            return self._execute_native(payload, total_sims, batch_size, on_progress)
            
        import pyodide
        
        while not all(w["ready"] for w in self.workers):
            time.sleep(0.1)
            
        batches = []
        rem = total_sims
        while rem > 0:
            sz = min(batch_size, rem)
            batches.append(sz)
            rem -= sz
            
        task_queue = list(enumerate(batches))
        active_tasks = {}
        results = []
        
        while task_queue or active_tasks:
            if self._cancelled:
                break
                
            for w_id, w in enumerate(self.workers):
                if not w["busy"] and task_queue:
                    task_idx, b_size = task_queue.pop(0)
                    task_id = str(uuid.uuid4())
                    w["busy"] = True
                    
                    active_tasks[task_id] = w_id
                    
                    msg = {
                        "type": "run",
                        "worker_id": w_id,
                        "task_id": task_id,
                        "payload": json.dumps(payload),
                        "seed": 42 + task_idx,
                        "batch_size": b_size
                    }
                    w["worker"].postMessage(pyodide.ffi.to_js(msg))
                    
            if active_tasks:
                time.sleep(0.1)
                
                done_tasks = list(self._results.keys())
                for t_id in done_tasks:
                    res_data, w_id = self._results.pop(t_id)
                    del active_tasks[t_id]
                    self.workers[w_id]["busy"] = False
                    
                    if res_data.get("status") == "success":
                        data = res_data.get("data", [])
                        results.extend(data)
                        if on_progress:
                            on_progress(len(results) / total_sims)
                    else:
                        logger.error(f"Worker error: {res_data.get('error')}")
                        
        return results

    def _execute_native(self, payload, total_sims, batch_size=100, on_progress=None):
        batches = []
        rem = total_sims
        while rem > 0:
            sz = min(batch_size, rem)
            batches.append(sz)
            rem -= sz
            
        results = []
        futures = []
        for i, b_size in enumerate(batches):
            futures.append(self.executor.submit(_run_batch, payload, b_size, i))
            
        import concurrent.futures
        completed_count = 0
        for f in concurrent.futures.as_completed(futures):
            if self._cancelled:
                break
            res = f.result()
            results.extend(res)
            completed_count += len(res)
            if on_progress:
                on_progress(completed_count / total_sims)
        return results

    def cancel(self):
        self._cancelled = True
        if self.is_pyodide:
            for w in self.workers:
                w["worker"].terminate()
        else:
            self.executor.shutdown(wait=False, cancel_futures=True)

