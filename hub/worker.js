// Background Web Worker for Pyodide Clinical Trial Simulations

// Load the Pyodide runtime from CDN
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js");

let pyodideInstance = null;
let isInitializing = false;

async function initPyodide() {
  if (pyodideInstance) return pyodideInstance;
  if (isInitializing) {
    while (isInitializing) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return pyodideInstance;
  }

  isInitializing = true;
  self.postMessage({ type: "status", message: "Initializing WebAssembly environment..." });

  try {
    const pyodide = await loadPyodide({
      stdout: (text) => console.log("Pyodide stdout:", text),
      stderr: (text) => console.error("Pyodide stderr:", text)
    });

    self.postMessage({ type: "status", message: "Loading package manager..." });
    await pyodide.loadPackage("micropip");

    self.postMessage({ type: "status", message: "Loading scientific dependencies (numpy, scipy, pandas, statsmodels)..." });
    await pyodide.loadPackage(["numpy", "scipy", "pandas", "statsmodels"]);

    // Resolve absolute URLs relative to worker location
    const baseUrl = self.location.href.substring(0, self.location.href.lastIndexOf('/') + 1);
    const wheelUrl = baseUrl + "clintrials-0.1.4-py3-none-any.whl";
    const runnerUrl = baseUrl + "runner.py";

    self.postMessage({ type: "status", message: "Installing clintrials scientific wheel..." });
    const micropip = pyodide.pyimport("micropip");
    await micropip.install(wheelUrl);

    self.postMessage({ type: "status", message: "Configuring runner environment..." });
    const runnerRes = await fetch(runnerUrl);
    if (!runnerRes.ok) {
      throw new Error(`Failed to fetch runner.py: ${runnerRes.statusText}`);
    }
    const runnerCode = await runnerRes.text();
    pyodide.FS.writeFile("runner.py", runnerCode, { encoding: "utf8" });

    pyodideInstance = pyodide;
    self.postMessage({ type: "status", message: "Ready to run simulations." });
  } catch (err) {
    self.postMessage({ type: "error", error: "Initialization failed: " + err.message });
    isInitializing = false;
    throw err;
  } finally {
    isInitializing = false;
  }
  return pyodideInstance;
}

self.onmessage = async function(e) {
  const { schemaName, payload } = e.data;

  try {
    const pyodide = await initPyodide();
    self.postMessage({ type: "status", message: "Executing simulation..." });

    // Expose progress callback to Python
    self.progressCallback = (percent) => {
      self.postMessage({ type: "progress", percent: percent });
    };

    const payloadStr = JSON.stringify(payload);

    // Run the runner function
    const resultJson = await pyodide.runPythonAsync(`
import js
from runner import run_simulation_py
run_simulation_py("${schemaName}", ${JSON.stringify(payloadStr)}, js.self.progressCallback)
    `);

    const result = JSON.parse(resultJson);
    self.postMessage({ type: "success", data: result });
  } catch (err) {
    self.postMessage({ type: "error", error: "Execution failed: " + err.message });
  }
};
