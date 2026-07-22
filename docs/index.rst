clintrials Documentation
========================

Welcome to **clintrials**, a Python library providing clinical trial designs and utilities.

.. raw:: html

   <div id="homepage-sim-placeholder" style="width: 100%; height: 800px; border: 1px solid #e1e4e8; border-radius: 8px; background: #f8f9fa; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 24px;">
       <!-- Mock window header -->
       <div style="position: absolute; top: 0; left: 0; right: 0; height: 40px; background: #e9ecef; border-bottom: 1px solid #e1e4e8; display: flex; align-items: center; padding: 0 15px;">
           <div style="width: 12px; height: 12px; border-radius: 50%; background: #ff5f56; margin-right: 8px;"></div>
           <div style="width: 12px; height: 12px; border-radius: 50%; background: #ffbd2e; margin-right: 8px;"></div>
           <div style="width: 12px; height: 12px; border-radius: 50%; background: #27c93f;"></div>
       </div>
       <!-- Mock dashboard body -->
       <div style="width: 80%; height: 60%; background: white; border: 1px dashed #ccc; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-direction: column; opacity: 0.6;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#496D89" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
            <p style="margin-top: 15px; color: #496D89; font-size: 18px; font-family: sans-serif;">Clinical Trials Simulation Hub Dashboard</p>
       </div>
       <!-- Call to action button -->
       <button id="launch-sim-btn" style="position: absolute; z-index: 10; padding: 14px 28px; background: #496D89; color: white; border: none; border-radius: 28px; font-size: 16px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 12px rgba(73, 109, 137, 0.4); transition: transform 0.1s, background 0.2s; display: flex; align-items: center; gap: 8px;">
           <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
           Launch Simulator
       </button>
       <!-- Loading spinner -->
       <div id="sim-loading-spinner" style="display: none; position: absolute; z-index: 5; flex-direction: column; align-items: center;">
           <div style="width: 40px; height: 40px; border: 4px solid #e1e4e8; border-top: 4px solid #496D89; border-radius: 50%; animation: spin 1s linear infinite;"></div>
           <p style="margin-top: 12px; color: #496D89; font-weight: 600; font-family: sans-serif;">Loading Simulator...</p>
       </div>
       <style>
           @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
           #launch-sim-btn:hover { background: #385b73; transform: scale(1.05); }
           #launch-sim-btn:active { transform: scale(0.95); }
       </style>
   </div>

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:

   README
   getting_started
   accessibility
   tutorials/*
   tutorials/matchpoint/*
   win_ratio_simulation
   reference/index
   CONTRIBUTING
   changelog

Clintrials aims to make trial simulations and design exploration easier. Use the sections above to get started, browse tutorials, or dig into the API reference.
