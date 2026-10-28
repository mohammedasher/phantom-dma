"""Step 4: make every figure of the paper that comes from data (Figs. 1-7, Figs. S1 and S2).

    python step4_figures.py

Figures are written to output/figures/. Each figure also has its own script in
figures/, which can be run on its own; this file just runs them one after another.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ["figures/fig1_fit_workflow.py"],
    ["figures/fig2_resistance.py"],
    ["figures/fig3_ir_correction.py"],
    ["figures/fig4_hysteresis.py"],
    ["figures/fig_bias_modes.py", "5"],
    ["figures/fig_bias_modes.py", "6"],
    ["figures/fig_bias_modes.py", "7"],
    ["figures/figS1_peak_alignment.py"],
    ["figures/figS2_current_partition.py"],
]

if __name__ == "__main__":
    for script in SCRIPTS:
        print("running", " ".join(script))
        subprocess.run([sys.executable, os.path.join(HERE, script[0])] + script[1:], check=True)
    print("Figures written to", os.path.join(HERE, "output", "figures"))
