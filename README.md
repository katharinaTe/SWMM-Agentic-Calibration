# SWMM-Agentic-Calibration

**Multi-agent LLM-based calibration of urban drainage models with EPA-SWMM, SPOTPY and locally hosted LLMs**

This repository contains the code accompanying the paper

> Teuber, K. (2026). Call my Agents – Multi-Agent LLM-Based Calibration of Urban Drainage Models: Performance and Reliability Across Model Scales. *Water Science & Technology*, in press.

A supervisor agent receives a calibration task in natural language and coordinates three specialised agents that prepare the SWMM model and observation data, run an SCE-UA calibration with SPOTPY and analyse calibration quality and parameter uncertainty (GLUE). All LLMs run locally via Ollama.


## Framework

| Agent | Role | Tools |
|---|---|---|
| Supervisor agent | decomposes the task, routes it to the specialised agents, writes the final summary | – |
| Model agent | loads and prepares the SWMM model, provides observation data | `prepare_swmm_model`, `generate_synthetic_observation`, `prepare_real_observation` (Bellinge only) |
| Calibration agent | runs the SCE-UA calibration and selects its settings (repetitions, k<sub>stop</sub>, threshold) | `run_spotpy_calibration` |
| Analysis agent | evaluates best-fit run and GLUE uncertainty band, plots the hydrograph | `analyze_calibration_results` |

Calibration objectives: Nash–Sutcliffe efficiency (NSE) or combined event score (NSE, volume error and peak error, scaled 1–5, after Henrichs 2015).

## Notebooks

Each notebook is self-contained and corresponds to one experiment in the paper.

| Notebook | Case study | Objective | Setting | Runs | Paper |
|---|---|---|---|---|---|
| `dp_nse_controlled_gpt.ipynb` | Detention pond | NSE | controlled | 10 | Tables 1, 2 |
| `dp_es_controlled_gpt.ipynb` | Detention pond | event score | controlled | 10 | Tables 1, 2 |
| `dp_nse_stochastic_gpt.ipynb` | Detention pond | NSE | stochastic | 10 | text only (not shown) |
| `dp_es_stochastic_gpt.ipynb` | Detention pond | event score | stochastic | 10 | Figure 8 |
| `Bel_NSE.ipynb` | Bellinge | NSE | – | 3 | Table 3, Figure 9 |
| `Bel_event_score.ipynb` | Bellinge | event score | – | 3 | Table 3, Figure 9 |

**Experimental settings**

- *Controlled:* the synthetic benchmark (random model variant and artificial observation) is regenerated with seed 501 and the SCE-UA optimizer is initialised with seed 502 in every run. Remaining variability stems from the LLM.
- *Stochastic:* random-number generators are seeded once (501) at the start and not reset between runs.
- *Bellinge:* measured rainfall and water-level data are identical across runs; the optimizer initialisation varies.
- LLM temperature: 0.1. The LLM itself is not seeded.

**Other LLMs.** The notebooks are configured for `gpt-oss:20b`. The runs with `qwen3:8b`, `llama3.2:3b` and `llama3.2:1b` were performed by changing the `model` argument in the cell *Choose LLM-model*; prompts and all other settings are identical.

**Pre-development subcatchments.** For the prompt experiment on pre-development subcatchments, the phrase *“remove subcatchments in predevelopment state”* was removed from the user prompt in the main cell.

## Requirements

- Linux or macOS (the notebooks use the Unix-only Python module `resource` for memory tracking)
- Python ≥ 3.10 with the packages listed in `requirements.txt`
- EPA SWMM 5 command-line engine (`runswmm`), e.g. built from the [EPA SWMM source code](https://github.com/USEPA/Stormwater-Management-Model)
- [Ollama](https://ollama.com) running locally (default `http://localhost:11434`) with the required models

```bash
pip install -r requirements.txt
ollama pull gpt-oss:20b
```
Package versions are not pinned. The LangChain/LangGraph APIs change frequently; if an import fails, use package versions released before October 2026.

## Input data

The input files are not included in this repository and must be placed in the working directory of the notebooks.

**Detention pond.** `Detention_Pond_Model.inp` from the example models distributed with EPA SWMM 5.2.

**Bellinge.** From the Bellinge data set (Nedergaard Pedersen et al. 2021):

- `BellingeSWMM_v021_nopervious.inp` – SWMM model
- `rg_bellinge_Jun2010_Aug2021.dat` – rainfall data
- `data-pG71F04R-D1-W1-BL3_M2-PV-SMO.txt` – water level at sensor G71F04R (compared with SWMM node G71F090)

Observed water levels are converted from cm to m, resampled to 1-min intervals and shifted by −2 h to match the model time base.

## Usage

1. Set the path to the SWMM executable in the cell that defines `CONFIG["exe_path"]`.
2. Make sure Ollama is running and the input files are in the working directory.
3. Run all cells of the notebook.

Results are written per run to `multi_run_outputs_<mode>/run_XX/` (detention pond) or `multi_run_outputs/run_XX/` (Bellinge):

- `message_history.txt` – full agent conversation
- `final_supervisor_message.txt` – final answer of the supervisor agent
- SPOTPY result database and `*_final_calibration_results.csv`
- `calib_results.png` – hydrograph with uncertainty band (if the analysis agent was invoked)
- `run_summary.json` – best objective, runtimes and number of model evaluations

An overview of all runs is saved as `all_run_summaries.csv` / `.json`.

**Notes**

- Start each experiment in a clean directory. In the detention pond notebooks, an existing SPOTPY database with the same name is not overwritten and leads to an error.
- LLM outputs are not deterministic. Agent routing, selected calibration settings and texts may differ between runs and between model versions.
- The Bellinge calibrations take several days per experiment (about 6.4 d for three runs on an NVIDIA RTX 2000 Ada workstation).

## Citation

If you use this code, please cite the paper above. The version of the code used in the paper is archived on Zenodo: *DOI follows*.

## License

This code is licensed under the [MIT License](LICENSE).

The generation of synthetic observations and parts of the SPOTPY calibration and GLUE analysis are adapted from the swmm_api example [“Calibration and uncertainty analysis”](https://markuspichler.gitlab.io/swmm_api/examples/calibration_and_uncertainty.html) by Markus Pichler, which is licensed under the MIT License.

## Author

Katharina Teuber, Jade University of Applied Sciences, Oldenburg, Germany

## References

- Henrichs, M. (2015). Einfluss von Unsicherheiten auf die Kalibrierung urban-hydrologischer Modelle. Dissertation, TU Dresden.
- Houska, T., Kraft, P., Chamorro-Chavez, A. & Breuer, L. (2015). SPOTting model parameters using a ready-made Python package. *PLoS ONE*, 10(12), e0145180. https://doi.org/10.1371/journal.pone.0145180
- Nedergaard Pedersen, A. et al. (2021). The Bellinge data set: open data and models for community-wide urban drainage systems research. *Earth System Science Data*, 13(10), 4779–4798. https://doi.org/10.5194/essd-13-4779-2021
- Pichler, M. (2025). swmm_api: A Python Package for Automation, Customization, and Visualization in SWMM-Based Urban Drainage Modeling. *Water*, 17(9), 1373. https://doi.org/10.3390/w17091373
