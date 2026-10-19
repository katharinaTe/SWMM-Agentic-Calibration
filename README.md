# SWMM-Agentic-Calibration

**Agent-Based Calibration of Urban Drainage Models Using SWMM and LLMs

This repository contains the code accompanying the paper listed under [Citation](#citation). It demonstrates a multi-agent LLM system that calibrates a SWMM urban drainage model (EPA detention pond example). A supervisor agent coordinates three specialised agents, which use tools to prepare the model, generate synthetic observations, run the calibration with SPOTPY and analyse calibration quality and parameter uncertainty.

## Workflow

1. **Model setup** – the SWMM model is loaded and prepared programmatically (`swmm_api`).
2. **Synthetic observations** – a randomised model variant is simulated and perturbed with relative and absolute error to create an artificial observation time series.
3. **Parameter calibration** – SPOTPY (SCE-UA) calibrates the model by maximising the Nash–Sutcliffe efficiency (NSE).
4. **Uncertainty analysis** – parameter uncertainty is estimated with a GLUE-type approach (runs with NSE > threshold).
5. **Agent orchestration** – LLM agents (LangGraph supervisor + ReAct agents) decide on and call the tools for steps 1–4 and summarise the results.

| Agent | Tools |
|---|---|
| Supervisor | delegates tasks to the agents below |
| Model agent | `prepare_swmm_model`, `generate_synthetic_observation` |
| Calibration agent | `run_spotpy_calibration` |
| Analysis agent | `analyze_calibration_results` |

## Repository structure
.

├── sewer_calibration_agentic_generic_detention_pond-WST.ipynb

├── README.md

## Methodology

The workflow consists of the following steps:

### Model Setup
A SWMM model is loaded and configured programmatically.
Synthetic Observations
Simulation outputs are used to generate synthetic observation data for calibration.
Parameter Calibration
The SPOTPY library is used to perform parameter sampling and optimization based on an objective function.
Agent-Based Interaction
LLM-based agents interact with tools to:
run simulations
adjust model parameters
evaluate calibration performance

## Requirements

- Python ≥ 3.10
- Python packages (see `requirements.txt`):
  `numpy`, `pandas`, `matplotlib`, `spotpy`, `swmm-api`, `langchain`, `langchain-core`, `langchain-ollama`, `langgraph`, `langgraph-supervisor`
- SWMM 5 engine as required by `swmm_api` (see the [swmm_api documentation](https://markuspichler.gitlab.io/swmm_api))
- [Ollama](https://ollama.com) running locally with the model `gpt-oss:20b`

```bash
pip install -r requirements.txt
ollama pull gpt-oss:20b
```

## Citation

If you use this code, please cite the accompanying paper:

> Teuber, K. (2026). Call my Agents - Multi-Agent LLM-Based Calibration of Urban Drainage Models: Performance and Reliability Across Model Scales. Water Science & Technology, in press.

The version of the code used in the paper is archived on Zenodo: [DOI follows].

## License

This code is licensed under the [MIT License](LICENSE).

Parts of the code (synthetic observation generation, SPOTPY calibration setup and GLUE uncertainty analysis) are adapted from the swmm_api example [“Calibration and uncertainty analysis”](https://markuspichler.gitlab.io/swmm_api/examples/calibration_and_uncertainty.html) by Markus Pichler, which is also licensed under the MIT License.

The example model `Detention_Pond_Model.inp` is taken from the EPA SWMM example set.

## Author

Katharina Teuber, Jade Hochschule – University of Applied Sciences, Oldenburg, Germany


### References and Thanks
Pichler, M. (2025). swmm_api: A Python Package for Automation, Customization, and Visualization in SWMM-Based Urban Drainage Modeling. Water, 17(9), 1373. https://doi.org/10.3390/w17091373
