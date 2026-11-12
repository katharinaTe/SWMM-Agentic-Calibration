"""Bellinge, objective event score.

The code in this module is taken unchanged from the notebook
`Bel_event_score.ipynb` (v1.0.0, https://doi.org/10.5281/zenodo.23478514).
Only the cell boundaries were removed and the block `if __name__ == "__main__":` of the main cell was turned into the function `run_experiment()`.
"""

from .common import *  # noqa: F401,F403  (imports, llm, tool, printing helpers)

# fixed seeds for reproduction
random.seed(501)
np.random.seed(501)

@dataclass
class CalibrationState:
    """common state"""
    inp_path: str | None = None
    node_label: str | None = None

    inp_base: SwmmInput | None = None
    inp_random: SwmmInput | None = None
    ts_base_sim: pd.Series | None = None
    ts_random_sim: pd.Series | None = None
    df_observation: pd.DataFrame | None = None
    value_col: str |None = None
    ts_observation: pd.Series | None = None
    df_results: pd.DataFrame | None = None
    best_index: int | None = None
    best_event_score: float | None = None

STATE = CalibrationState()

# Functions

# SWMM and synthetic observations

def load_base_model(inp_path: str, remove_predevelop: bool = True, start_date: str = "06/29/2012", end_date: str = "07/01/2012", start_time: str = "00:00:00",
    end_time: str = "23:59:59") -> SwmmInput:
    """
    Loads a SWMM model, optionally removes 'PreDevelop',
    and sets the SWMM simulation period.
    """
    inp = SwmmInput(inp_path)
    if remove_predevelop:
        delete_subcatchment(inp, "PreDevelop")

    inp.OPTIONS["START_DATE"] = start_date
    inp.OPTIONS["START_TIME"] = start_time
#    inp.OPTIONS["REPORT_START_DATE"] = start_date
#    inp.OPTIONS["REPORT_START_TIME"] = start_time
    inp.OPTIONS["END_DATE"] = end_date
    inp.OPTIONS["END_TIME"] = end_time
    return inp


def run_swmm_timeseries(inp: SwmmInput, node_label: str)-> pd.Series:
    """Runs SWMM and returns node depth time series for one node."""
    with swmm5_run_temporary(inp) as res:
        ts = res.out.get_part(label=node_label, variable=VARIABLES.NODE.DEPTH)
    # If swmm_api returns a DataFrame, extract the requested node column
    if isinstance(ts, pd.DataFrame):
        if node_label is None:
            raise ValueError("node_label is None. Set STATE.node_label first.")
        matches = [c for c in ts.columns if node_label in str(c)]
        if not matches:
            raise KeyError(
                f"Could not find node_label={node_label!r} in simulation columns. "
                f"Example columns: {list(ts.columns[:10])}"
            )
        ts = ts[matches[0]]

    ts.index = pd.to_datetime(ts.index)
    ts = pd.to_numeric(ts, errors="coerce").dropna().sort_index()
    ts.name = "simulation"

    return ts


def randomize_subcatchments(inp: SwmmInput) -> SwmmInput:
    """
    Generates a random variant of the model:
    - imperviousness randomly between 30 and 90
    - width randomly scaled
    (Based on the official example, but slightly reworded)
    """
    inp_rand = inp.copy()
    for sc in inp_rand.SUBCATCHMENTS.values():
        sc.imperviousness = random.randint(30, 90)
        sc.width = round(sc.width * random.randint(5, 15), 0)
    return inp_rand


def random_series_like(ts: pd.Series, amplitude: float) -> pd.Series:
    """Generates a random noise series with the same index length as ts."""
    noise = amplitude * (1 - 2 * np.random.rand(ts.index.size))
    return pd.Series(index=ts.index, data=noise)


def make_artificial_observation(ts_random: pd.Series) -> pd.Series:
    """
    Generates a synthetic observation from a simulated time series
    by adding relative and absolute error structures via rolling filters.
    """
    obs = ts_random.copy()

    # relative error
    obs *= 1 + random_series_like(ts_random, 0.15).rolling(4, center=True, min_periods=0).mean()
    obs *= 1 + random_series_like(ts_random, 0.15).rolling(11, center=True, min_periods=0).mean()

    # absolute error
    mean_val = obs.mean()
    obs += (mean_val * random_series_like(ts_random, 0.15)).rolling(7, center=True, min_periods=0).mean()
    obs += (mean_val * random_series_like(ts_random, 0.15)).rolling(15, center=True, min_periods=0).mean()

    return obs

def make_real_observation(obs_df: pd.DataFrame, value_col:str)-> pd.Series:
    #obs = obs_df[value_col].astype(float)

    obs = obs_df.copy()

    obs["datetime"] = pd.to_datetime(obs["TIME"].astype(str).str.strip(),
         format="%Y.%m.%d %H:%M:%S", errors="coerce")
    obs[value_col] = pd.to_numeric(obs[value_col], errors="coerce")

    obs = (
        obs
        .dropna(subset=["datetime", value_col])
        .sort_values("datetime")
        .set_index("datetime")[value_col]
    )

    obs_min = obs.copy()

    obs_min.index = pd.to_datetime(obs_min.index)
    obs_min = pd.to_numeric(obs_min, errors="coerce")
    obs_min = obs_min.dropna().sort_index()

    # One value per full minute
    obs_min = obs_min.resample("1min").mean()

    obs_utc = obs_min.copy()
    obs_utc.index = obs_utc.index - pd.Timedelta(hours=2)

    return obs_utc

# SPOTPY

class SpotpyCalibrationConfig:
    """
    SPOTPY configuration:
    Parameters to be calibrated:
    - global conduit Manning roughness factor
    - global subcatchment imperviousness factor

    Target variable:
    - event score between simulated and observed water depth at G71F090
    """

    # Parameter definition for SPOTPY
    conduit_n_factor = spotpy.parameter.Uniform(
        low=0.7,
        high=1.3
    )

    imp_factor = spotpy.parameter.Uniform(
        low=0.7,
        high=1.3
    )

    def __init__(self, observation: pd.Series, end_date:str, node_label:str):
        self.observation = observation
        # end of evaluation (as in the example)
        self._end_date = pd.to_datetime(end_date)
        self.node_label = node_label

    def _build_model(self, vector) -> SwmmInput:
        conduit_n_factor, imp_factor = vector

        if STATE.inp_base is None:
            raise RuntimeError(
                "Base model is not loaded (STATE.inp_base is None)."
            )

        inp = STATE.inp_base.copy()

        # -------------------------------------------------
        # Hydrological parameter:
        # global imperviousness factor
        # -------------------------------------------------
        for sc in inp.SUBCATCHMENTS.values():
            sc.imperviousness = round(
                max(
                    0,
                    min(
                        100,
                        sc.imperviousness * imp_factor
                    )
                ),
                3
            )

        # -------------------------------------------------
        # Hydraulic parameter:
        # global conduit Manning roughness factor
        # -------------------------------------------------
        for conduit in inp.CONDUITS.values():
            conduit.roughness = round(
                conduit.roughness * conduit_n_factor,
                5
            )

        return inp

    def simulation(self, vector):
        """
        Called by SPOTPY for each parameter combination.
        """
        inp = self._build_model(vector)
        sim = run_swmm_timeseries(inp, node_label=self.node_label)
        return sim

    def evaluation(self):
        """
        Delivers the observation time series to SPOTPY.
        """
        return self.observation

    def _clip_score(self, score):
        return min(5, max(1, score))

    def model_quality_score(self,obs, sim):
        """
        Score according to E2 / VOL / DYMax concept.

        E2     = Nash-Sutcliffe efficiency
        VOL    = relative volume deviation
        DYMax  = relative peak-flow deviation

        Final score:
            1 = perfect
            5 = unusable
        """

        obs = np.asarray(obs, dtype=float)
        sim = np.asarray(sim, dtype=float)

        mask = np.isfinite(obs) & np.isfinite(sim)
        obs = obs[mask]
        sim = sim[mask]

        if len(obs) == 0:
         return 5

        # E2 / NSE
        e2 = spotpy.objectivefunctions.nashsutcliffe(obs, sim)

        # Relative volume deviation
        vol = (np.sum(sim) - np.sum(obs)) / np.sum(obs)

        # Relative maximum / peak deviation
        dymax = (np.max(sim) - np.max(obs)) / np.max(obs)

        # Convert metrics to scores from 1 to 5
        # E2: 1 -> score 1, 0 -> score 5
        score_e2 = 1 + (1 - e2) * 4

        # VOL: 0 -> score 1, 0.5 -> score 5
        score_vol = 1 + (abs(vol) / 0.5) * 4

        # DYMax: 0 -> score 1, 0.5 -> score 5
        score_dymax = 1 + (abs(dymax) / 0.5) * 4

        # Clip worse results to 5
        score_e2 = self._clip_score(score_e2)
        score_vol = self._clip_score(score_vol)
        score_dymax = self._clip_score(score_dymax)

        # Event score
        score_event = (score_e2 + score_vol + score_dymax) / 3

        return score_event

    def objectivefunction(self, simulation, evaluation):
        """
        best fit: 1, worst fit: 5
        """
        obs = evaluation.loc[: self._end_date]
        sim = simulation.loc[: self._end_date]

        # Align time indices, important for pandas Series
        obs, sim = obs.align(sim, join="inner")

        if len(obs) == 0:
            raise RuntimeError(
                "No overlapping timestamps between observation and simulation."
           )

        return self.model_quality_score(obs, sim)

# Tools for agents

# Model tools


@tool("prepare_swmm_model")
def prepare_swmm_model(inp_path: str, remove_predevelop: bool = True, start_date: str = "06/29/2012", end_date: str = "07/01/2012", start_time: str = "00:00:00",
    end_time: str = "23:59:59") -> str:
    """
    Loads a SWMM model, optionally removes 'PreDevelop' and saves it in the global STATE.
    """
    STATE.inp_path = inp_path
    STATE.inp_base = load_base_model(inp_path=inp_path, remove_predevelop=remove_predevelop, start_date=start_date, end_date=end_date, start_time=start_time, end_time=end_time)
    return (
        f"Base model '{inp_path}' loaded."
        f"Removed 'PreDevelop': {remove_predevelop}."
        f"Simulation period: {start_date} {start_time} to {end_date} {end_time}. "
        f"Number of Subcatchments: {len(STATE.inp_base.SUBCATCHMENTS)}"
    )

@tool("generate_synthetic_observation")
def generate_synthetic_observation(node_label: str) -> str:
    """
    Creates a random model variant, simulates the base and random models,
    and generates a synthetic observation. Results are stored in STATE.
    """
    if STATE.inp_base is None:
        raise RuntimeError("Base model is not loaded. Please use prepare_swmm_model first.")

    inp_rand = randomize_subcatchments(STATE.inp_base)
    STATE.inp_random = inp_rand

    STATE.node_label = node_label
    STATE.ts_base_sim = run_swmm_timeseries(STATE.inp_base, node_label=node_label)
    STATE.ts_random_sim = run_swmm_timeseries(inp_rand, node_label=node_label)
    STATE.ts_observation = make_artificial_observation(STATE.ts_random_sim)
    

    return (
        "Synthetic observations generated. "
        f"length of time series: {len(STATE.ts_observation)}. "
        f"nodes: {node_label}."
    )

@tool("prepare_real_observation")
def prepare_real_observation() -> str:
    """
    Prepares real observation data for SWMM model calibration by storing the observed time series in STATE.ts_observation.
    """

    STATE.ts_observation = make_real_observation(STATE.df_observation, STATE.value_col)

    return (
        "Real observations prepared. "
        f"length of time series: {len(STATE.ts_observation)}. "
        
    )

# Calibration tools

@tool("run_spotpy_calibration")
def run_spotpy_calibration(
    end_date: str,
    repetitions: int = 1000,
    kstop: int = 100,
    score_threshold: float = 2.0,
    dbname: str = "data_interim_SCE_UA_multi_agent",
) -> str:
    """
    Performs an SCE-UA calibration with SPOTPY, saves the results
    in STATE.df_results, and calculates the best event score.
    Parameters:
      - repetitions: Number of model runs (default 1000)
      - kstop: termination criterion of SCE-UA
      - score_threshold: threshold for "good" runs, e.g. score <= 2.0 (between 1 and 5) 
    """
#    if STATE.ts_observation is None:
#        raise RuntimeError("No observation time series in STATE. Please call generate_synthetic_observation or prepare_real_observation first.")

    if STATE.ts_observation is None:
        if STATE.df_observation is not None and STATE.value_col is not None:
            STATE.ts_observation = make_real_observation(
                STATE.df_observation,
                STATE.value_col
            )
        else:
            raise RuntimeError(
                "No observation time series in STATE, and no dataframe/value column "
               "available to create one. Please set STATE.df_observation and STATE.value_col first."
            )
    cfg = SpotpyCalibrationConfig(STATE.ts_observation, end_date, STATE.node_label)

    fn = Path(f"{dbname}.csv")

    # Always remove old result file, especially if it is empty
    if fn.exists():
        print(f"Removing old SPOTPY result file: {fn}")
        fn.unlink()
    
    # create sampler
    sampler = spotpy.algorithms.sceua(
        cfg,
        dbname=dbname,
        dbformat="csv",
    )

    sampler.sample(
        repetitions=repetitions,
        kstop=kstop,
       pcento=1e-8,
    )

    # check that SPOTPY produced results
    if not fn.exists():
        raise RuntimeError(f"SPOTPY result file was not created: {fn}")

    if fn.stat().st_size == 0:
        raise RuntimeError(
            f"SPOTPY result file is empty: {fn}. "
           "The sampler did not write valid results."
        )

    results = spotpy.analyser.load_csv_results(dbname)
    df_results = pd.DataFrame(results)
    STATE.df_results = df_results


    output_file = Path(f"{dbname}_final_calibration_results.csv")
    df_results.to_csv(output_file, index=False)

    # Determine the best result
    best_index, best_obj = get_minlikeindex(results, verbose=False)
    best_event_score = best_obj  # lower is better

    STATE.best_index = int(best_index)
    STATE.best_event_score = float(best_event_score)

    # Number of "good" runs for uncertainty band
    best_row = df_results.iloc[int(best_index)]
    sims_cols = get_simulation_fields(results)
    count_good = (df_results["like1"] <= score_threshold).sum()

    summary = (
        f"Calibration complete.\n"
        f"Number of simulations: {len(df_results)}\n"
        f"Best event score: {best_event_score:.3f}\n"
        f"Number of runs with event score <= {score_threshold:.2f}: {count_good}"
        f"Results saved to: {output_file}"
    )

    return summary

# Analysis tools

@tool("analyze_calibration_results")
def analyze_calibration_results(score_threshold: float = 2.0) -> str:
    """
    Analyzes DF results in STATE, generates:
    - Uncertainty band (GLUE) for low event score runs
    - Plot of observation, uncertainty band, and best-fit simulation
    Returns a textual summary.

        Event score:
      - 1 = perfect
      - 5 = unusable
      - lower is better
    """
    if STATE.df_results is None or STATE.ts_observation is None:
        raise RuntimeError("No calibration results or observation data in STATE.")

    df = STATE.df_results
    obs = STATE.ts_observation

    results_rec = df.to_records(index=False)
    sim_cols = get_simulation_fields(results_rec)

    # all simulations
    sim_all = df[sim_cols].T
    sim_all.index = obs.index

    # filter for good runs
    mask_good = df["like1"] <= score_threshold
    sim_good = df.loc[mask_good, sim_cols].T
    sim_good.index = obs.index

    # uncertainty bounds
    unc_bounds = sim_good.agg(["min", "max"], axis=1)

    # best result
    if STATE.best_index is None or STATE.best_event_score is None:
        best_index, best_obj = get_minlikeindex(results_rec, verbose=False)
        STATE.best_index = int(best_index)
        STATE.best_event_score = float(best_obj)

    best_ts = sim_all[STATE.best_index]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    obs.plot(ax=ax, color="C1", marker=".", lw=0, ms=3, label="observation data")

    ax.fill_between(
        unc_bounds.index,
        unc_bounds["min"],
        unc_bounds["max"],
        alpha=0.3,
        label=f"parameter uncertainty (event score ≤ {score_threshold:.2f})",
    )

    best_ts.plot(ax=ax, label=f"best-fit simulation (best score ={STATE.best_event_score:.2f})")

    ax.set_xlabel("time")
    ax.set_ylabel("observation value")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('calib_results.png')

    return (
        f"Analysis complete. Best event score:  {STATE.best_event_score:.3f}. "
        f"Number of ‘good’ runs (event score ≤ {score_threshold:.2f}): {mask_good.sum()}."
    )

# Define agents

# Tools, auf die der Model-Agent zugreifen darf
model_tools = [prepare_swmm_model, generate_synthetic_observation, prepare_real_observation]

model_prompt = (
    "You are a simulation agent for SWMM.\n\n"
    "CRITICAL:\n"
    "- You MUST use the available tools to do your work.\n"
    "- Do NOT output Python code or step-by-step recipes.\n"
    "- Instead, call the tools until the SWMM base model is prepared and a synthetic\n"
    "  observation time series is generated or time series observation data is available.\n"
    "- When you are done, reply with a SHORT confirmation of what was done."    
    )

model_agent = create_react_agent(
    model=llm,
    tools=model_tools,
    prompt=model_prompt,
    name="model_agent",
)

calib_tools = [run_spotpy_calibration]

calib_prompt =(
    "You are a calibration agent.\n\n"
    "CRITICAL:\n"
    "- You MUST use the available tools to actually run the calibration.\n"
    "- Do NOT describe code or give instructions. Just call the tool.\n"
    "- Choose reasonable values for 'repetitions' and 'kstop' (default repetitions: 1000, kstop: 100),\n"
    "  pass them to the tool, and let it run.\n"
    "- When done, reply only with the text returned by the tool."
    )


calib_agent = create_react_agent(
    model=llm,
    tools=calib_tools,
    prompt=calib_prompt,
    name="calib_agent",
)

analysis_tools = [analyze_calibration_results]

analysis_prompt = (
    "You are an expert for uncertainty analysis in urban drainage models.\n"
    "CRITICAL:\n"
    "- You MUST use the tool 'analyze_calibration_results' to do your work.\n"
    "- Do NOT output code or recipes. Just call the tool and then summarize the result."
        )

analysis_agent = create_react_agent(
    model=llm,
    tools=analysis_tools,
    prompt=analysis_prompt,
    name="analysis_agent",
)

supervisor = create_supervisor(
    model=llm,
    agents=[model_agent, analysis_agent, calib_agent],
    prompt=("You are a supervisor agent for an urban drainage model calibration system."
"You have three specialized sub-agents available as tools:\n"
"- model_agent: handles SWMM model and observation data.\n"
"- calib_agent: performs SPOTPY calibration.\n"
            "- analysis_agent: analyzes the results and uncertainty."
"Plan the steps in a logical order and call up the appropriate tools."
"Your goal is to calibrate an urban drainage network and describe the quality of the calibration."
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Assign work at least once.\n"
        "After the agents return, you MUST produce ONE final assistant message to the USER "
        "that summarizes the result in plain text. "
        "This FINAL message must NOT contain any tool calls or handoffs. "
        "Begin the final message with: 'FINAL ANSWER:'."
    ),
    add_handoff_back_messages=True,
    #output_mode="assistant",
    output_mode="full_history",
).compile()

# Main method

from pathlib import Path
import shutil
import json

def message_content(msg):
    """Works for both dict messages and LangChain message objects."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", str(msg))

def run_experiment():

    n_runs = 3
    output_root = Path("multi_run_outputs")
    output_root.mkdir(exist_ok=True)   

    run_summaries = []

    # load rainfall and sensor data
    rainfile = "rg_bellinge_Jun2010_Aug2021.dat"
    sensorfile = "data-pG71F04R-D1-W1-BL3_M2-PV-SMO.txt"

    # Date range
    start = "06/29/2012 04:00:00"
    end = "07/01/2012 23:59:59"

    # -------------------------
    # Load water-level data
    # -------------------------
    water = pd.read_csv(
        sensorfile,
        sep=None,
        engine="python"
    )

    # Remove spaces from column names
    water.columns = water.columns.str.strip()

    # Remove leading/trailing spaces from TIME values
    water["TIME"] = water["TIME"].astype(str).str.strip()

    # Convert TIME to datetime
    water["datetime"] = pd.to_datetime(
        water["TIME"],
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce"
    )

    # Convert VALUE to numeric water level
    water["water_level"] = pd.to_numeric(
        water["VALUE"],
        errors="coerce"
    )

    water["water_level_m"] = water["water_level"]/100

    # Remove rows with invalid datetime or water level
    water = water.dropna(subset=["datetime", "water_level_m"])

    # Filter date range
    water = water[(water["datetime"] >= start) & (water["datetime"] <= end)]

    STATE.df_observation = water
    STATE.value_col = "water_level_m"
    STATE.node_label = "G71F090"

    for run_id in range(1, n_runs + 1):
        # Reset run-specific state
        STATE.inp_path = None
        STATE.inp_base = None
        STATE.inp_random = None
        STATE.ts_base_sim = None
        STATE.ts_random_sim = None
        STATE.ts_observation = None
        STATE.df_results = None
        STATE.best_index = None
        STATE.best_nse = None
        STATE.best_parameters = None

        # Keep externally provided observation information
        STATE.df_observation = water.copy()
        STATE.value_col = "water_level_m"

        # Sensor G71F04R inlet corresponds to SWMM node G71F090
        STATE.node_label = "G71F090"
        print(f"\n{'=' * 80}")
        print(f"STARTING SUPERVISOR RUN {run_id}/{n_runs}")
        print(f"{'=' * 80}\n")

        run_dir = output_root / f"run_{run_id:02d}"
        run_dir.mkdir(exist_ok=True)

        dbname = f"data_interim_SCE_UA_run_{run_id:02d}"

        user_prompt = ( "Calibrate the SWMM-model 'BellingeSWMM_v021_nopervious.inp' from 06/29/2012 00:00:00 to 07/01/2012 23:59:59."
                        "First, call prepare_swmm_model to load and prepare the SWMM model. "
                        "The measured sensor data originate from the G71F04R inlet level sensor. "
                        "The corresponding SWMM model node for comparison is G71F090. "
                        "Then call prepare_real_observation using the already loaded observation "
                        "data stored in STATE.df_observation with STATE.value_col='water_level_m'. "
                        "Only after both model and observation data have been prepared, "
                        "call the Calibration Agent and run the SPOTPY calibration. "
                        "Select appropriate settings for SPOTPY, perform the calibration, and evaluate the model quality and uncertainty."
                      f"Use the SPOTPY database name '{dbname}' so that this run is saved separately.")

        last_chunk = None

        for chunk in supervisor.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ]
            },
        ):
            last_chunk = chunk
            pretty_print_messages(chunk, last_message=True)

        if last_chunk is None:
            raise RuntimeError(f"No output received from supervisor in run {run_id}.")

        final_message_history = last_chunk["supervisor"]["messages"]
        last = final_message_history[-1]
        final_text = message_content(last)

        print("\nFINAL SUPERVISOR MESSAGE:\n")
        print(final_text)

        # Save final supervisor message
        final_message_file = run_dir / "final_supervisor_message.txt"
        final_message_file.write_text(final_text, encoding="utf-8")

        # Save full message history as text
        history_file = run_dir / "message_history.txt"
        with history_file.open("w", encoding="utf-8") as f:
            for m in convert_to_messages(final_message_history):
                f.write(m.pretty_repr())
                f.write("\n\n")

        # Save STATE.df_results, if available
        if getattr(STATE, "df_results", None) is not None:
            state_results_file = run_dir / "STATE_df_results.csv"
            STATE.df_results.to_csv(state_results_file, index=False)

        # Copy calibration CSV created by run_spotpy_calibration
        csv_file = Path(f"{dbname}_final_calibration_results.csv")
        if csv_file.exists():
            shutil.copy2(csv_file, run_dir / csv_file.name)

        raw_spotpy_file = Path(f"{dbname}.csv")
        if raw_spotpy_file.exists():
            shutil.copy2(raw_spotpy_file, run_dir / raw_spotpy_file.name)

        # Copy calibration result plot, if analyze_calibration_results created it
        plot_file = Path("calib_results.png")
        if plot_file.exists():
            shutil.copy2(plot_file, run_dir / "calib_results.png")

        # Save compact run summary
        summary = {
            "run_id": run_id,
            "dbname": dbname,
            "run_dir": str(run_dir),
            "best_index": getattr(STATE, "best_index", None),
            "best_event_score": getattr(STATE, "best_event_score", None),
            "best_parameters": getattr(STATE, "best_parameters", None),
            "final_message_file": str(final_message_file),
        }

        run_summaries.append(summary)

        with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Save overview of all 10 runs
    df_summary = pd.DataFrame(run_summaries)
    df_summary.to_csv(output_root / "all_run_summaries.csv", index=False)

    with (output_root / "all_run_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(run_summaries, f, indent=2)

    print("\nAll simulation runs complete.")
    print(f"Results saved in: {output_root.resolve()}")
