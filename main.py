import LiteralMessagePassing as lmp
import argparse
import asyncio
import datetime
from dataclasses import dataclass
import logging
import os
import json
import random
import traceback
import networkx as nx
from networkx.readwrite import json_graph
import subprocess
import pandas as pd
from datasets import load_dataset


TASKS = {
    "matching": lmp.Matching,
    "consensus": lmp.Consensus,
    "coloring": lmp.Coloring,
    "leader_election": lmp.LeaderElection,
    "vertex_cover": lmp.VertexCover,
}

MODEL_PROVIDER = {
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4o": "openai",
    "gpt-5.4": "openai",
    "o1": "openai",
    "o3-mini": "openai",
    "o4-mini": "openai",
    "llama3.1": "ollama",
    "gemini-2.0-flash": "google-genai",
    "gemini-2.0-flash-lite": "google-genai",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-3-7-sonnet-20250219-thinking": "anthropic",
    "gemini-2.5-flash-preview-04-17": "google-genai",
    "gemini-2.5-flash-preview-04-17-thinking": "google-genai",
    "gemini-2.5-pro-exp-03-25": "google-genai",
    "gemini-2.5-pro-preview-03-25": "google-genai",
    "gemini-2.5-pro-preview-05-06": "google-genai",
    "gemini-1.5-pro": "google-genai",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "together",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "together"
}


def resolve_model_provider(model_name):
    provider = MODEL_PROVIDER[model_name]
    if (
        provider == "openai"
        and os.getenv("USE_AZURE_OPENAI_AAD", "").lower() in {"1", "true", "yes"}
    ):
        return "azure-openai-aad"
    return provider


@dataclass(frozen=True)
class ExperimentSpec:
    task: str
    model: str
    graph_model: str
    graph_index: int
    run_index: int
    rounds: int
    graph: nx.Graph
    run_id: str


class KeyValueFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        fields = getattr(record, "fields", None)
        if not fields:
            return message
        kv = " ".join(
            f"{key}={json.dumps(value, default=str)}" for key, value in fields.items()
        )
        return f"{message} {kv}"


def log_event(logger, level, event, **fields):
    logger.log(level, event, extra={"fields": {"event": event, **fields}})


def setup_logging(log_level, log_file=None):
    logger = logging.getLogger("agentsnet")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = KeyValueFormatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_graph(graph_model, graph_size, num_sample, logger=None):
    dataset = load_dataset("disco-eth/AgentsNet", split="train")
    _loaded_hf_df = pd.DataFrame(dataset)

    row = _loaded_hf_df[
        (_loaded_hf_df["graph_generator"] == graph_model) &
        (_loaded_hf_df["num_nodes"] == graph_size) &
        (_loaded_hf_df["index"] == num_sample)
    ]

    if len(row) == 0:
        raise ValueError(f"Graph not found: {graph_model}_{graph_size}_{num_sample}")

    graph_dict = json.loads(row.iloc[0]["graph"])
    if logger:
        log_event(
            logger,
            logging.INFO,
            "graph_loaded",
            graph_model=graph_model,
            graph_size=graph_size,
            graph_index=num_sample,
        )
    return json_graph.node_link_graph(graph_dict["graph"], edges="links")

def get_git_commit_hash():
    '''This function is failsafe even if git is not installed on the system.'''
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return commit_hash
    except Exception as e:
        return "None"

def safe_filename_part(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def save_results(answers, transcripts, graph, rounds, model_name, task, score, commit_hash, graph_generator, graph_index, successful, error_message, chain_of_thought, num_fallbacks, num_failed_json_parsings_after_retry, num_failed_answer_parsings_after_retry, model_errors=None, run_index=0, run_id=None, logger=None):
    """Saves the experiment results and message transcripts to a JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = safe_filename_part(model_name.split("/")[-1])
    safe_graph_generator = safe_filename_part(graph_generator)
    filename = (
        f"{task}_results_{timestamp}_{safe_graph_generator}"
        f"_graph{graph_index}_run{run_index}_rounds{rounds}_{safe_model}"
        f"_nodes{len(graph.nodes())}.json"
    )

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump({
            'answers': answers,
            'transcripts': transcripts,
            'graph': json_graph.node_link_data(graph),
            'num_nodes': len(graph.nodes()),
            'diameter': nx.diameter(graph),
            'max_degree': max(dict(graph.degree()).values()),
            'rounds': rounds,
            'model_name': model_name,
            'task': task,
            'score': score,
            'commit_hash': commit_hash,
            'graph_generator': graph_generator,
            'graph_index': graph_index,
            'run_index': run_index,
            'run_id': run_id,
            'successful': successful,
            'error_message': error_message,
            'chain_of_thought': chain_of_thought,
            'num_fallbacks': num_fallbacks,
            'num_failed_json_parsings_after_retry': num_failed_json_parsings_after_retry,
            'num_failed_answer_parsings_after_retry': num_failed_answer_parsings_after_retry,
            'model_errors': model_errors or [],
        }, f, indent=4)

    if logger:
        log_event(logger, logging.INFO, "results_saved", run_id=run_id, filepath=filepath)
    return filepath


LOCAL_ROUNDS = {
    10: [4, 6, 8],
    20: [6, 8, 10],
    50: [8, 10], 
}


def determine_rounds(task, graph, num_sample, num_samples, rounds):
    if task in ["consensus", "leader_election"] or graph.number_of_nodes() > 16:
        return 2 * nx.diameter(graph) + 1
    else:
        return rounds


async def run(args):
    logger = setup_logging(args.log_level, args.log_file)
    if args.max_parallel_experiments < 1:
        raise ValueError("--max_parallel_experiments must be >= 1")
    commit_hash = get_git_commit_hash()
    random.seed(args.seed)

    if args.missing_run_file is not None:
        recovery_mode = True
        missing_run_df = pd.read_csv(args.missing_run_file)
    else:
        recovery_mode = False

    specs = build_experiment_specs(args, missing_run_df if recovery_mode else None, logger)
    if not specs:
        log_event(logger, logging.INFO, "run_summary", total=0, succeeded=0, failed=0)
        return []

    semaphore = asyncio.Semaphore(args.max_parallel_experiments)
    completed = 0
    succeeded = 0
    failed = 0
    results = []

    async def run_with_progress(spec):
        nonlocal completed, succeeded, failed
        async with semaphore:
            result = await run_single_experiment(args, spec, commit_hash, logger)
        completed += 1
        if result["successful"]:
            succeeded += 1
        else:
            failed += 1
        log_event(
            logger,
            logging.INFO,
            "progress_update",
            completed=completed,
            total=len(specs),
            succeeded=succeeded,
            failed=failed,
        )
        return result

    results = await asyncio.gather(*(run_with_progress(spec) for spec in specs))
    log_event(
        logger,
        logging.INFO,
        "run_summary",
        total=len(results),
        succeeded=succeeded,
        failed=failed,
    )
    return results


def build_experiment_specs(args, missing_run_df=None, logger=None):
    specs = []
    recovery_mode = missing_run_df is not None

    if recovery_mode and logger:
        log_event(logger, logging.INFO, "recovery_mode_start", missing_run_file=args.missing_run_file)

    for graph_model in args.graph_models:
        for i in range(args.start_from_sample, args.samples_per_graph_model):
            graph = get_graph(graph_model, args.graph_size, i, logger=logger)
            rounds = determine_rounds(args.task, graph, i, args.samples_per_graph_model, args.rounds)
            if logger:
                log_event(
                    logger,
                    logging.INFO,
                    "rounds_selected",
                    graph_model=graph_model,
                    graph_index=i,
                    rounds=rounds,
                )

            runs_to_execute = 1
            if recovery_mode:
                graph_string = str(json_graph.node_link_data(graph))
                filtered_df = missing_run_df[
                    (missing_run_df.num_nodes == len(graph.nodes))
                    & (missing_run_df.task == args.task)
                    & (missing_run_df.graph_generator == graph_model)
                    & (missing_run_df.graph == graph_string)
                    & (missing_run_df.model_name == args.model)
                ]
                if len(filtered_df) > 0:
                    runs_to_execute = int(filtered_df.iloc[0].missing_runs)
                    if logger:
                        log_event(
                            logger,
                            logging.INFO,
                            "recovery_runs_selected",
                            graph_model=graph_model,
                            graph_index=i,
                            runs_to_execute=runs_to_execute,
                        )
                else:
                    runs_to_execute = 0
                    if logger:
                        log_event(
                            logger,
                            logging.INFO,
                            "recovery_run_skipped",
                            graph_model=graph_model,
                            graph_index=i,
                        )

            for run_index in range(runs_to_execute):
                run_id = (
                    f"{args.task}:{args.model}:{graph_model}:"
                    f"graph{i}:run{run_index}"
                )
                spec = ExperimentSpec(
                    task=args.task,
                    model=args.model,
                    graph_model=graph_model,
                    graph_index=i,
                    run_index=run_index,
                    rounds=rounds,
                    graph=graph,
                    run_id=run_id,
                )
                specs.append(spec)
                if logger:
                    log_event(
                        logger,
                        logging.INFO,
                        "experiment_queued",
                        run_id=run_id,
                        graph_model=graph_model,
                        graph_index=i,
                        run_index=run_index,
                        rounds=rounds,
                    )
    return specs


async def run_single_experiment(args, spec, commit_hash, logger):
    task_class = TASKS[spec.task]
    model_provider = resolve_model_provider(spec.model)
    chain_of_thought = not args.disable_chain_of_thought
    lmp_model = None
    answers = [None for _ in range(spec.graph.order())]
    score = None
    successful = False
    error_message = None

    def progress_callback(event, **fields):
        log_event(logger, logging.INFO, event, run_id=spec.run_id, **fields)

    log_event(
        logger,
        logging.INFO,
        "experiment_start",
        run_id=spec.run_id,
        graph_model=spec.graph_model,
        graph_index=spec.graph_index,
        run_index=spec.run_index,
        rounds=spec.rounds,
    )

    try:
        lmp_model = task_class(
            graph=spec.graph,
            rounds=spec.rounds,
            model_name=spec.model,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            run_id=spec.run_id,
            logger=logger,
            progress_callback=progress_callback,
        )
        await lmp_model.bootstrap()
        log_event(logger, logging.INFO, "bootstrap_done", run_id=spec.run_id)
        answers = await lmp_model.pass_messages()
        score = lmp_model.get_score(answers)
        successful = len(lmp_model.model_errors) == 0
        if not successful:
            error_message = f"{len(lmp_model.model_errors)} model call(s) failed; see model_errors."
    except Exception as e:
        successful = False
        error_message = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        log_event(
            logger,
            logging.WARNING,
            "experiment_exception",
            run_id=spec.run_id,
            error_type=type(e).__name__,
            error_message=str(e),
        )

    transcripts = lmp_model.get_transcripts() if lmp_model else {}
    result_graph = lmp_model.graph if lmp_model else spec.graph
    model_name = lmp_model.model_name if lmp_model else spec.model
    num_fallbacks = lmp_model.num_fallbacks if lmp_model else [0 for _ in spec.graph.nodes()]
    failed_json = (
        lmp_model.num_failed_json_parsings_after_retry
        if lmp_model
        else [0 for _ in spec.graph.nodes()]
    )
    failed_answer = (
        lmp_model.num_failed_answer_parsings_after_retry
        if lmp_model
        else [0 for _ in spec.graph.nodes()]
    )
    model_errors = lmp_model.model_errors if lmp_model else []

    try:
        filepath = save_results(
            answers=answers,
            transcripts=transcripts,
            graph=result_graph,
            rounds=spec.rounds,
            model_name=model_name,
            task=spec.task,
            score=score,
            commit_hash=commit_hash,
            graph_generator=spec.graph_model,
            graph_index=spec.graph_index,
            successful=successful,
            error_message=error_message,
            chain_of_thought=chain_of_thought,
            num_fallbacks=num_fallbacks,
            num_failed_json_parsings_after_retry=failed_json,
            num_failed_answer_parsings_after_retry=failed_answer,
            model_errors=model_errors,
            run_index=spec.run_index,
            run_id=spec.run_id,
            logger=logger,
        )
    except Exception as e:
        successful = False
        filepath = None
        save_error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        error_message = f"{error_message or ''}\nResult save failed:\n{save_error}".strip()
        log_event(
            logger,
            logging.ERROR,
            "results_save_failed",
            run_id=spec.run_id,
            error_type=type(e).__name__,
            error_message=str(e),
        )

    if successful:
        log_event(
            logger,
            logging.INFO,
            "experiment_done",
            run_id=spec.run_id,
            score=score,
            results_file=filepath,
        )
    else:
        log_event(
            logger,
            logging.ERROR,
            "experiment_failed",
            run_id=spec.run_id,
            score=score,
            results_file=filepath,
            model_error_count=len(model_errors),
        )

    return dict(
        model=spec.model,
        task=spec.task,
        rounds=spec.rounds,
        seed=args.seed,
        score=score,
        successful=successful,
        run_id=spec.run_id,
        results_file=filepath,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--task", type=str, default="coloring")
    parser.add_argument("--graph_models", type=str, nargs="+", default=["ws", "ba", "dt"])
    parser.add_argument("--start_from_sample", type=int, default=0)
    parser.add_argument("--samples_per_graph_model", type=int, default=3)
    parser.add_argument("--graph_size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_chain_of_thought", action="store_true")
    parser.add_argument("--missing_run_file", type=str, default=None)
    parser.add_argument("--max_parallel_experiments", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()
    if args.max_parallel_experiments < 1:
        parser.error("--max_parallel_experiments must be >= 1")
    asyncio.run(run(args))
