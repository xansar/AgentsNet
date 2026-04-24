import asyncio
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import networkx as nx

import main


def make_args(max_parallel_experiments=1, samples_per_graph_model=3):
    return SimpleNamespace(
        model="gpt-4o-mini",
        task="coloring",
        graph_models=["ws"],
        start_from_sample=0,
        samples_per_graph_model=samples_per_graph_model,
        graph_size=3,
        rounds=1,
        seed=42,
        disable_chain_of_thought=True,
        missing_run_file=None,
        max_parallel_experiments=max_parallel_experiments,
        log_level="ERROR",
        log_file=None,
        output_dir="results",
    )


def fake_graph(_graph_model, _graph_size, num_sample, logger=None):
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["name"] = f"agent_{num_sample}_{node}"
    return graph


class FakeTask:
    active = 0
    max_active = 0
    starts = []
    fail_run_ids = set()

    def __init__(
        self,
        graph,
        rounds,
        model_name,
        model_provider,
        chain_of_thought,
        run_id=None,
        logger=None,
        progress_callback=None,
    ):
        self.graph = graph
        self.rounds = rounds
        self.model_name = model_name
        self.model_errors = []
        self.num_fallbacks = [0 for _ in graph.nodes]
        self.num_failed_json_parsings_after_retry = [0 for _ in graph.nodes]
        self.num_failed_answer_parsings_after_retry = [0 for _ in graph.nodes]
        self.run_id = run_id
        self.progress_callback = progress_callback

    async def bootstrap(self):
        FakeTask.active += 1
        FakeTask.max_active = max(FakeTask.max_active, FakeTask.active)
        FakeTask.starts.append(self.run_id)
        await asyncio.sleep(0.05)

    async def pass_messages(self):
        try:
            if self.progress_callback:
                self.progress_callback("round_start", round=1, rounds=self.rounds)
            await asyncio.sleep(0.05)
            if self.run_id in FakeTask.fail_run_ids:
                raise RuntimeError("planned failure")
            if self.progress_callback:
                self.progress_callback("round_done", round=1, rounds=self.rounds)
            return ["Group 1" for _ in self.graph.nodes]
        finally:
            FakeTask.active -= 1

    def get_score(self, answers):
        return 1.0

    def get_transcripts(self):
        return {}


class ParallelExperimentTests(unittest.TestCase):
    def setUp(self):
        FakeTask.active = 0
        FakeTask.max_active = 0
        FakeTask.starts = []
        FakeTask.fail_run_ids = set()
        self.patches = [
            mock.patch.object(main, "get_graph", fake_graph),
            mock.patch.dict(main.TASKS, {"coloring": FakeTask}),
            mock.patch.object(main, "get_git_commit_hash", lambda: "test-hash"),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def test_max_parallel_experiments_one(self):
        saved = []
        with mock.patch.object(
            main,
            "save_results",
            lambda **kwargs: saved.append(kwargs) or "result.json",
        ):
            results = asyncio.run(main.run(make_args(max_parallel_experiments=1)))

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["successful"] for result in results))
        self.assertEqual(FakeTask.max_active, 1)
        self.assertEqual(len(saved), 3)
        self.assertTrue(all(item["output_dir"] == "results" for item in saved))

    def test_max_parallel_experiments_two(self):
        with mock.patch.object(main, "save_results", lambda **kwargs: "result.json"):
            results = asyncio.run(main.run(make_args(max_parallel_experiments=2)))

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["successful"] for result in results))
        self.assertEqual(FakeTask.max_active, 2)
        self.assertEqual(len(FakeTask.starts), 3)

    def test_failed_experiment_does_not_stop_others(self):
        saved = []
        FakeTask.fail_run_ids = {"coloring:gpt-4o-mini:ws:graph1:run0"}
        with mock.patch.object(
            main,
            "save_results",
            lambda **kwargs: saved.append(kwargs) or "result.json",
        ):
            results = asyncio.run(main.run(make_args(max_parallel_experiments=2)))

        self.assertEqual(len(results), 3)
        self.assertEqual(sum(result["successful"] for result in results), 2)
        self.assertEqual(sum(not result["successful"] for result in results), 1)
        self.assertEqual(len(saved), 3)

    def test_parallel_save_filenames_are_unique(self):
        graph = fake_graph("ws", 3, 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [
                main.save_results(
                    answers=["Group 1"] * 3,
                    transcripts={},
                    graph=graph,
                    rounds=1,
                    model_name="gpt-4o-mini",
                    task="coloring",
                    score=1.0,
                    commit_hash="test-hash",
                    graph_generator="ws",
                    graph_index=0,
                    successful=True,
                    error_message=None,
                    chain_of_thought=False,
                    num_fallbacks=[0, 0, 0],
                    num_failed_json_parsings_after_retry=[0, 0, 0],
                    num_failed_answer_parsings_after_retry=[0, 0, 0],
                    run_index=run_index,
                    run_id=f"run-{run_index}",
                    output_dir=tmpdir,
                )
                for run_index in range(2)
            ]

        self.assertEqual(len(set(paths)), 2)
        self.assertTrue(
            all(f"_graph0_run{run_index}_" in paths[run_index] for run_index in range(2))
        )
        self.assertTrue(
            all(os.path.dirname(path) == os.path.join(tmpdir, "coloring", "ws") for path in paths)
        )

    def test_save_results_uses_task_and_graph_subdirectories_under_output_dir(self):
        graph = fake_graph("ws", 3, 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = main.save_results(
                answers=["Group 1"] * 3,
                transcripts={},
                graph=graph,
                rounds=1,
                model_name="gpt-4o-mini",
                task="coloring",
                score=1.0,
                commit_hash="test-hash",
                graph_generator="ws",
                graph_index=0,
                successful=True,
                error_message=None,
                chain_of_thought=False,
                num_fallbacks=[0, 0, 0],
                num_failed_json_parsings_after_retry=[0, 0, 0],
                num_failed_answer_parsings_after_retry=[0, 0, 0],
                output_dir=tmpdir,
            )

            self.assertTrue(path.startswith(os.path.join(tmpdir, "coloring", "ws", "")))
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
