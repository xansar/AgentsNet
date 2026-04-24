import os
import tempfile
import unittest

import networkx as nx

from LiteralMessagePassing import LiteralMessagePassing


class LiteralMessagePassingErrorTests(unittest.TestCase):
    def make_lmp(self):
        graph = nx.path_graph(1)
        graph.nodes[0]["name"] = "Terry"

        lmp = LiteralMessagePassing.__new__(LiteralMessagePassing)
        lmp.graph = graph
        lmp.run_id = "test/run"
        lmp.model_name = "gpt-test"
        lmp.logger = None
        lmp.model_errors = []
        return lmp

    def test_invalid_prompt_error_saves_prompt_file(self):
        lmp = self.make_lmp()
        exc = Exception("Error code: 400 - invalid_prompt")
        exc.model_request_messages = [
            {
                "type": "human",
                "data": {
                    "content": "prompt that triggered the provider rejection",
                },
            }
        ]

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                lmp._record_model_error(0, "message_passing", exc)

                self.assertEqual(len(lmp.model_errors), 1)
                prompt_file = lmp.model_errors[0]["prompt_file"]
                self.assertTrue(os.path.exists(prompt_file))
                self.assertTrue(prompt_file.startswith(os.path.join("invalid_prompts", "test_run")))
            finally:
                os.chdir(old_cwd)

    def test_non_invalid_prompt_error_does_not_save_prompt_file(self):
        lmp = self.make_lmp()
        exc = Exception("temporary network failure")
        exc.model_request_messages = [
            {
                "type": "human",
                "data": {
                    "content": "normal prompt",
                },
            }
        ]

        lmp._record_model_error(0, "message_passing", exc)

        self.assertEqual(len(lmp.model_errors), 1)
        self.assertNotIn("prompt_file", lmp.model_errors[0])


if __name__ == "__main__":
    unittest.main()
