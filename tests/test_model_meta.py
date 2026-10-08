"""Quantisation and inference-engine version, collected for the run log.

What is asserted: the quant comes from what the server says before what the
name says; the engine is the server kind with its version; nothing is guessed
where nothing says; and the four columns land on the row for the right pass.
"""

import unittest
from unittest import mock

import backends
import runlog

URL = "http://meta-test:9100"
EX_URL = "http://meta-extract:9200"


class _Res:
    def __init__(self, code, body=None):
        self.status_code = code
        self._body = body

    def json(self):
        return self._body


class QuantOfTests(unittest.TestCase):
    def test_gguf_types_in_a_file_name(self):
        self.assertEqual(backends.quant_of("typhoon-ocr-1.5-2b-Q8_0.gguf"), "Q8_0")
        self.assertEqual(backends.quant_of("/m/qwen3.6-35b-a3b-IQ3_XXS.gguf"), "IQ3_XXS")
        self.assertEqual(backends.quant_of("model.Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(backends.quant_of("hf.co/x/dots.ocr-GGUF:IQ4_XS"), "IQ4_XS")
        self.assertEqual(backends.quant_of("mmproj-model-f16.gguf"), "F16")

    def test_vllm_formats_in_a_model_id(self):
        self.assertEqual(backends.quant_of("Qwen/Qwen2.5-VL-7B-Instruct-AWQ"), "AWQ")
        self.assertEqual(backends.quant_of("RedHatAI/Qwen3-8B-FP8-dynamic"), "FP8")
        self.assertEqual(backends.quant_of("org/model-W4A16"), "W4A16")

    def test_a_name_that_says_nothing_is_blank(self):
        for name in ("qwen3.5:9b", "scb10x/typhoon-ocr1.5-3b:latest",
                     "gemma4:e4b", "", None, "marq4ket"):
            self.assertEqual(backends.quant_of(name), "", name)

    def test_the_last_match_wins(self):
        self.assertEqual(backends.quant_of("Q4_0-finetune-Q8_0.gguf"), "Q8_0")


class ModelMetaTests(unittest.TestCase):
    def setUp(self):
        backends._versions.clear()

    def _with(self, info, get=None):
        patches = [mock.patch.object(backends, "known", return_value=info)]
        if get is not None:
            patches.append(mock.patch.object(backends.requests, "get", get))
        return patches

    def test_ollama_states_the_quant_and_its_version(self):
        info = {"kind": "ollama", "reachable": True,
                "models": [{"name": "gemma4:e4b", "quant": "Q4_K_M"}]}
        asked = []

        def get(url, timeout=None):
            asked.append(url)
            return _Res(200, {"version": "0.32.14"})
        with mock.patch.object(backends, "known", return_value=info), \
                mock.patch.object(backends.requests, "get", get):
            first = backends.model_meta(URL, "gemma4:e4b")
            second = backends.model_meta(URL, "gemma4:e4b")
        self.assertEqual(first, {"quant": "Q4_K_M", "engine": "ollama 0.32.14"})
        self.assertEqual(second, first)
        # Asked once, then cached: not a poll.
        self.assertEqual(asked, [f"{URL}/api/version"])

    def test_llama_cpp_reads_the_probe_and_asks_nothing(self):
        info = {"kind": "llama.cpp", "reachable": True, "build": "b1-67a17c1",
                "models": [{"name": "typhoon", "quant": "Q8_0"}]}

        def get(*a, **k):
            raise AssertionError("llama.cpp must not be asked anything")
        with mock.patch.object(backends, "known", return_value=info), \
                mock.patch.object(backends.requests, "get", get):
            meta = backends.model_meta(URL, "typhoon")
        self.assertEqual(meta, {"quant": "Q8_0", "engine": "llama.cpp b1-67a17c1"})

    def test_a_server_with_no_version_is_its_kind_alone(self):
        info = {"kind": "openai", "reachable": True,
                "models": [{"name": "some-model", "quant": ""}]}

        def get(url, timeout=None):
            return _Res(404)
        with mock.patch.object(backends, "known", return_value=info), \
                mock.patch.object(backends, "_pooled", return_value=False), \
                mock.patch.object(backends.requests, "get", get):
            meta = backends.model_meta(URL, "some-model")
        self.assertEqual(meta, {"quant": "", "engine": "openai"})

    def test_an_unreachable_server_is_not_asked_its_version(self):
        info = {"kind": None, "reachable": False, "models": []}
        with mock.patch.object(backends, "known", return_value=info):
            self.assertEqual(backends.model_meta(URL, "x"), {"quant": "", "engine": ""})

    def test_it_never_raises(self):
        with mock.patch.object(backends, "known", side_effect=RuntimeError("boom")):
            self.assertEqual(backends.model_meta(URL, "x"), {"quant": "", "engine": ""})


class RowTests(unittest.TestCase):
    def test_each_pass_is_described_by_its_own_model(self):
        import app

        def meta(url, model):
            return {"quant": {"typhoon": "Q8_0", "qwen": "Q4_K_M"}[model],
                    "engine": {URL: "llama.cpp b1", EX_URL: "vllm 0.11.0"}[url]}
        summary = {"model": "typhoon", "url": URL,
                   "extracted": {"model": "qwen", "url": EX_URL, "fields": {}}}
        with mock.patch.object(backends, "model_meta", side_effect=meta):
            app._stamp_model_meta(summary)
        self.assertEqual((summary["quant"], summary["engine"]), ("Q8_0", "llama.cpp b1"))
        cells = runlog._extract_cells(summary)
        self.assertEqual(cells["extract_quant"], "Q4_K_M")
        self.assertEqual(cells["extract_inference_engine"], "vllm 0.11.0")

    def test_blank_where_pass_2_never_ran(self):
        cells = runlog._extract_cells({"model": "typhoon"})
        self.assertEqual(cells["extract_quant"], "")
        self.assertEqual(cells["extract_inference_engine"], "")

    def test_the_columns_are_appended_and_re_extracted(self):
        tail = runlog.COLUMNS[-4:]
        self.assertEqual(tail, ["model_quant", "inference_engine",
                                "extract_quant", "extract_inference_engine"])
        self.assertIn("extract_quant", runlog.EXTRACT_COLUMNS)
        self.assertIn("extract_inference_engine", runlog.EXTRACT_COLUMNS)


class SummaryTests(unittest.TestCase):
    def test_a_group_names_what_its_runs_ran_on_newest_first(self):
        runs = [{"model_quant": "Q8_0", "inference_engine": "ollama 0.33.0"},
                {"model_quant": "Q4_K_M", "inference_engine": "ollama 0.32.14"},
                {"model_quant": "Q8_0", "inference_engine": ""},
                {}]
        self.assertEqual(runlog._built(runs, *runlog._READ_BUILT),
                         {"quant": "Q8_0 / Q4_K_M",
                          "engine": "ollama 0.33.0 / ollama 0.32.14"})

    def test_blank_where_no_run_says(self):
        self.assertEqual(runlog._built([{}, {"model_quant": ""}], *runlog._READ_BUILT),
                         {"quant": "", "engine": ""})

    def test_the_tables_carry_it_without_splitting_a_setting(self):
        base = {"run_type": "ocr", "status": "ok", "model": "m", "backend": "ollama",
                "detail": "low", "ocr_profile": "typhoon", "case": "sol002",
                "char_accuracy": "90", "seconds": "10"}
        rows = [{**base, "timestamp": "2026-10-08T10:00:00",
                 "model_quant": "Q4_K_M", "inference_engine": "ollama 0.32.14"},
                {**base, "timestamp": "2026-10-01T10:00:00"}]
        table = runlog.by_ocr(rows)
        self.assertEqual(len(table), 1)
        self.assertEqual((table[0]["quant"], table[0]["engine"]),
                         ("Q4_K_M", "ollama 0.32.14"))


class FilterTests(unittest.TestCase):
    ROWS = [
        {"run_type": "ocr", "model": "typhoon", "model_quant": "Q8_0",
         "inference_engine": "llama.cpp b1", "extract_mode": "single",
         "p1_present": "9", "extract_quant": "Q4_K_M",
         "extract_inference_engine": "vllm 0.11.0"},
        {"run_type": "ocr", "model": "typhoon", "model_quant": "Q4_K_M",
         "inference_engine": "ollama 0.32.14"},
        # A fields-only row: its model_quant is the EXTRACTOR's.
        {"run_type": "extract", "model": "qwen", "model_quant": "Q4_K_M",
         "extract_quant": "Q4_K_M", "extract_mode": "agentic", "p1_present": "9"},
        {"run_type": "ocr", "model": "typhoon"},
    ]

    def test_each_pass_answers_only_for_itself(self):
        f = runlog.FILTER_FIELDS
        self.assertEqual([f["model_quant"](r) for r in self.ROWS],
                         ["Q8_0", "Q4_K_M", "", ""])
        self.assertEqual([f["extract_quant"](r) for r in self.ROWS],
                         ["Q4_K_M", "", "Q4_K_M", ""])
        self.assertEqual([f["inference_engine"](r) for r in self.ROWS],
                         ["llama.cpp b1", "ollama 0.32.14", "", ""])

    def test_facets_offer_them_and_a_filter_keeps_them(self):
        facets = runlog.facets(self.ROWS)
        self.assertEqual({v["value"] for v in facets["extract_inference_engine"]},
                         {"vllm 0.11.0"})
        kept = runlog.filter_rows(self.ROWS, include={"model_quant": ["Q8_0"]})
        self.assertEqual(len(kept), 1)
        dropped = runlog.filter_rows(self.ROWS, exclude={"inference_engine": ["ollama 0.32.14"]})
        self.assertEqual(len(dropped), 3)


if __name__ == "__main__":
    unittest.main()
