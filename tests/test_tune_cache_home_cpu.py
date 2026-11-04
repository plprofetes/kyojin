"""CPU test of the per-user cache root shared by the dense-GEMM tune file writers (util/tune_cache.py home_dir(); hgemm.cu
cache_path() applies the same rule). tune_cache.py is loaded by path: it needs no torch, no GPU and no built extension."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tune_cache_under_test", ROOT / "exllamav3" / "util" / "tune_cache.py")
tune_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune_cache)


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("HOME", "USERPROFILE", "XDG_CACHE_HOME", "EXLLAMAV3_TUNE_CACHE", "EXL3_DENSE_GEMM_TUNE_FILE"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_home_wins(clean_env):
    clean_env.setenv("HOME", "/h")
    clean_env.setenv("USERPROFILE", "C:/Users/u")
    assert tune_cache.home_dir() == "/h"


def test_userprofile_when_home_is_unset(clean_env):
    clean_env.setenv("USERPROFILE", "C:/Users/u")
    assert tune_cache.home_dir() == "C:/Users/u"


def test_tmp_when_neither_is_set(clean_env):
    assert tune_cache.home_dir() == "/tmp"


def test_empty_home_is_kept_as_before(clean_env):
    # os.environ.get("HOME", "/tmp") returned "" for HOME="" before home_dir() existed; it still does
    clean_env.setenv("HOME", "")
    clean_env.setenv("USERPROFILE", "C:/Users/u")
    assert tune_cache.home_dir() == ""


def test_dense_tune_path_follows_the_rule(clean_env):
    clean_env.setenv("USERPROFILE", "C:/Users/u")
    assert tune_cache.dense_live_path() == Path("C:/Users/u") / ".cache/exllamav3/dense_gemm_tune.txt"
    clean_env.setenv("EXL3_DENSE_GEMM_TUNE_FILE", "/x/t.txt")
    assert tune_cache.dense_live_path() == Path("/x/t.txt")


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX branch")
def test_autotune_cache_path_posix(clean_env):
    clean_env.setenv("HOME", "/h")
    assert tune_cache.live_cache_path() == Path("/h/.cache/exllamav3/autotune/coop_autotune_v1.bin")
    clean_env.setenv("XDG_CACHE_HOME", "/xdg")
    assert tune_cache.live_cache_path() == Path("/xdg/exllamav3/autotune/coop_autotune_v1.bin")
