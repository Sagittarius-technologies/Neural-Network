"""
Small utility to load and inspect .joblib files in the `model_output` folder.

Usage (PowerShell):
    python "./load_joblibs.py"

This script prints the filename, size and a short summary of the loaded object.
If loading fails, it prints the exception and full traceback so you can see why.

Note: .joblib files are binary pickle-like files. They are not readable in a text
editor. Loading them requires Python and the same (or compatible) libraries used
to create them (e.g. scikit-learn, joblib).
"""
import os
import sys
import traceback
from pathlib import Path
import importlib
import inspect

import clust_nn_pipeline_sklearn  # makes KmerVectorizer available
from joblib import load

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model_output"

if not MODEL_DIR.exists():
    print(f"No model_output folder found at: {MODEL_DIR}")
    sys.exit(1)

files = sorted(MODEL_DIR.glob("*.joblib"))
if not files:
    print(f"No .joblib files found in {MODEL_DIR}")
    sys.exit(0)

def short(x, n=200):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."

for p in files:
    print("\n===", p.name)
    # If a custom class was defined in this repo and pickled (e.g. KmerVectorizer),
    # ensure the class is available in the unpickling namespace. Many scripts
    # define classes in __main__ when pickled which makes unpickling fail when
    # running from a different script. Try to import the local pipeline module
    # and inject its classes into __main__ so pickle can find them.
    try:
        repo_module_name = "clust_nn_pipeline_sklearn"
        repo_path = ROOT / (repo_module_name + ".py")
        if repo_path.exists():
            spec = importlib.util.spec_from_file_location(repo_module_name, str(repo_path))
            repo_mod = importlib.util.module_from_spec(spec)
            sys.modules[repo_module_name] = repo_mod
            spec.loader.exec_module(repo_mod)
            # inject classes into __main__ for pickle compatibility
            main_mod = sys.modules.get("__main__")
            for name, obj in inspect.getmembers(repo_mod, inspect.isclass):
                # avoid overwriting important names
                if not hasattr(main_mod, name):
                    setattr(main_mod, name, obj)
    except Exception:
        # non-fatal; proceed to load and show any error
        pass

    try:
        size = p.stat().st_size
        print(f"size: {size} bytes")
    except Exception:
        print("size: (could not stat file)")

    try:
        obj = load(p)
        print("loaded:", type(obj))

        # common SKLearn-ish objects
        if hasattr(obj, "get_params"):
            print("sklearn estimator/transformer detected -> class:", obj.__class__.__module__ + "." + obj.__class__.__name__)
            try:
                params = obj.get_params()
                print("get_params keys (showing first 10):", list(params.keys())[:10])
            except Exception as e:
                print("get_params() raised:", e)

        elif isinstance(obj, dict):
            print("dict with keys:", list(obj.keys()))
        elif isinstance(obj, (list, tuple)):
            print("sequence len:", len(obj))
        elif hasattr(obj, "__dict__"):
            print("object attrs:", list(vars(obj).keys())[:20])
        else:
            print("repr:", short(obj))

    except Exception as e:
        print("ERROR loading file:", e)
        traceback.print_exc()

print("\nDone.")
