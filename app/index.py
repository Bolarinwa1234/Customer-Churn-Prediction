import sys
import traceback
from pathlib import Path
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    report = {}

    for name in ["pydantic", "numpy", "pandas", "joblib", "sklearn"]:
        try:
            mod = __import__(name)
            report[name] = getattr(mod, "__version__", "ok")
        except Exception:
            report[name] = traceback.format_exc()[-400:]

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import schemas
        report["schemas"] = "ok"
    except Exception:
        report["schemas"] = traceback.format_exc()[-400:]

    models = Path(__file__).resolve().parent.parent / "models"
    report["models_folder_exists"] = models.exists()
    report["model_files"] = [p.name for p in models.glob("*")] if models.exists() else []

    return report