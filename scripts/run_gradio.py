"""Run the Gradio inspection UI standalone (http://127.0.0.1:7860).

    python scripts/run_gradio.py [--share]

Login with any QualityGuard account that has the 'inspect' permission
(initial accounts are written to storage/INITIAL_CREDENTIALS.txt on first start).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

from app.core.bootstrap import init_db  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.gradio_ui import authenticate, build_ui  # noqa: E402
from ml.config import CHECKPOINT_DIR  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    setup_logging()
    init_db()
    import gradio as gr
    build_ui().launch(server_name=a.host, server_port=a.port, share=a.share, auth=authenticate,
                      allowed_paths=[str(get_settings().storage_dir), str(CHECKPOINT_DIR)],
                      theme=gr.themes.Soft(primary_hue="blue"))
