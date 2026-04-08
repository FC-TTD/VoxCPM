import os

import gradio as gr
from fastapi.responses import RedirectResponse

from api.server import app as api_app
from app import I18N, _APP_THEME, _CUSTOM_CSS, VoxCPMDemo, create_demo_interface


def _normalize_mount_path(value: str) -> str:
    path = (value or "/gradio").strip() or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    if path != "/":
        path = path.rstrip("/")
    return path


GRADIO_MOUNT_PATH = _normalize_mount_path(os.environ.get("GRADIO_MOUNT_PATH", "/gradio"))
os.environ["GRADIO_MOUNT_PATH"] = GRADIO_MOUNT_PATH

demo = VoxCPMDemo(model_dir=os.environ.get("VOXCPM_MODEL_DIR") or None)
gradio_blocks = create_demo_interface(demo)
gradio_blocks.theme = _APP_THEME
gradio_blocks.css = _CUSTOM_CSS
gradio_blocks.queue(max_size=10, default_concurrency_limit=1)


if GRADIO_MOUNT_PATH != "/":
    @api_app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url=f"{GRADIO_MOUNT_PATH}/")


app = gr.mount_gradio_app(
    api_app,
    gradio_blocks,
    path=GRADIO_MOUNT_PATH,
    show_error=True,
    i18n=I18N,
)
