"""One Hub GPU runtime with the existing /generate API and /gradio native UI."""
import json
import os
import sys
from fastapi.responses import RedirectResponse
from .adapter import load_model, completion, release, cleanup
from .api import build_api


def create_app(runtime=None):
    import gradio as gr
    from ttd_model_runtime import Runtime
    from ttd_model_runtime.integrations.fastapi import attach
    from .ui import VoxCPMDemo, create_demo_interface, I18N, _APP_THEME, _CUSTOM_CSS
    runtime = runtime or Runtime(load_model, completion=completion, release=release, cleanup=cleanup)
    app = build_api(runtime)
    # Native production contract uses /gradio, with / redirecting there.
    mount_path = (os.getenv("GRADIO_MOUNT_PATH", "/gradio").strip() or "/gradio")
    if not mount_path.startswith("/"):
        mount_path = "/" + mount_path
    mount_path = mount_path.rstrip("/") or "/"
    os.environ["GRADIO_MOUNT_PATH"] = mount_path
    if mount_path != "/":
        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(mount_path + "/")
    demo = create_demo_interface(VoxCPMDemo(runtime))
    demo.queue(max_size=10, default_concurrency_limit=1)
    # Gradio 6 mounting sets theme/css itself; passing original options here
    # preserves I18N and styling without extending or monkeypatching the SDK.
    gr.mount_gradio_app(app, demo, path=mount_path, show_error=True,
                        i18n=I18N, theme=_APP_THEME, css=_CUSTOM_CSS)
    @app.middleware("http")
    async def native_proxy_headers(request, call_next):
        if request.url.path == mount_path or request.url.path.startswith(mount_path + "/"):
            request.scope["headers"] = [(name, value) for name, value in request.scope["headers"]
                                         if name.lower() != b"x-forwarded-host"]
        return await call_next(request)
    # The UI is already mounted with its native lifecycle. SDK wraps that
    # lifecycle and starts control before entering the Gradio app.
    return attach(app, runtime=runtime)


def describe():
    class Contract:
        def task(self, fn): return fn
    return build_api(Contract()).openapi()


def main():
    if sys.argv[1:] == ["describe"]:
        print(json.dumps(describe(), ensure_ascii=False));return
    if sys.argv[1:]: raise SystemExit("Use python -m hub_runtime [describe]")
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__": main()
