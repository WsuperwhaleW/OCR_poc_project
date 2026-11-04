"""The OCR worker service: the User pages and their API on the shared engine.

Built 2026-11-04 at the user's request (worker-service plan, step 2): "แยกจาก
app.py ให้เป็นเป้าหมายหลัก ใช้ entrypoint เช่น service_app.py และ engine กลาง
ส่วน Benchmark เรียก engine เดียวกัน". Production runs this file, never app.py:
it imports `engine` (input, read, grouping, extraction, checks), `userweb` (the
/user portal and /api/user, /api/schema) and `webcommon` (sub-path mount, per-tab
workspace, gzip, JSON-body check) -- and no benchmark module, route, template or
fixture. `tests/test_service_app.py` holds it to that
(`dryreplay` and `scoring` count as engine: the read path uses them).

One process, one scheduler: the User queue starts here at boot, not on the first
request. Run it the way `python app.py` ran:

    python service_app.py            # HOST / PORT from the environment (config.py)
"""
import sys

from flask import Flask, redirect, url_for

import config
import engine
import serviceapi
import userweb
import webcommon
import workspace

# No /static: that folder is the benchmark's scripts and styles; /user is self-contained.
app = Flask(__name__, static_folder=None)
# Upload cap, sub-path mount, per-tab workspace and gzip, in the benchmark's order.
webcommon.install(app, engine.MAX_UPLOAD_MB)
get_user_service = userweb.register(app, engine)
# The Worker API for the main service (API keys) and the /user API keys tab.
serviceapi.register(app, engine, get_user_service)
# After the User routes' origin check, as in the benchmark app.
app.before_request(webcommon.validate_api_json_object)


@app.get('/')
def index():
    """The service has one page: the User portal."""
    return redirect(url_for('user_index'))


def main():
    # What every browser tab starts from; the User pipeline sets its own
    # servers and models per job from the saved settings.
    workspace.seal()
    get_user_service().start()
    try:
        from waitress import serve
    except ImportError:
        config.say(f"[service] http://{config.HOST}:{config.PORT}/user (Werkzeug, threaded)", stream=sys.stderr)
        app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
    else:
        config.say(f"[service] http://{config.HOST}:{config.PORT}/user (waitress)", stream=sys.stderr)
        serve(app, host=config.HOST, port=config.PORT, threads=32)


if __name__ == '__main__':
    main()
