"""In-process User adapter for compare.py; real model requests, isolated data."""
import copy
import json
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import config
import workspace


def defaults_snapshot(store, variant):
    snapshot = copy.deepcopy(store.prompt_set_snapshot())
    if variant == 'before':
        # Exactly the three missing bindings before migration 2; T02/T03/T06
        # already inherited Invoice/Receipt. No live team settings are changed.
        for code in ('TAX_INVOICE','DELIVERY_TAX_INVOICE','ABBREVIATED_TAX_INVOICE'):
            snapshot['bindings'].pop(code,None)
    return snapshot


class Reply:
    def __init__(self, response):
        try:
            self.status_code = response.status_code
            self.ok = 200 <= response.status_code < 400
            self.text = response.get_data(as_text=True)
            self.value = response.get_json(silent=True)
        finally:
            response.close()

    def json(self):
        return self.value


class LocalApp:
    """Only compare's HTTP calls use this adapter; model HTTP remains real."""
    def __init__(self, application, ws):
        self.application, self.ws = application, ws

    def request(self, method, url, **kwargs):
        parsed = urlsplit(url)
        path = parsed.path + ('?' + parsed.query if parsed.query else '')
        options = {'headers':{workspace.HEADER:self.ws.id}}
        if 'json' in kwargs:
            options['json'] = kwargs['json']
        if kwargs.get('files'):
            options['data'] = dict(kwargs.get('data') or {})
            for key, (name, stream) in kwargs['files'].items():
                options['data'][key] = (stream,name)
        elif 'data' in kwargs:
            options['data'] = kwargs['data']
        with workspace.use(self.ws):
            return Reply(self.application.test_client().open(path,method=method,**options))

    def get(self, url, **kwargs):
        return self.request('GET',url,**kwargs)

    def post(self, url, **kwargs):
        return self.request('POST',url,**kwargs)


def run(args, compare):
    import app as engine
    from userstore import UserStore
    variant = args.user_prompt_defaults
    output = config.BASE_DIR / '.work' / 'compare-user-prompts' / variant
    output.mkdir(parents=True,exist_ok=True)
    old_requests, old_output, old_log = compare.requests, compare.scoring.OUT, engine.log_extract
    original_args = args.app, args.mode
    try:
        with tempfile.TemporaryDirectory(dir=output) as directory, workspace.use(None):
            store = UserStore(directory,engine.MAX_UPLOAD_MB)
            snapshot = defaults_snapshot(store,variant)
            (output / 'prompt-snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
            ws = workspace.enter('compare_user_'+uuid.uuid4().hex)
            ws.put('user_pipeline',True)
            ws.put('user_prompt_sets',snapshot)
            ws.put('user_prompt_profiles',{})
            compare.requests = LocalApp(engine.app,ws)
            compare.scoring.OUT = output
            engine.log_extract = lambda *a,**kw: None  # compare saves the full result here instead of team history
            args.app, args.mode = 'http://local-user-compare', 'hybrid'
            compare.say('User Hybrid defaults: '+variant+' · output '+str(output))
            compare.say('Field scopes differ: compare common field values, errors, request counts and timing; a smaller form is not proof of better accuracy.')
            return compare.run(args)
    finally:
        compare.requests, compare.scoring.OUT, engine.log_extract = old_requests, old_output, old_log
        args.app, args.mode = original_args
