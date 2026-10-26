"""The deploy archive must contain the live engine and complete inline User UI."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import package


class DeploymentPackageTests(unittest.TestCase):
    def test_every_local_python_dependency_is_included_without_logs_or_private_files(self):
        picked = {name.as_posix() for _, name in package.collect(no_fixtures=True)}
        for name in package.FILES:
            if not name.endswith('.py'):
                continue
            tree = ast.parse((package.BASE / name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                modules = ([item.name for item in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for module in modules:
                    local = module.split('.')[0] + '.py'
                    if (package.BASE / local).is_file():
                        self.assertIn(local, picked, f'{name} imports missing module {local}')
        for name in ('templates/_user_style.html', 'templates/_user_app_script.html',
                     'templates/_user_settings_script.html', 'templates/_workspace_session.html'):
            self.assertIn(name, picked)
        self.assertFalse(any(name.startswith(('logs/', '.work/', 'outputs/')) for name in picked))
        self.assertNotIn('.env', picked)

    def test_extracted_upload_only_archive_serves_user_without_a_static_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'app.zip'
            subprocess.run([sys.executable, 'package.py', '--no-fixtures', '--out', str(archive)],
                           cwd=package.BASE, check=True, capture_output=True, text=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(Path(directory) / 'unpacked')
            root = next((Path(directory) / 'unpacked').iterdir())
            self.assertFalse((root / 'static').exists())
            code = '''
from unittest.mock import patch
from pathlib import Path
with patch('requests.sessions.Session.request', side_effect=AssertionError('No network in deployment smoke test')):
    import app
    assert Path(app.__file__).resolve().parent == Path.cwd()
    client = app.app.test_client()
    for path in ('/user', '/ocr/user'):
        response = client.get(path)
        assert response.status_code == 200, response.status_code
        text = response.get_data(as_text=True)
        assert '.workspace{display:grid' in text
        assert 'async function loadSettings()' in text
        assert '/static/user/' not in text
        assert 'Whale' not in text
    print('archive-user-page: 200, embedded CSS/JS')
'''
            env = dict(os.environ, PYTHONPATH='')
            result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('archive-user-page: 200', result.stdout)


if __name__ == '__main__':
    unittest.main()
