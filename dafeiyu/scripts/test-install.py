"""Regression checks for safe repeated installation and character generation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import install
import run as launcher


class InstallTests(unittest.TestCase):
    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                install.inside(Path(folder), '../outside')

    def test_overlay_is_repeatable_and_preserves_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original_root = install.ROOT
            install.ROOT = root
            try:
                (root / 'overlay').mkdir()
                repo = root / 'repo'
                repo.mkdir()
                (root / 'overlay/one.txt').write_bytes(b'new')
                (repo / 'one.txt').write_bytes(b'old')
                item = {'path': 'one.txt', 'sha256': hashlib.sha256(b'new').hexdigest(),
                        'base_sha256': hashlib.sha256(b'old').hexdigest()}
                manifest = {'overlay': [item]}
                install.apply_overlay(repo, manifest)
                install.apply_overlay(repo, manifest)
                self.assertEqual((repo / 'one.txt').read_bytes(), b'new')
                (repo / 'one.txt').write_bytes(b'user edit')
                with self.assertRaises(RuntimeError):
                    install.apply_overlay(repo, manifest)
                self.assertEqual((repo / 'one.txt').read_bytes(), b'user edit')
            finally:
                install.ROOT = original_root

    def test_overlay_checks_every_file_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original_root = install.ROOT
            install.ROOT = root
            try:
                (root / 'overlay').mkdir()
                repo = root / 'repo'
                repo.mkdir()
                items = []
                for name in ('one.txt', 'two.txt'):
                    (root / 'overlay' / name).write_bytes(b'new')
                    (repo / name).write_bytes(b'old')
                    items.append({'path': name, 'sha256': hashlib.sha256(b'new').hexdigest(),
                                  'base_sha256': hashlib.sha256(b'old').hexdigest()})
                (repo / 'two.txt').write_bytes(b'user edit')
                with self.assertRaises(RuntimeError):
                    install.apply_overlay(repo, {'overlay': items})
                self.assertEqual((repo / 'one.txt').read_bytes(), b'old')
            finally:
                install.ROOT = original_root

    def test_card_voice_modes_and_provider_inheritance(self):
        original_runtime = install.RUNTIME
        with tempfile.TemporaryDirectory() as folder:
            install.RUNTIME = Path(folder)
            try:
                for mode, expected in [('edge', 'edge-tts'), ('gptsovits', 'gpt-sovits')]:
                    with zipfile.ZipFile(install.make_card(mode)) as package:
                        card = json.loads(package.read('card.json'))
                        modules = card['data']['extensions']['airi']['modules']
                        self.assertEqual(modules['speech']['model'], expected)
                        self.assertEqual(modules['consciousness'], {'provider': '', 'model': ''})
                        self.assertIn('models/body-model.zip', package.namelist())
            finally:
                install.RUNTIME = original_runtime

    def test_busy_foreign_port_never_starts_or_stops_processes(self):
        with patch.object(launcher, 'owned', return_value=None), \
                patch.object(launcher, 'port_busy', return_value=True), \
                patch.object(launcher.subprocess, 'Popen') as popen:
            with self.assertRaises(RuntimeError):
                launcher.spawn('web', ['node', 'server.js'], Path.cwd(), port=5173)
            popen.assert_not_called()

    def test_process_id_reuse_is_not_owned(self):
        original_runtime = launcher.RUNTIME
        with tempfile.TemporaryDirectory() as folder:
            launcher.RUNTIME = Path(folder)
            (launcher.RUNTIME / 'desktop.process.json').write_text(json.dumps({
                'pid': 123, 'created': 1.0, 'command': ['electron', 'app']}))
            fake_process = SimpleNamespace(create_time=lambda: 2.0, cmdline=lambda: ['electron', 'app'])
            fake_psutil = SimpleNamespace(Process=lambda pid: fake_process, NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)
            try:
                with patch.object(launcher, 'process_module', return_value=fake_psutil):
                    self.assertIsNone(launcher.owned('desktop'))
            finally:
                launcher.RUNTIME = original_runtime

    def test_changed_process_command_is_not_owned(self):
        original_runtime = launcher.RUNTIME
        with tempfile.TemporaryDirectory() as folder:
            launcher.RUNTIME = Path(folder)
            (launcher.RUNTIME / 'desktop.process.json').write_text(json.dumps({
                'pid': 123, 'created': 1.0, 'command': ['electron', 'app']}))
            fake_process = SimpleNamespace(create_time=lambda: 1.0, cmdline=lambda: ['electron', 'other-app'])
            fake_psutil = SimpleNamespace(Process=lambda pid: fake_process, NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)
            try:
                with patch.object(launcher, 'process_module', return_value=fake_psutil):
                    self.assertIsNone(launcher.owned('desktop'))
            finally:
                launcher.RUNTIME = original_runtime


if __name__ == '__main__':
    unittest.main()
