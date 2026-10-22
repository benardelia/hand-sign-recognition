"""API tests. Run from the repo root: python -m unittest discover tests"""
import base64
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_cwd = None
_tmp = None
app_module = None


def setUpModule():
    # Data/ and Model/ are relative to the working directory: isolate them.
    global _cwd, _tmp, app_module
    _cwd = os.getcwd()
    _tmp = tempfile.mkdtemp()
    os.chdir(_tmp)
    os.environ['ADMIN_TOKEN'] = 'secret-admin'
    os.environ.pop('COLLECT_TOKEN', None)
    os.environ.pop('LOCAL_CAMERA', None)
    import app as app_module_  # noqa: E402
    app_module = app_module_


def tearDownModule():
    os.chdir(_cwd)
    shutil.rmtree(_tmp, ignore_errors=True)


def sample(t=20, seed=0):
    return np.random.default_rng(seed).uniform(0.1, 0.9, (t, 1662)).astype('<f4')


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def upload(self, label='HELLO', seq=None, **kw):
        seq = sample() if seq is None else seq
        body = {"label": label, "frames_b64": base64.b64encode(seq.tobytes()).decode(),
                "meta": {"source": "test", "fps": 20}}
        body.update(kw)
        return self.client.post('/api/samples', json=body)

    def test_config(self):
        cfg = self.client.get('/api/config').get_json()
        self.assertEqual(set(cfg['profiles']), {'low', 'high', 'experimental'})
        self.assertFalse(cfg['local_camera'])
        self.assertIn('holistic_landmarker', cfg['mediapipe']['model'])

    def test_upload_list_play_delete(self):
        r = self.upload('thank you')
        self.assertEqual(r.status_code, 200, r.get_json())
        path = r.get_json()['path']
        self.assertTrue(path.endswith('.npy') and os.path.exists(path))
        self.assertTrue(os.path.exists(path[:-4] + '.json'))

        signs = self.client.get('/api/signs').get_json()['signs']
        entry = next(s for s in signs if s['name'] == 'THANK YOU')
        self.assertEqual(entry['sources'], {'test': 1})

        data = self.client.get('/api/npy_data', query_string={'path': path}).get_json()
        self.assertEqual(len(data['frames']), 20)
        self.assertEqual(data['meta']['source'], 'test')

        r = self.client.post('/api/delete_file', json={'path': path})
        self.assertEqual(r.status_code, 401)
        r = self.client.post('/api/delete_file', json={'path': path}, headers={'X-Handsign-Token': 'secret-admin'})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(path) or os.path.exists(path[:-4] + '.json'))

    def test_rejects_bad_samples(self):
        self.assertEqual(self.upload('../evil').status_code, 400)
        self.assertEqual(self.upload('OK', seq=sample(t=3)).status_code, 400)
        no_hands = sample()
        no_hands[:, 1536:] = 0
        self.assertEqual(self.upload('OK', seq=no_hands).status_code, 400)
        r = self.client.post('/api/samples', json={"label": "OK", "frames_b64": base64.b64encode(b'123').decode()})
        self.assertEqual(r.status_code, 400)

    def test_path_traversal_blocked(self):
        r = self.client.get('/api/npy_data', query_string={'path': '/etc/passwd'})
        self.assertEqual(r.status_code, 403)
        r = self.client.post('/api/delete_sign', json={'label': '..'}, headers={'X-Handsign-Token': 'secret-admin'})
        self.assertEqual(r.status_code, 403)

    def test_stream_without_model(self):
        frames = sample(t=3).tobytes()
        r = self.client.post('/api/stream/frames', data=frames, headers={'X-Session-Id': 'session-123456'})
        self.assertEqual(r.status_code, 200, r.get_json())
        body = r.get_json()
        self.assertFalse(body['model_loaded'])
        self.assertEqual(body['profile'], 'high')
        r = self.client.post('/api/stream/frames', data=b'xx', headers={'X-Session-Id': 'session-123456'})
        self.assertEqual(r.status_code, 400)
        r = self.client.post('/api/stream/frames', data=frames)
        self.assertEqual(r.status_code, 400)

    def test_models_and_predict_without_model(self):
        self.assertEqual(self.client.get('/api/models').get_json()['profiles'], {})
        self.assertEqual(self.client.get('/api/models/high/model.tflite').status_code, 404)
        self.assertEqual(self.client.get('/api/models/high/../../app.py').status_code, 404)
        r = self.client.post('/api/predict', json={"profile": "low", "frames": sample(t=15).tolist()})
        self.assertEqual(r.status_code, 404)

    def test_local_routes_disabled(self):
        self.assertEqual(self.client.get('/api/status').status_code, 404)
        self.assertEqual(self.client.get('/api/cameras').status_code, 404)

    def test_train_requires_token(self):
        self.assertEqual(self.client.post('/api/train', json={'profile': 'low'}).status_code, 401)
        r = self.client.post('/api/train', json={'profile': 'nope'}, headers={'X-Handsign-Token': 'secret-admin'})
        self.assertEqual(r.status_code, 400)


if __name__ == '__main__':
    unittest.main()
