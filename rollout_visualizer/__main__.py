"""Run with python -m rollout_visualizer PATH."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .data import read_document

STATIC = Path(__file__).with_name('static')


def make_handler(directory):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == '/api/files':
                    files = sorted((p.name for p in directory.glob('*.jsonl') if p.is_file()), key=lambda n: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', n)])
                    return self.send_json({'directory': str(directory), 'files': files})
                if url.path == '/api/file':
                    name = parse_qs(url.query).get('name', [''])[0]
                    path = (directory / name).resolve()
                    if path.parent != directory or path.suffix != '.jsonl':
                        return self.send_json({'error': 'Invalid file name'}, 400)
                    return self.send_json(read_document(path))
                name = {'/': 'index.html', '/app.js': 'app.js', '/decision_model.js': 'decision_model.js', '/style.css': 'style.css'}.get(url.path)
                if name is None:
                    return self.send_json({'error': 'Not found'}, 404)
                mime = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[name.rsplit('.', 1)[-1]]
                self.send_body((STATIC / name).read_bytes(), mime)
            except (OSError, ValueError, TypeError) as exc:
                self.send_json({'error': str(exc)}, 400)

        def send_json(self, value, status=200):
            self.send_body(json.dumps(value).encode(), 'application/json', status)

        def send_body(self, body, mime, status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Directory containing one JSONL file per training step')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f'Not a directory: {directory}')
    server = ThreadingHTTPServer((args.host, args.port), make_handler(directory))
    print(f'Rollout visualizer: http://{args.host}:{server.server_port}\nReading {directory}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
