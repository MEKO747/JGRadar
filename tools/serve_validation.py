#!/usr/bin/env python3
"""本机验证报告服务，支持视频 HTTP Range，便于直接跳转异常片段。"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import re
from pathlib import Path


class VideoHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def send_head(self):
        self.range_remaining = None
        path = self.translate_path(self.path)
        requested = self.headers.get('Range')
        if not requested or not os.path.isfile(path):
            return super().send_head()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
        size = os.path.getsize(path)
        if match and any(match.groups()) and size:
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size-1) if right else size-1
            else:
                start, end = max(0, size-int(right)), size-1
            if 0 <= start <= end < size:
                source = open(path, 'rb')
                source.seek(start)
                self.range_remaining = end-start+1
                self.send_response(206)
                self.send_header('Content-Type', self.guess_type(path))
                self.send_header('Content-Length', str(self.range_remaining))
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.send_header('Last-Modified', self.date_time_string(os.fstat(source.fileno()).st_mtime))
                self.end_headers()
                return source
        self.send_response(416)
        self.send_header('Content-Range', f'bytes */{size}')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return None

    def copyfile(self, source, target):
        try:
            if self.range_remaining is None:
                return super().copyfile(source, target)
            remaining = self.range_remaining
            while remaining:
                chunk = source.read(min(remaining, 1024*1024))
                if not chunk:
                    break
                target.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 拖动进度条会取消旧的视频请求。


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(VideoHandler, directory=str(args.directory.resolve())))
    print(f'Validation viewer: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()
