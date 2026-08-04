import http.server
import socketserver

PORT = 8888

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

NoCacheHandler.extensions_map['.wasm'] = 'application/wasm'
NoCacheHandler.extensions_map['.onnx'] = 'application/octet-stream'

with socketserver.TCPServer(("", PORT), NoCacheHandler) as httpd:
    print(f"Serving at http://127.0.0.1:{PORT}")
    httpd.serve_forever()