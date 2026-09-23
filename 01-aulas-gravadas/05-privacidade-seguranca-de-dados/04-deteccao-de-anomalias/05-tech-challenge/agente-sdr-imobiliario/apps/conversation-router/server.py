"""Servidor HTTP standalone para conversation-router no ECS Fargate.

Adapta requisições HTTP para o payload padrão de API Gateway HTTP v2 que o
handler.handler(event) já espera, mantendo 100% de compatibilidade sem alterar
a lógica de negócio.
"""
from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import handler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("router-server")


class ApiGatewayHttpAdapter(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        path = self.path
        query_string = ""
        if "?" in path:
            path, query_string = path.split("?", 1)

        body = ""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length).decode("utf-8")

        headers_dict = {k.lower(): v for k, v in self.headers.items()}

        # Constrói o evento no formato API Gateway HTTP API v2 payload format
        event: dict[str, Any] = {
            "version": "2.0",
            "routeKey": f"{method} {path}",
            "rawPath": path,
            "rawQueryString": query_string,
            "headers": headers_dict,
            "body": body,
            "isBase64Encoded": False,
            "requestContext": {
                "http": {
                    "method": method,
                    "path": path,
                    "protocol": "HTTP/1.1",
                    "sourceIp": self.client_address[0] if self.client_address else "127.0.0.1",
                    "userAgent": self.headers.get("User-Agent", ""),
                }
            },
        }

        try:
            res = handler.handler(event, None)
            status_code = res.get("statusCode", 200)
            res_headers = res.get("headers", {})
            res_body = res.get("body", "")

            self.send_response(status_code)
            for k, v in res_headers.items():
                self.send_header(k, str(v))
            if "content-type" not in [k.lower() for k in res_headers]:
                self.send_header("Content-Type", "application/json")
            self.end_headers()

            if isinstance(res_body, str):
                self.wfile.write(res_body.encode("utf-8"))
            elif isinstance(res_body, (dict, list)):
                self.wfile.write(json.dumps(res_body).encode("utf-8"))
            elif res_body:
                self.wfile.write(str(res_body).encode("utf-8"))

        except Exception as e:
            logger.error("Erro interno no dispatch do router: %s", e, exc_info=True)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Internal Server Error", "detail": str(e)}).encode("utf-8"))


def run() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, ApiGatewayHttpAdapter)
    logger.info("Conversation Router Server escutando em http://0.0.0.0:%d", port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
