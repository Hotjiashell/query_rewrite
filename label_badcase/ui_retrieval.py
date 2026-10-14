"""Optional browser check with local mock retrieval; never calls production."""
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright, expect
from server import ROOT, Store, build_dataset, handler, load, retrieve_cases


def main():
    demo = load(ROOT / "demo_data.json")
    records = build_dataset(demo["badcases"], demo["analysis"], demo["cases"])
    gt = records[0]["gt"]["case_id"]

    class RetrievalHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            query = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["retrieval_query"]
            if query == "slow":
                time.sleep(1)
            result = {"top3": {"case_id": gt, "case_title": "GT测试案例", "content": "测试正文", "score": 0.9},
                      "top1": {"case_id": "OTHER", "case_title": '<script>alert("x")</script>'}}
            if query == "miss":
                result.pop("top3")
            if query == "empty":
                result = {}
            payload = json.dumps({"retrieval_result": result}).encode()
            self.send_response(503 if query == "error" else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), RetrievalHandler)
        endpoint = f"http://127.0.0.1:{upstream.server_port}"
        store = Store(records, Path(tmp) / "labels.json", {"demo": True},
                      retriever=lambda query: retrieve_cases(query, url=endpoint, timeout=3))
        app = ThreadingHTTPServer(("127.0.0.1", 0), handler(store))
        for server in (upstream, app):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel="chrome", headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    errors = []
                    page.on("pageerror", lambda e: errors.append(str(e)))
                    page.goto(f"http://127.0.0.1:{app.server_port}")
                    page.wait_for_load_state("networkidle")
                    expect(page.locator("#trial-panel")).to_be_visible()
                    page.locator('[data-trial-fill="gt"]').click()
                    expect(page.locator("#trial-query")).to_have_value(records[0]["gt"]["title"])
                    page.locator("#trial-submit").click()
                    expect(page.locator("#trial-results")).to_contain_text("命中 GT · 排名 #3")
                    expect(page.locator(".trial-hit")).to_have_count(1)
                    expect(page.locator("#trial-results script")).to_have_count(0)
                    page.locator(".trial-hit summary").click()
                    expect(page.locator(".trial-hit .content")).to_have_text("测试正文")
                    for query, message in (("miss", "未命中 GT"), ("empty", "本次未返回任何案例"), ("error", "检索失败")):
                        page.locator("#trial-query").fill(query)
                        page.locator("#trial-submit").click()
                        expect(page.locator("#trial-results")).to_contain_text(message)
                    page.locator("#trial-query").fill("slow")
                    page.locator("#trial-submit").click()
                    page.locator("#next").click()
                    expect(page.locator("#trial-results")).to_contain_text("输入 query")
                    page.wait_for_timeout(1200)
                    expect(page.locator("#trial-results")).to_contain_text("输入 query")
                    page.locator("#prev").click()
                    expect(page.locator("#trial-results")).to_contain_text("命中 GT · 排名 #3")
                    expect(page.locator("#trial-query")).to_have_value("slow")
                    assert not store.state["annotations"]
                    assert not store.output.exists()
                    page.screenshot(path="/tmp/badcase-trial-desktop.png", full_page=True)
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                    page.screenshot(path="/tmp/badcase-trial-mobile.png", full_page=True)
                    assert not errors, errors
                    print("Retrieval UI passed: real search.py client + mock upstream, hit/miss/empty/error, navigation, escaping, no annotation writes.")
                finally:
                    browser.close()
        finally:
            for server in (app, upstream):
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    main()
