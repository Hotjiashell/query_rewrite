"""Run against fresh demo data to verify browser-specific default owners."""
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    a, b = browser.new_context().new_page(), browser.new_context().new_page()
    errors = []
    for page in (a, b):
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8765")
        page.wait_for_load_state("networkidle")
    a.locator("#owners-open").click()
    for name in ("张三", "李四"):
        a.locator("#owner-name").fill(name)
        a.locator("#owner-form button").click()
        expect(a.locator("#owners-message")).to_contain_text(f"已保存负责人：{name}")
    a.locator("#default-owner").select_option("张三")
    a.locator("#owners-close").click()
    expect(a.locator('[data-review-field="owner"]')).to_have_value("张三")
    expect(b.locator('[data-review-field="owner"] option[value="李四"]')).to_have_count(1, timeout=10000)
    b.locator("#owners-open").click()
    b.locator("#default-owner").select_option("李四")
    b.locator("#owners-close").click()
    expect(b.locator('[data-review-field="owner"]')).to_have_value("李四")
    snapshot = a.request.get("http://127.0.0.1:8765/api/data").json()
    assert not snapshot["state"]["annotations"], "Viewing and default settings must not create annotations"
    a.reload()
    expect(a.locator('[data-review-field="owner"]')).to_have_value("张三")
    a.locator('[data-verdict="yes"]').click()
    a.locator("#complete").click()
    expect(a.locator("#record-state")).to_contain_text("已完成")
    # Existing attribution from A must survive B's different local default.
    b.locator("h1").click()
    expect(b.locator('[data-review-field="owner"]')).to_have_value("张三", timeout=10000)
    b.locator("#jump").fill("3")
    b.locator("#jump-go").click()
    expect(b.locator("#error-owner")).to_have_value("李四")
    b.locator("#error-type").select_option("其他")
    b.locator("#error-detail").fill("query遗漏用户诉求")
    b.locator("#complete").click()
    expect(b.locator("#record-state")).to_contain_text("已完成")
    saved = b.request.get("http://127.0.0.1:8765/api/data").json()["state"]["annotations"]
    assert saved["12"]["reviews"]["DEMO_RESET"]["owner"] == "张三"
    assert saved["20"]["owner"] == "李四"
    b.locator("#next").click()
    b.locator('[data-add="DEMO_RESET"]').click()
    expect(b.locator('[data-review-field="owner"]')).to_have_value("李四")
    # Explicit per-record selection continues to override local preference.
    b.locator('[data-review-field="owner"]').select_option("张三")
    b.locator('[data-verdict="no"]').click()
    b.locator('[data-review-field="reason"]').fill("不适用于锁定问题")
    expect(b.locator("#save-state")).to_contain_text("已保存")
    b.reload()
    b.locator("#jump").fill("4")
    b.locator("#jump-go").click()
    expect(b.locator('[data-review-field="owner"]')).to_have_value("张三")
    b.locator("#owners-open").click()
    b.locator("#default-owner").select_option("")
    b.locator("#owners-close").click()
    expect(b.locator("#default-owner-status")).to_contain_text("未设置")
    b.locator('[data-verdict="uncertain"]').click()
    b.locator('[data-review-field="reason"]').fill("")
    b.locator("#complete").click()
    expect(b.locator("#record-state")).to_contain_text("已完成")
    expect(b.locator("#classification")).to_contain_text("待进一步确认")
    snapshot = b.request.get("http://127.0.0.1:8765/api/data").json()
    assert snapshot["summary"]["uncertain"] == 1
    assert snapshot["summary"]["genuine"] == 1
    assert snapshot["state"]["annotations"]["27"]["outcome"] == "uncertain"
    b.locator("#filter").select_option("uncertain")
    expect(b.locator("#page-info")).to_have_text("1 / 1 条")
    assert not errors, errors
    print("Default owner test passed: isolation, persistence, automatic attribution, existing owners, manual override, no writes on viewing.")
    browser.close()
