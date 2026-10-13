"""Optional browser smoke test. Run against a fresh --demo output on port 8765."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8765")
    page.wait_for_load_state("networkidle")
    expect(page.locator("#page-info")).to_have_text("1 / 4 条")
    page.screenshot(path="/tmp/badcase-annotator-desktop.png", full_page=True)
    page.locator("#complete").click()
    expect(page.locator("#notice")).to_contain_text("每个待核验案例")
    card = page.locator('[data-cid="DEMO_RESET"]')
    card.locator('[data-verdict="yes"]').click()
    card.locator('[data-review-field="reason"]').fill("核验通过：身份验证后重置密码能解决诉求")
    card.locator('[data-review-field="owner"]').fill("测试负责人")
    page.locator("#complete").click()
    expect(page.locator("#record-state")).to_contain_text("已完成")
    page.reload()
    expect(page.locator("#record-state")).to_contain_text("已完成")
    expect(page.locator('[data-review-field="reason"]')).to_have_value("核验通过：身份验证后重置密码能解决诉求")
    page.locator("#next").click()
    expect(page.locator("#record-state")).to_contain_text("样本 15")
    page.locator("#next").click()
    expect(page.locator("#record-state")).to_contain_text("样本 20")
    page.locator("#error-type").select_option("检索排序问题")
    page.locator("#error-detail").fill("query没有表达转账限额诉求")
    page.locator("#error-owner").fill("分类负责人")
    page.locator("#complete").click()
    expect(page.locator("#record-state")).to_contain_text("已完成")
    page.locator("#types-open").click()
    page.locator("#type-name").fill("测试自定义类型")
    page.locator("#type-form button").click()
    expect(page.locator("#types-list")).to_contain_text("测试自定义类型")
    page.on("dialog", lambda dialog: dialog.accept())
    page.locator('[data-delete-type="检索排序问题"]').click()
    expect(page.locator('[data-delete-type="检索排序问题"]')).to_have_count(0)
    page.locator("#types-close").click()
    expect(page.locator("#error-type")).to_have_value("其他")
    expect(page.locator("#error-detail")).to_contain_text("原错误类型")
    page.locator("#next").click()
    expect(page.locator("#record-state")).to_contain_text("样本 27")
    page.locator('[data-add="DEMO_RESET"]').click()
    expect(page.locator('[data-cid="DEMO_RESET"]')).to_be_visible()
    page.locator('[data-cid="DEMO_RESET"] [data-verdict="no"]').click()
    page.locator('[data-review-field="reason"]').fill("密码重置不等于解锁")
    page.locator('[data-review-field="owner"]').fill("复核负责人")
    page.locator("#prev").click()
    page.locator("#next").click()
    expect(page.locator('[data-review-field="reason"]')).to_have_value("密码重置不等于解锁")
    response = page.request.get("http://127.0.0.1:8765/api/data").json()
    assert response["summary"]["completed"] == 2
    assert response["summary"]["alternative"] == 1
    assert response["summary"]["genuine"] == 1
    assert response["summary"]["error_counts"]["其他"] == 1
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path="/tmp/badcase-annotator-mobile.png", full_page=True)
    assert not errors, errors
    print("Browser smoke test passed: completion, autosave, reload, paging, types, manual reviews, statistics.")
    browser.close()
