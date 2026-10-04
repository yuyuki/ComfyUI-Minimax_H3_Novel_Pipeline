import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const extensions = [];
globalThis.frontendTestApp = { registerExtension: (extension) => extensions.push(extension) };
const source = (await readFile(new URL("../web/js/minimax_h3_novel.js", import.meta.url), "utf8"))
    .replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.frontendTestApp;');
await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
test("frontend registers current features without configuration migration", () => {
    assert.deepEqual(extensions.map((item) => item.name).sort(), [
        "minimax_h3_novel.chapter_picker", "minimax_h3_novel.lmstudio_settings",
    ]);
});

test("LM Studio settings send the authorized endpoint and key together before queuing", async () => {
    const settingsExtension = extensions.find((item) => item.name === "minimax_h3_novel.lmstudio_settings");
    const urlSetting = settingsExtension.settings.find((item) => item.id.endsWith(".ApiUrl"));
    assert.equal(urlSetting.default, "http://127.0.0.1:1234/v1");
    const values = new Map([
        [urlSetting.id, "https://trusted.example/v1"],
        ["MiniMaxH3Novel.LMStudio.ApiKey", "test-only"],
    ]);
    globalThis.frontendTestApp.ui = { settings: { getSettingValue: (id) => values.get(id) } };
    const originalFetch = globalThis.fetch;
    const sent = [];
    globalThis.fetch = async (url, options) => {
        assert.equal(url, "/minimax_h3_novel/lmstudio-settings");
        assert.equal(options.headers.get("X-MiniMax-H3-Request"), "1");
        sent.push(JSON.parse(options.body));
        return { ok: true };
    };
    try {
        await settingsExtension.beforeQueuing();
        assert.deepEqual(sent[0], { api_url: "https://trusted.example/v1", api_key: "test-only" });
        values.delete(urlSetting.id);
        await settingsExtension.beforeQueuing();
        assert.deepEqual(sent[1], { api_url: urlSetting.default, api_key: "test-only" });
    } finally {
        globalThis.fetch = originalFetch;
        delete globalThis.frontendTestApp.ui;
    }
});
