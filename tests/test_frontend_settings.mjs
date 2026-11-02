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

test("model options follow the selected family and preserve workflow order", async () => {
    const extension = extensions.find((item) => item.name === "minimax_h3_novel.chapter_picker");
    const widgets = ["api_url", "qwen35_top_k", "qwen35_min_p", "model_family"]
        .map((name) => ({ name, type: "number", value: name === "model_family" ? "Qwen" : name }));
    const node = {
        comfyClass: "LMStudioConfigurationNode", widgets,
        computeSize: () => [300, 100], setSize() {},
        serialize() { return this.widgets.map((item) => item.value); },
        configure(values) { this.widgets.forEach((item, index) => { item.value = values[index]; }); },
    };
    await extension.nodeCreated(node);
    assert.deepEqual(node.widgets.map((item) => item.name), ["api_url", "model_family", "qwen35_top_k", "qwen35_min_p"]);
    const model = node.widgets[1];
    model.value = "Gemma";
    model.callback(model.value);
    assert.equal(node.widgets[2].type, "hidden");
    assert.deepEqual(node.serialize(), ["api_url", "qwen35_top_k", "qwen35_min_p", "Gemma"]);
    node.configure(["url", 42, 0.2, "Qwen"]);
    assert.equal(node.widgets[1].value, "Qwen");
    assert.equal(node.widgets[2].value, 42);
    assert.equal(node.widgets[2].type, "number");
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
