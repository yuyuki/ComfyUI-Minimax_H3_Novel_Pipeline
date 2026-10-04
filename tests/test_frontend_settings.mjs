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
        "minimax_h3_novel.reference_links_path",
    ]);
});

test("reference links path stays visible on creation, toggles and workflow loading", () => {
    const extension = extensions.find((item) => item.name === "minimax_h3_novel.reference_links_path");
    const computeSize = () => [200, 20];
    const path = { name: "reference_links_path", type: "text", value: "", computeSize };
    const callback = function(value) { this.value = value; };
    const toggle = { name: "links_only", value: false, callback };
    const onConfigure = () => { toggle.value = false; };
    const node = {
        comfyClass: "ConsolidateReferencesNode", widgets: [path, toggle], onConfigure,
    };
    extension.nodeCreated(node);
    assert.equal(toggle.callback, callback);
    assert.equal(node.onConfigure, onConfigure);
    for (const value of [false, true, false]) {
        toggle.callback(value);
        assert.equal(path.type, "text");
        assert.equal(path.computeSize, computeSize);
    }
    node.onConfigure({});
    assert.equal(path.type, "text");
    assert.equal(path.computeSize, computeSize);
});

test("saved reference links fill an empty path and preserve manual paths and execution handlers", () => {
    const extension = extensions.find((item) => item.name === "minimax_h3_novel.reference_links_path");
    const path = { name: "reference_links_path", type: "text", value: "" };
    const toggle = { name: "links_only", value: true };
    let executions = 0;
    let callbacks = 0;
    path.callback = (value) => { assert.equal(value, "run/references/reference_links.json"); callbacks++; };
    const node = {
        comfyClass: "ConsolidateReferencesNode", widgets: [path, toggle],
        computeSize: () => [300, 124], setSize() {},
        onExecuted() { assert.equal(this, node); executions++; return "handled"; },
    };
    extension.nodeCreated(node);
    assert.equal(node.onExecuted({text: ["No saved path"]}), "handled");
    assert.equal(path.value, "");
    node.onExecuted({reference_links_path: ["run/references/reference_links.json"]});
    assert.equal(path.value, "run/references/reference_links.json");
    toggle.value = false;
    assert.equal(path.value, "run/references/reference_links.json");
    path.value = "my/edited.json";
    node.onExecuted({reference_links_path: ["new/references/reference_links.json"]});
    assert.equal(path.value, "my/edited.json");
    assert.equal(executions, 3);
    assert.equal(callbacks, 1);
    path.value = "references/reference_links.json";
    node.onExecuted({reference_links_path: ["run/references/reference_links.json"]});
    assert.equal(path.value, "run/references/reference_links.json");
    assert.equal(callbacks, 2);
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
