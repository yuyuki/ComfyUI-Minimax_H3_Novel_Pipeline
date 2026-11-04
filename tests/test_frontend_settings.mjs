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
        "minimax_h3_novel.chapter_picker", "minimax_h3_novel.provider_settings",
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

test("provider cards sync a complete snapshot before queueing and propagate failures", async () => {
    const extension = extensions.find((item) => item.name.endsWith(".provider_settings"));
    assert.equal(extension.settings.length, 1);
    assert.ok(extension.settings.every((setting) => typeof setting.type === "function"));
    const values = new Map([["MiniMaxH3Novel.Connections",
        { lmstudio: { enabled: true, api_url: "https://trusted.example/v1", api_key: "test-only" } }]]);
    globalThis.frontendTestApp.ui = { settings: { getSettingValue: (id) => values.get(id) } };
    const originalFetch = globalThis.fetch;
    const sent = [];
    globalThis.fetch = async (url, options) => {
        assert.equal(url, "/minimax_h3_novel/provider-settings");
        assert.equal(options.headers.get("X-MiniMax-H3-Request"), "1");
        sent.push(JSON.parse(options.body));
        return { ok: true };
    };
    try {
        await extension.beforeQueuing();
        assert.deepEqual(sent[0], {
            lmstudio: { enabled: true, api_url: "https://trusted.example/v1", api_key: "test-only" },
            openai: { enabled: false, api_url: "https://api.openai.com/v1", api_key: "", model: "" },
        });
        globalThis.fetch = async () => ({ ok: false });
        await assert.rejects(extension.beforeQueuing(), /Could not save/);
    } finally { globalThis.fetch = originalFetch; }
});

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.events = {}; }
    appendChild(child) { this.children.push(child); return child; }
    append(...children) { children.forEach((child) => this.appendChild(child)); }
    replaceChildren(...children) { this.children = [...children]; }
    setAttribute(name, value) { this[name] = value; }
    addEventListener(name, callback) { this.events[name] = callback; }
}

test("provider cards collapse, mask keys, test drafts without saving, and save explicitly", async () => {
    const extension = extensions.find((item) => item.name.endsWith(".provider_settings"));
    const values = new Map();
    globalThis.frontendTestApp.ui = { settings: { getSettingValue: (id) => values.get(id) } };
    globalThis.document = { createElement: (tag) => new Element(tag), createTextNode: (text) => text };
    const originalFetch = globalThis.fetch;
    const requests = [];
    globalThis.fetch = async (url, options) => {
        requests.push({ url, body: JSON.parse(options.body) });
        return { ok: true, json: async () => url.endsWith("/provider-models")
            ? { models: ["gpt-6.1-sol", "embedding-model", "audio-model", "gpt-6.1-sol"] } : { message: "Connected." } };
    };
    try {
        const setting = extension.settings[0];
        const saves = [];
        const panel = setting.type(setting.name, async (value) => { saves.push(value); values.set(setting.id, value); }, setting.default);
        assert.equal(panel.children.length, 2);
        const card = panel.children[1];
        const toggle = card.children[0].children[0];
        const body = card.children[1];
        assert.equal(body.hidden, true);
        assert.equal(toggle.role, "switch");
        toggle.checked = true;
        await toggle.events.change();
        assert.equal(body.hidden, false);
        assert.equal(saves.length, 1);
        const [url, key, model, status, actions] = body.children;
        assert.equal(url.children[0].readOnly, true);
        assert.equal(key.children[0].type, "password");
        key.children[0].value = "draft-key";
        assert.equal(model.children[0].tag, "select");
        assert.equal(model.children[0].children.length, 0);
        assert.equal(model.children[0].value, "");
        model.children[0].value = "gpt-6.1-sol";
        await actions.children[1].events.click();
        assert.equal(saves.length, 1);
        assert.equal(requests.at(-1).url, "/minimax_h3_novel/provider-test");
        assert.equal(requests.at(-1).body.api_key, "draft-key");
        assert.equal(status.textContent, "Connected.");
        assert.deepEqual(actions.children.map((button) => button.textContent), ["Fetch models", "Test", "Save"]);
        await actions.children[0].events.click();
        assert.equal(saves.length, 1);
        assert.equal(requests.at(-1).url, "/minimax_h3_novel/provider-models");
        assert.equal(requests.at(-1).body.api_key, "draft-key");
        assert.deepEqual(model.children[0].children.map((option) => option.value), [
            "audio-model", "embedding-model", "gpt-6.1-sol",
        ]);
        assert.equal(model.children[0].value, "gpt-6.1-sol");
        const successfulFetch = globalThis.fetch;
        globalThis.fetch = async () => ({ ok: false, json: async () => ({ error: "Authentication failed" }) });
        await actions.children[0].events.click();
        assert.equal(status.textContent, "Authentication failed");
        assert.equal(model.children[0].children.length, 3);
        assert.equal(saves.length, 1);
        globalThis.fetch = async () => ({ ok: true, json: async () => ({ models: [] }) });
        await actions.children[0].events.click();
        assert.match(status.textContent, /current list is unchanged/);
        assert.equal(model.children[0].children.length, 3);
        globalThis.fetch = async () => ({ ok: false, status: 405,
            json: async () => { throw new SyntaxError("Unexpected non-whitespace character after JSON"); } });
        await actions.children[0].events.click();
        assert.match(status.textContent, /Restart ComfyUI/);
        assert.equal(model.children[0].children.length, 3);
        globalThis.fetch = async () => ({ ok: true, status: 200,
            json: async () => { throw new SyntaxError("Invalid JSON"); } });
        await actions.children[0].events.click();
        assert.match(status.textContent, /non-JSON response/);
        assert.equal(model.children[0].children.length, 3);
        globalThis.fetch = successfulFetch;
        await actions.children[2].events.click();
        assert.equal(saves.length, 2);
        assert.deepEqual(saves[1].openai.available_models, ["audio-model", "embedding-model", "gpt-6.1-sol"]);
        const reopened = setting.type(setting.name, async () => {}, saves[1]);
        assert.deepEqual(reopened.children[1].children[1].children[2].children[0].children.map((option) => option.value),
            ["audio-model", "embedding-model", "gpt-6.1-sol"]);
        assert.equal(saves[1].openai.model, "gpt-6.1-sol");
        toggle.checked = false;
        await toggle.events.change();
        assert.equal(body.hidden, true);
        assert.equal(values.get(setting.id).openai.api_key, "draft-key");
    } finally { globalThis.fetch = originalFetch; delete globalThis.document; }
});