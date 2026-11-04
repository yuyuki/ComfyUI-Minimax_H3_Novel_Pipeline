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

test("provider models follow saved availability, preserve workflow values and manual resizing", async () => {
    const extension = extensions.find((item) => item.name === "minimax_h3_novel.chapter_picker");
    const settings = extensions.find((item) => item.name.endsWith(".provider_settings"));
    const widgets = ["backend", "qwen35_top_k", "qwen35_min_p"]
        .map((name) => ({ name, type: "number", options: {}, value: name === "backend" ? "LM Studio" : name }));
    const node = {
        comfyClass: "LMStudioConfigurationNode", widgets, size: [620, 420],
        computeSize: () => [300, 100], setSize(size) { this.size = size; },
        addWidget(type, name, value, callback, options) {
            const item = { type, name, value, callback, options };
            this.widgets.push(item);
            return item;
        },
        serialize() { return this.widgets.filter((item) => item.serialize !== false).map((item) => item.value); },
        configure(values) { this.widgets.forEach((item, index) => { item.value = values[index]; }); },
    };
    const originalFetch = globalThis.fetch;
    const providers = [
        { backend: "LM Studio", model: "qwen3.5-9b", model_family: "Qwen" },
        { backend: "OpenAI", model: "gpt-test", model_family: "Mistral" },
    ];
    globalThis.frontendTestApp.ui = { settings: { getSettingValue: () => ({}) } };
    globalThis.frontendTestApp.graph = { _nodes: [node], setDirtyCanvas() {} };
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ providers }) });
    try {
        await extension.nodeCreated(node);
        assert.equal(node.widgets[0].value, "LM Studio");
        await settings.beforeQueuing();
        assert.deepEqual(node.widgets.map((item) => item.name), ["backend", "model", "qwen35_top_k", "qwen35_min_p"]);
        const [backend, model] = node.widgets;
        assert.deepEqual(backend.options.values, ["LM Studio", "OpenAI"]);
        assert.equal(model.value, "qwen3.5-9b");
        assert.equal(model.disabled, true);
        backend.value = "OpenAI";
        backend.callback(backend.value);
        assert.equal(model.value, "gpt-test");
        assert.equal(node.widgets[2].type, "hidden");
        assert.deepEqual(node.serialize(), ["OpenAI", "qwen35_top_k", "qwen35_min_p"]);
        node.configure(["LM Studio", 42, 0.2]);
        assert.equal(model.value, "qwen3.5-9b");
        assert.equal(node.widgets[2].value, 42);
        assert.equal(node.widgets[2].type, "number");
        node.size = [840, 600];
        for (let i = 0; i < 5; i++) node.serialize();
        assert.deepEqual(node.size, [840, 600]);
        providers[0] = { backend: "LM Studio", model: "gemma-3", model_family: "Gemma" };
        await settings.beforeQueuing();
        assert.equal(model.value, "gemma-3");
        assert.equal(node.widgets[2].type, "hidden");
        providers.shift();
        await settings.beforeQueuing();
        assert.equal(backend.value, "OpenAI");
        providers.pop();
        await settings.beforeQueuing();
        assert.deepEqual(backend.options.values, []);
        assert.equal(backend.value, "");
        assert.match(model.value, /Configure and save/);
        assert.deepEqual(node.size, [840, 600]);
    } finally { globalThis.fetch = originalFetch; delete globalThis.frontendTestApp.graph; }
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
        return { ok: true, json: async () => ({ providers: [] }) };
    };
    try {
        await extension.beforeQueuing();
        assert.deepEqual(sent[0], {
            lmstudio: { enabled: true, api_url: "https://trusted.example/v1", api_key: "test-only", model: "" },
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

for (const [providerIndex, providerKey] of [[0, "lmstudio"], [1, "openai"]]) {
test(`${providerKey} cards fetch models, test drafts without saving, and save explicitly`, async () => {
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
        const panel = setting.type(setting.name, async (value) => { saves.push(value); values.set(setting.id, value); },
            { ...setting.default, lmstudio: { enabled: false } });
        assert.equal(panel.children.length, 2);
        const card = panel.children[providerIndex];
        const toggle = card.children[0].children[0];
        const body = card.children[1];
        assert.equal(body.hidden, true);
        assert.equal(toggle.role, "switch");
        toggle.checked = true;
        await toggle.events.change();
        assert.equal(body.hidden, false);
        assert.equal(saves.length, 1);
        const [url, key, model, status, actions] = body.children;
        assert.equal(Boolean(url.children[0].readOnly), providerKey === "openai");
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
        assert.equal(requests.at(-1).body.provider, providerKey);
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
        assert.deepEqual(saves[1][providerKey].available_models, ["audio-model", "embedding-model", "gpt-6.1-sol"]);
        const reopened = setting.type(setting.name, async () => {}, saves[1]);
        assert.deepEqual(reopened.children[providerIndex].children[1].children[2].children[0].children.map((option) => option.value),
            ["audio-model", "embedding-model", "gpt-6.1-sol"]);
        assert.equal(saves[1][providerKey].model, "gpt-6.1-sol");
        assert.equal(requests.at(-1).body[providerKey].model, "gpt-6.1-sol");
        assert.equal(reopened.children[providerIndex].children[1].children[2].children[0].value, "gpt-6.1-sol");
        toggle.checked = false;
        await toggle.events.change();
        assert.equal(body.hidden, true);
        assert.equal(values.get(setting.id)[providerKey].api_key, "draft-key");
    } finally { globalThis.fetch = originalFetch; delete globalThis.document; }
});
}
