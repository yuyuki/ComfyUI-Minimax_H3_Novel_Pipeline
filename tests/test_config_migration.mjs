import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const extensions = [];
globalThis.migrationTestApp = { registerExtension: (extension) => extensions.push(extension) };
const source = (await readFile(new URL("../web/js/minimax_h3_novel.js", import.meta.url), "utf8"))
    .replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.migrationTestApp;');
await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const extension = extensions.find((item) => item.name === "minimax_h3_novel.config_migration");

test("old configuration preserves every remaining widget and does not mutate source", () => {
    class Config {
        configure(info) { this.loaded = info; return "configured"; }
    }
    extension.beforeRegisterNodeDef(Config, { name: "LMStudioConfigurationNode" });
    const old = {
        widgets_values: ["url", false, 8000, 2, 3600, 20, 0, 1.05],
        widgets_values_named: { qwen35_max_output_tokens: 8000, qwen35_safe_chunk_chars: 3600, qwen35_length_retries: 2 },
    };
    const instance = new Config();
    assert.equal(instance.configure(old), "configured");
    assert.deepEqual(instance.loaded.widgets_values, ["url", false, 2, 20, 0, 1.05, "Qwen"]);
    assert.deepEqual(instance.loaded.widgets_values_named, { qwen35_length_retries: 2, model_family: "Qwen" });
    assert.equal(old.widgets_values.length, 8);
    const migrated = instance.loaded;
    instance.configure(migrated);
    assert.deepEqual(instance.loaded, migrated);
});

test("configuration with only the legacy chunk cap migrates", () => {
    class Config { configure(info) { this.loaded = info; } }
    extension.beforeRegisterNodeDef(Config, { name: "LMStudioConfigurationNode" });
    const instance = new Config();
    const old = { widgets_values: ["url", true, 4, 5000, 30, 0.1, 1.1] };
    instance.configure(old);
    assert.deepEqual(instance.loaded.widgets_values, ["url", true, 4, 30, 0.1, 1.1, "Qwen"]);
    assert.equal(old.widgets_values.length, 7);
});

test("unrelated node configuration is untouched", () => {
    class Other { configure(info) { this.loaded = info; } }
    const original = Other.prototype.configure;
    extension.beforeRegisterNodeDef(Other, { name: "ExtractChapterReferencesNode" });
    assert.equal(Other.prototype.configure, original);
});

for (const family of ["Qwen", "Mistral"]) {
    test(`current ${family} configuration survives repeated loading`, () => {
        class Config { configure(info) { this.loaded = info; } }
        extension.beforeRegisterNodeDef(Config, { name: "LMStudioConfigurationNode" });
        const instance = new Config();
        const info = { widgets_values: ["url", false, 2, 20, 0, 1.05, family],
            widgets_values_named: { model_family: family } };
        instance.configure(info);
        assert.deepEqual(instance.loaded, info);
        instance.configure(instance.loaded);
        assert.deepEqual(instance.loaded, info);
    });
}

test("six-widget configuration gains Qwen without shifting sampling values", () => {
    class Config { configure(info) { this.loaded = info; } }
    extension.beforeRegisterNodeDef(Config, { name: "LMStudioConfigurationNode" });
    const instance = new Config();
    instance.configure({ widgets_values: ["url", false, 2, 20, 0, 1.05] });
    assert.deepEqual(instance.loaded.widgets_values, ["url", false, 2, 20, 0, 1.05, "Qwen"]);
});


test("LM Studio settings send the authorized endpoint and key together before queuing", async () => {
    const settingsExtension = extensions.find((item) => item.name === "minimax_h3_novel.lmstudio_settings");
    const urlSetting = settingsExtension.settings.find((item) => item.id.endsWith(".ApiUrl"));
    assert.equal(urlSetting.default, "http://127.0.0.1:1234/v1");
    const values = new Map([
        [urlSetting.id, "https://trusted.example/v1"],
        ["MiniMaxH3Novel.LMStudio.ApiKey", "test-only"],
    ]);
    globalThis.migrationTestApp.ui = { settings: { getSettingValue: (id) => values.get(id) } };
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
        delete globalThis.migrationTestApp.ui;
    }
});
