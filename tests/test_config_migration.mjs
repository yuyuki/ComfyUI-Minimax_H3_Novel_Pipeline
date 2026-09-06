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
        widgets_values_named: { qwen35_max_output_tokens: 8000, qwen35_length_retries: 2 },
    };
    const instance = new Config();
    assert.equal(instance.configure(old), "configured");
    assert.deepEqual(instance.loaded.widgets_values, ["url", false, 2, 3600, 20, 0, 1.05]);
    assert.deepEqual(instance.loaded.widgets_values_named, { qwen35_length_retries: 2 });
    assert.equal(old.widgets_values.length, 8);
    const migrated = instance.loaded;
    instance.configure(migrated);
    assert.deepEqual(instance.loaded, migrated);
});

test("unrelated node configuration is untouched", () => {
    class Other { configure(info) { this.loaded = info; } }
    const original = Other.prototype.configure;
    extension.beforeRegisterNodeDef(Other, { name: "ExtractChapterReferencesNode" });
    assert.equal(Other.prototype.configure, original);
});
