import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

let extension;
globalThis.previewTestApp = { registerExtension: (value) => { extension = value; } };
globalThis.previewTestWidgets = {
    STRING(node, name, spec) {
        assert.equal(spec[1].multiline, true);
        const widget = { name, inputEl: { setAttribute() {} }, options: {} };
        node.widgets.push(widget);
        return { widget };
    },
};
const source = (await readFile(new URL("../web/js/text_preview.js", import.meta.url), "utf8"))
    .replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.previewTestApp;')
    .replace('import { ComfyWidgets } from "../../scripts/widgets.js";', 'const ComfyWidgets = globalThis.previewTestWidgets;');
await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

for (const name of ["ExtractChapterReferencesNode", "ConsolidateReferencesNode",
    "CinematicChapterAdapterNode", "GenerateH3PromptsNode"]) {
    test(`${name} displays output and reuses its read-only preview`, () => {
        class Node {
            widgets = [];
            size = [500, 700];
            calls = 0;
            onExecuted() { this.calls++; return "original"; }
            computeSize() { return [300, 200]; }
            setSize(size) { this.size = size; }
            setDirtyCanvas() { this.dirty = true; }
        }
        extension.beforeRegisterNodeDef(Node, { name });
        const node = new Node();
        assert.equal(node.onExecuted({ minimax_h3_completed: [true] }), "original");
        assert.equal(node.widgets.length, 0);
        node.onExecuted({ text: ["Résumé <script>literal</script>", "Image prompts"] });
        const preview = node.widgets[0];
        assert.equal(preview.value, "Résumé <script>literal</script>\n\nImage prompts");
        assert.equal(preview.inputEl.readOnly, true);
        assert.equal(preview.serialize, false);
        assert.equal(preview.options.serialize, false);
        assert.deepEqual(node.size, [500, 700]);
        assert.equal(node.dirty, true);
        node.onExecuted({ text: ["Updated"] });
        assert.equal(node.widgets.length, 1);
        assert.equal(preview.value, "Updated");
        node.onExecuted({ text: [] });
        assert.equal(preview.value, "");
        assert.equal(node.calls, 4);
    });
}

test("unrelated nodes retain their execution handler", () => {
    class Node { onExecuted() {} }
    const original = Node.prototype.onExecuted;
    extension.beforeRegisterNodeDef(Node, { name: "SelectChaptersNode" });
    assert.equal(Node.prototype.onExecuted, original);
});
