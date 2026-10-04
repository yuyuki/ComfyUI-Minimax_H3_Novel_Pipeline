import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const PREVIEW_NODES = new Set([
    "ExtractChapterReferencesNode",
    "ConsolidateReferencesNode",
    "CinematicChapterAdapterNode",
    "GenerateH3PromptsNode",
]);

app.registerExtension({
    name: "minimax_h3_novel.text_preview",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PREVIEW_NODES.has(nodeData.name)) return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            if (!Array.isArray(message?.text)) return result;
            let preview = this.widgets?.find((item) => item.name === "minimax_text_preview");
            if (!preview) {
                preview = ComfyWidgets.STRING(this, "minimax_text_preview", ["STRING", {
                    multiline: true,
                }], app).widget;
                preview.inputEl.readOnly = true;
                preview.inputEl.setAttribute("aria-label", "Output preview");
                preview.options = { ...preview.options, serialize: false };
                preview.serialize = false;
                const size = this.computeSize();
                this.setSize([Math.max(this.size[0], size[0], 400), Math.max(this.size[1], size[1], 300)]);
            }
            preview.value = message.text.join("\n\n");
            this.setDirtyCanvas(true, true);
            return result;
        };
    },
});
