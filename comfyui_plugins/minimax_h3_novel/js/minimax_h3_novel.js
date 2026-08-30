import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "minimax_h3_novel.chapter_picker";
const ACCEPTED = ".txt,.md,.markdown,.pdf";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function migrateLegacyExtractDefaults(node) {
    // Older workflows included endpoint fields before the extraction settings.
    // ComfyUI stores widget values by position, so after those fields were
    // removed their old defaults appeared under the wrong labels.
    const outDir = widget(node, "out_dir");
    const chunkChars = widget(node, "chunk_chars");
    const overlap = widget(node, "overlap_paragraphs");
    const temperature = widget(node, "temperature");
    const maxTokens = widget(node, "max_tokens");
    const seed = widget(node, "seed");
    const isLegacyDefaults = outDir?.value === "http://127.0.0.1:1234/v1"
        && chunkChars?.value === "lm-studio"
        && Number(overlap?.value) === 5500
        && Number(temperature?.value) === 2
        && Number(maxTokens?.value) === 0
        && Number(seed?.value) === 8192;
    if (!isLegacyDefaults) return;

    outDir.value = "";
    chunkChars.value = 5500;
    overlap.value = 2;
    temperature.value = 1.8;
    maxTokens.value = 8192;
    // Do not force a fixed seed: ComfyUI's randomize control will supply one.
    seed.value = Math.floor(Math.random() * 0x100000000);
}

function migrateLegacyConsolidationDefaults(node) {
    // Older workflows included ``base_url`` and ``api_key`` before these
    // widgets. ComfyUI persists widget values by position, so removing those
    // inputs causes every saved value to be assigned to the following field.
    const candidateCount = widget(node, "candidate_count");
    const includeAllBelow = widget(node, "include_all_below");
    const pictureThreshold = widget(node, "picture_threshold");
    const audioThreshold = widget(node, "audio_threshold");
    const maxCharacterViews = widget(node, "max_character_base_views");
    const maxLocationViews = widget(node, "max_location_base_views");
    const maxObjectViews = widget(node, "max_object_base_views");
    const assetBatchSize = widget(node, "asset_batch_size");
    const noVariants = widget(node, "no_variants");
    const noAudit = widget(node, "no_audit");
    const auditMaxEntities = widget(node, "audit_max_entities");
    const temperature = widget(node, "temperature");
    const maxTokens = widget(node, "max_tokens");
    const delay = widget(node, "delay");
    const outDir = widget(node, "out_dir");

    // A URL/API-key pair shifted into integer inputs is clamped to 1, while
    // the old candidate count then appears as an invalid threshold value.
    // This signature distinguishes a legacy saved node from user settings.
    const thresholds = ["optional", "recommended", "required"];
    const isShiftedLegacyNode = Number(candidateCount?.value) === 1
        && Number(includeAllBelow?.value) === 1
        && !thresholds.includes(pictureThreshold?.value);
    if (!isShiftedLegacyNode) return;

    candidateCount.value = 12;
    includeAllBelow.value = 35;
    pictureThreshold.value = "recommended";
    audioThreshold.value = "recommended";
    maxCharacterViews.value = 4;
    maxLocationViews.value = 3;
    maxObjectViews.value = 2;
    assetBatchSize.value = 16;
    noVariants.value = false;
    noAudit.value = false;
    auditMaxEntities.value = 120;
    temperature.value = 0.12;
    maxTokens.value = 8500;
    delay.value = 0;
    outDir.value = "";
}

async function refreshSavedChapters(node) {
    const response = await fetch("/minimax_h3_novel/chapters");
    if (!response.ok) return;
    const result = await response.json();
    const widget = node.widgets?.find((item) => item.name === "saved_chapter");
    if (!widget) return;
    widget.options.values = result.files?.length ? result.files : [""];
    if (widget.value && !widget.options.values.includes(widget.value)) widget.value = "";
    app.graph?.setDirtyCanvas(true, true);
}

function chooseFiles(node, directory) {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ACCEPTED;
    if (directory) input.webkitdirectory = true;
    input.style.display = "none";
    document.body.appendChild(input);
    input.addEventListener("change", async () => {
        try {
            if (!input.files?.length) return;
            const form = new FormData();
            for (const file of input.files) form.append("chapters", file, file.name);
            const response = await fetch("/minimax_h3_novel/upload", { method: "POST", body: form });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || "Upload failed");
            const widget = node.widgets?.find((item) => item.name === "chapter_paths");
            if (!widget) throw new Error("chapter_paths widget was not found");
            widget.value = (result.files || []).join("\n");
            widget.callback?.(widget.value);
            await refreshSavedChapters(node);
            app.graph?.setDirtyCanvas(true, true);
        } catch (error) {
            alert(`MiniMax H3 chapter upload failed: ${error.message}`);
        } finally {
            input.remove();
        }
    }, { once: true });
    input.click();
}

app.registerExtension({
    name: EXTENSION_NAME,
    async nodeCreated(node) {
        if (node.comfyClass === "ConsolidateReferencesNode") {
            migrateLegacyConsolidationDefaults(node);
            return;
        }
        if (node.comfyClass !== "ExtractChapterReferencesNode") return;
        migrateLegacyExtractDefaults(node);
        node.addWidget("button", "Select chapter files", null, () => chooseFiles(node, false));
        node.addWidget("button", "Select chapter folder", null, () => chooseFiles(node, true));
        const saved = node.widgets?.find((item) => item.name === "saved_chapter");
        if (saved) {
            const originalCallback = saved.callback;
            saved.callback = (value) => {
                originalCallback?.(value);
                if (!value) return;
                const paths = node.widgets?.find((item) => item.name === "chapter_paths");
                if (!paths) return;
                paths.value = value;
                paths.callback?.(paths.value);
                app.graph?.setDirtyCanvas(true, true);
            };
            refreshSavedChapters(node);
        }
    },
});
