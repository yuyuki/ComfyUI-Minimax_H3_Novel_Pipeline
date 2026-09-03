import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "minimax_h3_novel.chapter_picker";
const ACCEPTED = ".txt,.md,.markdown,.pdf";
const CHAPTER_EXTENSIONS = new Set(ACCEPTED.split(","));
const PICKER_BUTTON_NAME = "saved_chapter_picker";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function migrateLegacyExtractDefaults(node) {
    // Version 1 used a CLIP input and had no endpoint widgets. ComfyUI stores
    // widget values by position, so recover its values after the LM Studio
    // fields were inserted at the beginning of the node.
    const apiUrl = widget(node, "api_url");
    const apiKey = widget(node, "api_key");
    const model = widget(node, "model");
    const chapterPaths = widget(node, "chapter_paths");
    const savedChapter = widget(node, "saved_chapter");
    const outDir = widget(node, "out_dir");
    const chunkChars = widget(node, "chunk_chars");
    const overlap = widget(node, "overlap_paragraphs");
    const temperature = widget(node, "temperature");
    const maxTokens = widget(node, "max_tokens");
    const isLegacyDefaults = Number(model?.value) === 5500
        && Number(chapterPaths?.value) === 2
        && Number(savedChapter?.value) === 0.35
        && Number(chunkChars?.value) === 8192;
    if (!isLegacyDefaults) return;

    const legacyPaths = apiUrl.value;
    const legacySavedChapter = apiKey.value;
    const legacyOutDir = overlap.value;
    apiUrl.value = "http://127.0.0.1:1234/v1";
    apiKey.value = "lm-studio";
    model.value = "";
    chapterPaths.value = legacyPaths;
    savedChapter.value = legacySavedChapter;
    chunkChars.value = 5500;
    overlap.value = 2;
    temperature.value = 0.35;
    maxTokens.value = 2200;
    outDir.value = legacyOutDir || outDir.options?.default;
}

function migrateLegacyConsolidationDefaults(node) {
    const apiUrl = widget(node, "api_url");
    const apiKey = widget(node, "api_key");
    const model = widget(node, "model");
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
    const outDir = widget(node, "out_dir");
    const isShiftedLegacyNode = Number(apiUrl?.value) === 12
        && Number(apiKey?.value) === 35
        && model?.value === "recommended";
    if (!isShiftedLegacyNode) return;

    const values = [
        apiUrl.value, apiKey.value, model.value, candidateCount.value,
        includeAllBelow.value, pictureThreshold.value, audioThreshold.value,
        maxCharacterViews.value, maxLocationViews.value, maxObjectViews.value,
        assetBatchSize.value, noVariants.value, noAudit.value,
        auditMaxEntities.value, temperature.value, maxTokens.value, outDir.value,
    ];
    apiUrl.value = "http://127.0.0.1:1234/v1";
    apiKey.value = "lm-studio";
    model.value = "";
    [candidateCount.value, includeAllBelow.value, pictureThreshold.value,
        audioThreshold.value, maxCharacterViews.value, maxLocationViews.value,
        maxObjectViews.value, assetBatchSize.value, noVariants.value,
        noAudit.value, auditMaxEntities.value, temperature.value,
        maxTokens.value, outDir.value] = values.slice(0, 14);
}

function migrateLegacyPromptDefaults(node) {
    const apiUrl = widget(node, "api_url");
    const apiKey = widget(node, "api_key");
    const model = widget(node, "model");
    const chapterPaths = widget(node, "chapter_paths");
    const savedChapter = widget(node, "saved_chapter");
    const duration = widget(node, "duration");
    const maxPictures = widget(node, "max_pictures");
    const maxAudio = widget(node, "max_audio");
    const outDir = widget(node, "out_dir");
    // The former CLIP-based node had: chapter paths, saved chapter, duration,
    // max pictures, max audio, output folder.
    const isLegacyNode = Number(model?.value) === 8
        && Number(chapterPaths?.value) === 8
        && Number(savedChapter?.value) === 4;
    if (!isLegacyNode) return;

    const legacyPaths = apiUrl.value;
    const legacySaved = apiKey.value;
    const legacyDuration = model.value;
    const legacyPictures = chapterPaths.value;
    const legacyAudio = savedChapter.value;
    const legacyOutDir = duration.value;
    apiUrl.value = "http://127.0.0.1:1234/v1";
    apiKey.value = "lm-studio";
    model.value = "";
    chapterPaths.value = legacyPaths;
    savedChapter.value = legacySaved;
    duration.value = legacyDuration;
    maxPictures.value = legacyPictures;
    maxAudio.value = legacyAudio;
    outDir.value = legacyOutDir || outDir.options?.default;
}

function pickerButton(node) {
    return widget(node, PICKER_BUTTON_NAME);
}

function updatePickerButton(node) {
    const button = pickerButton(node);
    const saved = widget(node, "saved_chapter");
    if (!button || !saved) return;
    button.label = saved.value || "Select chapters";
    app.graph?.setDirtyCanvas(true, true);
}

function setChapterPaths(node, value) {
    if (!value) return;
    const paths = widget(node, "chapter_paths");
    if (!paths) return;
    paths.value = value;
    paths.callback?.(paths.value);
}

function setSavedChapter(node, file) {
    const saved = widget(node, "saved_chapter");
    if (!saved) return;
    saved.value = file || "";
    saved.callback?.(saved.value);
    updatePickerButton(node);
}

async function fetchSavedChapters() {
    const response = await fetch("/minimax_h3_novel/chapters");
    if (!response.ok) throw new Error("Could not load saved chapters");
    const result = await response.json();
    return result.files || [];
}

async function refreshSavedChapters(node, selectedFile = null) {
    try {
        const files = await fetchSavedChapters();
        const saved = widget(node, "saved_chapter");
        if (!saved) return files;
        saved.options.values = files.length ? files : [""];
        if (selectedFile && files.includes(selectedFile)) {
            setSavedChapter(node, selectedFile);
        } else if (saved.value && !files.includes(saved.value)) {
            setSavedChapter(node, "");
        } else {
            updatePickerButton(node);
        }
        return files;
    } catch (error) {
        console.warn("[minimax_h3_novel] could not refresh saved chapters", error);
        return [];
    }
}

async function deleteSavedChapter(node, file, onComplete = null) {
    try {
        const response = await fetch("/minimax_h3_novel/chapters", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Delete failed");
        const paths = widget(node, "chapter_paths");
        if (paths?.value === file) {
            paths.value = "";
            paths.callback?.(paths.value);
        }
        if (widget(node, "saved_chapter")?.value === file) setSavedChapter(node, "");
        const files = await refreshSavedChapters(node);
        await onComplete?.(files);
    } catch (error) {
        alert(`MiniMax H3 chapter deletion failed: ${error.message}`);
    }
}

function chooseFiles(node, directory, onComplete = null) {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ACCEPTED;
    if (directory) input.webkitdirectory = true;
    input.style.display = "none";
    document.body.appendChild(input);
    input.addEventListener("change", async () => {
        let uploadedFiles = [];
        try {
            if (!input.files?.length) return;
            // Native directory pickers deliberately ignore ``accept``: they
            // select a directory, then expose every file below it.  Filter the
            // resulting FileList before upload so only supported chapters are
            // ever sent to the server.
            const chapterFiles = Array.from(input.files).filter((file) => {
                const dot = file.name.lastIndexOf(".");
                return dot >= 0 && CHAPTER_EXTENSIONS.has(file.name.slice(dot).toLowerCase());
            });
            if (!chapterFiles.length) {
                alert("The selected folder contains no supported chapter files (.txt, .md, .markdown, or .pdf).");
                return;
            }
            const form = new FormData();
            for (const file of chapterFiles) form.append("chapters", file, file.name);
            const response = await fetch("/minimax_h3_novel/upload", { method: "POST", body: form });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || "Upload failed");
            uploadedFiles = result.files || [];
            const paths = widget(node, "chapter_paths");
            if (paths) {
                paths.value = uploadedFiles.join("\n");
                paths.callback?.(paths.value);
            }
        } catch (error) {
            alert(`MiniMax H3 chapter upload failed: ${error.message}`);
        } finally {
            // The server may have accepted files before reporting an error (for
            // example, when a later file in a selected folder is unsupported).
            const files = await refreshSavedChapters(node, uploadedFiles[0] || null);
            await onComplete?.(files);
            input.remove();
        }
    }, { once: true });
    input.click();
}

function dialogButton(text, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.style.cssText = "padding:7px 10px; border:1px solid #555; border-radius:4px; background:#333; color:#eee; cursor:pointer;";
    button.addEventListener("click", onClick);
    return button;
}

async function openSavedChapterDialog(node) {
    node._minimaxH3ChapterDialog?.close();
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed; inset:0; z-index:10000; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,.55);";
    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.style.cssText = "width:min(560px, calc(100vw - 32px)); max-height:calc(100vh - 32px); display:flex; flex-direction:column; background:#222; color:#eee; border:1px solid #666; border-radius:6px; box-shadow:0 12px 36px #000; font:14px sans-serif;";
    const title = document.createElement("div");
    title.textContent = "Saved chapters";
    title.style.cssText = "padding:14px 16px 10px; font-weight:bold;";
    const list = document.createElement("div");
    list.style.cssText = "overflow:auto; min-height:72px; max-height:360px; border-top:1px solid #444; border-bottom:1px solid #444;";
    const footer = document.createElement("div");
    footer.style.cssText = "display:flex; gap:8px; justify-content:flex-end; padding:12px; flex-wrap:wrap;";
    dialog.append(title, list, footer);
    overlay.appendChild(dialog);

    let closed = false;
    const onKeyDown = (event) => {
        if (event.key === "Escape") {
            event.preventDefault();
            close();
        }
    };
    const close = () => {
        if (closed) return;
        closed = true;
        document.removeEventListener("keydown", onKeyDown, true);
        overlay.remove();
        if (node._minimaxH3ChapterDialog?.close === close) delete node._minimaxH3ChapterDialog;
    };
    overlay.addEventListener("pointerdown", (event) => {
        if (event.target === overlay) close();
    });
    document.addEventListener("keydown", onKeyDown, true);
    node._minimaxH3ChapterDialog = { close };

    const renderFiles = (files) => {
        list.replaceChildren();
        if (!files.length) {
            const empty = document.createElement("div");
            empty.textContent = "No saved chapters yet.";
            empty.style.cssText = "padding:18px 16px; color:#bbb;";
            list.appendChild(empty);
            return;
        }
        for (const file of files) {
            const row = document.createElement("div");
            row.tabIndex = 0;
            row.style.cssText = "display:flex; align-items:center; gap:10px; padding:9px 10px 9px 16px; cursor:pointer; border-bottom:1px solid #333;";
            const name = document.createElement("span");
            name.textContent = file;
            name.style.cssText = "overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;";
            const trash = dialogButton("🗑", (event) => {
                event.stopPropagation();
                deleteSavedChapter(node, file, (files) => {
                    if (!closed) renderFiles(files);
                });
            });
            trash.title = `Delete ${file}`;
            trash.setAttribute("aria-label", trash.title);
            trash.style.cssText += "padding:4px 7px;";
            const select = () => {
                setSavedChapter(node, file);
                close();
            };
            row.addEventListener("click", select);
            row.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    select();
                }
            });
            row.append(name, trash);
            list.appendChild(row);
        }
    };
    const uploadAndRefresh = (directory) => chooseFiles(node, directory, (files) => {
        if (!closed) renderFiles(files);
    });
    footer.append(
        dialogButton("Select chapter files", () => uploadAndRefresh(false)),
        dialogButton("Select chapter folder", () => uploadAndRefresh(true)),
    );
    document.body.appendChild(overlay);

    try {
        const files = await fetchSavedChapters();
        const saved = widget(node, "saved_chapter");
        if (saved) saved.options.values = files.length ? files : [""];
        if (closed) return;
        renderFiles(files);
    } catch (error) {
        if (!closed) {
            const message = document.createElement("div");
            message.textContent = `Could not load saved chapters: ${error.message}`;
            message.style.cssText = "padding:18px 16px; color:#f99;";
            list.appendChild(message);
        }
    }
}

function installSavedChapterPicker(node) {
    const saved = widget(node, "saved_chapter");
    if (!saved || pickerButton(node)) return;
    // Keep this hidden enum value so existing workflows and the Python nodes
    // still receive saved_chapter, while the visible control is a dialog button.
    saved.type = "hidden";
    saved.computeSize = () => [0, -4];
    const originalCallback = saved.callback;
    saved.callback = (value) => {
        originalCallback?.(value);
        setChapterPaths(node, value);
        updatePickerButton(node);
    };
    const button = node.addWidget("button", PICKER_BUTTON_NAME, null, () => openSavedChapterDialog(node));
    button.serialize = false;
    updatePickerButton(node);
    refreshSavedChapters(node);
}

app.registerExtension({
    name: EXTENSION_NAME,
    async nodeCreated(node) {
        if (node.comfyClass === "ConsolidateReferencesNode") migrateLegacyConsolidationDefaults(node);
        if (node.comfyClass === "ExtractChapterReferencesNode") migrateLegacyExtractDefaults(node);
        if (node.comfyClass === "GenerateH3PromptsNode") migrateLegacyPromptDefaults(node);
        installSavedChapterPicker(node);
    },
});

const LMSTUDIO_API_KEY_SETTING = "MiniMaxH3Novel.LMStudio.ApiKey";

function ensureLmStudioApiKeyFieldWidth() {
    const styleId = "minimax_h3_novel_lmstudio_api_key_field_width";
    if (document.getElementById(styleId)) return;
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
        #MiniMaxH3Novel\\.LMStudio\\.ApiKey,
        [id="MiniMaxH3Novel.LMStudio.ApiKey"] {
            width: 500px !important;
            max-width: 100% !important;
            min-width: 300px;
        }
    `;
    document.head.appendChild(style);
}

async function sendLMStudioApiKey() {
    try {
        const apiKey = app.ui.settings.getSettingValue(LMSTUDIO_API_KEY_SETTING) || "";
        const response = await fetch("/minimax_h3_novel/lmstudio-settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: apiKey }),
        });
        if (!response.ok) {
            console.error("MiniMax H3 Novel: could not save the LM Studio API-key setting.");
        }
    } catch (error) {
        console.error("MiniMax H3 Novel: failed to send the LM Studio API-key setting.", error);
    }
}

// This follows ComfyUI's normal global Settings integration. The key is saved
// by ComfyUI in the local browser profile, never embedded in a workflow, and
// is passed to the backend only immediately before a workflow is queued.
app.registerExtension({
    name: "minimax_h3_novel.lmstudio_settings",
    settings: [
        {
            id: LMSTUDIO_API_KEY_SETTING,
            name: "LM Studio API Key",
            type: "text",
            default: "",
            category: ["MiniMax H3 Novel", "LM Studio"],
            tooltip: "Stored locally in ComfyUI settings, not in workflow JSON. Leave blank to use the environment-variable fallback.",
        },
    ],
    async setup() {
        ensureLmStudioApiKeyFieldWidth();
        // Settings are loaded after extensions; sync once they are available.
        setTimeout(() => { sendLMStudioApiKey(); }, 500);
        const originalSetSettingValue = app.ui.settings.setSettingValue;
        app.ui.settings.setSettingValue = function(id, value) {
            originalSetSettingValue.call(this, id, value);
            if (id === LMSTUDIO_API_KEY_SETTING) sendLMStudioApiKey();
        };
    },
    async beforeQueuing() {
        await sendLMStudioApiKey();
        return null;
    },
});
