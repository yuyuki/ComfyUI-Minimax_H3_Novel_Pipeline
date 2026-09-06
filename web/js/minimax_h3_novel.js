import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "minimax_h3_novel.chapter_picker";
const ACCEPTED = ".txt,.md,.markdown,.pdf";
const CHAPTER_EXTENSIONS = new Set(ACCEPTED.split(","));
const PICKER_BUTTON_NAME = "saved_chapter_picker";

app.registerExtension({
    name: "minimax_h3_novel.config_migration",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LMStudioConfigurationNode") return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            let values = info.widgets_values;
            // Migrate both legacy layouts: with both shared limits, or chunk cap only.
            if (Array.isArray(values) && values.length === 8) {
                values = [...values.slice(0, 2), ...values.slice(3)];
            }
            if (Array.isArray(values) && values.length === 7) {
                values = [...values.slice(0, 3), ...values.slice(4)];
                info = { ...info, widgets_values: values };
            }
            if (info.widgets_values_named) {
                const named = { ...info.widgets_values_named };
                delete named.qwen35_max_output_tokens;
                delete named.qwen35_safe_chunk_chars;
                info = { ...info, widgets_values_named: named };
            }
            return configure?.apply(this, [info]);
        };
    },
});

function localFetch(url, options = {}) {
    const headers = new Headers(options.headers);
    headers.set("X-MiniMax-H3-Request", "1");
    return fetch(url, { ...options, headers });
}

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function pickerButton(node) {
    return widget(node, PICKER_BUTTON_NAME);
}

function updatePickerButton(node) {
    const button = pickerButton(node);
    const saved = widget(node, "saved_chapter");
    if (!button || !saved) return;
    const selected = String(widget(node, "chapter_paths")?.value || "")
        .split(/\r?\n/)
        .filter(Boolean);
    button.label = saved.value
        || (selected.length === 1 ? selected[0] : selected.length > 1 ? `${selected.length} chapters selected` : "Select chapters");
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

function setSavedChapters(node, files) {
    const selected = [...new Set(files)];
    const paths = widget(node, "chapter_paths");
    if (paths) {
        paths.value = selected.join("\n");
        paths.callback?.(paths.value);
    }
    setSavedChapter(node, selected.length === 1 ? selected[0] : "");
}

async function fetchSavedChapters() {
    const response = await localFetch("/minimax_h3_novel/chapters");
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
        const response = await localFetch("/minimax_h3_novel/chapters", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Delete failed");
        const selected = String(widget(node, "chapter_paths")?.value || "")
            .split(/\r?\n/)
            .filter((path) => path && path !== file);
        if (selected.length !== String(widget(node, "chapter_paths")?.value || "").split(/\r?\n/).filter(Boolean).length) {
            setSavedChapters(node, selected);
        } else if (widget(node, "saved_chapter")?.value === file) {
            setSavedChapter(node, "");
        }
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
            const response = await localFetch("/minimax_h3_novel/upload", { method: "POST", body: form });
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

    const selectedFiles = new Set(
        String(widget(node, "chapter_paths")?.value || widget(node, "saved_chapter")?.value || "")
            .split(/\r?\n/)
            .filter(Boolean),
    );
    let currentFiles = [];
    let addSelection;
    const updateAddSelectionButton = () => {
        if (!addSelection) return;
        addSelection.disabled = selectedFiles.size === 0;
        addSelection.style.opacity = addSelection.disabled ? "0.55" : "1";
        addSelection.style.cursor = addSelection.disabled ? "default" : "pointer";
    };

    const renderFiles = (files) => {
        currentFiles = files;
        for (const file of selectedFiles) {
            if (!files.includes(file)) selectedFiles.delete(file);
        }
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
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = selectedFiles.has(file);
            checkbox.title = `Select ${file}`;
            checkbox.setAttribute("aria-label", checkbox.title);
            checkbox.addEventListener("click", (event) => event.stopPropagation());
            checkbox.addEventListener("change", () => {
                if (checkbox.checked) selectedFiles.add(file);
                else selectedFiles.delete(file);
                updateAddSelectionButton();
            });
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
            const toggleSelection = () => {
                checkbox.checked = !checkbox.checked;
                checkbox.dispatchEvent(new Event("change"));
            };
            row.addEventListener("click", toggleSelection);
            row.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    toggleSelection();
                }
            });
            row.append(checkbox, name, trash);
            list.appendChild(row);
        }
        updateAddSelectionButton();
    };
    const uploadAndRefresh = (directory) => chooseFiles(node, directory, (files) => {
        if (!closed) renderFiles(files);
    });
    footer.append(
        dialogButton("Select chapter files", () => uploadAndRefresh(false)),
        dialogButton("Select chapter folder", () => uploadAndRefresh(true)),
        addSelection = dialogButton("Add selection", () => {
            const selected = currentFiles.filter((file) => selectedFiles.has(file));
            if (!selected.length) return;
            setSavedChapters(node, selected);
            close();
        }),
    );
    updateAddSelectionButton();
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
    if (node.comfyClass !== "SelectChaptersNode") return;
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
        const response = await localFetch("/minimax_h3_novel/lmstudio-settings", {
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
