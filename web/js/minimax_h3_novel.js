import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "minimax_h3_novel.chapter_picker";
const ACCEPTED = ".txt,.md,.markdown,.pdf";
const CHAPTER_EXTENSIONS = new Set(ACCEPTED.split(","));
const PICKER_BUTTON_NAME = "saved_chapter_picker";

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

function chooseFiles(node, onComplete = null) {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ACCEPTED;
    input.style.display = "none";
    document.body.appendChild(input);
    input.addEventListener("change", async () => {
        let uploadedFiles = [];
        try {
            if (!input.files?.length) return;
            // Validate extensions because file dialogs can override accept.
            const chapterFiles = Array.from(input.files).filter((file) => {
                const dot = file.name.lastIndexOf(".");
                return dot >= 0 && CHAPTER_EXTENSIONS.has(file.name.slice(dot).toLowerCase());
            });
            if (!chapterFiles.length) {
                alert("Select supported chapter files (.txt, .md, .markdown, or .pdf).");
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
            // The server may have accepted some files before reporting an error.
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
            updateAddSelectionButton();
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
    const uploadAndRefresh = () => chooseFiles(node, (files) => {
        if (!closed) renderFiles(files);
    });
    const selectionControls = document.createElement("div");
    selectionControls.style.cssText = "display:flex; gap:8px; margin-right:auto;";
    selectionControls.append(
        dialogButton("Select all", () => {
            for (const file of currentFiles) selectedFiles.add(file);
            renderFiles(currentFiles);
        }),
        dialogButton("Select none", () => {
            selectedFiles.clear();
            renderFiles(currentFiles);
        }),
    );
    footer.append(
        selectionControls,
        dialogButton("Select chapter files", uploadAndRefresh),
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

let availableProviders = null;

function updateProviderNodes(providers) {
    availableProviders = providers;
    for (const node of app.graph?._nodes || []) node._minimaxH3UpdateProvider?.();
}

function installModelOptions(node) {
    if (node.comfyClass !== "LMStudioConfigurationNode") return;
    const backend = widget(node, "backend");
    if (!backend || node._minimaxH3UpdateProvider) return;
    backend.label = "provider";
    backend.label = "provider";
    const originalOrder = [...node.widgets];
    const model = node.addWidget("text", "model", "", () => {
        model.value = selectedProvider()?.model || "Configure and save a provider in Settings";
    }, { serialize: false, read_only: true });
    model.serialize = false;
    model.disabled = true;
    const selectedProvider = () => availableProviders?.find((item) => item.backend === backend.value);
    const qwenWidgets = originalOrder.filter((item) => item.name.startsWith("qwen35_"));
    const originalSizes = new Map(qwenWidgets.map((item) => [item, { type: item.type, computeSize: item.computeSize }]));
    function arrange(grow = true) {
        const values = availableProviders?.map((item) => item.backend);
        if (values) {
            backend.options.values = values;
            if (!values.includes(backend.value)) backend.value = values[0] || "";
        }
        const selected = selectedProvider();
        model.value = selected?.model || "Configure and save a provider in Settings";
        const widgets = node.widgets;
        widgets.splice(widgets.indexOf(model), 1);
        widgets.splice(widgets.indexOf(backend) + 1, 0, model);
        const isQwen = selected?.backend === "LM Studio" && selected.model_family === "Qwen";
        for (const item of qwenWidgets) {
            const original = originalSizes.get(item);
            item.type = isQwen ? original.type : "hidden";
            item.computeSize = isQwen ? original.computeSize : () => [0, -4];
        }
        // Only grow to fit new controls. Serialization and settings refreshes must
        // never undo a user's resize or restore the node's minimum dimensions.
        if (grow) {
            const minimum = node.computeSize();
            const current = node.size || minimum;
            if (current[0] < minimum[0] || current[1] < minimum[1]) {
                node.setSize?.([Math.max(current[0], minimum[0]), Math.max(current[1], minimum[1])]);
            }
        }
        app.graph?.setDirtyCanvas(true, true);
    }
    node._minimaxH3UpdateProvider = arrange;
    // LiteGraph persists widget values by position; the display-only model must
    // not shift the backend or Qwen values when saving and loading a workflow.
    for (const method of ["configure", "serialize"]) {
        const original = node[method];
        if (!original) continue;
        node[method] = function (...args) {
            const extras = this.widgets.filter((item) => !originalOrder.includes(item));
            this.widgets.splice(0, this.widgets.length, ...originalOrder, ...extras);
            try { return original.apply(this, args); }
            finally { arrange(method !== "serialize"); }
        };
    }
    const callback = backend.callback;
    backend.callback = function (...args) {
        const result = callback?.apply(this, args);
        arrange();
        return result;
    };
    arrange();
}

app.registerExtension({
    name: EXTENSION_NAME,
    async nodeCreated(node) {
        installSavedChapterPicker(node);
        installModelOptions(node);
    },
});


const CONNECTIONS_SETTING = "MiniMaxH3Novel.Connections";
const PROVIDERS = [
    { name: "LM Studio", key: "lmstudio",
        defaults: { enabled: true, api_url: "http://127.0.0.1:1234/v1", api_key: "", model: "" } },
    { name: "OpenAI", key: "openai",
        defaults: { enabled: false, api_url: "https://api.openai.com/v1", api_key: "", model: "" } },
];

function providerValue(provider) {
    return { ...provider.defaults, ...app.ui.settings.getSettingValue(CONNECTIONS_SETTING)?.[provider.key] };
}

async function sendProviderSettings() {
    const response = await localFetch("/minimax_h3_novel/provider-settings", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            ...Object.fromEntries(PROVIDERS.map((provider) => [provider.key, providerValue(provider)])) }),
    });
    if (!response.ok) throw new Error("Could not save provider settings.");
    const result = await response.json();
    updateProviderNodes(result.providers || []);
}

function installProviderPanelStyles() {
    if (document.getElementById("minimax-h3-provider-styles")) return;
    const style = document.createElement("style");
    style.id = "minimax-h3-provider-styles";
    style.textContent = `
        /* Give the custom panel the entire native settings row. */
        tr:has(.minimax-h3-provider-panel) { display:block !important; width:100%; }
        tr:has(.minimax-h3-provider-panel) > td:first-child { display:none !important; }
        tr:has(.minimax-h3-provider-panel) > td:last-child { display:block !important; width:100% !important; }
        .setting-item:has(.minimax-h3-provider-panel) > div { display:block !important; width:100%; }
        .setting-item:has(.minimax-h3-provider-panel) > div > div:first-child { display:none !important; }
        .setting-item:has(.minimax-h3-provider-panel) > div > div:last-child { display:block !important; width:100%; }
        .minimax-h3-provider-panel {
            display:flex; flex-direction:column; gap:16px; width:100%; min-width:0;
            flex:1 1 100%; text-align:left; font:inherit; color:var(--input-text);
        }
        .minimax-h3-provider-card {
            padding:20px; width:100%; box-sizing:border-box;
            border:1px solid var(--border-color); border-radius:var(--p-border-radius-lg, 8px);
            background:var(--comfy-menu-bg); color:var(--input-text);
        }
        .minimax-h3-provider-header {
            display:flex; align-items:center; gap:12px; font-weight:600; cursor:pointer;
        }
        .minimax-h3-provider-body { margin-top:20px; }
        .minimax-h3-provider-field {
            display:block; margin:16px 0; color:var(--input-text); font-size:inherit;
        }
        .minimax-h3-provider-control {
            display:block; width:100%; box-sizing:border-box; margin-top:8px; padding:10px 12px;
            min-height:40px; font:inherit; border:1px solid var(--border-color);
            border-radius:var(--p-form-field-border-radius, 6px);
            background:var(--comfy-input-bg); color:var(--input-text);
        }
        .minimax-h3-provider-control:hover { border-color:var(--input-text); }
        .minimax-h3-provider-control[readonly] { color:var(--descrip-text, var(--input-text)); }
        .minimax-h3-provider-control option { background:var(--comfy-input-bg); color:var(--input-text); }
        .minimax-h3-provider-status {
            min-height:1.5em; margin:12px 0; overflow-wrap:anywhere; color:var(--input-text);
        }
        .minimax-h3-provider-actions { display:flex; justify-content:flex-end; flex-wrap:wrap; gap:12px; }
        .minimax-h3-provider-button {
            padding:10px 24px; min-height:40px; font:inherit; cursor:pointer;
            border:1px solid var(--border-color); border-radius:var(--p-form-field-border-radius, 6px);
            background:var(--comfy-input-bg); color:var(--input-text);
        }
        .minimax-h3-provider-button:hover:not(:disabled) {
            background:var(--comfy-menu-secondary-bg, var(--comfy-menu-bg)); border-color:var(--input-text);
        }
        .minimax-h3-provider-button-save { font-weight:600; }
        .minimax-h3-provider-panel :is(button, input):disabled { opacity:.5; cursor:wait; }
        .minimax-h3-provider-toggle {
            appearance:none; width:36px; height:20px; border-radius:20px;
            position:relative; background:var(--comfy-input-bg); cursor:pointer;
            flex-shrink:0; margin:0; border:1px solid var(--border-color);
        }
        .minimax-h3-provider-toggle:checked { background:var(--input-text); border-color:var(--input-text); }
        .minimax-h3-provider-toggle::before {
            content:""; position:absolute; width:14px; height:14px; border-radius:50%;
            background:var(--input-text); top:2px; left:2px; transition:transform .15s;
        }
        .minimax-h3-provider-toggle:checked::before {
            transform:translateX(16px); background:var(--comfy-input-bg);
        }
        .minimax-h3-provider-panel :is(input, select, button):focus-visible {
            outline:2px solid var(--input-text); outline-offset:3px;
        }
        @media (prefers-reduced-motion: reduce) {
            .minimax-h3-provider-toggle::before { transition:none; }
        }
    `;
    document.head.appendChild(style);
}
async function providerResponse(response) {
    if (response.status === 404 || response.status === 405) {
        throw new Error("ComfyUI has not loaded this endpoint. Restart ComfyUI, then refresh this page.");
    }
    let result;
    try {
        result = await response.json();
    } catch {
        throw new Error(`The server returned a non-JSON response (HTTP ${response.status}).`);
    }
    if (!response.ok) throw new Error(result.error || `Provider request failed (HTTP ${response.status}).`);
    return result;
}
function providerCard(provider, setter, value) {
    const saved = { ...provider.defaults, ...value };
    const card = document.createElement("section");
    card.className = "minimax-h3-provider-card";
    const header = document.createElement("label");
    header.className = "minimax-h3-provider-header";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-label", `Enable ${provider.name}`);
    toggle.checked = saved.enabled;
    toggle.className = "minimax-h3-provider-toggle";
    header.append(toggle, document.createTextNode(provider.name));
    card.appendChild(header);
    const body = document.createElement("div");
    body.hidden = !toggle.checked;
    body.className = "minimax-h3-provider-body";
    card.appendChild(body);
    const fields = {};
    let modelIds = Array.isArray(saved.available_models) && saved.available_models.length
        ? [...saved.available_models] : [];
    for (const [key, title, type] of [["api_url", "API base URL", "url"], ["api_key", "API key", "password"],
        ["model", "Model", "select"]]) {
        const label = document.createElement("label");
        label.className = "minimax-h3-provider-field";
        label.textContent = title;
        const input = document.createElement(type === "select" ? "select" : "input");
        if (type === "select") {
            for (const model of modelIds) {
                const option = document.createElement("option");
                option.value = model;
                option.textContent = model;
                input.appendChild(option);
            }
            input.value = modelIds.includes(saved[key]) ? saved[key] : "";
        } else {
            input.type = type;
            input.value = saved[key];
        }
        input.autocomplete = "off";
        input.spellcheck = false;
        input.className = "minimax-h3-provider-control";
        if (provider.key === "openai" && key === "api_url") {
            input.readOnly = true;
            input.title = "OpenAI credentials are restricted to the official OpenAI endpoint.";
        }
        fields[key] = input;
        label.appendChild(input);
        body.appendChild(label);
    }
    const status = document.createElement("div");
    status.setAttribute("role", "status");
    status.className = "minimax-h3-provider-status";
    body.appendChild(status);
    const actions = document.createElement("div");
    actions.className = "minimax-h3-provider-actions";
    body.appendChild(actions);
    const draft = () => ({ enabled: toggle.checked,
        available_models: [...modelIds],
        ...Object.fromEntries(Object.entries(fields).map(([key, input]) => [key, input.value.trim()])) });
    const buttons = [];
    for (const title of ["Fetch models", "Test", "Save"]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = title;
        button.className = `minimax-h3-provider-button${title === "Save" ? " minimax-h3-provider-button-save" : ""}`;
        button.addEventListener("click", async () => {
            buttons.forEach((item) => { item.disabled = true; });
            status.textContent = title === "Test" ? "Testing…" : title === "Fetch models" ? "Fetching models…" : "Saving…";
            try {
                const connection = draft();
                if (title === "Fetch models") {
                    const response = await localFetch("/minimax_h3_novel/provider-models", {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ provider: provider.key, ...connection }),
                    });
                    const result = await providerResponse(response);
                    if (!Array.isArray(result.models) || !result.models.length
                        || result.models.some((id) => typeof id !== "string" || !id.trim())) {
                        throw new Error("No available models returned. The current list is unchanged.");
                    }
                    const selected = fields.model.value;
                    modelIds = [...new Set(result.models)].sort();
                    const options = modelIds.map((id) => {
                        const option = document.createElement("option");
                        option.value = id;
                        option.textContent = id;
                        return option;
                    });
                    fields.model.replaceChildren(...options);
                    fields.model.value = modelIds.includes(selected) ? selected : modelIds[0];
                    status.textContent = `${modelIds.length} models retrieved. ${modelIds.includes(selected)
                        ? "Selection preserved." : "Previous model unavailable; first model selected."} Click Save to apply.`;
                } else if (title === "Test") {
                    const response = await localFetch("/minimax_h3_novel/provider-test", {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ provider: provider.key, ...connection }),
                    });
                    const result = await providerResponse(response);
                    status.textContent = result.message;
                } else {
                    // Validate and install on the server before persisting the card.
                    const payload = {
                        ...Object.fromEntries(PROVIDERS.map((item) => [item.key,
                            item.key === provider.key ? connection : providerValue(item)])) };
                    const response = await localFetch("/minimax_h3_novel/provider-settings", {
                        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
                    });
                    if (!response.ok) throw new Error("Could not save. Check the URL, key and model.");
                    const result = await response.json();
                    await setter(connection);
                    Object.assign(saved, connection);
                    updateProviderNodes(result.providers || []);
                    status.textContent = "Saved.";
                }
            } catch (error) {
                status.textContent = error.message;
            } finally {
                buttons.forEach((item) => { item.disabled = false; });
            }
        });
        buttons.push(button);
        actions.appendChild(button);
    }
    toggle.addEventListener("change", async () => {
        body.hidden = !toggle.checked;
        toggle.disabled = true;
        try {
            await setter({ ...saved, enabled: toggle.checked });
            await sendProviderSettings();
            saved.enabled = toggle.checked;
        } catch {
            toggle.checked = saved.enabled;
            body.hidden = !toggle.checked;
            status.textContent = "Could not change provider availability. Retry.";
        } finally { toggle.disabled = false; }
    });
    return card;
}

app.registerExtension({
    name: "minimax_h3_novel.provider_settings",
    settings: [{
        id: CONNECTIONS_SETTING, name: " ",
        category: ["📖 MiniMax H3 Novel", "Providers", "Connections"],
        default: Object.fromEntries(PROVIDERS.map((provider) => [provider.key, provider.defaults])),
        type: (_name, setter, value) => {
            const panel = document.createElement("div");
            panel.className = "minimax-h3-provider-panel";
            let connections = { ...value };
            for (const provider of PROVIDERS) {
                panel.appendChild(providerCard(provider, async (connection) => {
                    const next = { ...connections, [provider.key]: connection };
                    await setter(next);
                    connections = next;
                }, connections[provider.key]));
            }
            return panel;
        },
    }],
    async setup() {
        installProviderPanelStyles();
        await sendProviderSettings();
    },
    async beforeQueuing() {
        await sendProviderSettings();
        return null;
    },
});
