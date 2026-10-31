const searchInput = document.querySelector("#node-search");
const categoryNav = document.querySelector("#categories");
const workflowList = document.querySelector("#workflow");
const nodeList = document.querySelector("#nodes");
const nodeCount = document.querySelector("#node-count");
const status = document.querySelector("#status");

let catalog;
let activeCategory = "All nodes";

function makeElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

function renderWorkflow() {
  workflowList.replaceChildren();
  for (const [index, nodeId] of catalog.workflow.entries()) {
    const node = catalog.nodes.find((entry) => entry.id === nodeId);
    const item = makeElement("li");
    item.append(
      makeElement("span", "step-number", `STEP ${index + 1}`),
      makeElement("span", "step-name", node.name),
    );
    workflowList.append(item);
  }
}

function renderCategoryFilters() {
  categoryNav.replaceChildren();
  const availableCategories = ["All nodes", ...catalog.categories];

  for (const category of availableCategories) {
    const button = makeElement("button", "category-button", category);
    button.type = "button";
    button.setAttribute("aria-pressed", String(category === activeCategory));
    button.addEventListener("click", () => {
      activeCategory = category;
      renderCategoryFilters();
      renderNodes();
    });
    categoryNav.append(button);
  }
}

function renderNodes() {
  const query = searchInput.value.trim().toLocaleLowerCase();
  const matches = catalog.nodes.filter((node) => {
    const inCategory = activeCategory === "All nodes" || node.category === activeCategory;
    const matchesQuery = [node.id, node.name, node.category, node.purpose].some((value) =>
      value.toLocaleLowerCase().includes(query),
    );
    return inCategory && matchesQuery;
  });

  nodeList.replaceChildren();
  nodeCount.textContent = `${matches.length} / ${catalog.nodes.length} nodes`;
  status.textContent = matches.length
    ? `Showing ${matches.length} ${matches.length === 1 ? "node" : "nodes"}.`
    : "No nodes match those filters.";

  for (const node of matches) {
    const item = makeElement("li", "node-card");
    const details = makeElement("details");
    const summary = makeElement("summary");
    summary.append(
      makeElement("span", "node-name", node.name),
      makeElement("span", "node-category", node.category),
    );

    const purpose = makeElement("p", "node-purpose", node.purpose);
    const idRow = makeElement("div", "node-id-row");
    const id = makeElement("code", "", node.id);
    const copyButton = makeElement("button", "copy-button", "Copy ID");
    copyButton.type = "button";
    copyButton.setAttribute("aria-label", `Copy node ID ${node.id}`);
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(node.id);
        status.textContent = `Copied ${node.id}.`;
      } catch {
        status.textContent = `Could not copy ${node.id}. Select the node ID to copy it manually.`;
      }
    });

    idRow.append(id, copyButton);
    details.append(summary, purpose, idRow);
    item.append(details);
    nodeList.append(item);
  }
}

async function loadCatalog() {
  try {
    const response = await fetch("/api/catalog", { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`Catalog request returned HTTP ${response.status}.`);
    }

    const result = await response.json();
    if (
      !result ||
      !Array.isArray(result.nodes) ||
      !Array.isArray(result.categories) ||
      !Array.isArray(result.workflow) ||
      result.categories.some((category) => typeof category !== "string") ||
      result.workflow.some(
        (nodeId) => typeof nodeId !== "string" || !result.nodes.some((node) => node?.id === nodeId),
      ) ||
      result.nodes.some(
        (node) =>
          typeof node?.id !== "string" ||
          typeof node?.name !== "string" ||
          typeof node?.category !== "string" ||
          typeof node?.purpose !== "string",
      )
    ) {
      throw new Error("The node catalog response has an invalid shape.");
    }

    catalog = result;
    renderWorkflow();
    renderCategoryFilters();
    renderNodes();
  } catch (error) {
    nodeCount.textContent = "";
    nodeList.replaceChildren();
    status.textContent = `Unable to load the node catalog: ${error.message}`;
  }
}

searchInput.addEventListener("input", renderNodes);
loadCatalog();
