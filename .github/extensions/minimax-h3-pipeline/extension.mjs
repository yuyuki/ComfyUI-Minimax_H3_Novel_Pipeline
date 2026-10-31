import { CanvasError, createCanvas, joinSession } from "@github/copilot-sdk/extension";
import { findNode, getPipelineOverview, pipeline, searchNodes } from "./catalog.mjs";
import { startPipelineServer } from "./server.mjs";

const instances = new Map();
let session;

const searchSchema = {
  type: "object",
  properties: {
    query: { type: "string", minLength: 1, maxLength: 120 },
  },
  required: ["query"],
  additionalProperties: false,
};

const nodeDetailsSchema = {
  type: "object",
  properties: {
    nodeId: { type: "string", enum: pipeline.nodes.map((node) => node.id) },
  },
  required: ["nodeId"],
  additionalProperties: false,
};

function requireActionObject(input, property) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new CanvasError("invalid_input", `Expected an object containing ${property}.`);
  }
  const keys = Object.keys(input);
  if (keys.length !== 1 || keys[0] !== property) {
    throw new CanvasError("invalid_input", `Expected only the ${property} property.`);
  }
}

async function getInstanceServer(instanceId) {
  let pending = instances.get(instanceId);
  if (!pending) {
    pending = startPipelineServer();
    instances.set(instanceId, pending);
    try {
      const server = await pending;
      if (instances.get(instanceId) === pending) {
        pending = Promise.resolve(server);
        instances.set(instanceId, pending);
      }
      return server;
    } catch (error) {
      if (instances.get(instanceId) === pending) {
        instances.delete(instanceId);
      }
      throw error;
    }
  }
  return pending;
}

const canvas = createCanvas({
  id: "minimax-h3-novel-pipeline",
  displayName: "MiniMax H3 Pipeline",
  description:
    "Shows the registered MiniMax H3 Novel Pipeline nodes, their workflow stages, and quick node-ID lookup.",
  actions: [
    {
      name: "get_pipeline_overview",
      description: "Return the registered node count, categories, and primary workflow order.",
      handler: () => getPipelineOverview(),
    },
    {
      name: "search_nodes",
      description: "Find registered nodes by name, purpose, category, or ComfyUI node ID.",
      inputSchema: searchSchema,
      handler: ({ input }) => {
        requireActionObject(input, "query");
        if (typeof input.query !== "string" || input.query.trim().length === 0 || input.query.length > 120) {
          throw new CanvasError("invalid_input", "query must contain 1 to 120 non-whitespace characters.");
        }
        return searchNodes(input.query);
      },
    },
    {
      name: "get_node_details",
      description: "Return the purpose and category for one registered ComfyUI node ID.",
      inputSchema: nodeDetailsSchema,
      handler: ({ input }) => {
        requireActionObject(input, "nodeId");
        if (typeof input.nodeId !== "string") {
          throw new CanvasError("invalid_input", "nodeId must be a registered node ID.");
        }
        const node = findNode(input.nodeId);
        if (!node) {
          throw new CanvasError("node_not_found", `No registered node has ID ${input.nodeId}.`);
        }
        return node;
      },
    },
  ],
  open: async ({ instanceId }) => {
    if (typeof instanceId !== "string" || instanceId.length === 0) {
      throw new CanvasError("invalid_instance", "A non-empty canvas instance ID is required.");
    }
    try {
      const server = await getInstanceServer(instanceId);
      return { title: "MiniMax H3 Pipeline", url: server.url };
    } catch (error) {
      session.log(`Unable to start MiniMax H3 Pipeline canvas server: ${String(error)}`);
      throw new CanvasError("server_start_failed", "Unable to start the local pipeline canvas server.");
    }
  },
  onClose: async ({ instanceId }) => {
    const pending = instances.get(instanceId);
    if (!pending) {
      return;
    }
    instances.delete(instanceId);
    try {
      const server = await pending;
      await server.close();
    } catch (error) {
      session.log(`Unable to stop MiniMax H3 Pipeline canvas server: ${String(error)}`);
    }
  },
});

session = await joinSession({ canvases: [canvas] });
