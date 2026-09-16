import { builtinEnvironments, type Environment } from "vitest/runtime";

// Compile Vue templates for the client while mounting into a custom renderer.
// These component tests need lifecycle/watchers, but no browser or WebGL.
export default {
  ...builtinEnvironments.node,
  name: "vue-renderer",
  viteEnvironment: "client",
} satisfies Environment;
