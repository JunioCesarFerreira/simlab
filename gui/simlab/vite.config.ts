import { defineConfig } from "vitest/config";
import vue from "@vitejs/plugin-vue";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const { version } = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf-8"));

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  define: {
    __APP_VERSION__: JSON.stringify(version),
  },
  test: {
    environment: "node",
    alias: {
      "vitest-environment-vue-renderer": fileURLToPath(new URL("./src/test/vueRendererEnvironment.ts", import.meta.url)),
    },
    include: ["src/**/*.spec.ts"],
  },
});
