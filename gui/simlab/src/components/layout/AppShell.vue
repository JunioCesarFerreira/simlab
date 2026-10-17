<template>
  <div class="shell">
    <Sidebar class="sidebar" />
    <div class="main">
      <Topbar class="topbar" />
      <div class="content" :class="{ 'full-screen': route.meta.fullScreen }">
        <slot />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRoute } from "vue-router";
import Sidebar from "./Sidebar.vue";
import Topbar from "./TopBar.vue";

const route = useRoute();
</script>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  height: 100vh;
  background: var(--color-bg);
}

html.dark .shell {
  --color-text-muted: #a6adc8;
}

.sidebar {
  border-right: 1px solid var(--color-border);
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}

.main {
  display: grid;
  grid-template-rows: 48px 1fr;
  height: 100vh;
  overflow: hidden;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
}

.content {
  padding: 24px;
  overflow-y: auto;
}

.content.full-screen {
  padding: 0;
  overflow: hidden;
}
@media (max-width: 760px) {
  .shell {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: auto minmax(0, 1fr);
    height: 100dvh;
  }

  .sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--color-border);
  }

  .main {
    min-height: 0;
    height: auto;
  }

  .content {
    padding: 16px;
  }
}
</style>
