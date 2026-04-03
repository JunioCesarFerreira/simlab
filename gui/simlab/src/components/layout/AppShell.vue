<template>
  <div class="shell">
    <Sidebar class="sidebar" />
    <div class="main">
      <Topbar class="topbar" />
      <div class="content">
        <slot />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import Sidebar from "./Sidebar.vue";
import Topbar from "./TopBar.vue";
</script>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: 220px 1fr;
  min-height: 100vh;
  background: var(--color-bg);
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
  min-height: 100vh;
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
</style>
