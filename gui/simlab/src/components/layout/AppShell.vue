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
  grid-template-columns: 260px 1fr;
  min-height: 100vh;
}
.sidebar {
  border-right: 1px solid #e5e7eb;
}
.main {
  display: grid;
  grid-template-rows: 56px 1fr;
}
.topbar {
  border-bottom: 1px solid #e5e7eb;
}
.content {
  padding: 16px;
}
</style>
