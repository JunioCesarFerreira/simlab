<template>
  <header class="topbar">
    <div class="left">Gestão e análise de experimentos</div>
    <div class="right">
      <span class="pill">v0.1</span>
    </div>
  </header>
</template>

<script setup lang="ts"></script>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  font-size: 13px;
  color: var(--color-text-muted);
}

.pill {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border: 1px solid var(--color-border);
  border-radius: 999px;
  color: var(--color-text-muted);
  background: var(--color-bg);
}
</style>
