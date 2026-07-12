<template>
  <div class="notfound">
    <div class="code">404</div>
    <p class="msg">This page doesn't exist.</p>
    <RouterLink to="/dashboard" class="home-link">← Back to dashboard</RouterLink>
  </div>
</template>

<script setup lang="ts">
import { RouterLink } from "vue-router";
</script>

<style scoped>
.notfound {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 50vh;
  text-align: center;
}

.code {
  font-size: 56px;
  font-weight: 800;
  color: var(--color-text-muted);
  letter-spacing: 0.04em;
}

.msg {
  font-size: 14px;
  color: var(--color-text-muted);
  margin: 0;
}

.home-link {
  margin-top: 8px;
  font-size: 13px;
  font-weight: 500;
  color: var(--color-primary);
}

.home-link:hover {
  text-decoration: underline;
}
</style>
