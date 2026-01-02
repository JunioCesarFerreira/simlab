<template>
  <nav class="nav">
    <div class="title">SimLab Console</div>
    <RouterLink to="/dashboard" class="link">Dashboard</RouterLink>
    <RouterLink to="/experiments" class="link">Experimentos</RouterLink>
  </nav>
</template>

<script setup lang="ts"></script>

<style scoped>
.nav {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.title {
  font-weight: 700;
  margin-bottom: 12px;
}
.link {
  text-decoration: none;
  padding: 8px 10px;
  border-radius: 8px;
  color: #111827;
}
.link.router-link-active {
  background: #f3f4f6;
}
</style>
