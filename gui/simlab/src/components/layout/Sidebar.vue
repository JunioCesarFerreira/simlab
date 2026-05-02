<template>
  <nav class="sidebar">
    <div class="brand">
      <span class="brand-icon">⬡</span>
      <span class="brand-name">SimLab</span>
    </div>
    <div class="nav-section">
      <RouterLink to="/dashboard" class="nav-link">
        <span class="nav-icon">▦</span>
        Dashboard
      </RouterLink>
      <RouterLink to="/campaigns" class="nav-link">
        <span class="nav-icon">⊞</span>
        Campaigns
      </RouterLink>
      <RouterLink to="/experiments" class="nav-link">
        <span class="nav-icon">◈</span>
        Experiments
      </RouterLink>
    </div>
  </nav>
</template>

<script setup lang="ts"></script>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px 12px;
  background: var(--color-surface);
  height: 100%;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 8px 16px;
  border-bottom: 1px solid var(--color-border);
  margin-bottom: 4px;
}

.brand-icon {
  font-size: 22px;
  color: var(--color-primary);
  line-height: 1;
}

.brand-name {
  font-size: 18px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--color-text);
}

.nav-section {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border-radius: var(--radius-md);
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-muted);
  transition: background 0.12s, color 0.12s;
}

.nav-link:hover {
  background: var(--color-bg);
  color: var(--color-text);
}

.nav-link.router-link-active {
  background: var(--color-primary-light);
  color: var(--color-primary);
  font-weight: 600;
}

.nav-icon {
  font-size: 15px;
  width: 18px;
  text-align: center;
  flex-shrink: 0;
}
</style>
