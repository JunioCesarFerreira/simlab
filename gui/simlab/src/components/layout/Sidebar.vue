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
      <RouterLink to="/problems" class="nav-link">
        <span class="nav-icon">⊙</span>
        Problems
      </RouterLink>
      <RouterLink to="/sources" class="nav-link">
        <span class="nav-icon">⊛</span>
        Sources
      </RouterLink>
      <RouterLink to="/compare" class="nav-link">
        <span class="nav-icon">⊟</span>
        Compare
      </RouterLink>
    </div>

    <div class="sidebar-footer">
      <button class="theme-toggle" :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'" @click="toggle">
        <span class="toggle-icon">{{ isDark ? '☀' : '🌙' }}</span>
        <span class="toggle-label">{{ isDark ? 'Light mode' : 'Dark mode' }}</span>
      </button>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { useTheme } from '../../composables/useTheme'

const { isDark, toggle } = useTheme()
</script>

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
  flex: 1;
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
  background: var(--color-surface-hover);
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

/* Theme toggle */
.sidebar-footer {
  border-top: 1px solid var(--color-border);
  padding-top: 10px;
  margin-top: 4px;
}

.theme-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 9px 10px;
  border-radius: var(--radius-md);
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-muted);
  transition: background 0.12s, color 0.12s;
  cursor: pointer;
}

.theme-toggle:hover {
  background: var(--color-surface-hover);
  color: var(--color-text);
}

.toggle-icon {
  font-size: 16px;
  width: 18px;
  text-align: center;
  flex-shrink: 0;
}

.toggle-label {
  font-size: 13px;
}
</style>
