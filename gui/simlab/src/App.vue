<template>
  <AppShell>
    <!-- Keyed by path: navigating /experiments/A → /experiments/B must remount
         the page (vue-router reuses the instance otherwise, and pages fetch in
         onMounted / bind per-id state at setup time). -->
    <RouterView :key="$route.path" />
  </AppShell>

  <!-- Global promise-based confirm dialog (see useConfirm) -->
  <ConfirmDialog />

  <!-- Global runtime-error banner (populated by app.config.errorHandler) -->
  <Transition name="err-slide">
    <div v-if="runtimeError" class="runtime-error" role="alert">
      <span class="runtime-error-msg">⚠ {{ runtimeError }}</span>
      <button class="runtime-error-close" aria-label="Dismiss error" @click="dismiss">✕</button>
    </div>
  </Transition>
</template>

<script setup lang="ts">
import AppShell from "./components/layout/AppShell.vue";
import ConfirmDialog from "./components/common/ConfirmDialog.vue";
import { useRuntimeError } from "./composables/useRuntimeError";

const { runtimeError, dismiss } = useRuntimeError();
</script>

<style scoped>
.runtime-error {
  position: fixed;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10000;
  display: flex;
  align-items: center;
  gap: 12px;
  max-width: min(680px, calc(100vw - 32px));
  padding: 10px 16px;
  background: #7f1d1d;
  color: #fecaca;
  border: 1px solid #b91c1c;
  border-radius: var(--radius-lg, 10px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
  font-size: 13px;
}

.runtime-error-msg {
  overflow-wrap: anywhere;
}

.runtime-error-close {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  padding: 2px;
  opacity: 0.8;
  flex-shrink: 0;
}

.runtime-error-close:hover {
  opacity: 1;
}

.err-slide-enter-active,
.err-slide-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}

.err-slide-enter-from,
.err-slide-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(8px);
}
</style>
