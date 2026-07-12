<template>
  <Teleport to="body">
    <div v-if="active" class="confirm-backdrop" @click.self="settle(false)">
      <div
        ref="dialogEl"
        class="confirm-modal"
        role="alertdialog"
        aria-modal="true"
        :aria-label="active.title"
        tabindex="-1"
        @keydown.esc="settle(false)"
      >
        <h2 class="confirm-title">{{ active.title }}</h2>
        <p class="confirm-message">{{ active.message }}</p>
        <div class="confirm-actions">
          <button class="btn-secondary" @click="settle(false)">
            {{ active.cancelLabel ?? "Cancel" }}
          </button>
          <button
            :class="active.danger ? 'btn-danger' : 'btn-primary'"
            @click="settle(true)"
          >
            {{ active.confirmLabel ?? "Confirm" }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from "vue";
import { useConfirmState } from "../../composables/useConfirm";

const { active, settle } = useConfirmState();

// Focus the dialog when it opens so Esc works without clicking first.
const dialogEl = ref<HTMLElement | null>(null);
watch(active, async (a) => {
  if (a) {
    await nextTick();
    dialogEl.value?.focus();
  }
});
</script>

<style scoped>
.confirm-backdrop {
  position: fixed;
  inset: 0;
  z-index: 10001;
  background: rgba(15, 23, 42, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.confirm-modal {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg, 10px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
  width: 100%;
  max-width: 440px;
  padding: 20px 24px;
  outline: none;
}

.confirm-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--color-text);
  margin: 0 0 8px;
}

.confirm-message {
  font-size: 13px;
  color: var(--color-text-muted);
  line-height: 1.6;
  margin: 0 0 18px;
  white-space: pre-line;
}

.confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.btn-secondary {
  padding: 8px 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm, 5px);
  background: none;
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text);
  cursor: pointer;
}

.btn-secondary:hover {
  border-color: var(--color-text-muted);
}

.btn-primary {
  padding: 8px 18px;
  border-radius: var(--radius-sm, 5px);
  border: none;
  background: var(--color-primary);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.btn-danger {
  padding: 8px 18px;
  border-radius: var(--radius-sm, 5px);
  border: none;
  background: #ef4444;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.btn-danger:hover {
  background: #dc2626;
}
</style>
