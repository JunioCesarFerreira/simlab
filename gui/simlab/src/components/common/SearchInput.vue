<template>
  <div class="search-input">
    <span class="icon" aria-hidden="true">⌕</span>
    <input
      :value="modelValue"
      class="field"
      type="search"
      :placeholder="placeholder"
      :aria-label="placeholder"
      @input="$emit('update:modelValue', ($event.target as HTMLInputElement).value)"
      @keydown.esc="$emit('update:modelValue', '')"
    />
    <button
      v-if="modelValue"
      class="clear-btn"
      type="button"
      title="Clear search"
      @click="$emit('update:modelValue', '')"
    >
      ✕
    </button>
  </div>
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{
    modelValue: string;
    placeholder?: string;
  }>(),
  { placeholder: "Search by name…" },
);

defineEmits<{
  (e: "update:modelValue", value: string): void;
}>();
</script>

<style scoped>
.search-input {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  transition: border-color 0.15s;
}

.search-input:focus-within {
  border-color: #bfdbfe;
}

.icon {
  font-size: 15px;
  color: var(--color-text-muted);
  flex-shrink: 0;
}

.field {
  flex: 1;
  min-width: 0;
  padding: 7px 0;
  border: none;
  outline: none;
  background: transparent;
  font-size: 13px;
  color: var(--color-text);
}

/* The native affordance duplicates our own clear button. */
.field::-webkit-search-cancel-button {
  display: none;
}

.clear-btn {
  flex-shrink: 0;
  padding: 0 2px;
  border: none;
  background: transparent;
  font-size: 12px;
  line-height: 1;
  color: var(--color-text-muted);
  transition: color 0.15s;
}

.clear-btn:hover {
  color: var(--color-text);
}
</style>
