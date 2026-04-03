<template>
  <span class="badge" :class="`badge--${normalizedStatus}`">{{ status }}</span>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{ status: string }>();

const normalizedStatus = computed(() => props.status.toLowerCase());
</script>

<style scoped>
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.badge--waiting {
  background: #fef3c7;
  color: var(--status-waiting);
}
.badge--running {
  background: #dbeafe;
  color: var(--status-running);
}
.badge--done {
  background: #d1fae5;
  color: var(--status-done);
}
.badge--error {
  background: #fee2e2;
  color: var(--status-error);
}
</style>
