<template>
  <!--
    `expr` is benchmark-formula markup (sub/sup tags + math entities) built
    internally in BenchmarkPreviewPanel from numeric parameters and static
    strings — it never contains user input, so v-html carries no XSS risk.
    This component is the single audited trusted-HTML sink for those formulas,
    which keeps `vue/no-v-html` active everywhere else in the codebase.
    The container class (e.g. `formula-line`) is passed by the caller and
    styled by the caller's scoped CSS via its root-node scope id.
  -->
  <!-- eslint-disable-next-line vue/no-v-html -->
  <div v-html="expr"></div>
</template>

<script setup lang="ts">
defineProps<{ expr: string }>()
</script>
