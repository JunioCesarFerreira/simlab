<template>
  <div>
    <h1>Experimento {{ id }}</h1>
    <p>Detalhes: parâmetros, gerações, simulações, logs e artefatos (GridFS).</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{ id: string }>();
</script>
