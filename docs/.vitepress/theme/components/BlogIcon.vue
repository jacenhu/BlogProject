<script setup>
defineProps({ name: { type: String, required: true } })

const paths = {
  system: 'M3 4h18v12H3z M8 20h8 M12 16v4',
  model: 'M12 3l8 5v8l-8 5-8-5V8z M4 8l8 5 8-5 M12 13v8 M8 5.5l8 5',
  cache: 'M4 6c0-4 16-4 16 0s-16 4-16 0v12c0 4 16 4 16 0V6 M4 12c0 4 16 4 16 0',
  layers: 'M12 3l9 5-9 5-9-5z M3 12l9 5 9-5 M3 16l9 5 9-5',
  book: 'M12 5C9 3 5 3 3 4v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-2-1-6-1-9 1v15',
  arrow: 'M5 19L19 5 M7 5h12v12',
  right: 'M4 12h16 M13 5l7 7-7 7'
}
</script>

<template>
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
    <path :d="paths[name]" />
  </svg>
</template>
