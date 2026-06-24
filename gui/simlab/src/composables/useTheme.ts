import { ref, watch } from "vue";

const stored = localStorage.getItem("simlab-theme");
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const isDark = ref(stored === "dark" || (!stored && prefersDark));

document.documentElement.classList.toggle("dark", isDark.value);

watch(isDark, (dark) => {
  document.documentElement.classList.toggle("dark", dark);
  localStorage.setItem("simlab-theme", dark ? "dark" : "light");
});

export function useTheme() {
  function toggle() {
    isDark.value = !isDark.value;
  }
  return { isDark, toggle };
}
