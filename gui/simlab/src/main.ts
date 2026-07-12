import { createApp } from "vue";
import { createPinia } from "pinia";
import router from "./app/router";
import App from "./App.vue";
import { reportRuntimeError } from "./composables/useRuntimeError";
import "./style.css";

const app = createApp(App);

// Without this, an exception thrown in a component silently kills its subtree
// (console-only). Keep the console.error for the stack trace, and surface a
// dismissible banner so the user knows something actually broke.
app.config.errorHandler = (err, _instance, info) => {
  console.error(`[vue error] (${info})`, err);
  reportRuntimeError(err, "Unexpected error");
};

app.use(createPinia()).use(router).mount("#app");
