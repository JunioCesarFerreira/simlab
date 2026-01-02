module.exports = {
  root: true,
  env: { browser: true, es2021: true },
  extends: [
    "eslint:recommended",
    "plugin:vue/vue3-recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:prettier/recommended",
  ],
  parserOptions: { ecmaVersion: "latest", sourceType: "module" },
  rules: {
    "vue/multi-word-component-names": "off",
  },
};
