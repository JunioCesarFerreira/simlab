import js from '@eslint/js'
import globals from 'globals'
import vue from 'eslint-plugin-vue'
import tsParser from '@typescript-eslint/parser'
import tsPlugin from '@typescript-eslint/eslint-plugin'
import prettier from 'eslint-config-prettier'

// Shared TypeScript rules (used for both .ts files and <script lang="ts"> blocks)
const tsRules = {
  ...tsPlugin.configs.recommended.rules,
  // TypeScript itself resolves globals/DOM lib types and build-time defines
  // (e.g. __APP_VERSION__, HeadersInit); no-undef would false-positive here.
  'no-undef': 'off',
  '@typescript-eslint/no-unused-vars': [
    'error',
    { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
  ],
  '@typescript-eslint/no-explicit-any': 'error',
  // Page/layout components are conventionally single-word (Dashboard, Sidebar…)
  'vue/multi-word-component-names': 'off',
}

export default [
  { ignores: ['dist/**', 'node_modules/**', 'coverage/**'] },

  js.configs.recommended,
  ...vue.configs['flat/recommended'],

  // Node context for config files
  {
    files: ['*.config.{js,ts,mjs}', 'vite.config.*'],
    languageOptions: { globals: { ...globals.node } },
  },

  // Plain TypeScript modules
  {
    files: ['**/*.ts'],
    languageOptions: {
      parser: tsParser,
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser },
    },
    plugins: { '@typescript-eslint': tsPlugin },
    rules: tsRules,
  },

  // Vue SFCs — the vue flat config wires vue-eslint-parser; we point its
  // <script> parser at the TypeScript parser and reuse the same TS rules.
  {
    files: ['**/*.vue'],
    languageOptions: {
      parserOptions: { parser: tsParser },
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.browser },
    },
    plugins: { '@typescript-eslint': tsPlugin },
    rules: tsRules,
  },

  // Test files may use loose typing for fixtures and assertions
  {
    files: ['**/*.spec.ts', '**/*.test.ts'],
    languageOptions: {
      globals: { ...globals.node },
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },

  // Disable formatting rules that conflict with Prettier (run `npm run format`)
  prettier,
]

