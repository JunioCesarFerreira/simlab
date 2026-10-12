<template>
  <Teleport to="body">
    <div class="backdrop" @click.self="close">
      <div
        ref="dialogEl"
        class="modal"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
        tabindex="-1"
        @keydown.esc="close"
      >
        <header>
          <h2>{{ title }}</h2>
          <button class="icon-btn" aria-label="Close" @click="close">✕</button>
        </header>

        <div class="body">
          <p v-if="error" class="banner banner--error">{{ error }}</p>

          <p class="hint">
            Where this worker's sshd answers, reached <em>through</em>
            <span class="mono">{{ hostLabel }}</span>. Everything left blank
            below falls back to the host's defaults, so a worker that differs
            only in its port needs only that field.
          </p>

          <div class="grid">
            <label class="field">
              <span>Worker name</span>
              <input v-model.trim="form.name" placeholder="vm7-cooja1" :disabled="saving" />
            </label>
            <label class="field">
              <span>Address <em>(as seen from the host)</em></span>
              <input v-model.trim="form.address" placeholder="127.0.0.1" :disabled="saving" />
            </label>
            <label class="field field--narrow">
              <span>SSH port</span>
              <input v-model.number="form.port" type="number" min="1" max="65535" :disabled="saving" />
            </label>
          </div>

          <p class="hint hint--aside">
            <strong>Address</strong> is resolved on the host, not here. A port
            published by Docker is usually
            <span class="mono">127.0.0.1</span> with the published port; a
            container on the host's bridge is its container IP with port 22.
          </p>

          <section>
            <h3>Overrides <em>(optional)</em></h3>
            <div class="grid">
              <label class="field field--narrow">
                <span>SSH user</span>
                <input
                  v-model.trim="form.username"
                  :placeholder="hostDefaults.username"
                  :disabled="saving"
                />
              </label>
              <label class="field field--narrow">
                <span>Method</span>
                <select v-model="form.authMethod" :disabled="saving">
                  <option value="">host default ({{ hostDefaults.authMethod }})</option>
                  <option value="key">SSH key</option>
                  <option value="password">Password</option>
                </select>
              </label>
              <label class="field">
                <span>Cooja directory</span>
                <input
                  v-model.trim="form.remoteDir"
                  placeholder="/opt/contiki-ng/tools/cooja"
                  :disabled="saving"
                />
              </label>
            </div>

            <div v-if="keyError" class="banner banner--error">{{ keyError }}</div>

            <div v-if="isEdit && !replaceSecret" class="stored">
              <span v-if="hasStoredSecret">
                This worker has its own credential, sealed against
                <span class="mono">{{ storedKeyId }}</span>.
              </span>
              <span v-else>
                Uses the host's credential.
              </span>
              <button class="link-btn" :disabled="saving" @click="replaceSecret = true">
                {{ hasStoredSecret ? "Replace it" : "Give it its own" }}
              </button>
            </div>

            <template v-else>
              <label v-if="form.authMethod === 'key'" class="field">
                <span>Private key <em>(this worker only)</em></span>
                <textarea
                  v-model="secretInput"
                  rows="4"
                  spellcheck="false"
                  placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"
                  :disabled="saving"
                />
              </label>
              <label v-else class="field">
                <span>Password <em>(this worker only)</em></span>
                <input
                  v-model="secretInput"
                  type="password"
                  autocomplete="new-password"
                  :disabled="saving"
                />
              </label>
              <small class="hint">
                Encrypted in your browser before it is sent. Leave empty to use
                the host's credential instead.
              </small>
              <button v-if="isEdit" class="link-btn" :disabled="saving" @click="cancelReplace">
                Keep what is stored
              </button>
            </template>
          </section>
        </div>

        <footer>
          <label class="inline">
            <input v-model="form.enabled" type="checkbox" :disabled="saving" />
            <span>Available for simulations</span>
          </label>
          <div class="actions">
            <button class="btn-secondary" :disabled="saving" @click="close">Cancel</button>
            <button class="btn-primary" :disabled="saving || !canSave" @click="save">
              {{ saving ? "Saving…" : isEdit ? "Save" : "Add worker" }}
            </button>
          </div>
        </footer>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref } from "vue";
import {
  createResource,
  getPublicKey,
  updateResourceConnection,
} from "../../api/infra";
import { SealingError, sealSecret } from "../../services/secretEnvelope";
import type {
  HostDto,
  PublicKeyDto,
  ResourceConnectionInput,
  ResourceDto,
  SealedSecret,
} from "../../types/infra";

/**
 * Add or edit one Cooja worker, with its own SSH parameters.
 *
 * The counterpart to Docker discovery, and the only way that works on a host
 * where the SSH user cannot reach the Docker daemon — a shared machine. It is
 * also simply clearer wherever the ports are already known: the address is
 * stated rather than derived from a mode.
 *
 * Only what is filled in is sent. Everything else falls back to the host's
 * `container_access` at connect time, so configuring eight workers that differ
 * only in their port means typing eight ports.
 */
const props = defineProps<{ host: HostDto; worker?: ResourceDto | null }>();

const emit = defineEmits<{
  (event: "close"): void;
  (event: "saved", id: string): void;
}>();

const isEdit = computed(() => Boolean(props.worker));
const title = computed(() =>
  isEdit.value ? `Edit ${props.worker?.name}` : `Add a worker to ${props.host.name}`,
);

const hostLabel = computed(() => {
  const access = props.host.access;
  return access ? `${access.username}@${access.address}:${access.port}` : "this machine";
});

const hostDefaults = computed(() => ({
  username: props.host.container_access?.username || "root",
  port: props.host.container_access?.port ?? 22,
  authMethod: props.host.container_access?.auth_method || "password",
}));

const endpoint = (props.worker?.endpoint ?? {}) as { address?: string; port?: number };

const form = reactive({
  name: props.worker?.name ?? "",
  address: endpoint.address ?? "",
  port: props.worker?.connection?.port ?? endpoint.port ?? hostDefaults.value.port,
  username: props.worker?.connection?.username ?? "",
  authMethod: props.worker?.connection?.auth_method ?? ("" as "" | "key" | "password"),
  remoteDir: props.worker?.remote_dir ?? "",
  enabled: props.worker?.enabled ?? true,
});

const secretInput = ref("");
const replaceSecret = ref(false);
const saving = ref(false);
const error = ref("");
const keyError = ref("");
const publicKey = ref<PublicKeyDto | null>(null);
const dialogEl = ref<HTMLElement | null>(null);

const hasStoredSecret = computed(
  () => props.worker?.connection?.secret?.has_secret ?? false,
);
const storedKeyId = computed(() => props.worker?.connection?.secret?.key_id ?? "—");

const canSave = computed(() => {
  if (!form.name || !form.address) return false;
  // A credential is optional here — the host's is the fallback — but if one is
  // being set, there has to be a key to seal it against.
  if (secretInput.value && !publicKey.value) return false;
  return true;
});

onMounted(async () => {
  await nextTick();
  dialogEl.value?.focus();
  try {
    publicKey.value = await getPublicKey();
  } catch (e) {
    keyError.value =
      `This worker cannot be given its own credential: ${describe(e)}. ` +
      `It will use the host's.`;
  }
});

function cancelReplace() {
  secretInput.value = "";
  replaceSecret.value = false;
}

async function save() {
  error.value = "";
  saving.value = true;
  try {
    let sealed: SealedSecret | undefined;
    if (secretInput.value) {
      sealed = await sealSecret(secretInput.value, publicKey.value!);
      secretInput.value = "";
    }

    const connection: ResourceConnectionInput = {};
    if (form.port) connection.port = form.port;
    if (form.username) connection.username = form.username;
    if (form.authMethod) connection.auth_method = form.authMethod;
    if (sealed) connection.secret = sealed;

    const payload = {
      name: form.name,
      endpoint: { address: form.address, port: form.port || undefined },
      connection,
      enabled: form.enabled,
      ...(form.remoteDir ? { remote_dir: form.remoteDir } : {}),
    };

    const saved = isEdit.value
      ? await updateResourceConnection(props.worker!.id, payload)
      : await createResource(props.host.id, payload);
    emit("saved", saved.id);
  } catch (e) {
    error.value = describe(e);
  } finally {
    saving.value = false;
  }
}

function describe(e: unknown): string {
  if (e instanceof SealingError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

function close() {
  if (!saving.value) emit("close");
}
</script>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  z-index: 10000;
  background: rgba(15, 23, 42, 0.5);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 32px 24px;
  overflow-y: auto;
}

.modal {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  width: 100%;
  max-width: 580px;
  outline: none;
}

header,
footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
}

header { border-bottom: 1px solid var(--color-border); }
footer { border-top: 1px solid var(--color-border); gap: 12px; }

h2 { font-size: 16px; font-weight: 700; }

h3 {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
  margin-bottom: 4px;
}

h3 em, .field em { font-weight: 400; font-style: normal; text-transform: none; }

.body {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

section { display: flex; flex-direction: column; gap: 8px; }

.hint {
  font-size: 12px;
  color: var(--color-text-muted);
  line-height: 1.5;
}

.hint--aside {
  padding: 8px 10px;
  background: var(--color-surface-hover);
  border-radius: var(--radius-md);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
}

.field--narrow { max-width: 150px; }

input[type="text"],
input:not([type]),
input[type="number"],
input[type="password"],
select,
textarea {
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 13px;
  font-weight: 400;
  font-family: inherit;
}

textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  resize: vertical;
}

input:focus, select:focus, textarea:focus {
  outline: 2px solid var(--color-primary-light);
  border-color: var(--color-primary);
}

.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

.stored {
  font-size: 12px;
  color: var(--color-text-muted);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.banner {
  padding: 10px 12px;
  border-radius: var(--radius-md);
  font-size: 12px;
  line-height: 1.5;
}

.banner--error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: var(--status-error);
}

.inline {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.actions { display: flex; gap: 8px; }

.btn-primary, .btn-secondary {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.btn-primary {
  background: var(--color-primary);
  color: #fff;
  border: 1px solid var(--color-primary);
}

.btn-secondary {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
}

.btn-primary:disabled, .btn-secondary:disabled { opacity: 0.5; cursor: default; }

.link-btn {
  background: none;
  border: none;
  color: var(--color-primary);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  padding: 0;
  align-self: flex-start;
}

.icon-btn {
  background: none;
  border: none;
  color: var(--color-text-muted);
  font-size: 14px;
  cursor: pointer;
}
</style>
