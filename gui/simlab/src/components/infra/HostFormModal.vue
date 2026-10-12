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

          <!-- Hop 1 -->
          <section>
            <h3>Host access</h3>
            <p class="hint">
              The master-node connects here first, then tunnels into this
              host's containers. No container port has to be published.
            </p>
            <div class="grid">
              <label class="field">
                <span>Name</span>
                <input v-model.trim="form.name" placeholder="lab-01" :disabled="saving" />
              </label>
              <label class="field">
                <span>Address</span>
                <input v-model.trim="form.address" placeholder="192.168.0.42" :disabled="saving" />
              </label>
              <label class="field field--narrow">
                <span>SSH port</span>
                <input v-model.number="form.port" type="number" min="1" max="65535" :disabled="saving" />
              </label>
              <label class="field">
                <span>SSH user</span>
                <input v-model.trim="form.username" placeholder="simlab" :disabled="saving" />
              </label>
            </div>
          </section>

          <!-- The credential -->
          <section>
            <h3>Credential</h3>
            <p class="hint">
              Encrypted in your browser before it is sent, and stored that way.
              Only the master-node can open it — the API holds no key.
              <span v-if="publicKey" class="mono">
                Sealing against {{ publicKey.fingerprint }}
              </span>
            </p>

            <div v-if="keyError" class="banner banner--error">
              {{ keyError }}
            </div>

            <div class="grid">
              <label class="field field--narrow">
                <span>Method</span>
                <select v-model="form.authMethod" :disabled="saving">
                  <option value="key">SSH key</option>
                  <option value="password">Password</option>
                </select>
              </label>
            </div>

            <div v-if="isEdit && !replaceSecret" class="stored">
              <span v-if="hasStoredSecret">
                A credential is stored, sealed against key
                <span class="mono">{{ storedKeyId }}</span>.
              </span>
              <span v-else>No credential stored yet.</span>
              <button class="link-btn" :disabled="saving" @click="replaceSecret = true">
                {{ hasStoredSecret ? "Replace it" : "Add one" }}
              </button>
            </div>

            <template v-else>
              <label v-if="form.authMethod === 'key'" class="field">
                <span>Private key</span>
                <textarea
                  v-model="secretInput"
                  rows="5"
                  spellcheck="false"
                  placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"
                  :disabled="saving"
                />
                <input
                  type="file"
                  class="file"
                  accept=".pem,.key,text/plain"
                  :disabled="saving"
                  @change="readKeyFile"
                />
                <small class="hint">
                  Must be unencrypted — the sealed envelope is what protects it
                  at rest. Never leaves this page in the clear.
                </small>
              </label>
              <label v-else class="field">
                <span>Password</span>
                <input
                  v-model="secretInput"
                  type="password"
                  autocomplete="new-password"
                  :disabled="saving"
                />
              </label>
              <button
                v-if="isEdit"
                class="link-btn"
                :disabled="saving"
                @click="cancelReplace"
              >
                Keep the stored credential
              </button>
            </template>
          </section>

          <!-- Hop 2 -->
          <section>
            <h3>Container access</h3>
            <p class="hint">
              How this host reaches its own Cooja containers. Leave the
              credential empty to use the image's built-in login, which is
              what the local containers have always used.
            </p>
            <div class="grid">
              <label class="field field--narrow">
                <span>User</span>
                <input v-model.trim="form.containerUser" :disabled="saving" />
              </label>
              <label class="field field--narrow">
                <span>Port</span>
                <input v-model.number="form.containerPort" type="number" min="1" :disabled="saving" />
              </label>
              <label class="field">
                <span>Address mode</span>
                <select v-model="form.addressMode" :disabled="saving">
                  <option value="container_ip">Bridge IP (default for a remote host)</option>
                  <option value="published_port">Published port on loopback</option>
                  <option value="container_name">Container name (Docker DNS)</option>
                </select>
              </label>
            </div>
          </section>

          <!-- Where the workers come from -->
          <section>
            <h3>Workers</h3>
            <div class="grid">
              <label class="field">
                <span>How to list them</span>
                <select v-model="form.discoveryMode" :disabled="saving">
                  <option value="docker">Ask Docker on the host</option>
                  <option value="manual">Configure each one by hand</option>
                </select>
              </label>
            </div>
            <p v-if="form.discoveryMode === 'manual'" class="hint hint--aside">
              You add each worker yourself, with its own address and port. The
              only option that works when the SSH user above cannot reach the
              host's Docker socket — a shared machine, typically.
            </p>
            <template v-else>
              <p class="hint">
                The master-node runs <span class="mono">docker ps</span> on the
                host and finds the containers carrying this label. Requires the
                SSH user to be able to use the Docker socket there.
              </p>
              <div class="grid">
                <label class="field">
                  <span>Label filter</span>
                  <input v-model.trim="form.labelFilter" :disabled="saving" />
                </label>
                <label class="field">
                  <span>Name prefix <em>(optional)</em></span>
                  <input v-model.trim="form.namePrefix" placeholder="cooja" :disabled="saving" />
                </label>
              </div>
            </template>
          </section>
        </div>

        <footer>
          <label class="inline">
            <input v-model="form.enabled" type="checkbox" :disabled="saving" />
            <span>Add to the pool straight away</span>
          </label>
          <div class="actions">
            <button class="btn-secondary" :disabled="saving" @click="close">Cancel</button>
            <button class="btn-primary" :disabled="saving || !canSave" @click="save">
              {{ saving ? sealingLabel : isEdit ? "Save" : "Register host" }}
            </button>
          </div>
        </footer>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref } from "vue";
import { getPublicKey } from "../../api/infra";
import { SealingError, sealSecret } from "../../services/secretEnvelope";
import type {
  HostCreateInput,
  HostDto,
  HostUpdateInput,
  PublicKeyDto,
  SealedSecret,
} from "../../types/infra";

/**
 * Register or edit a remote host.
 *
 * The credential is sealed here, in the browser, against the master-node's
 * published public key. The plaintext lives in `secretInput` for as long as
 * the form is open and is dropped as soon as it has been sealed; nothing in
 * this component logs it or keeps a copy.
 */
const props = defineProps<{ host?: HostDto | null }>();

const emit = defineEmits<{
  (event: "close"): void;
  (event: "saved", id: string): void;
}>();

const isEdit = computed(() => Boolean(props.host));
const title = computed(() =>
  isEdit.value ? `Edit ${props.host?.name}` : "Register a remote host",
);

const form = reactive({
  name: props.host?.name ?? "",
  address: props.host?.access?.address ?? "",
  port: props.host?.access?.port ?? 22,
  username: props.host?.access?.username ?? "",
  authMethod: props.host?.access?.auth_method ?? ("key" as "key" | "password"),
  containerUser: props.host?.container_access?.username ?? "root",
  containerPort: props.host?.container_access?.port ?? 22,
  addressMode: props.host?.container_access?.address_mode ?? "container_ip",
  discoveryMode: props.host?.discovery?.mode ?? ("docker" as "docker" | "manual"),
  labelFilter: props.host?.discovery?.label_filter ?? "simlab.type=cooja",
  namePrefix: props.host?.discovery?.name_prefix ?? "",
  enabled: props.host?.enabled ?? true,
});

const secretInput = ref("");
const replaceSecret = ref(!isEdit.value);
const saving = ref(false);
const sealingLabel = ref("Saving…");
const error = ref("");
const keyError = ref("");
const publicKey = ref<PublicKeyDto | null>(null);
const dialogEl = ref<HTMLElement | null>(null);

const hasStoredSecret = computed(
  () => props.host?.access?.secret?.has_secret ?? false,
);
const storedKeyId = computed(() => props.host?.access?.secret?.key_id ?? "—");

const canSave = computed(() => {
  if (!form.name || !form.address || !form.username) return false;
  // A new host needs a credential; an edit may keep the stored one.
  if (replaceSecret.value && !secretInput.value) return false;
  if (replaceSecret.value && !publicKey.value) return false;
  return true;
});

onMounted(async () => {
  await nextTick();
  dialogEl.value?.focus();
  try {
    publicKey.value = await getPublicKey();
  } catch (e) {
    // Without a published key nothing can be sealed, so say why rather than
    // letting the save fail later with a 422.
    keyError.value =
      `No public key to encrypt against: ${describe(e)}. ` +
      `The master-node publishes it on startup; start it once and reopen this form.`;
  }
});

async function readKeyFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  secretInput.value = await file.text();
  // Clear the picker so the same file can be chosen again after an edit.
  input.value = "";
}

function cancelReplace() {
  secretInput.value = "";
  replaceSecret.value = false;
}

async function save() {
  error.value = "";
  saving.value = true;
  try {
    let sealed: SealedSecret | undefined;
    if (replaceSecret.value) {
      sealingLabel.value = "Encrypting…";
      sealed = await sealSecret(secretInput.value, publicKey.value!);
      // The plaintext has served its purpose; drop it before the request so
      // it is not sitting in memory during the round trip.
      secretInput.value = "";
    }
    sealingLabel.value = "Saving…";

    const id = isEdit.value
      ? await saveEdit(sealed)
      : await saveNew(sealed!);
    emit("saved", id);
  } catch (e) {
    error.value = describe(e);
  } finally {
    saving.value = false;
  }
}

async function saveNew(sealed: SealedSecret): Promise<string> {
  const { createHost } = await import("../../api/infra");
  const payload: HostCreateInput = {
    name: form.name,
    enabled: form.enabled,
    access: {
      address: form.address,
      port: form.port,
      username: form.username,
      auth_method: form.authMethod,
      secret: sealed,
    },
    container_access: {
      username: form.containerUser,
      port: form.containerPort,
      address_mode: form.addressMode,
    },
    discovery: {
      mode: form.discoveryMode,
      label_filter: form.labelFilter,
      name_prefix: form.namePrefix || null,
    },
  };
  return createHost(payload);
}

async function saveEdit(sealed?: SealedSecret): Promise<string> {
  const { updateHost } = await import("../../api/infra");
  const updates: HostUpdateInput = {
    name: form.name,
    enabled: form.enabled,
    access: {
      address: form.address,
      port: form.port,
      username: form.username,
      auth_method: form.authMethod,
      // Omitted when not replacing, so the stored credential is left alone.
      ...(sealed ? { secret: sealed } : {}),
    },
    container_access: {
      username: form.containerUser,
      port: form.containerPort,
      address_mode: form.addressMode,
    },
    discovery: {
      mode: form.discoveryMode,
      label_filter: form.labelFilter,
      name_prefix: form.namePrefix || null,
    },
  };
  await updateHost(props.host!.id, updates);
  return props.host!.id;
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
  max-width: 640px;
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

.body {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
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
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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

.field em { font-weight: 400; font-style: normal; }

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

input:focus,
select:focus,
textarea:focus {
  outline: 2px solid var(--color-primary-light);
  border-color: var(--color-primary);
}

.file { font-size: 11px; font-weight: 400; }

.stored {
  font-size: 12px;
  color: var(--color-text-muted);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
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

.btn-primary,
.btn-secondary {
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

.btn-primary:disabled,
.btn-secondary:disabled { opacity: 0.5; cursor: default; }

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
