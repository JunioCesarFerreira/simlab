<template>
  <Teleport to="body">
    <div class="backdrop" @click.self="close">
      <div class="modal" role="dialog" aria-modal="true" aria-label="Experiment created">

        <div class="success-header">
          <span class="check-icon">✓</span>
          <div>
            <h2 class="title">Experiment created!</h2>
            <p class="exp-id">ID: <span class="mono">{{ experimentId }}</span></p>
          </div>
        </div>

        <!-- Campaign assignment -->
        <div class="section">
          <p class="section-label">Would you like to add it to a campaign?</p>

          <div class="campaign-options">
            <div class="option-card" :class="{ selected: mode === 'existing' }" @click="mode = 'existing'">
              <span class="option-radio" />
              <div>
                <div class="option-title">Existing campaign</div>
                <div class="option-sub">Select a campaign already created</div>
              </div>
            </div>
            <div class="option-card" :class="{ selected: mode === 'new' }" @click="mode = 'new'">
              <span class="option-radio" />
              <div>
                <div class="option-title">New campaign</div>
                <div class="option-sub">Create a campaign and add this experiment</div>
              </div>
            </div>
          </div>

          <!-- Existing campaign select -->
          <div v-if="mode === 'existing'" class="sub-form">
            <div v-if="loadingCampaigns" class="loading-text">Loading campaigns…</div>
            <div v-else-if="campaigns.length === 0" class="empty-text">No campaigns available.</div>
            <select v-else v-model="selectedCampaignId" class="field-select">
              <option value="">Select a campaign…</option>
              <option v-for="c in campaigns" :key="c.id" :value="c.id">
                {{ c.name }} ({{ c.experiment_count }} exp.)
              </option>
            </select>
          </div>

          <!-- New campaign form -->
          <div v-if="mode === 'new'" class="sub-form">
            <input
              v-model="newCampaignName"
              type="text"
              placeholder="Campaign name *"
              class="field-input"
            />
            <input
              v-model="newCampaignDesc"
              type="text"
              placeholder="Description (optional)"
              class="field-input"
            />
          </div>
        </div>

        <div v-if="assignError" class="err-banner">{{ assignError }}</div>

        <div class="footer">
          <button class="btn-secondary" @click="navigateToExperiment">View experiment →</button>
          <button
            v-if="mode === 'existing' || mode === 'new'"
            class="btn-primary"
            :disabled="!canAssign || assigning"
            @click="assignToCampaign"
          >
            {{ assigning ? 'Adding…' : 'Add to campaign' }}
          </button>
          <button class="btn-secondary" @click="close">Close</button>
        </div>

      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getAllCampaigns, createCampaign, addExperimentToCampaign } from '../../../api/campaigns'
import type { CampaignInfoDto } from '../../../types/simlab'

const props = defineProps<{ experimentId: string }>()
const emit = defineEmits<{ close: [] }>()

const router = useRouter()

const mode = ref<'existing' | 'new'>('existing')
const campaigns = ref<CampaignInfoDto[]>([])
const loadingCampaigns = ref(false)
const selectedCampaignId = ref('')
const newCampaignName = ref('')
const newCampaignDesc = ref('')
const assigning = ref(false)
const assignError = ref<string | null>(null)

const canAssign = computed(() => {
  if (mode.value === 'existing') return selectedCampaignId.value !== ''
  if (mode.value === 'new') return newCampaignName.value.trim().length > 0
  return false
})

onMounted(async () => {
  loadingCampaigns.value = true
  try {
    campaigns.value = await getAllCampaigns()
  } catch {
    // silently ignore — user can still create a new campaign
  } finally {
    loadingCampaigns.value = false
  }
})

async function assignToCampaign() {
  assigning.value = true
  assignError.value = null
  try {
    let campaignId = selectedCampaignId.value
    if (mode.value === 'new') {
      campaignId = await createCampaign({
        name: newCampaignName.value.trim(),
        description: newCampaignDesc.value.trim(),
        created_time: null,
        experiment_ids: [],
      })
    }
    await addExperimentToCampaign(campaignId, props.experimentId)
    router.push(`/campaigns/${campaignId}`)
    emit('close')
  } catch (e: unknown) {
    assignError.value = e instanceof Error ? e.message : 'Failed to add to campaign.'
  } finally {
    assigning.value = false
  }
}

function navigateToExperiment() {
  router.push(`/experiments/${props.experimentId}`)
  emit('close')
}

function close() { emit('close') }
</script>

<style scoped>
.backdrop {
  position: fixed; inset: 0; z-index: 10000;
  background: rgba(15, 23, 42, 0.5);
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
}
.modal {
  background: var(--color-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius-lg); box-shadow: 0 20px 60px rgba(0,0,0,0.25);
  width: 100%; max-width: 480px;
  padding: 24px; display: flex; flex-direction: column; gap: 20px;
}
.success-header { display: flex; align-items: flex-start; gap: 14px; }
.check-icon {
  width: 40px; height: 40px; border-radius: 50%; background: #dcfce7;
  color: #16a34a; font-size: 18px; font-weight: 700;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.title { font-size: 16px; font-weight: 700; color: var(--color-text); margin: 0 0 3px; }
.exp-id { font-size: 12px; color: var(--color-text-muted); margin: 0; }
.mono { font-family: 'SFMono-Regular', Consolas, monospace; }
.section { display: flex; flex-direction: column; gap: 10px; }
.section-label { font-size: 13px; font-weight: 600; color: var(--color-text); }
.campaign-options { display: flex; flex-direction: column; gap: 8px; }
.option-card {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px; border: 1px solid var(--color-border);
  border-radius: var(--radius-md); cursor: pointer;
  transition: border-color 0.12s;
}
.option-card:hover { border-color: var(--color-primary); }
.option-card.selected { border-color: var(--color-primary); background: var(--color-primary-light); }
.option-radio {
  width: 16px; height: 16px; border-radius: 50%;
  border: 2px solid var(--color-border); flex-shrink: 0;
  transition: border-color 0.12s;
}
.option-card.selected .option-radio { border-color: var(--color-primary); background: var(--color-primary); }
.option-title { font-size: 13px; font-weight: 600; color: var(--color-text); }
.option-sub { font-size: 11px; color: var(--color-text-muted); }
.sub-form { display: flex; flex-direction: column; gap: 8px; margin-top: 2px; }
.loading-text, .empty-text { font-size: 13px; color: var(--color-text-muted); }
.field-select, .field-input {
  width: 100%; padding: 8px 10px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); font-size: 13px;
  color: var(--color-text); background: var(--color-surface); outline: none;
}
.field-select:focus, .field-input:focus { border-color: var(--color-primary); }
.err-banner {
  padding: 8px 12px; background: #fef2f2; border: 1px solid #fecaca;
  border-radius: var(--radius-sm); font-size: 12px; color: #ef4444;
}
.footer { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }
.btn-secondary {
  padding: 8px 16px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); background: none;
  font-size: 13px; font-weight: 500; color: var(--color-text); cursor: pointer;
}
.btn-secondary:hover { border-color: var(--color-text-muted); }
.btn-primary {
  padding: 8px 18px; border-radius: var(--radius-sm); border: none;
  background: var(--color-primary); color: #fff;
  font-size: 13px; font-weight: 600; cursor: pointer;
}
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
