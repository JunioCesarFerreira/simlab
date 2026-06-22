import { defineStore } from 'pinia'
import { ref } from 'vue'

export type EditorTool = 'select' | 'place-sink' | 'place-candidate' | 'place-target' | 'place-relay' | 'chromosome-pick' | 'draw-line' | 'draw-ellipse' | 'measure' | 'scale-calibrate'

export type SelectedElement =
  | { type: 'sink' }
  | { type: 'candidate'; id: string }
  | { type: 'target'; id: string }
  | { type: 'relay'; id: string }

export const useEditorStore = defineStore('editor', () => {
  const activeTool = ref<EditorTool>('select')
  const backgroundImage = ref<string | null>(null)
  const activeNodeId = ref<string | null>(null)
  const showJsonPreview = ref(false)
  const selected = ref<SelectedElement | null>(null)

  /**
   * Optional calibration: which world-coordinate bounding box does the image cover?
   * [xmin, ymin, xmax, ymax] in world units.
   * null = uncalibrated (image is fit-inside the region viewport, preserving aspect ratio).
   */
  const imageWorldBounds = ref<[number, number, number, number] | null>(null)
  const showConnectivity = ref(false)
  const showChromosomeConnectivity = ref(false)

  function setTool(tool: EditorTool) {
    activeTool.value = tool
    selected.value = null
  }

  function setBackground(dataUrl: string | null) {
    backgroundImage.value = dataUrl
    if (!dataUrl) imageWorldBounds.value = null
  }

  function setActiveNode(id: string | null) { activeNodeId.value = id }
  function toggleJsonPreview() { showJsonPreview.value = !showJsonPreview.value }
  function setSelected(el: SelectedElement | null) { selected.value = el }
  function setImageWorldBounds(bounds: [number, number, number, number] | null) {
    imageWorldBounds.value = bounds
  }

  function toggleConnectivity() { showConnectivity.value = !showConnectivity.value }
  function toggleChromosomeConnectivity() {
    showChromosomeConnectivity.value = !showChromosomeConnectivity.value
  }

  return {
    activeTool,
    backgroundImage,
    activeNodeId,
    showJsonPreview,
    selected,
    imageWorldBounds,
    showConnectivity,
    showChromosomeConnectivity,
    setTool,
    setBackground,
    setActiveNode,
    toggleJsonPreview,
    setSelected,
    setImageWorldBounds,
    toggleConnectivity,
    toggleChromosomeConnectivity,
  }
})
