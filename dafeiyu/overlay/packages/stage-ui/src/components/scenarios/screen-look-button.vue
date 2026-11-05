<script setup lang="ts">
import { errorMessageFrom } from '@moeru/std'
import { useLLM } from '@proj-airi/stage-ui/stores/ai/chat-llm/llm'
import { useSpeakingStore } from '@proj-airi/stage-ui/stores/audio'
import { useChatStore } from '@proj-airi/stage-ui/stores/chat'
import { useChatSessionStore } from '@proj-airi/stage-ui/stores/chat/session-store'
import { useChatStreamStore } from '@proj-airi/stage-ui/stores/chat/stream-store'
import { useAiriCardStore } from '@proj-airi/stage-ui/stores/modules/airi-card'
import { useConsciousnessStore } from '@proj-airi/stage-ui/stores/modules/consciousness'
import { useVisionStore } from '@proj-airi/stage-ui/stores/modules/vision'
import { useProviderStore } from '@proj-airi/stage-ui/stores/providers/provider'
import { useVoiceStore } from '@proj-airi/stage-ui/stores/voice'
import { useVoiceMessagesStore } from '@proj-airi/stage-ui/stores/voice-messages'
import { Button } from '@proj-airi/ui'
import { useLocalStorage } from '@vueuse/core'
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { toast } from 'vue-sonner'

import { ScreenChangeGate } from '../../composables/screen-change-gate'
import { ScreenSnapshot } from '../../composables/screen-snapshot'

const props = defineProps<{ requestStream?: () => Promise<MediaStream>, sourceName?: string, inlineStatus?: boolean }>()
const emit = defineEmits<{ active: [value: boolean], locked: [value: boolean] }>()

const { t } = useI18n()
const chat = useChatStore()
const sessions = useChatSessionStore()
const cards = useAiriCardStore()
const consciousness = useConsciousnessStore()
const capture = new ScreenSnapshot(props.requestStream)
const busy = ref(false)
const configured = computed(() => consciousness.configured)
let cancellation: AbortController | undefined
const voice = useVoiceStore()
const voiceMessages = useVoiceMessagesStore()
const speaking = useSpeakingStore()
const chatStream = useChatStreamStore()
const vision = useVisionStore()
const llm = useLLM()
const providers = useProviderStore()
const quiet = ref(false)
const forced = useLocalStorage('settings/screen-watch/forced', false, { listenToStorageChanges: false })
let recentEvents: string[] = []
const watching = ref(false)
watch(watching, value => emit('active', value), { flush: 'sync' })
const opening = ref(false)
watch(() => busy.value || opening.value || watching.value, value => emit('locked', value), { flush: 'sync' })
const commenting = ref(false)
const source = ref('')
const lastAt = ref('')
const watchOwner = shallowRef<AbortController>()
const commentOwner = shallowRef<AbortController>()
const ownTurn = ref('')
let timer: ReturnType<typeof setInterval> | undefined
// Busy speech keeps only the newest sampled frame, never an expanding queue of old screenshots.
let pendingFrame: ReturnType<ScreenSnapshot['frame']> | undefined
let gate = new ScreenChangeGate()
const watchCapture = new ScreenSnapshot(props.requestStream)
const userTalking = computed(() => voiceMessages.isRecording || voice.state?.phase === 'capturing' || voice.state?.phase === 'finalizing')
const otherTurn = computed(() => [...chat.pendingRequests, ...chatStream.activeTurns].some(turn => turn.turnId !== ownTurn.value))
const occupied = computed(() => userTalking.value || otherTurn.value || speaking.nowSpeaking || voice.activeTurns.length > 0)
const canSee = computed(() => configured.value && (
  (consciousness.activeProvider === 'deepseek' && consciousness.activeModel === 'deepseek-flash')
  || consciousness.providerModels.some(model => model.id === consciousness.activeModel && model.metadata?.abilities?.vision === true)
  || (vision.configured && vision.useForChat)
))

function stopWatching() {
  watchOwner.value?.abort()
  commentOwner.value?.abort()
  watchCapture.close()
  clearInterval(timer)
  timer = undefined
  pendingFrame = undefined
  watching.value = false
  source.value = ''
}

watch([userTalking, otherTurn], ([talking, other]) => {
  if (talking || other)
    commentOwner.value?.abort()
}, { flush: 'sync' })
watch(occupied, (value) => {
  if (!value && forced.value && pendingFrame)
    void comment(false)
})

// A delayed source chooser must not send the image to a different conversation, character or provider.
watch(() => [sessions.activeSessionId, cards.activeCardId, consciousness.activeProvider, consciousness.activeModel], () => {
  cancellation?.abort()
  stopWatching()
}, { flush: 'sync' })
watch(() => [vision.activeProvider, vision.activeModel, vision.enabled, vision.useForChat], stopWatching, { flush: 'sync' })
onBeforeUnmount(() => {
  cancellation?.abort()
  stopWatching()
})

async function startWatching() {
  if (opening.value || watching.value || busy.value)
    return
  if (!canSee.value) {
    toast.error(t('stage.screen-look.configure'))
    return
  }
  const owner = new AbortController()
  watchOwner.value = owner
  watching.value = true
  opening.value = true
  lastAt.value = ''
  quiet.value = false
  recentEvents = []
  gate = new ScreenChangeGate()
  try {
    await watchCapture.open(owner.signal, stopWatching)
    owner.signal.throwIfAborted()
    source.value = props.sourceName || watchCapture.source
    timer = setInterval(sample, 5000)
    sample()
  }
  catch (error) {
    const cancelled = owner.signal.aborted || (error instanceof DOMException && error.name === 'NotAllowedError')
    stopWatching()
    if (!cancelled)
      toast.error(t('stage.screen-look.failed'), { description: errorMessageFrom(error) })
  }
  finally {
    opening.value = false
  }
}

function sample() {
  if (!watching.value || watchOwner.value?.signal.aborted)
    return
  try {
    if (forced.value)
      pendingFrame = watchCapture.frame()
    void comment(false)
  }
  catch (error) {
    stopWatching()
    toast.error(t('stage.screen-look.failed'), { description: errorMessageFrom(error) })
  }
}

// Screening has no chat turn or speech output. Only explicit decisions can create an observation turn.
async function interestingEvent(imageBase64: string, signal: AbortSignal): Promise<string | undefined> {
  const separateVision = vision.configured && vision.useForChat
    && !consciousness.providerModels.some(model => model.id === consciousness.activeModel && model.metadata?.abilities?.vision === true)
  const model = separateVision ? vision.activeModel : consciousness.activeModel
  const provider = separateVision
    ? await providers.getChatProviderInstance(vision.activeProvider)
    : await consciousness.getChatProviderInstance(consciousness.activeProvider)
  signal.throwIfAborted()
  const context = (sessions.sessionMessages[sessions.activeSessionId] ?? [])
    .filter(message => message.role === 'user' || message.role === 'assistant')
    .slice(-8)
    .map(message => ({ role: message.role, text: typeof message.content === 'string' ? message.content.slice(0, 500) : '' }))
  let decision = ''
  const deadline = AbortSignal.any([signal, AbortSignal.timeout(30_000)])
  await llm.stream(model, provider, { turns: [
    { id: 'screen-interest-policy', type: 'system', authority: 'system', content: [{ type: 'text', text: t('stage.screen-look.interest-prompt') }] },
    { id: 'screen-interest-frame', type: 'user', content: [
      { type: 'text', text: JSON.stringify({ recentConversation: context, previouslyDiscussedEvents: recentEvents }) },
      { type: 'image', url: `data:image/jpeg;base64,${imageBase64}` },
    ] },
  ] }, {
    supportsTools: false,
    taskTools: [],
    abortSignal: deadline,
    onStreamEvent: (event) => {
      if (event.type === 'text-delta')
        decision += event.text
    },
  })
  deadline.throwIfAborted()
  // Unknown or malformed decisions stay silent, rather than becoming spoken control text.
  return /^COMMENT:\s*(\S[^\r\n]{0,219})$/u.exec(decision.trim())?.[1]
}

async function comment(manual: boolean) {
  if (!watching.value || commenting.value || watchOwner.value?.signal.aborted)
    return
  if (!watchCapture.active) {
    stopWatching()
    return
  }
  if (occupied.value)
    return
  const owner = new AbortController()
  const sessionId = sessions.activeSessionId
  const messageId = crypto.randomUUID()
  commentOwner.value = owner
  ownTurn.value = messageId
  const cancelTurn = () => {
    void chat.cancelTurn({ sessionId, turnId: messageId }).catch(() => {})
  }
  try {
    const frame = forced.value && !manual && pendingFrame ? pendingFrame : watchCapture.frame()
    pendingFrame = undefined
    const selective = !manual && !forced.value
    if (selective && !gate.accepts(frame.thumbnail, Date.now()))
      return
    commenting.value = true
    owner.signal.addEventListener('abort', cancelTurn, { once: true })
    const event = selective ? await interestingEvent(frame.data, owner.signal) : undefined
    owner.signal.throwIfAborted()
    if (selective && !event) {
      gate.acknowledge(frame.thumbnail, Date.now())
      quiet.value = true
      return
    }
    quiet.value = false
    const result = await chat.send({
      sessionId,
      messageId,
      observation: true,
      text: `${t(forced.value ? 'stage.screen-look.forced-prompt' : 'stage.screen-look.watch-prompt')}${event ? `\n${JSON.stringify({ interestingEvent: event })}` : ''}`,
      attachments: [{ type: 'image', data: frame.data, mimeType: 'image/jpeg' }],
    })
    owner.signal.throwIfAborted()
    if (result.messages.some(message => message.role === 'error'))
      throw new Error(t('stage.screen-look.inference-failed'))
    gate.acknowledge(frame.thumbnail, Date.now())
    if (event)
      recentEvents = [...recentEvents.slice(-7), event]
    lastAt.value = new Date().toLocaleTimeString()
  }
  catch (error) {
    if (!owner.signal.aborted) {
      stopWatching()
      toast.error(t('stage.screen-look.failed'), { description: errorMessageFrom(error) })
    }
  }
  finally {
    owner.signal.removeEventListener('abort', cancelTurn)
    if (commentOwner.value === owner) {
      commentOwner.value = undefined
      ownTurn.value = ''
      commenting.value = false
      if (forced.value && pendingFrame && !occupied.value)
        void comment(false)
    }
  }
}

async function lookOnce() {
  if (busy.value || watching.value || opening.value)
    return
  if (!configured.value) {
    toast.error(t('stage.screen-look.configure'))
    return
  }
  const sessionId = sessions.activeSessionId
  const owner = new AbortController()
  const messageId = crypto.randomUUID()
  const cancelTurn = () => {
    void chat.cancelTurn({ sessionId, turnId: messageId }).catch(() => {})
  }
  cancellation = owner
  busy.value = true
  try {
    const data = await capture.capture(owner.signal)
    owner.signal.throwIfAborted()
    owner.signal.addEventListener('abort', cancelTurn, { once: true })
    await chat.send({
      sessionId,
      messageId,
      observation: true,
      text: t('stage.screen-look.prompt'),
      attachments: [{ type: 'image', data, mimeType: 'image/jpeg' }],
    })
  }
  catch (error) {
    if (error instanceof DOMException && (error.name === 'AbortError' || error.name === 'NotAllowedError'))
      return
    const description = error instanceof DOMException && error.name === 'NotSupportedError'
      ? t('stage.screen-look.unsupported')
      : errorMessageFrom(error)
    toast.error(t('stage.screen-look.failed'), { description })
  }
  finally {
    owner.signal.removeEventListener('abort', cancelTurn)
    if (cancellation === owner)
      cancellation = undefined
    busy.value = false
  }
}
</script>

<template>
  <div :class="['relative flex flex-wrap items-center gap-2']">
    <Button
      v-if="!watching"
      size="sm"
      :class="['w-full']"
      :disabled="busy || opening"
      :aria-label="t('stage.screen-look.forced')"
      :aria-pressed="forced"
      @click="forced = !forced"
    >
      {{ t(forced ? 'stage.screen-look.forced-on' : 'stage.screen-look.forced-off') }}
    </Button>
    <Button
      v-if="!watching"
      size="sm"
      :class="['min-h-10']"
      :loading="busy"
      :title="t('stage.screen-look.description')"
      @click="lookOnce"
    >
      <span :class="['i-solar:monitor-smartphone-bold-duotone size-4']" aria-hidden="true" />
      {{ busy ? t('stage.screen-look.busy') : t('stage.screen-look.button') }}
    </Button>
    <Button v-if="!watching" size="sm" :disabled="busy || opening" :title="t('stage.screen-look.watch-description')" @click="startWatching">
      {{ t('stage.screen-look.watch') }}
    </Button>
    <template v-else>
      <Button size="sm" :disabled="opening || commenting || occupied" @click="comment(true)">
        {{ t('stage.screen-look.now') }}
      </Button>
      <Button size="sm" @click="stopWatching">
        {{ t('stage.screen-look.stop') }}
      </Button>
      <div role="status" :class="[props.inlineStatus ? 'mt-2 w-full rounded-xl p-3' : 'absolute right-0 top-full mt-2 w-72 max-w-[85vw] rounded-xl p-3', 'bg-white/95 text-neutral-800 shadow-lg dark:bg-neutral-900/95 dark:text-neutral-100']">
        <p :class="['text-sm font-semibold']">
          {{ opening ? t('stage.screen-look.choosing') : t('stage.screen-look.watching') }}
        </p>
        <p :class="['truncate text-xs']" :title="source">
          {{ source }}
        </p>
        <p :class="['mt-1 text-xs']">
          {{ t(forced ? 'stage.screen-look.forced-cadence' : 'stage.screen-look.cadence') }}
        </p>
        <p :class="['text-xs']">
          {{ occupied ? t('stage.screen-look.waiting') : (commenting ? t('stage.screen-look.busy') : t(quiet ? 'stage.screen-look.quiet' : 'stage.screen-look.observing')) }}
        </p>
        <p v-if="lastAt" :class="['text-xs']">
          {{ t('stage.screen-look.last', { time: lastAt }) }}
        </p>
      </div>
    </template>
  </div>
</template>
