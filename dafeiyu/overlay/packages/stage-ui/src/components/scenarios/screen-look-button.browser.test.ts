import type { GenerationProvider } from '@proj-airi/provider-inference'

import en from '@proj-airi/i18n/locales/en'

import { useLLM } from '@proj-airi/stage-ui/stores/ai/chat-llm/llm'
import { useChatStore } from '@proj-airi/stage-ui/stores/chat'
import { useChatSessionStore } from '@proj-airi/stage-ui/stores/chat/session-store'
import { useModsServerChannelStore } from '@proj-airi/stage-ui/stores/mods/api/channel-server'
import { useConsciousnessStore } from '@proj-airi/stage-ui/stores/modules/consciousness'
import { useVoiceMessagesStore } from '@proj-airi/stage-ui/stores/voice-messages'
import { createPinia, disposePinia } from 'pinia'
import { expect, it, vi } from 'vitest'
import { render } from 'vitest-browser-vue'
import { createI18n } from 'vue-i18n'

import ScreenLookButton from './screen-look-button.vue'

it.each(['SKIP', 'Maybe interesting', 'COMMENT: A surprising game result', 'forced'])(
  'keeps screening private and speaks only for an explicit interesting event: %s',
  async (decision) => {
    const modules = {
      consciousness: { provider: 'deepseek', model: 'deepseek-flash' },
      vision: { provider: '', model: '' },
      speech: { provider: '', model: '', voice_id: '' },
    }
    localStorage.setItem('airi-cards', JSON.stringify([['default', { name: 'Character', version: '1.0', extensions: { airi: { agents: {}, modules } } }]]))
    localStorage.setItem('airi-card-module-defaults', JSON.stringify(modules))
    const pinia = createPinia()
    const screen = render(ScreenLookButton, { global: { plugins: [pinia, createI18n({ legacy: false, locale: 'en', messages: { en } })] } })
    const chat = useChatStore(pinia)
    const sessions = useChatSessionStore(pinia)
    await sessions.initialize()
    await sessions.createSession(sessions.sessionMetas[sessions.activeSessionId]!.characterId)
    const consciousness = useConsciousnessStore(pinia)
    consciousness.activeProvider = 'deepseek'
    consciousness.activeModel = 'deepseek-flash'
    vi.spyOn(consciousness, 'getChatProviderInstance').mockResolvedValue({ generation: model => ({ protocol: 'chat-completions', config: { model, baseURL: 'https://example.com/' } }) })
    vi.spyOn(consciousness, 'getModelsForProvider').mockResolvedValue([])
    const inference = vi.spyOn(useLLM(pinia), 'stream').mockImplementation(async (_model, _provider, _context, options) => {
      await options?.onStreamEvent?.({ type: 'text-delta', text: decision })
      await options?.onStreamEvent?.({ type: 'finish' })
    })
    const send = vi.spyOn(chat, 'send')
    const initialCount = sessions.sessionMessages[sessions.activeSessionId]?.length ?? 0
    const canvas = document.createElement('canvas')
    const stream = canvas.captureStream(20)
    const timer = setInterval(() => canvas.getContext('2d')!.fillRect(0, 0, 50, 50), 50)
    vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockResolvedValue(stream)
    try {
      if (decision === 'forced')
        await screen.getByRole('button', { name: 'Five-second comments', exact: true }).click()
      await screen.getByRole('button', { name: 'Watch my screen', exact: true }).click()
      await expect.poll(() => inference.mock.calls.length, { timeout: 5000 }).toBeGreaterThan(0)
      await expect.element(screen.getByRole('button', { name: 'Comment now', exact: true })).toBeEnabled()
      expect(inference.mock.calls[0]![3]?.taskTools).toEqual([])
      if (decision === 'forced') {
        await expect.poll(() => send.mock.calls.length, { timeout: 8000 }).toBe(2)
        await expect.poll(() => inference.mock.calls.length).toBe(2)
        expect(send.mock.calls[0]![0].attachments).toEqual(send.mock.calls[1]![0].attachments)
        expect(inference.mock.calls[0]![2].turns.some(turn => turn.id === 'screen-interest-policy')).toBe(false)
        await screen.getByRole('button', { name: 'Stop watching', exact: true }).click()
        return
      }
      expect(inference.mock.calls[0]![2].turns).toContainEqual(expect.objectContaining({
        type: 'user',
        content: expect.arrayContaining([expect.objectContaining({ type: 'image', url: expect.stringMatching(/^data:image\/jpeg;base64,/) })]),
      }))
      if (decision.startsWith('COMMENT:')) {
        expect(send).toHaveBeenCalledTimes(1)
        expect(send.mock.calls[0]![0].observation).toBe(true)
      }
      else {
        expect(send).not.toHaveBeenCalled()
        expect(sessions.sessionMessages[sessions.activeSessionId]?.length ?? 0).toBe(initialCount)
        await screen.getByRole('button', { name: 'Comment now', exact: true }).click()
        await expect.poll(() => send.mock.calls.length).toBe(1)
        await expect.poll(() => inference.mock.calls.length).toBe(2)
      }
      await screen.getByRole('button', { name: 'Stop watching', exact: true }).click()
    }
    finally {
      clearInterval(timer)
      stream.getTracks().forEach(track => track.stop())
      screen.unmount()
      await chat.cancelPendingSends()
      useModsServerChannelStore(pinia).dispose()
      chat.dispose()
      disposePinia(pinia)
      vi.restoreAllMocks()
      localStorage.clear()
    }
  },
)

it.each(['typing', 'recording', 'source-ended'] as const)('interrupts screen inference on %s without cancelling the user turn', async (interruption) => {
  const modules = {
    consciousness: { provider: 'deepseek', model: 'deepseek-flash' },
    vision: { provider: '', model: '' },
    speech: { provider: '', model: '', voice_id: '' },
  }
  localStorage.setItem('airi-cards', JSON.stringify([['default', { name: 'Character', version: '1.0', extensions: { airi: { agents: {}, modules } } }]]))
  localStorage.setItem('airi-card-module-defaults', JSON.stringify(modules))
  const pinia = createPinia()
  const screen = render(ScreenLookButton, { global: { plugins: [pinia, createI18n({ legacy: false, locale: 'en', messages: { en } })] } })
  const chat = useChatStore(pinia)
  const sessions = useChatSessionStore(pinia)
  await sessions.initialize()
  await sessions.createSession(sessions.sessionMetas[sessions.activeSessionId]!.characterId)
  const consciousness = useConsciousnessStore(pinia)
  consciousness.activeProvider = 'deepseek'
  consciousness.activeModel = 'deepseek-flash'
  vi.spyOn(consciousness, 'getChatProviderInstance').mockResolvedValue({ generation: model => ({ protocol: 'chat-completions', config: { model, baseURL: 'https://example.com/' } }) })
  vi.spyOn(consciousness, 'getModelsForProvider').mockResolvedValue([])
  const inference = vi.spyOn(useLLM(pinia), 'stream').mockImplementationOnce(async (_model, _provider, _context, options) => {
    await new Promise<void>((resolve) => {
      if (options?.abortSignal?.aborted)
        resolve()
      else
        options?.abortSignal?.addEventListener('abort', () => resolve(), { once: true })
    })
  }).mockImplementation(async (_model, _provider, _context, options) => {
    await options?.onStreamEvent?.({ type: 'text-delta', text: 'I heard you.' })
    await options?.onStreamEvent?.({ type: 'finish' })
  })
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')!
  const stream = canvas.captureStream(20)
  const timer = setInterval(() => context.fillRect(0, 0, 50, 50), 50)
  vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockResolvedValue(stream)
  let userRequest: ReturnType<typeof chat.send> | undefined
  try {
    await screen.getByRole('button', { name: 'Watch my screen', exact: true }).click()
    await expect.poll(() => inference.mock.calls.length, { timeout: 5000 }).toBe(1)
    const signal = inference.mock.calls[0]![3]!.abortSignal!
    if (interruption === 'typing')
      userRequest = chat.send({ sessionId: sessions.activeSessionId, text: 'Listen to me.', messageId: 'user-priority' })
    else if (interruption === 'recording')
      useVoiceMessagesStore(pinia).isRecording = true
    else
      stream.getVideoTracks()[0].dispatchEvent(new Event('ended'))
    await expect.poll(() => signal.aborted).toBe(true)
    if (userRequest) {
      await userRequest
      expect(inference).toHaveBeenCalledTimes(2)
      expect(inference.mock.calls[1]![3]?.abortSignal?.aborted).toBe(false)
    }
    if (interruption === 'source-ended')
      expect(stream.getTracks().every(track => track.readyState === 'ended')).toBe(true)
    else
      await screen.getByRole('button', { name: 'Stop watching', exact: true }).click()
  }
  finally {
    clearInterval(timer)
    stream.getTracks().forEach(track => track.stop())
    screen.unmount()
    await chat.cancelPendingSends()
    await userRequest?.catch(() => {})
    useModsServerChannelStore(pinia).dispose()
    chat.dispose()
    disposePinia(pinia)
    vi.restoreAllMocks()
    localStorage.clear()
  }
})

it('allows stopping a pending source chooser and releases its late stream without a chat request', async () => {
  const pinia = createPinia()
  const screen = render(ScreenLookButton, { global: { plugins: [pinia, createI18n({ legacy: false, locale: 'en', messages: { en } })] } })
  const consciousness = useConsciousnessStore(pinia)
  consciousness.activeProvider = 'deepseek'
  consciousness.activeModel = 'deepseek-flash'
  const pending = Promise.withResolvers<MediaStream>()
  vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockReturnValue(pending.promise)
  const inference = vi.spyOn(useLLM(pinia), 'stream').mockResolvedValue(undefined)
  const stream = document.createElement('canvas').captureStream()
  try {
    await screen.getByRole('button', { name: 'Watch my screen', exact: true }).click()
    await screen.getByRole('button', { name: 'Stop watching', exact: true }).click()
    pending.resolve(stream)
    await expect.poll(() => stream.getTracks()[0].readyState).toBe('ended')
    expect(inference).not.toHaveBeenCalled()
    await expect.element(screen.getByRole('button', { name: 'Watch my screen', exact: true })).toBeVisible()
  }
  finally {
    pending.resolve(stream)
    screen.unmount()
    useModsServerChannelStore(pinia).dispose()
    useChatStore(pinia).dispose()
    disposePinia(pinia)
    vi.restoreAllMocks()
    localStorage.clear()
  }
})

it('cancels the pending image turn when the model changes during provider preparation', async () => {
  const modules = {
    consciousness: { provider: 'openai', model: 'test-model' },
    vision: { provider: '', model: '' },
    speech: { provider: '', model: '', voice_id: '' },
  }
  localStorage.setItem('airi-cards', JSON.stringify([['default', { name: 'Character', version: '1.0', extensions: { airi: { agents: {}, modules } } }]]))
  localStorage.setItem('airi-card-module-defaults', JSON.stringify(modules))
  const pinia = createPinia()
  const screen = render(ScreenLookButton, { global: { plugins: [pinia, createI18n({ legacy: false, locale: 'en', messages: { en } })] } })
  const sessions = useChatSessionStore(pinia)
  await sessions.initialize()
  await sessions.createSession(sessions.sessionMetas[sessions.activeSessionId]!.characterId)
  const consciousness = useConsciousnessStore(pinia)
  consciousness.activeProvider = 'openai'
  consciousness.activeModel = 'test-model'
  const pendingProvider = Promise.withResolvers<GenerationProvider>()
  const resolution = vi.spyOn(consciousness, 'getChatProviderInstance').mockReturnValue(pendingProvider.promise)
  vi.spyOn(consciousness, 'getModelsForProvider').mockResolvedValue([])
  const inference = vi.spyOn(useLLM(pinia), 'stream').mockResolvedValue(undefined)
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')!
  const stream = canvas.captureStream(20)
  const timer = setInterval(() => context.fillRect(0, 0, 50, 50), 50)
  vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockResolvedValue(stream)
  try {
    await screen.getByRole('button', { name: 'Look at my screen' }).click()
    await expect.poll(() => resolution.mock.calls.length).toBeGreaterThan(0)
    expect(stream.getTracks().every(track => track.readyState === 'ended')).toBe(true)
    consciousness.activeModel = 'another-model'
    pendingProvider.resolve({ generation: model => ({ protocol: 'chat-completions', config: { model, baseURL: 'https://example.com/' } }) })
    await expect.element(screen.getByRole('button', { name: 'Look at my screen' })).toBeVisible()
    expect(inference).not.toHaveBeenCalled()
  }
  finally {
    pendingProvider.resolve({ generation: model => ({ protocol: 'chat-completions', config: { model, baseURL: 'https://example.com/' } }) })
    clearInterval(timer)
    stream.getTracks().forEach(track => track.stop())
    screen.unmount()
    useModsServerChannelStore(pinia).dispose()
    useChatStore(pinia).dispose()
    disposePinia(pinia)
    vi.restoreAllMocks()
    localStorage.clear()
  }
})
