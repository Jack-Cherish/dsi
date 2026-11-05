import en from '@proj-airi/i18n/locales/en'

import { useChatStore } from '@proj-airi/stage-ui/stores/chat'
import { useModsServerChannelStore } from '@proj-airi/stage-ui/stores/mods/api/channel-server'
import { useConsciousnessStore } from '@proj-airi/stage-ui/stores/modules/consciousness'
import { useVoiceMessagesStore } from '@proj-airi/stage-ui/stores/voice-messages'
import { createPinia, disposePinia } from 'pinia'
import { expect, it, vi } from 'vitest'
import { render } from 'vitest-browser-vue'
import { createI18n } from 'vue-i18n'

import ScreenCompanion from './screen-companion.vue'

const ipc = vi.hoisted(() => ({ getSources: vi.fn(), selectWithSource: vi.fn() }))
vi.mock('@proj-airi/electron-screen-capture/vue', () => ({ useElectronScreenCapture: () => ipc }))
vi.mock('@proj-airi/electron-vueuse', () => ({ getElectronEventaContext: () => ({}), useElectronEventaInvoke: () => vi.fn() }))

it('requires a selected source, keeps watching when collapsed, and stops the selected stream', async () => {
  const pinia = createPinia()
  const screen = render(ScreenCompanion, { global: { plugins: [pinia, createI18n({ legacy: false, locale: 'en', messages: { en } })] } })
  const consciousness = useConsciousnessStore(pinia)
  consciousness.activeProvider = 'deepseek'
  consciousness.activeModel = 'deepseek-flash'
  useVoiceMessagesStore(pinia).isRecording = true
  ipc.getSources.mockResolvedValue([{ id: 'window:1', name: 'Shared video', thumbnail: new Uint8Array() }])
  const canvas = document.createElement('canvas')
  const stream = canvas.captureStream(20)
  const timer = setInterval(() => canvas.getContext('2d')!.fillRect(0, 0, 20, 20), 50)
  ipc.selectWithSource.mockImplementation(async (select) => {
    expect(select()).toBe('window:1')
    return stream
  })
  try {
    expect(ipc.selectWithSource).not.toHaveBeenCalled()
    await screen.getByRole('button', { name: 'Look at my screen', exact: true }).click()
    await expect.poll(() => ipc.getSources.mock.calls.length).toBe(1)
    expect(ipc.selectWithSource).not.toHaveBeenCalled()
    await screen.getByRole('combobox').click()
    await screen.getByRole('option', { name: 'Shared video' }).click()
    await screen.getByRole('button', { name: 'Watch my screen', exact: true }).click()
    await expect.poll(() => ipc.selectWithSource.mock.calls.length).toBe(1)
    await screen.getByRole('button', { name: 'Watching your shared screen', exact: true }).click()
    expect(stream.getVideoTracks()[0].readyState).toBe('live')
    await screen.getByRole('button', { name: 'Watching your shared screen', exact: true }).click()
    await screen.getByRole('button', { name: 'Stop watching', exact: true }).click()
    await expect.poll(() => stream.getVideoTracks()[0].readyState).toBe('ended')
  }
  finally {
    screen.unmount()
    clearInterval(timer)
    stream.getTracks().forEach(track => track.stop())
    useModsServerChannelStore(pinia).dispose()
    useChatStore(pinia).dispose()
    disposePinia(pinia)
    localStorage.clear()
    vi.clearAllMocks()
  }
})

it('locks source selection during a delayed one-shot capture and releases a late stream after unmount', async () => {
  const pinia = createPinia()
  const screen = render(ScreenCompanion, { global: { plugins: [pinia, createI18n({ legacy: false, locale: 'en', messages: { en } })] } })
  const consciousness = useConsciousnessStore(pinia)
  consciousness.activeProvider = 'deepseek'
  consciousness.activeModel = 'deepseek-flash'
  ipc.getSources.mockResolvedValue([{ id: 'window:1', name: 'Shared video', thumbnail: new Uint8Array() }])
  const pending = Promise.withResolvers<MediaStream>()
  ipc.selectWithSource.mockReturnValue(pending.promise)
  const stream = document.createElement('canvas').captureStream()
  try {
    await screen.getByRole('button', { name: 'Look at my screen', exact: true }).click()
    await screen.getByRole('combobox').click()
    await screen.getByRole('option', { name: 'Shared video' }).click()
    await screen.getByRole('button', { name: 'Look at my screen', exact: true }).nth(1).click()
    await expect.poll(() => ipc.selectWithSource.mock.calls.length).toBe(1)
    await expect.element(screen.getByRole('combobox')).toBeDisabled()
    await expect.element(screen.getByRole('button', { name: 'Refresh list' })).toBeDisabled()
    screen.unmount()
    pending.resolve(stream)
    await expect.poll(() => stream.getVideoTracks()[0].readyState).toBe('ended')
  }
  finally {
    pending.resolve(stream)
    stream.getTracks().forEach(track => track.stop())
    useModsServerChannelStore(pinia).dispose()
    useChatStore(pinia).dispose()
    disposePinia(pinia)
    localStorage.clear()
    vi.clearAllMocks()
  }
})
