import { afterEach, describe, expect, it, vi } from 'vitest'

import { ScreenSnapshot } from './screen-snapshot'

const streams: MediaStream[] = []
const timers: ReturnType<typeof setInterval>[] = []

function canvasStream() {
  const canvas = document.createElement('canvas')
  canvas.width = 2000
  canvas.height = 1000
  const context = canvas.getContext('2d')!
  const paint = () => {
    context.fillStyle = '#2468ac'
    context.fillRect(0, 0, canvas.width, canvas.height)
  }
  paint()
  const stream = canvas.captureStream(20)
  timers.push(setInterval(paint, 50))
  streams.push(stream)
  return stream
}

afterEach(() => {
  streams.splice(0).forEach(stream => stream.getTracks().forEach(track => track.stop()))
  timers.splice(0).forEach(timer => clearInterval(timer))
  vi.restoreAllMocks()
})

describe('one-shot screen capture', () => {
  it('uses the desktop-selected stream and releases it on cancellation', async () => {
    const stream = canvasStream()
    const browserChooser = vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockRejectedValue(new Error('Unexpected browser chooser'))
    const desktopChooser = vi.fn().mockResolvedValue(stream)
    const capture = new ScreenSnapshot(desktopChooser)
    const owner = new AbortController()
    await capture.open(owner.signal)
    expect(desktopChooser).toHaveBeenCalledOnce()
    expect(browserChooser).not.toHaveBeenCalled()
    expect(capture.frame().data.length).toBeGreaterThan(100)
    owner.abort()
    expect(stream.getVideoTracks()[0].readyState).toBe('ended')
  })

  it('retains a shared source across frames until cancelled', async () => {
    const stream = canvasStream()
    vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockResolvedValue(stream)
    const capture = new ScreenSnapshot()
    const owner = new AbortController()
    await capture.open(owner.signal)
    expect(capture.frame().data.length).toBeGreaterThan(100)
    expect(capture.frame().thumbnail.length).toBe(64 * 36 * 4)
    expect(stream.getTracks()[0].readyState).toBe('live')
    owner.abort()
    expect(stream.getTracks()[0].readyState).toBe('ended')
    expect(() => capture.frame()).toThrow()
  })

  it('reports a source ending and refuses stale frames', async () => {
    const stream = canvasStream()
    vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockResolvedValue(stream)
    const ended = vi.fn()
    const capture = new ScreenSnapshot()
    await capture.open(new AbortController().signal, ended)
    stream.getVideoTracks()[0].dispatchEvent(new Event('ended'))
    expect(ended).toHaveBeenCalledOnce()
    expect(capture.active).toBe(false)
    expect(() => capture.frame()).toThrow()
  })

  it('returns one resized frame and stops every capture track before returning', async () => {
    const stream = canvasStream()
    const chooser = vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockResolvedValue(stream)
    const data = await new ScreenSnapshot().capture(new AbortController().signal)
    const image = new Image()
    image.src = `data:image/jpeg;base64,${data}`
    await image.decode()

    expect(image.width).toBe(1600)
    expect(image.height).toBe(800)
    expect(stream.getTracks().every(track => track.readyState === 'ended')).toBe(true)
    expect(chooser).toHaveBeenCalledExactlyOnceWith({ video: true, audio: false })
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d')!
    context.drawImage(image, 0, 0)
    const pixel = context.getImageData(0, 0, 1, 1).data
    expect(pixel[2]).toBeGreaterThan(150)
    expect(pixel[0]).toBeLessThan(70)
  })

  it('propagates a cancelled chooser without producing an image', async () => {
    vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockRejectedValue(new DOMException('Cancelled', 'NotAllowedError'))
    await expect(new ScreenSnapshot().capture(new AbortController().signal)).rejects.toMatchObject({ name: 'NotAllowedError' })
  })

  it('stops a late permission result after the owner cancels', async () => {
    const stream = canvasStream()
    let grant!: (stream: MediaStream) => void
    vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockReturnValue(new Promise(resolve => grant = resolve))
    const cancellation = new AbortController()
    const pending = new ScreenSnapshot().capture(cancellation.signal)
    const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    cancellation.abort()
    grant(stream)
    await rejected
    expect(stream.getTracks().every(track => track.readyState === 'ended')).toBe(true)
  })

  it('rejects duplicate captures while a source chooser is open', async () => {
    let rejectChooser!: (reason: Error) => void
    const chooser = vi.spyOn(navigator.mediaDevices, 'getDisplayMedia').mockReturnValue(new Promise((_resolve, reject) => rejectChooser = reject))
    const capture = new ScreenSnapshot()
    const pending = capture.capture(new AbortController().signal)
    const rejected = expect(pending).rejects.toMatchObject({ name: 'NotAllowedError' })
    await expect(capture.capture(new AbortController().signal)).rejects.toMatchObject({ name: 'InvalidStateError' })
    rejectChooser(new DOMException('Cancelled', 'NotAllowedError'))
    await rejected
    expect(chooser).toHaveBeenCalledTimes(1)
  })
})
