/** Owns one browser source selection. open retains tracks until close or cancellation. capture releases them after one frame. */
export class ScreenSnapshot {
  /** The desktop supplies its selected stream. The web uses the browser chooser. */
  constructor(private readonly requestStream = () => navigator.mediaDevices.getDisplayMedia({ video: true, audio: false })) {}

  private pending = false
  private stream?: MediaStream
  private video?: HTMLVideoElement
  private release?: () => void

  get source() {
    return this.stream?.getVideoTracks()[0]?.label ?? ''
  }

  get active() {
    return this.stream?.getVideoTracks().some(track => track.readyState === 'live') ?? false
  }

  close() {
    this.release?.()
  }

  /** Returns JPEG base64 bytes. Cancellation also releases a stream granted after the chooser was cancelled by its owner. */
  async capture(signal: AbortSignal): Promise<string> {
    await this.open(signal)
    try {
      return this.frame().data
    }
    finally {
      this.close()
    }
  }

  async open(signal: AbortSignal, onEnded?: () => void): Promise<void> {
    signal.throwIfAborted()
    if (this.pending || this.active)
      throw new DOMException('A capture is already pending', 'InvalidStateError')
    if (!navigator.mediaDevices?.getDisplayMedia)
      throw new DOMException('Screen sharing is unavailable', 'NotSupportedError')

    this.pending = true
    let stream: MediaStream | undefined
    const video = document.createElement('video')
    const ended = () => {
      this.close()
      onEnded?.()
    }
    const stop = () => {
      stream?.getTracks().forEach((track) => {
        track.removeEventListener('ended', ended)
        track.stop()
      })
      signal.removeEventListener('abort', stop)
      video.pause()
      video.srcObject = null
      if (this.video === video) {
        this.stream = undefined
        this.video = undefined
        this.release = undefined
      }
    }
    signal.addEventListener('abort', stop, { once: true })
    try {
      // Keep this call before the first await so the browser retains the user's click activation.
      stream = await this.requestStream()
      signal.throwIfAborted()
      this.stream = stream
      this.video = video
      this.release = stop
      stream.getVideoTracks().forEach(track => track.addEventListener('ended', ended, { once: true }))
      video.muted = true
      video.playsInline = true
      video.srcObject = stream
      await this.waitForFrame(video, signal)
      signal.throwIfAborted()
      if (!video.videoWidth || !video.videoHeight)
        throw new Error('The selected source has no image')
    }
    catch (error) {
      stop()
      throw error
    }
    finally {
      this.pending = false
    }
  }

  /** Reads the current shared frame. The thumbnail stays local and is used only for change detection. */
  frame(): { data: string, thumbnail: Uint8ClampedArray } {
    const video = this.video
    if (!video || !this.active || video.readyState < 2)
      throw new DOMException('Screen sharing ended', 'InvalidStateError')
    const scale = Math.min(1, 1600 / Math.max(video.videoWidth, video.videoHeight))
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale))
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale))
    const context = canvas.getContext('2d')
    if (!context)
      throw new Error('A canvas context is unavailable')
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    const data = canvas.toDataURL('image/jpeg', 0.85).slice('data:image/jpeg;base64,'.length)
    canvas.width = 64
    canvas.height = 36
    context.drawImage(video, 0, 0, 64, 36)
    return { data, thumbnail: context.getImageData(0, 0, 64, 36).data }
  }

  private waitForFrame(video: HTMLVideoElement, signal: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      let timeout: ReturnType<typeof setTimeout>
      let frame: number
      function cancel() {
        finish(signal.reason)
      }
      function failed() {
        finish(new Error('The selected source could not be read'))
      }
      function finish(error?: unknown) {
        clearTimeout(timeout)
        video.cancelVideoFrameCallback(frame)
        signal.removeEventListener('abort', cancel)
        video.removeEventListener('error', failed)
        if (error)
          reject(error)
        else
          resolve()
      }
      timeout = setTimeout(() => finish(new DOMException('Screen capture timed out', 'TimeoutError')), 10_000)
      frame = video.requestVideoFrameCallback(() => finish())
      signal.addEventListener('abort', cancel, { once: true })
      video.addEventListener('error', failed, { once: true })
      void video.play().catch(finish)
    })
  }
}
