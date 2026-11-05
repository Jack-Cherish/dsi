/** Compares thumbnails with the last completed analysis. Frames rejected before analysis never become the baseline. */
export class ScreenChangeGate {
  private previous?: Uint8ClampedArray
  private nextAt = 0

  accepts(pixels: Uint8ClampedArray, now: number): boolean {
    if (now < this.nextAt)
      return false
    if (!this.previous || this.previous.length !== pixels.length)
      return true
    let changed = 0
    for (let i = 0; i < pixels.length; i += 4) {
      const delta = Math.abs(pixels[i] - this.previous[i])
        + Math.abs(pixels[i + 1] - this.previous[i + 1])
        + Math.abs(pixels[i + 2] - this.previous[i + 2])
      if (delta > 60)
        changed++
    }
    // Ignore compression noise and cursor motion. A changed panel must cover at least 6% of the thumbnail.
    return changed / (pixels.length / 4) >= 0.06
  }

  acknowledge(pixels: Uint8ClampedArray, now: number) {
    this.previous = pixels.slice()
    this.nextAt = now + 15_000
  }
}
