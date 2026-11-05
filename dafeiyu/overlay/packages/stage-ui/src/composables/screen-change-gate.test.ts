import { expect, it } from 'vitest'

import { ScreenChangeGate } from './screen-change-gate'

function pixels(value: number) {
  return new Uint8ClampedArray(64 * 36 * 4).fill(value)
}

it('accepts the first frame and skips unchanged frames after cooldown', () => {
  const gate = new ScreenChangeGate()
  expect(gate.accepts(pixels(20), 0)).toBe(true)
  gate.acknowledge(pixels(20), 0)
  expect(gate.accepts(pixels(20), 20_000)).toBe(false)
  expect(gate.accepts(pixels(21), 20_000)).toBe(false)
})

it('waits fifteen seconds after success and compares against the acknowledged frame', () => {
  const gate = new ScreenChangeGate()
  gate.acknowledge(pixels(20), 1000)
  expect(gate.accepts(pixels(200), 15_999)).toBe(false)
  expect(gate.accepts(pixels(200), 16_000)).toBe(true)
  expect(gate.accepts(pixels(200), 20_000)).toBe(true)
  gate.acknowledge(pixels(200), 20_000)
  expect(gate.accepts(pixels(200), 40_000)).toBe(false)
})

it('ignores a small cursor-sized change but detects a changed panel', () => {
  const gate = new ScreenChangeGate()
  gate.acknowledge(pixels(20), 0)
  const frame = pixels(20)
  frame.fill(240, 0, 16)
  expect(gate.accepts(frame, 20_000)).toBe(false)
  frame.fill(240, 0, 1200)
  expect(gate.accepts(frame, 20_000)).toBe(true)
})
