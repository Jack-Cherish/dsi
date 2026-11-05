<script setup lang="ts">
import { errorMessageFrom } from '@moeru/std'
import { useElectronEventaInvoke } from '@proj-airi/electron-vueuse'
import { ScreenLookButton } from '@proj-airi/stage-ui/components'
import { Button, FieldSelect } from '@proj-airi/ui'
import { computed, onBeforeUnmount, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { electronOpenSettings } from '../../shared/eventa'
import { useVisionScreenCapture } from '../composables/use-vision-screen-capture'

const { t } = useI18n()
const expanded = ref(false)
const watching = ref(false)
const locked = ref(false)
const error = ref('')
const openSettings = useElectronEventaInvoke(electronOpenSettings)
const capture = useVisionScreenCapture({ types: ['window', 'screen'], fetchWindowIcons: false })
const { sources, activeSourceId, activeSource, isRefetching } = capture
const options = computed(() => sources.value.map(source => ({ label: source.name, value: source.id })))

async function refresh() {
  const previous = activeSourceId.value
  error.value = ''
  try {
    await capture.refetchSources()
    // A refreshed list never grants consent for its first source.
    activeSourceId.value = sources.value.some(source => source.id === previous) ? previous : ''
  }
  catch (cause) {
    error.value = errorMessageFrom(cause) ?? t('stage.screen-look.failed')
  }
}

function toggle() {
  expanded.value = !expanded.value
  if (expanded.value && !sources.value.length)
    void refresh()
}

onBeforeUnmount(capture.cleanup)
</script>

<template>
  <div :class="['absolute left-2 top-2 z-40 max-w-[calc(100%_-_1rem)]']">
    <Button size="sm" :aria-expanded="expanded" @click="toggle">
      <span :class="['i-solar:monitor-smartphone-bold-duotone size-4']" aria-hidden="true" />
      {{ watching ? t('stage.screen-look.watching') : t('stage.screen-look.button') }}
    </Button>
    <div
      v-show="expanded"
      :class="['mt-2 max-h-[calc(100dvh_-_4rem)] w-80 max-w-full overflow-y-auto rounded-xl p-3 shadow-lg', 'bg-white/95 text-neutral-900 dark:bg-neutral-900/95 dark:text-neutral-100']"
    >
      <FieldSelect
        v-model="activeSourceId"
        :label="t('stage.screen-look.source')"
        :placeholder="t('stage.screen-look.choose-source')"
        :options="options"
        :disabled="locked || isRefetching"
        layout="vertical"
      />
      <p :class="['my-2 text-xs']">
        {{ t('stage.screen-look.watch-description') }}
      </p>
      <div :class="['mb-2 flex gap-2']">
        <Button size="sm" :disabled="locked || isRefetching" @click="refresh">
          {{ t('stage.screen-look.refresh-sources') }}
        </Button>
        <Button size="sm" @click="openSettings({ route: '/settings/modules/vision' })">
          {{ t('stage.screen-look.vision-settings') }}
        </Button>
      </div>
      <p v-if="error" role="alert" :class="['text-sm text-red-500']">
        {{ error }}
      </p>
      <!-- Keep the capture owner mounted when this panel collapses. -->
      <ScreenLookButton
        v-show="!!activeSourceId"
        :request-stream="capture.startStream"
        :source-name="activeSource?.name"
        inline-status
        @active="watching = $event"
        @locked="locked = $event"
      />
    </div>
  </div>
</template>
