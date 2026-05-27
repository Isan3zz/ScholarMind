<template>
  <div class="relative overflow-hidden rounded-xl border border-sky-100/80 bg-white/70 backdrop-blur-xl shadow-xl shadow-sky-950/5 p-6 transition-all hover:shadow-2xl">
    <div class="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-[#0B1F3A] via-[#1D74B7] to-[#BFE9FF]"></div>

    <h3 class="relative text-xs font-bold text-gray-400 mb-6 uppercase tracking-[0.2em] font-sans">
      论文分析流程
    </h3>
    
    <div class="relative space-y-8 pl-2">
      <div class="absolute left-[19px] top-4 bottom-4 w-0.5 bg-gray-200/50 rounded-full"></div>

      <div 
        v-for="(step, index) in steps" 
        :key="step.id"
        class="relative flex items-center gap-5 group"
      >
        <div class="relative z-10 flex items-center justify-center w-10 h-10 rounded-xl border transition-all duration-500"
          :class="getIconStyles(index)"
        >
          <div v-if="currentStepIndex === index && currentStep !== 'done'" 
               class="absolute inset-0 rounded-xl bg-sky-500 blur-lg opacity-40 animate-pulse">
          </div>

          <component 
            :is="step.icon" 
            class="w-5 h-5 transition-all duration-300" 
            :class="currentStepIndex === index && currentStep !== 'done' ? 'animate-bounce-slight text-white' : ''" 
          />
        </div>

        <div class="flex flex-col">
          <span 
            class="text-sm font-bold tracking-wide transition-all duration-300"
            :class="getTextStyles(index)"
          >
            {{ step.label }}
          </span>
          <span class="text-[10px] text-gray-400 font-medium tracking-wider uppercase">
             {{ step.desc }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { BrainCircuitIcon, SearchIcon, FileTextIcon, ShieldCheckIcon } from 'lucide-vue-next';

const props = defineProps({
  currentStep: { type: String, default: 'idle' }
});

const steps = [
  { id: 'planner', label: '问题解析', desc: '识别论文意图与回答路径', icon: BrainCircuitIcon },
  { id: 'researcher', label: '证据检索', desc: '混合召回、重排序与上下文回填', icon: SearchIcon },
  { id: 'writer', label: '学术生成', desc: '基于证据生成带引用回答', icon: FileTextIcon },
  { id: 'reviewer', label: '引用校验', desc: '检查证据、逻辑与可溯源性', icon: ShieldCheckIcon },
];

const currentStepIndex = computed(() => {
    if (props.currentStep === 'idle') return -1;
    if (props.currentStep === 'done') return steps.length;
    return steps.findIndex(s => s.id === props.currentStep);
});

// 样式辅助函数
const getIconStyles = (index) => {
    // 已完成
    if (props.currentStep === 'done' || currentStepIndex.value > index) {
        return 'bg-gradient-to-br from-emerald-400 to-teal-600 border-transparent text-white shadow-lg shadow-emerald-500/25 scale-100';
    }
    // 进行中
    if (currentStepIndex.value === index) {
        return 'bg-gradient-to-br from-[#0B1F3A] to-[#1D74B7] border-transparent text-white shadow-lg shadow-sky-500/35 scale-110';
    }
    // 未开始
    return 'bg-white border-gray-200 text-gray-300 scale-90';
};

const getTextStyles = (index) => {
    if (props.currentStep === 'done' || currentStepIndex.value > index) return 'text-green-600';
    if (currentStepIndex.value === index) return 'text-sky-800 scale-105 origin-left';
    return 'text-gray-300';
};
</script>

<style scoped>
/* 自定义微动效 */
@keyframes bounce-slight {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-2px); }
}
.animate-bounce-slight {
    animation: bounce-slight 2s infinite;
}
</style>
