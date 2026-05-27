<template>
    <transition name="slide-down">
        <div v-if="showWarning" class="fixed top-24 left-1/2 transform -translate-x-1/2 z-50 bg-amber-50 border border-amber-200 text-amber-800 px-6 py-3 rounded-full shadow-lg flex items-center gap-3">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-amber-500" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
            <span class="text-sm font-medium">{{ warningMessage }}</span>
            <button @click="showWarning = false" class="text-amber-500 hover:text-amber-700">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
            </button>
        </div>
    </transition>
  <div class="min-h-screen bg-[#F6FAFD] text-slate-950 font-sans relative overflow-x-hidden selection:bg-sky-100 selection:text-sky-950">
    
    <div class="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div class="liquid-oxygen-field"></div>
        <div class="academic-grid"></div>
    </div>

    <header class="sticky top-0 z-50 border-b border-sky-100/70 bg-white/80 backdrop-blur-xl">
      <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="relative w-10 h-10 flex items-center justify-center rounded-lg bg-[#0B1F3A] shadow-lg shadow-sky-900/15 text-white">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 32 32" fill="none">
              <path d="M7 18.5C8.8 13.4 13.4 10 19.1 10H24l-3.1 3.2L26 16l-5.1 2.8L24 22h-5.4C13.3 22 9.2 20.8 7 18.5Z" fill="currentColor"/>
              <path d="M9.5 18.3C7.6 17.7 5.7 16.4 4.2 14.5C4 17.7 5.1 20.1 7.5 21.6C7.6 20.3 8.2 19.1 9.5 18.3Z" fill="#BFE9FF"/>
              <path d="M18 11.7V21" stroke="#BFE9FF" stroke-width="1.5" stroke-linecap="round"/>
              <path d="M14.5 14H20.5M13.5 17H21.5" stroke="#BFE9FF" stroke-width="1.2" stroke-linecap="round"/>
            </svg>
          </div>
          <div class="flex flex-col">
            <h1 class="text-2xl font-black text-[#0B1F3A]">
              小蓝鲸论文助手
            </h1>
            <span class="text-[10px] font-semibold text-sky-700 tracking-[0.12em] uppercase -mt-1">
              Academic Paper Reading Agent
            </span>
          </div>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8 mt-4">
      
      <div class="lg:col-span-4 space-y-6">
        
        <div class="bg-white/86 backdrop-blur-xl p-6 rounded-xl shadow-xl shadow-sky-950/5 border border-sky-100/80 space-y-5">
          
            <div>
                <div class="flex justify-between items-center mb-2">
                    <label class="text-xs font-bold text-slate-500 uppercase tracking-wider">论文知识库</label>
                    <span class="text-[10px] text-sky-700 bg-sky-50 px-2 py-0.5 rounded-full border border-sky-100">最多 5 篇 PDF</span>
                </div>
                
                <div 
                    @dragover.prevent="isDragging = true"
                    @dragleave.prevent="isDragging = false"
                    @drop.prevent="handleDrop"
                    class="relative group cursor-pointer border-2 border-dashed rounded-xl p-4 transition-all duration-300 flex flex-col items-center justify-center text-center min-h-[100px]"
                    :class="isDragging ? 'border-sky-500 bg-sky-50/70' : 'border-slate-300 hover:border-sky-400 hover:bg-sky-50/40'"
                >
                    <input type="file" multiple accept=".pdf" class="absolute inset-0 opacity-0 cursor-pointer" @change="handleFileSelect" />
                    
                    <div v-if="uploadedFiles.length === 0" class="pointer-events-none flex flex-col items-center">
                        <div class="w-8 h-8 mb-2 rounded-full bg-sky-50 flex items-center justify-center text-sky-600">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                        </div>
                        <p class="text-xs text-slate-500 font-medium">拖拽论文 PDF 到这里</p>
                    </div>

                    <div v-else class="w-full space-y-2 pointer-events-none z-10">
                        <div v-for="(file, i) in uploadedFiles" :key="i" class="flex items-center justify-between bg-white px-3 py-2 rounded-lg shadow-sm border border-gray-100 text-xs animate-fade-in-up">
                            <div class="flex items-center gap-2 overflow-hidden">
                                <span class="text-red-500">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
                                </span>
                                <span class="truncate max-w-[150px] text-gray-700 font-medium">{{ file.name }}</span>
                            </div>
                            <span class="text-green-500">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <div
                class="rounded-xl border px-4 py-3 transition-all duration-300"
                :class="knowledgeStatusPanelClass"
            >
                <div class="flex items-center justify-between gap-3">
                    <div class="flex items-center gap-2 min-w-0">
                        <span
                            class="relative flex h-2.5 w-2.5 shrink-0 rounded-full"
                            :class="knowledgeStatusDotClass"
                        >
                            <span
                                v-if="knowledgeStatus === 'building'"
                                class="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-75"
                            ></span>
                        </span>
                        <span class="truncate text-sm font-semibold text-slate-800">
                            {{ knowledgeStatusTitle }}
                        </span>
                    </div>
                    <span
                        class="shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide"
                        :class="knowledgeStatusBadgeClass"
                    >
                        {{ knowledgeStatusBadge }}
                    </span>
                </div>
                <p class="mt-1 text-xs leading-relaxed text-slate-500">
                    {{ knowledgeStatusDescription }}
                </p>
                <div v-if="knowledgeStatus === 'building'" class="mt-3 h-1.5 overflow-hidden rounded-full bg-sky-100">
                    <div class="h-full w-1/2 rounded-full bg-gradient-to-r from-sky-500 to-cyan-300 animate-vector-build"></div>
                </div>
            </div>

            <div class="bg-slate-100/80 p-1 rounded-xl flex items-center relative">
                <div 
                    class="absolute top-1 bottom-1 w-[48%] bg-white rounded-lg shadow-sm transition-all duration-300 ease-out z-0"
                    :class="searchMode === 'document' ? 'left-1' : 'left-[51%]'"
                ></div>

                <button 
                    @click="setMode('document')"
                    :disabled="uploadedFiles.length === 0"
                    class="flex-1 py-2 text-[10px] font-bold tracking-wide rounded-lg z-10 transition-colors duration-300 uppercase flex items-center justify-center gap-1.5"
                    :class="[
                        searchMode === 'document' ? 'text-sky-700' : 'text-slate-400 hover:text-slate-600',
                        uploadedFiles.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
                    ]"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>
                    仅论文
                </button>
                <button 
                    @click="setMode('hybrid')"
                    class="flex-1 py-2 text-[10px] font-bold tracking-wide rounded-lg z-10 transition-colors duration-300 uppercase flex items-center justify-center gap-1.5"
                    :class="searchMode === 'hybrid' ? 'text-[#1D74B7]' : 'text-slate-400 hover:text-slate-600'"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
                    论文 + 网络
                </button>
            </div>

            <div>
                <div class="relative group">
                    <div class="absolute -inset-0.5 bg-gradient-to-r from-sky-500 to-cyan-300 rounded-xl opacity-20 group-hover:opacity-35 transition duration-500 blur"></div>
                    <textarea 
                    v-model="query" 
                    class="relative w-full p-4 bg-white border-0 rounded-xl shadow-inner text-slate-700 placeholder-slate-400 focus:ring-0 text-sm leading-relaxed resize-none transition-all"
                    rows="3"
                    placeholder="输入你想分析的论文问题，例如：这篇论文的方法创新是什么？"
                    :disabled="isLoading"
                    @keydown.enter.prevent="handleEnterSubmit"
                    ></textarea>
                </div>
            
                <div class="mt-4 space-y-3">
                    <div
                        v-if="hasReport"
                        class="flex items-center gap-2 rounded-lg border border-sky-100 bg-sky-50/80 px-3 py-2 text-xs text-sky-800"
                    >
                        <FileCheck2 class="h-4 w-4 shrink-0 text-sky-600" />
                        <span>当前已有一版报告，可继续修改或开始新分析。</span>
                    </div>

                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        <button
                            @click="startResearch('new_topic')"
                            :disabled="isLoading || !query"
                            class="group relative min-h-12 overflow-hidden rounded-xl bg-[#0B1F3A] px-4 py-3 text-white shadow-lg shadow-sky-950/20 transition-all duration-200 hover:bg-[#123154] hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-sky-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none sm:col-span-2"
                        >
                            <div class="absolute inset-0 bg-gradient-to-r from-[#0B1F3A] via-[#1D74B7] to-[#47B8E8] opacity-0 transition-opacity duration-300 group-hover:opacity-100"></div>
                            <span class="relative flex items-center justify-center gap-2 text-xs font-semibold tracking-wide">
                                <span>{{ activeIntent === 'new_topic' ? '正在启动分析...' : '开始新分析' }}</span>
                                <ArrowRight v-if="!isLoading" class="h-4 w-4" />
                            </span>
                        </button>

                        <button
                            @click="startResearch('edit_report')"
                            :disabled="isLoading || !query || !hasReport"
                            class="group min-h-12 rounded-xl border border-sky-200 bg-white/90 px-4 py-3 text-sky-800 shadow-sm transition-all duration-200 hover:border-sky-300 hover:bg-sky-50 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-sky-300 focus:ring-offset-2 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-50 disabled:text-slate-400 disabled:shadow-none"
                            :title="hasReport ? '只修改表达、结构或篇幅' : '生成报告后可用'"
                        >
                            <span class="flex items-center justify-center gap-2 text-xs font-semibold tracking-wide">
                                <PencilLine class="h-4 w-4" />
                                <span>{{ activeIntent === 'edit_report' ? '正在修改表达...' : '修改表达' }}</span>
                            </span>
                            <span v-if="!hasReport" class="mt-0.5 block text-[10px] font-medium text-slate-400">生成报告后可用</span>
                        </button>

                        <button
                            @click="startResearch('augment_report')"
                            :disabled="isLoading || !query || !hasReport"
                            class="group min-h-12 rounded-xl border border-cyan-200 bg-cyan-50/80 px-4 py-3 text-cyan-900 shadow-sm transition-all duration-200 hover:border-cyan-300 hover:bg-cyan-100/70 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-cyan-300 focus:ring-offset-2 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-50 disabled:text-slate-400 disabled:shadow-none"
                            :title="hasReport ? '检索补充证据后修改当前报告' : '生成报告后可用'"
                        >
                            <span class="flex items-center justify-center gap-2 text-xs font-semibold tracking-wide">
                                <SearchCheck class="h-4 w-4" />
                                <span>{{ activeIntent === 'augment_report' ? '正在补充证据...' : '补充证据/条件' }}</span>
                            </span>
                            <span v-if="!hasReport" class="mt-0.5 block text-[10px] font-medium text-slate-400">生成报告后可用</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <StatusFlow :currentStep="currentStep" />

        <div class="rounded-xl bg-[#071827] border border-slate-800 shadow-2xl overflow-hidden ring-1 ring-white/10">
            <div class="flex items-center gap-1.5 px-4 py-2 bg-slate-900/50 border-b border-slate-800">
                <div class="w-2.5 h-2.5 rounded-full bg-red-500/80"></div>
                <div class="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></div>
                <div class="w-2.5 h-2.5 rounded-full bg-green-500/80"></div>
                <span class="ml-2 text-[10px] font-mono text-slate-500">lab@bluewhale-paper:~</span>
            </div>
            <div ref="logsContainer" class="h-32 p-4 overflow-y-auto font-mono text-[11px] leading-5 space-y-1 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                <div v-if="logs.length === 0" class="text-slate-600 italic">论文分析台已就绪，等待输入...</div>
                <div v-for="(log, i) in logs" :key="i" class="flex gap-2">
                    <span class="text-sky-400 shrink-0">&gt;</span>
                    <span class="text-slate-300 break-all">{{ log }}</span>
                </div>
                <div v-if="isLoading" class="animate-pulse text-sky-400 mt-2">_</div>
            </div>
        </div>
      </div>

      <div class="lg:col-span-8 flex flex-col min-h-0 h-[72vh] min-h-[520px] lg:h-auto lg:min-h-0 lg:self-stretch">
        <div class="flex-1 min-h-0 lg:h-full bg-white/86 backdrop-blur-xl rounded-xl shadow-xl shadow-sky-950/5 border border-sky-100/80 p-8 lg:p-12 relative overflow-hidden flex flex-col">
            
            <div class="absolute top-0 right-0 p-8 opacity-[0.03] pointer-events-none">
                 <h1 class="text-8xl font-black font-sans">小蓝鲸</h1>
            </div>

            <div v-if="!displayedReport && !isLoading" class="flex-1 flex flex-col items-center justify-center text-gray-400 space-y-4">
                <div class="w-20 h-20 rounded-2xl bg-gradient-to-tr from-slate-50 to-sky-100 flex items-center justify-center shadow-inner border border-sky-100">
                    <svg class="w-10 h-10 text-sky-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"></path></svg>
                </div>
                <div class="text-center">
                    <h3 class="text-lg font-medium text-slate-900">等待论文分析</h3>
                    <p class="text-sm">上传论文或输入研究问题后开始。</p>
                </div>
            </div>

            <div v-else-if="isLoading && !displayedReport" class="flex-1 flex flex-col items-center justify-center relative">
                <div class="oxygen-stage">
                    <div class="oxygen-wave oxygen-wave-1"></div>
                    <div class="oxygen-wave oxygen-wave-2"></div>
                    <div class="oxygen-core">
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-14 h-14 text-white/90" viewBox="0 0 32 32" fill="none">
                            <path d="M7 18.5C8.8 13.4 13.4 10 19.1 10H24l-3.1 3.2L26 16l-5.1 2.8L24 22h-5.4C13.3 22 9.2 20.8 7 18.5Z" fill="currentColor"/>
                            <path d="M9.5 18.3C7.6 17.7 5.7 16.4 4.2 14.5C4 17.7 5.1 20.1 7.5 21.6C7.6 20.3 8.2 19.1 9.5 18.3Z" fill="#BFE9FF"/>
                        </svg>
                    </div>
                </div>

                <div class="mt-12 text-center space-y-2 relative z-10">
                    <h3 class="text-xl font-bold text-[#0B1F3A] tracking-widest animate-pulse">
                        正在解析论文证据
                    </h3>
                    <p class="text-xs text-sky-700 font-mono uppercase tracking-[0.2em]">
                        STRUCTURE SEARCH · RERANK · CITATION TRACE
                    </p>
                </div>
            </div>

            <div v-else class="report-scroll flex-1 min-h-0 overflow-y-scroll pr-3 prose prose-slate max-w-none prose-headings:font-display prose-headings:font-bold prose-headings:tracking-tight prose-a:text-sky-700 prose-img:rounded-xl">
                <div v-html="renderedReport"></div>
                <span v-if="isTyping" class="inline-block w-2 h-5 bg-sky-600 ml-1 animate-pulse align-middle"></span>
            </div>

        </div>
      </div>

    </main>
  </div>
</template>

<script setup>
import { ref, computed, nextTick } from 'vue';
import { uploadFiles, streamChat, clearContext, resetSessionThreadId } from './services/api';
import { createReportReviewGate } from './reportReviewGate';
import StatusFlow from './components/StatusFlow.vue';
import MarkdownIt from 'markdown-it';
import { ArrowRight, FileCheck2, PencilLine, SearchCheck } from 'lucide-vue-next';
// 【修复步骤 1】引入数学公式插件 (必须先 npm install markdown-it-katex)
import mk from 'markdown-it-katex';

// 【修复步骤 2】挂载插件
const md = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true
});
md.use(mk);
const showWarning = ref(false);
const warningMessage = ref('');
const triggerWarning = (msg) => {
    warningMessage.value = msg;
    showWarning.value = true;
    // 5秒后自动消失
    setTimeout(() => {
        showWarning.value = false;
    }, 5000);
};


// 状态变量
const query = ref('');
const isLoading = ref(false);
const activeIntent = ref(null);
const currentStep = ref('idle'); 
const logs = ref([]);
const logsContainer = ref(null);
const uploadedFiles = ref([]); 
const isDragging = ref(false);
const searchMode = ref('hybrid'); 
const knowledgeStatus = ref('idle');
const vectorChunkCount = ref(0);
const knowledgeStatusError = ref('');
const reportReviewGate = createReportReviewGate();

const knowledgeStatusTitle = computed(() => {
    if (knowledgeStatus.value === 'building') return '正在构建论文向量库';
    if (knowledgeStatus.value === 'ready') return '论文向量库已构建完成';
    if (knowledgeStatus.value === 'error') return '论文向量库构建失败';
    return '论文向量库未构建';
});

const knowledgeStatusBadge = computed(() => {
    if (knowledgeStatus.value === 'building') return '构建中';
    if (knowledgeStatus.value === 'ready') return '可提问';
    if (knowledgeStatus.value === 'error') return '需重试';
    return '待上传';
});

const knowledgeStatusDescription = computed(() => {
    if (knowledgeStatus.value === 'building') {
        return `正在解析 ${uploadedFiles.value.length} 篇 PDF，并写入向量索引，请稍等。`;
    }
    if (knowledgeStatus.value === 'ready') {
        return `已索引 ${vectorChunkCount.value} 个文本块，可以选择“仅论文”或“论文 + 网络”开始提问。`;
    }
    if (knowledgeStatus.value === 'error') {
        return knowledgeStatusError.value || '构建过程中发生错误，请重新上传 PDF。';
    }
    return '上传 PDF 后，这里会显示解析和向量库构建状态。';
});

const knowledgeStatusPanelClass = computed(() => {
    if (knowledgeStatus.value === 'building') return 'border-sky-200 bg-sky-50/80 shadow-sm shadow-sky-900/5';
    if (knowledgeStatus.value === 'ready') return 'border-emerald-200 bg-emerald-50/80 shadow-sm shadow-emerald-900/5';
    if (knowledgeStatus.value === 'error') return 'border-rose-200 bg-rose-50/80 shadow-sm shadow-rose-900/5';
    return 'border-slate-200 bg-slate-50/80';
});

const knowledgeStatusDotClass = computed(() => {
    if (knowledgeStatus.value === 'building') return 'bg-sky-500';
    if (knowledgeStatus.value === 'ready') return 'bg-emerald-500';
    if (knowledgeStatus.value === 'error') return 'bg-rose-500';
    return 'bg-slate-300';
});

const knowledgeStatusBadgeClass = computed(() => {
    if (knowledgeStatus.value === 'building') return 'bg-sky-100 text-sky-700';
    if (knowledgeStatus.value === 'ready') return 'bg-emerald-100 text-emerald-700';
    if (knowledgeStatus.value === 'error') return 'bg-rose-100 text-rose-700';
    return 'bg-slate-200 text-slate-500';
});

// 打字机变量
const displayedReport = ref('');
const isTyping = ref(false);
const hasReport = computed(() => displayedReport.value.trim().length > 0);

// 【修复步骤 3】增强渲染逻辑：把后端返回的 \[...\] 替换成插件能识别的 $$...$$
const renderedReport = computed(() => {
    let raw = displayedReport.value || '';
    
    // 1. 预处理：修复 LaTeX 定界符
    // 将 \[ ... \] 替换为 $$ ... $$ (块级公式)
    raw = raw.replace(/\\\[/g, '$$$').replace(/\\\]/g, '$$$');
    
    // 将 \( ... \) 替换为 $ ... $ (行内公式)
    raw = raw.replace(/\\\(/g, '$').replace(/\\\)/g, '$');

    // 2. 额外补丁：有些模型会输出不带反斜杠的 [ formula ]，这比较少见但要防备
    // 注意：这里需要小心不要误伤 Markdown 链接 [text](url)
    // 简单的策略是：如果 [ 后面跟着 \text 或 \frac 等 LaTeX 关键字，就认为是公式
    raw = raw.replace(/\[\s*(\\text|\\frac|\\sum|\\int)/g, '$$$$ $1'); 
    // 对应的闭合 ] 很难精准匹配，通常标准的 \[ \] 替换就够了。
    
    // 3. 渲染
    return md.render(raw);
});

const scrollToBottom = async () => {
    await nextTick();
    if (logsContainer.value) logsContainer.value.scrollTop = logsContainer.value.scrollHeight;
};

// --- 文件处理逻辑 ---
const handleFileSelect = async (event) => {
    processFiles(event.target.files);
};

const handleDrop = async (event) => {
    isDragging.value = false;
    processFiles(event.dataTransfer.files);
};

const processFiles = async (files) => {
    if (files.length > 5) {
        alert("最多上传 5 篇论文 PDF");
        knowledgeStatus.value = 'error';
        knowledgeStatusError.value = '一次最多只能上传 5 篇 PDF。';
        return;
    }
    
    uploadedFiles.value = Array.from(files);
    vectorChunkCount.value = 0;
    knowledgeStatusError.value = '';
    
    if (uploadedFiles.value.length > 0) {
        knowledgeStatus.value = 'building';
        logs.value.push(`[系统] 正在上传 ${files.length} 篇论文...`);
        try {
            const res = await uploadFiles(uploadedFiles.value);
            vectorChunkCount.value = res.chunks_stored || 0;
            knowledgeStatus.value = 'ready';
            logs.value.push(`[系统] 论文知识库构建完成，已索引 ${res.chunks_stored} 个段落 chunk。`);
        } catch (e) {
            knowledgeStatus.value = 'error';
            knowledgeStatusError.value = e.message || '上传失败，请检查后端服务或 PDF 文件。';
            logs.value.push(`[错误] 上传失败：${e.message}`);
            alert("上传失败：" + e.message);
            uploadedFiles.value = []; 
        }
    } else {
        knowledgeStatus.value = 'idle';
    }
};

const setMode = (mode) => {
    searchMode.value = mode;
};

let typingInterval = null;
const typeWriterEffect = (text) => {
    isTyping.value = true;
    
    // 【关键修复】：如果当前有正在运行的打字机，立刻干掉它！防止文字重叠并发
    if (typingInterval) {
        clearInterval(typingInterval);
    }
    
    let index = 0;
    typingInterval = setInterval(() => {
        if (index < text.length) {
            displayedReport.value += text.slice(index, index + 3);
            index += 3;
        } else {
            clearInterval(typingInterval);
            typingInterval = null; // 清空记录
            isTyping.value = false;
        }
    }, 10);
};

// --- 回车提交（不指定 intent，由 LLM 兜底判断） ---
const handleEnterSubmit = () => {
    if (isLoading.value) return;
    if (!query.value.trim()) return;
    startResearch(null);
};

// --- 开始研究 ---
const startResearch = async (intent = 'new_topic') => { 
    if (!query.value) return;
    if (intent && (intent === 'edit_report' || intent === 'augment_report') && !hasReport.value) return;
    
    isLoading.value = true;
    activeIntent.value = intent || 'auto';
    currentStep.value = (intent === 'edit_report') ? 'writer' : 'planner'; 
    logs.value = []; 
    logs.value.push(`[初始化] 小蓝鲸论文助手已启动，模式：${searchMode.value.toUpperCase()}`);
    if (intent === 'new_topic') {
        displayedReport.value = '';
    }
    reportReviewGate.reset();
    
    const actualMode = uploadedFiles.value.length === 0 ? 'hybrid' : searchMode.value;

    try {
        if (intent === 'new_topic' && uploadedFiles.value.length === 0) {
            logs.value.push(`[系统] 正在清空上一轮论文上下文...`);
            await clearContext();
            logs.value.push(`[系统] 上下文已清空，当前使用网络补充模式。`);
        }

        if (intent === 'new_topic') {
            resetSessionThreadId();
        }

        streamChat(
            query.value,
            actualMode,
            intent,
            (data) => {
                    // 1. 同步后端当前步骤
                    if (data.step) currentStep.value = data.step;

                    // --- 步骤 1: 规划 (Planner) ---
                    if (data.step === 'planner') {
                        currentStep.value = 'researcher'; // 视觉上跳到下一步
                        logs.value.push(`[问题解析] 路径：${data.data.plan.join(' / ')}`);
                    }

                    // --- 步骤 2: 搜索 (Researcher) ---
                    else if (data.step === 'researcher') {
                        const results = data.data.search_results || [];
                        // 检测熔断停止（仅论文模式下证据不足）
                        if (data.data.should_stop) {
                            triggerWarning("⛔️ 本地论文未找到相关证据，请尝试切换到「论文 + 网络」混合搜索模式");
                            logs.value.push(`[系统] 任务终止：仅论文模式下未找到相关证据。`);
                            currentStep.value = 'done';
                            return;
                        }

                        // 如果没有停止，则正常流转到 writer
                        currentStep.value = 'writer';
                        logs.value.push(`[证据检索] 已收集证据条目：${results.length}`);
                        results.slice(0, 3).forEach((item) => {
                            const firstLine = String(item).split('\n')[0];
                            if (firstLine.includes('[source:')) {
                                logs.value.push(`[证据] ${firstLine}`);
                            }
                        });
                    }

                    // --- 步骤 3: 写作 (Writer) ---
                    else if (data.step === 'writer') {
                        currentStep.value = 'reviewer';
                        logs.value.push(`[学术生成] 正在生成带引用回答...`);
                        if (data.data.final_report) {
                            reportReviewGate.holdDraft(data.data.final_report);
                        }
                    }

                    // --- 步骤 4: 审查 (Reviewer) ---
                    else if (data.step === 'reviewer') {
                        const reviewResult = reportReviewGate.review(data.data.review_status);
                        const revision = data.data.revision_number || 0;
                        if (data.data.review_status === 'FAIL') {
                            if (revision >= 3) {
                                triggerWarning(`⚠️ 报告质量未达标，已达最大3次修订上限，仅供参考`);
                                logs.value.push(`[引用校验] 未通过（已达3次上限）：${data.data.critique}`);
                                logs.value.push(`[系统] 报告质量未达标，请检查报告内容。`);
                                const lastDraft = reportReviewGate.peekDraft();
                                reportReviewGate.reset();
                                if (lastDraft) {
                                    displayedReport.value = '';
                                    typeWriterEffect(lastDraft);
                                }
                            } else {
                                logs.value.push(`[引用校验] 未通过：${data.data.critique}，正在重写`);
                                currentStep.value = 'planner';
                            }
                        } else {
                            logs.value.push(`[引用校验] 已通过。`);
                            if (reviewResult.action === 'publish') {
                                displayedReport.value = '';
                                typeWriterEffect(reviewResult.report);
                            }
                        }
                    }
                    else if (data.step === 'refiner') {
                    currentStep.value = 'writer'; // UI上复用写作状态
                    logs.value.push(`[修订] 正在根据反馈调整回答...`);
                    if (data.data.final_report) {
                        displayedReport.value = '';
                        // 重新打字输出修改后的报告
                        typeWriterEffect(data.data.final_report);
                    }
                }

                    scrollToBottom();
                },
            () => {
                isLoading.value = false;
                activeIntent.value = null;
                currentStep.value = 'done';
                logs.value.push('[完成] 论文分析完成。');
                scrollToBottom();
            },
            (err) => {
                isLoading.value = false;
                activeIntent.value = null;
                logs.value.push(`[错误] ${err.message}`);
                scrollToBottom();
            }
        );
    } catch (e) {
        isLoading.value = false;
        activeIntent.value = null;
        logs.value.push(`[错误] 初始化失败：${e.message}`);
        alert("系统错误：" + e.message);
    }
};
</script>

<style>
@import 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';

/* 保持原有的动画样式 */
.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.3s ease;
}
.slide-down-enter-from,
.slide-down-leave-to {
  transform: translate(-50%, -100%);
  opacity: 0;
}
@keyframes blob {
    0% { transform: translate(0px, 0px) scale(1); }
    33% { transform: translate(30px, -50px) scale(1.1); }
    66% { transform: translate(-20px, 20px) scale(0.9); }
    100% { transform: translate(0px, 0px) scale(1); }
}
.animate-blob {
    animation: blob 7s infinite;
}
.animation-delay-2000 { animation-delay: 2s; }
.animation-delay-4000 { animation-delay: 4s; }

/* 简单的淡入动画 */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in-up {
    animation: fadeInUp 0.3s ease-out;
}

@keyframes vectorBuild {
  0% { transform: translateX(-110%); }
  100% { transform: translateX(220%); }
}

.animate-vector-build {
  animation: vectorBuild 1.15s ease-in-out infinite;
}

.report-scroll {
  scrollbar-width: thin;
  scrollbar-color: rgba(29, 116, 183, 0.38) rgba(226, 242, 252, 0.72);
}

.report-scroll::-webkit-scrollbar {
  width: 8px;
}

.report-scroll::-webkit-scrollbar-track {
  background: rgba(226, 242, 252, 0.72);
  border-radius: 9999px;
}

.report-scroll::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(29, 116, 183, 0.52), rgba(71, 184, 232, 0.42));
  border-radius: 9999px;
  border: 2px solid rgba(255, 255, 255, 0.82);
}

.report-scroll::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(11, 31, 58, 0.62), rgba(29, 116, 183, 0.55));
}

.liquid-oxygen-field {
  position: absolute;
  inset: -20%;
  background:
    radial-gradient(60% 45% at 18% 16%, rgba(191, 233, 255, 0.42), transparent 62%),
    radial-gradient(50% 40% at 82% 12%, rgba(29, 116, 183, 0.20), transparent 60%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(229, 246, 255, 0.78) 45%, rgba(247, 251, 253, 0.96));
  filter: saturate(112%);
  animation: liquidOxygenShift 18s ease-in-out infinite alternate;
}

.academic-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(11, 31, 58, 0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(11, 31, 58, 0.045) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.8), transparent 78%);
}

@keyframes liquidOxygenShift {
  0% { transform: translate3d(-2%, -1%, 0) scale(1); }
  50% { transform: translate3d(2%, 1%, 0) scale(1.04); }
  100% { transform: translate3d(1%, -2%, 0) scale(1.02); }
}

.oxygen-stage {
  position: relative;
  width: 11rem;
  height: 11rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.oxygen-core {
  position: relative;
  z-index: 2;
  width: 6.5rem;
  height: 6.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 9999px;
  background:
    radial-gradient(circle at 34% 24%, rgba(255, 255, 255, 0.9), rgba(191, 233, 255, 0.55) 24%, transparent 38%),
    linear-gradient(145deg, #0B1F3A 0%, #1D74B7 58%, #69D7FF 100%);
  box-shadow:
    0 24px 70px rgba(29, 116, 183, 0.28),
    inset 0 1px 10px rgba(255, 255, 255, 0.42);
  animation: oxygenBreath 2.8s ease-in-out infinite;
}

.oxygen-core::after {
  content: '';
  position: absolute;
  inset: 12%;
  border-radius: inherit;
  border: 1px solid rgba(255, 255, 255, 0.35);
}

.oxygen-wave {
  position: absolute;
  inset: 1rem;
  border-radius: 9999px;
  border: 1px solid rgba(29, 116, 183, 0.16);
  animation: oxygenRipple 3.4s ease-out infinite;
}

.oxygen-wave-2 {
  animation-delay: 1.4s;
}

@keyframes oxygenBreath {
  0%, 100% { transform: scale(0.96); filter: brightness(0.98); }
  50% { transform: scale(1.06); filter: brightness(1.08); }
}

@keyframes oxygenRipple {
  0% { transform: scale(0.72); opacity: 0.52; }
  80%, 100% { transform: scale(1.42); opacity: 0; }
}

/* 2. 关键修复：解决 Tailwind 与 KaTeX 的样式冲突 */
/* Tailwind 默认将所有元素设为 border-box，这会破坏 KaTeX 的布局算法 */
.katex * {
    box-sizing: content-box !important;
}

/* 3. 公式滚动条优化 */
.katex-display {
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.5em 0;
    margin: 1em 0 !important; /* 修正外边距 */
}

/* --- 全局字体优化 --- */
body {
  font-family: theme('fontFamily.sans');
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* --- 舒适阅读风格 --- */
.prose {
  font-size: 1.05rem;
  color: #374151;
  line-height: 1.75;
}

/* 标题 */
.prose h1 {
  @apply text-3xl font-bold text-gray-900 mb-8 pb-4 border-b border-gray-100;
  font-family: theme('fontFamily.sans');
  line-height: 1.3;
}

.prose h2 {
  @apply text-xl font-bold text-gray-800 mt-10 mb-4 flex items-center;
  font-family: theme('fontFamily.sans');
  position: relative;
  padding-left: 1rem;
}
.prose h2::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 1em;
  @apply bg-blue-600 rounded-full;
}

.prose h3 {
  @apply text-lg font-bold text-gray-800 mt-8 mb-3;
  font-family: theme('fontFamily.sans');
}

/* 正文 */
.prose p {
  @apply text-justify mb-5 leading-relaxed text-gray-700;
}

/* 重点文字 */
.prose strong {
  @apply font-bold text-gray-900;
}

/* 摘要/引用块 */
.prose blockquote {
  font-style: normal !important;
  @apply my-8 pl-6 pr-4 py-5;
  @apply bg-gray-50 rounded-r-lg border-l-4 border-blue-500;
  @apply text-gray-700 text-base leading-relaxed; 
}

/* 列表 */
.prose ul {
  @apply list-disc list-outside ml-6 space-y-2 mb-6 text-gray-700;
}
.prose ol {
  @apply list-decimal list-outside ml-6 space-y-2 mb-6 text-gray-700;
}

/* 表格 */
.prose table {
  @apply w-full text-left border-collapse my-8 rounded-lg overflow-hidden border border-gray-200;
}
.prose thead {
  @apply bg-gray-50;
}
.prose th {
  @apply px-4 py-3 font-semibold text-gray-900 text-sm uppercase tracking-wide border-b border-gray-200;
}
.prose td {
  @apply px-4 py-3 text-sm text-gray-600 border-b border-gray-100;
}
.prose tr:hover td {
  @apply bg-blue-50/30 transition-colors;
}

/* 代码块 */
.prose pre {
  @apply bg-[#1e293b] text-gray-100 rounded-xl p-5 my-6 overflow-x-auto shadow-lg;
  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
  font-size: 0.9em;
}
.prose code {
  @apply text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded text-sm font-medium mx-0.5;
  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
}
.prose pre code {
  @apply bg-transparent text-gray-100 p-0 text-xs;
}

/* KaTeX 字体微调 */
.katex {
  font-size: 1.15em;
  font-family: 'Times New Roman', serif;
}
</style>
