package com.example.ui

import ai.onnxruntime.OrtEnvironment
import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.SwapAuditDatabase
import com.example.data.SwapAuditLog
import com.example.data.SwapAuditRepository
import com.example.onnx.CompleteVisualValidationSuite
import com.example.onnx.DetectedFace
import com.example.onnx.FaceAlignment
import com.example.onnx.FaceReactionSourceMode
import com.example.onnx.FaceSwapAndRestorationService
import com.example.onnx.FaceSwapExecutionResult
import com.example.onnx.HeadSegmentationAndInpainting
import com.example.onnx.InpaintModelOption
import com.example.onnx.DepthEstimationEngine
import com.example.onnx.AccessoryMaskEngine
import com.example.onnx.AccessoryMaskLayers
import com.example.onnx.AccessoryTarget
import com.example.onnx.DepthMapResult
import com.example.service.BackgroundProcessingCoordinator
import com.example.service.ProcessingForegroundService
import com.example.onnx.InteractiveInpaintEngine
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxMemoryServiceState
import com.example.onnx.OnnxModelInspection
import com.example.onnx.OnnxProtobufInspector
import com.example.onnx.OnnxRuntimeModelService
import com.example.onnx.SdInpaintPackManager
import com.example.onnx.SdInpaintPackType
import com.example.onnx.SkinToneSourceMode
import com.example.onnx.SwapStageProgress
import com.example.onnx.VisualValidationBenchmark
import com.example.util.ImageGalleryHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class AppTab {
    STUDIO,
    MODELS,
    HISTORY,
    ETHICS
}

enum class StudioSubPage {
    FRONT_HOME,
    SETTINGS_AND_DIAGNOSTICS,
    ALIGNMENT_PREVIEW,
    INPAINT_PAGE
}

enum class StudioMode(val title: String, val badge: String) {
    FACE_SWAP("FACE SWAP", "MODE 1 — HyperSwap 1b (256px) Primary Production Engine"),
    HEAD_REPLACEMENT("FULL HEAD REPLACEMENT", "MODE 2 — GHOST 2.0 Android Full Head/Hair/Neck"),
    IDENTITY_PRESERVED("IDENTITY PRESERVED", "MODE 3 — High-Fidelity Source Identity & Texture (No AI Hallucination)")
}

enum class ProcessingQualityLevel(
    val title: String,
    val maxDecodeDimensionPx: Int,
    val subtitle: String
) {
    FAST("FAST", 1280, "1280px Max • Fast Blend • Safe for <4GB RAM"),
    BALANCED("BALANCED", 2048, "2048px Max • 512×512 HD Restore • 4–7.5GB RAM"),
    HIGH_QUALITY("HIGH QUALITY", 3072, "3072px Max • Full Tiled HD & Hair Matting • 8GB+ RAM")
}

enum class OutputResolutionOption(
    val title: String,
    val maxExportDimensionPx: Int
) {
    ORIGINAL("Original Size", 0),
    RES_1080P("Full HD (1920px)", 1920),
    RES_2K_QHD("2K QHD (2560px)", 2560),
    RES_4K_UHD("4K UHD (3840px)", 3840)
}

data class HeadAlignmentPreviewState(
    val segmentedSourceHeadBitmap: Bitmap,
    val alignedTargetHeadBitmap: Bitmap,
    val sourceRollDeg: Float,
    val targetRollDeg: Float,
    val segmentationLabel: String
)

data class FaceSwapUiState(
    val currentTab: AppTab = AppTab.STUDIO,
    val studioSubPage: StudioSubPage = StudioSubPage.FRONT_HOME,
    val studioMode: StudioMode = StudioMode.FACE_SWAP,
    val qualityLevel: ProcessingQualityLevel = ProcessingQualityLevel.BALANCED,
    val outputResolution: OutputResolutionOption = OutputResolutionOption.ORIGINAL,
    val deviceTotalRamGb: Float = 6.0f,
    val deviceAvailRamMb: Long = 2048L,
    val modelInspections: List<OnnxModelInspection> = emptyList(),
    val memoryServiceState: OnnxMemoryServiceState = OnnxMemoryServiceState(),
    val isInspectingModels: Boolean = false,
    val sourceBitmap: Bitmap? = null,
    val sourceFaces: List<DetectedFace> = emptyList(),
    val selectedSourceFaceIndex: Int = 0,
    val isDetectingSource: Boolean = false,
    val targetBitmap: Bitmap? = null,
    val targetFaces: List<DetectedFace> = emptyList(),
    val selectedTargetFaceIndex: Int = 0,
    val replaceAllTargetFaces: Boolean = false,
    val isDetectingTarget: Boolean = false,
    val headPreviewState: HeadAlignmentPreviewState? = null,
    val isGeneratingHeadPreview: Boolean = false,
    val consentConfirmed: Boolean = true,
    val enableColorTransfer: Boolean = true,
    val skinToneMode: SkinToneSourceMode = SkinToneSourceMode.SOURCE_IDENTITY,
    val faceReactionMode: FaceReactionSourceMode = FaceReactionSourceMode.TARGET_REACTION,
    val enableOcclusionProtection: Boolean = true,
    val enableNegativeArtifactSuppression: Boolean = true,
    val enableLcmInpainting: Boolean = true,
    val faceAgingFactor: Float = 0.0f,
    val portraitBlurStrength: Float = 0.0f,
    val blendStrength: Float = 1.0f,
    val enhancementStrength: Float = 0.96f,
    val faceOffsetX: Float = 0f,
    val faceOffsetY: Float = 0f,
    val faceScaleAdjust: Float = 1.0f,
    val enableProvenanceWatermark: Boolean = false,
    val allowTwoModelFallbackForTesting: Boolean = false,
    val preferHardwareAcceleration: Boolean = true,
    val lowMemoryMode: Boolean = false,
    val isSwapping: Boolean = false,
    val swapProgress: SwapStageProgress? = null,
    val swapResult: FaceSwapExecutionResult? = null,
    val visualValidationSuite: CompleteVisualValidationSuite? = null,
    val isRunningVisualValidation: Boolean = false,
    val lastAuditLogId: Int? = null,
    val showOriginalInComparison: Boolean = false,
    val inpaintSourceBitmap: Bitmap? = null,
    val inpaintMaskBitmap: Bitmap? = null,
    val inpaintPrompt: String = "",
    val inpaintFastMode: Boolean = true,
    val inpaintStrength: Float = 0.70f,
    val inpaintCfgScale: Float = 7.0f,
    val inpaintDepthGuardEnabled: Boolean = false,
    val inpaintDepthMap: DepthMapResult? = null,
    val isComputingInpaintDepth: Boolean = false,
    val inpaintAccessoryLayers: AccessoryMaskLayers? = null,
    val isComputingAccessoryMask: Boolean = false,
    val selectedAccessoryTargets: Set<AccessoryTarget> = emptySet(),
    val segformerAutoSelectedForAccessory: Boolean = false,
    val inpaintAccessoryExpandPx: Float = 6f,
    val accessoryMaskNotice: String? = null,
    val isInpainting: Boolean = false,
    val inpaintProgressText: String? = null,
    val inpaintResultBitmap: Bitmap? = null,
    val inpaintElapsedSec: Float? = null,
    val inpaintEngineLabel: String? = null,
    val inpaintModelOption: InpaintModelOption = InpaintModelOption.AUTO,
    val installedSdInpaintPacks: List<SdInpaintPackType> = emptyList(),
    val isLamaInpaintReady: Boolean = false,
    val inpaintModelLabel: String = "Coming Soon",
    val statusBannerMessage: String? = null,
    val errorBannerMessage: String? = null
) {
    val isDetectorReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.DETECTOR && it.isValidOnnx }

    val isRecognizerReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.RECOGNIZER && it.isValidOnnx }

    val isSwapperReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.SWAPPER && it.isValidOnnx }

    val isSegmentationOnnxReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.SEGMENTATION && it.isValidOnnx }

    val isDepthModelReady: Boolean
        get() = modelInspections.any { it.slot == ModelSlot.DEPTH && it.isValidOnnx }

    val canExecuteSwap: Boolean
        get() = !isSwapping &&
            consentConfirmed &&
            sourceBitmap != null &&
            sourceFaces.isNotEmpty() &&
            targetBitmap != null &&
            targetFaces.isNotEmpty()
}

class FaceSwapViewModel(application: Application) : AndroidViewModel(application) {

    private val ortEnv: OrtEnvironment? = runCatching { OrtEnvironment.getEnvironment() }.getOrNull()
    val modelService: FaceSwapAndRestorationService = OnnxRuntimeModelService(application, ortEnv)
    private val auditRepository: SwapAuditRepository = SwapAuditRepository(
        SwapAuditDatabase.getInstance(application).swapAuditDao()
    )

    private val initialRamConfig = detectDeviceRamProfile(application)

    private val _uiState = MutableStateFlow(
        FaceSwapUiState(
            lowMemoryMode = initialRamConfig.first,
            qualityLevel = initialRamConfig.second,
            deviceTotalRamGb = initialRamConfig.third.first,
            deviceAvailRamMb = initialRamConfig.third.second
        )
    )
    val uiState: StateFlow<FaceSwapUiState> = _uiState.asStateFlow()

    val auditLogs: StateFlow<List<SwapAuditLog>> = auditRepository.allLogs.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = emptyList()
    )

    init {
        refreshModelInspections()
        viewModelScope.launch {
            BackgroundProcessingCoordinator.inpaintSync.collect { sync ->
                if (sync.isRunning || sync.resultBitmap != null || sync.errorMessage != null) {
                    _uiState.update { cur ->
                        cur.copy(
                            isInpainting = sync.isRunning,
                            inpaintProgressText = sync.progressText,
                            inpaintResultBitmap = sync.resultBitmap ?: cur.inpaintResultBitmap,
                            inpaintElapsedSec = sync.elapsedSec ?: cur.inpaintElapsedSec,
                            inpaintEngineLabel = sync.engineLabel ?: cur.inpaintEngineLabel,
                            errorBannerMessage = sync.errorMessage ?: cur.errorBannerMessage,
                            statusBannerMessage = sync.statusMessage ?: cur.statusBannerMessage
                        )
                    }
                }
            }
        }
        viewModelScope.launch {
            BackgroundProcessingCoordinator.swapSync.collect { sync ->
                if (sync.isRunning || sync.result != null || sync.errorMessage != null) {
                    _uiState.update { cur ->
                        cur.copy(
                            isSwapping = sync.isRunning,
                            swapProgress = if (sync.isRunning && sync.stageTitle != null) {
                                (cur.swapProgress ?: SwapStageProgress(1, 5, sync.stageTitle, sync.detailMessage ?: "", 0.2f))
                            } else if (!sync.isRunning) {
                                null
                            } else {
                                cur.swapProgress
                            },
                            swapResult = sync.result ?: cur.swapResult,
                            lastAuditLogId = sync.auditLogId ?: cur.lastAuditLogId,
                            errorBannerMessage = sync.errorMessage ?: cur.errorBannerMessage,
                            statusBannerMessage = sync.statusMessage ?: cur.statusBannerMessage
                        )
                    }
                }
            }
        }
    }

    fun loadSourceAndTargetBitmaps(
        sourceBitmap: Bitmap,
        targetBitmap: Bitmap,
        sourceFace: DetectedFace? = null,
        targetFace: DetectedFace? = null
    ) {
        val sFaces = if (sourceFace != null) listOf(sourceFace) else modelService.detectFaces(sourceBitmap)
        val tFaces = if (targetFace != null) listOf(targetFace) else modelService.detectFaces(targetBitmap)
        _uiState.update {
            it.copy(
                sourceBitmap = sourceBitmap,
                sourceFaces = sFaces,
                selectedSourceFaceIndex = 0,
                isDetectingSource = false,
                targetBitmap = targetBitmap,
                targetFaces = tFaces,
                selectedTargetFaceIndex = 0,
                isDetectingTarget = false,
                swapResult = null,
                errorBannerMessage = null
            )
        }
    }

    private fun detectDeviceRamProfile(
        context: Context
    ): Triple<Boolean, ProcessingQualityLevel, Pair<Float, Long>> {
        return runCatching {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            val memInfo = ActivityManager.MemoryInfo()
            am?.getMemoryInfo(memInfo)
            val totalGb = if (memInfo.totalMem > 0L) {
                memInfo.totalMem.toFloat() / (1024f * 1024f * 1024f)
            } else {
                6.0f
            }
            val availMb = if (memInfo.availMem > 0L) {
                memInfo.availMem / (1024L * 1024L)
            } else {
                2048L
            }
            val isLow = (am?.isLowRamDevice == true) || totalGb < 4.0f || availMb < 750L
            val autoQuality = when {
                isLow -> ProcessingQualityLevel.FAST
                totalGb >= 7.5f && availMb >= 1800L -> ProcessingQualityLevel.HIGH_QUALITY
                else -> ProcessingQualityLevel.BALANCED
            }
            Triple(isLow, autoQuality, totalGb to availMb)
        }.getOrDefault(Triple(false, ProcessingQualityLevel.BALANCED, 6.0f to 2048L))
    }

    fun selectTab(tab: AppTab) {
        _uiState.update {
            it.copy(
                currentTab = tab,
                studioSubPage = if (tab == AppTab.STUDIO) StudioSubPage.FRONT_HOME else it.studioSubPage
            )
        }
    }

    fun selectStudioSubPage(subPage: StudioSubPage) {
        _uiState.update {
            it.copy(
                currentTab = AppTab.STUDIO,
                studioSubPage = subPage
            )
        }
    }

    fun selectStudioMode(mode: StudioMode) {
        _uiState.update {
            it.copy(
                studioMode = mode,
                swapResult = null,
                headPreviewState = null,
                errorBannerMessage = null,
                statusBannerMessage = null
            )
        }
    }

    fun clearMessages() {
        _uiState.update { it.copy(statusBannerMessage = null, errorBannerMessage = null) }
    }

    fun setConsentConfirmed(confirmed: Boolean) {
        _uiState.update { it.copy(consentConfirmed = confirmed, errorBannerMessage = null) }
    }

    fun setEnableColorTransfer(enabled: Boolean) {
        _uiState.update { it.copy(enableColorTransfer = enabled) }
    }

    fun setSkinToneMode(mode: SkinToneSourceMode) {
        _uiState.update { it.copy(skinToneMode = mode, enableColorTransfer = true) }
    }

    fun setFaceReactionMode(mode: FaceReactionSourceMode) {
        _uiState.update { it.copy(faceReactionMode = mode) }
    }

    fun setQualityLevel(level: ProcessingQualityLevel) {
        _uiState.update {
            it.copy(
                qualityLevel = level,
                lowMemoryMode = (level == ProcessingQualityLevel.FAST),
                statusBannerMessage = "Quality set to ${level.title} (Max ${level.maxDecodeDimensionPx}px)"
            )
        }
    }

    fun setOutputResolution(option: OutputResolutionOption) {
        _uiState.update { it.copy(outputResolution = option) }
    }

    fun setBlendStrength(strength: Float) {
        _uiState.update { it.copy(blendStrength = strength.coerceIn(0.20f, 1.0f)) }
    }

    fun setEnhancementStrength(strength: Float) {
        _uiState.update { it.copy(enhancementStrength = strength.coerceIn(0f, 1.0f)) }
    }

    fun setEnableOcclusionProtection(enabled: Boolean) {
        _uiState.update { it.copy(enableOcclusionProtection = enabled) }
    }

    fun setNegativeArtifactSuppression(enabled: Boolean) {
        _uiState.update { it.copy(enableNegativeArtifactSuppression = enabled) }
    }

    fun setLcmInpainting(enabled: Boolean) {
        _uiState.update { it.copy(enableLcmInpainting = enabled) }
    }

    fun setFaceAgingFactor(factor: Float) {
        _uiState.update { it.copy(faceAgingFactor = factor.coerceIn(-1.0f, 1.0f)) }
    }

    fun setPortraitBlurStrength(strength: Float) {
        _uiState.update { it.copy(portraitBlurStrength = strength.coerceIn(0f, 1.0f)) }
    }

    fun setFaceOffsetX(offsetX: Float) {
        _uiState.update { it.copy(faceOffsetX = offsetX.coerceIn(-32f, 32f)) }
    }

    fun setFaceOffsetY(offsetY: Float) {
        _uiState.update { it.copy(faceOffsetY = offsetY.coerceIn(-32f, 32f)) }
    }

    fun setFaceScaleAdjust(scale: Float) {
        _uiState.update { it.copy(faceScaleAdjust = scale.coerceIn(0.85f, 1.15f)) }
    }

    fun resetPositionAdjustments() {
        _uiState.update {
            it.copy(
                faceOffsetX = 0f,
                faceOffsetY = 0f,
                faceScaleAdjust = 1.0f,
                blendStrength = 1.0f,
                enhancementStrength = 0.85f,
                portraitBlurStrength = 0.0f
            )
        }
    }

    fun setEnableProvenanceWatermark(enabled: Boolean) {
        _uiState.update { it.copy(enableProvenanceWatermark = enabled) }
    }

    fun setAllowTwoModelFallback(allowed: Boolean) {
        _uiState.update { it.copy(allowTwoModelFallbackForTesting = allowed) }
    }

    fun setPreferHardwareAcceleration(enabled: Boolean) {
        _uiState.update { it.copy(preferHardwareAcceleration = enabled) }
    }

    fun setLowMemoryMode(enabled: Boolean) {
        _uiState.update {
            it.copy(
                lowMemoryMode = enabled,
                qualityLevel = if (enabled) ProcessingQualityLevel.FAST else it.qualityLevel
            )
        }
    }

    fun setReplaceAllTargetFaces(replaceAll: Boolean) {
        _uiState.update { it.copy(replaceAllTargetFaces = replaceAll) }
    }

    fun toggleComparisonMode(showOriginal: Boolean) {
        _uiState.update { it.copy(showOriginalInComparison = showOriginal) }
    }

    fun selectSourceFace(index: Int) {
        _uiState.update { state ->
            if (index in state.sourceFaces.indices) {
                state.copy(selectedSourceFaceIndex = index, headPreviewState = null)
            } else {
                state
            }
        }
    }

    fun selectTargetFace(index: Int) {
        _uiState.update { state ->
            if (index in state.targetFaces.indices) {
                state.copy(
                    selectedTargetFaceIndex = index,
                    replaceAllTargetFaces = false,
                    headPreviewState = null
                )
            } else {
                state
            }
        }
    }

    fun refreshModelInspections() {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isInspectingModels = true,
                    memoryServiceState = it.memoryServiceState.copy(isLoading = true)
                )
            }
            val (inspections, memState, sdPackTypes, lamaReady) = withContext(Dispatchers.IO) {
                val ctx = getApplication<Application>()
                val insp = OnnxProtobufInspector.inspectAllModels(ctx, ortEnv)
                val mem = modelService.loadModelsIntoMemory(
                    preferHardwareAccel = _uiState.value.preferHardwareAcceleration,
                    lowMemoryMode = _uiState.value.lowMemoryMode
                )
                val packs = SdInpaintPackManager.discoverInstalledPacks(ctx).map { it.packType }
                val lamaFile = OnnxProtobufInspector.resolveModelFile(ctx, ModelSlot.INPAINTING)
                val hasLama = lamaFile.exists() && lamaFile.length() > 1024L
                listOf(insp, mem, packs, hasLama)
            }.let { list ->
                @Suppress("UNCHECKED_CAST")
                Quadruple(
                    list[0] as List<OnnxModelInspection>,
                    list[1] as OnnxMemoryServiceState,
                    list[2] as List<SdInpaintPackType>,
                    list[3] as Boolean
                )
            }
            val lamaFile = OnnxProtobufInspector.resolveModelFile(getApplication(), ModelSlot.INPAINTING)
            val resolvedOption = if (_uiState.value.inpaintModelOption == InpaintModelOption.AUTO) {
                SdInpaintPackManager.determineDefaultModelOption(getApplication(), if (lamaReady) lamaFile else null)
            } else {
                _uiState.value.inpaintModelOption
            }
            val label = SdInpaintPackManager.formatDropdownLabel(
                getApplication(),
                resolvedOption,
                if (lamaReady) lamaFile else null
            )
            _uiState.update {
                it.copy(
                    modelInspections = inspections,
                    memoryServiceState = memState,
                    installedSdInpaintPacks = sdPackTypes,
                    isLamaInpaintReady = lamaReady,
                    inpaintModelOption = resolvedOption,
                    inpaintModelLabel = label,
                    isInspectingModels = false
                )
            }
            val detReady = inspections.any { it.slot == ModelSlot.DETECTOR && it.isValidOnnx }
            if (detReady) {
                _uiState.value.sourceBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = true) }
                _uiState.value.targetBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = false) }
            }
        }
    }

    fun preloadModelsIntoMemory() {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    memoryServiceState = it.memoryServiceState.copy(isLoading = true),
                    statusBannerMessage = "Loading w600k_r50.onnx, gfpgan_1.4.onnx, and segformer_B5_ce.onnx into ONNX Runtime memory...",
                    errorBannerMessage = null
                )
            }
            val memState = withContext(Dispatchers.IO) {
                modelService.loadModelsIntoMemory(
                    preferHardwareAccel = _uiState.value.preferHardwareAcceleration,
                    lowMemoryMode = _uiState.value.lowMemoryMode
                )
            }
            _uiState.update {
                it.copy(
                    memoryServiceState = memState,
                    statusBannerMessage = memState.summaryMessage
                )
            }
        }
    }

    fun releaseModelsFromMemory() {
        viewModelScope.launch {
            val memState = withContext(Dispatchers.IO) {
                modelService.releaseAllSessions()
                modelService.getMemoryState()
            }
            _uiState.update {
                it.copy(
                    memoryServiceState = memState,
                    statusBannerMessage = "Released in-memory ONNX Runtime sessions. Models will reload on-demand."
                )
            }
        }
    }

    fun importOnnxModel(uri: Uri, slot: ModelSlot) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isInspectingModels = true,
                    statusBannerMessage = "Importing ${slot.canonicalFileName} into ${slot.categoryTitle}...",
                    errorBannerMessage = null
                )
            }
            val result = withContext(Dispatchers.IO) {
                OnnxProtobufInspector.importModelFromUri(getApplication(), uri, slot)
            }
            result.fold(
                onSuccess = { file ->
                    val (inspections, memState) = withContext(Dispatchers.IO) {
                        val insp = OnnxProtobufInspector.inspectAllModels(getApplication(), ortEnv)
                        modelService.loadModelSlotIntoMemory(
                            slot = slot,
                            preferHardwareAccel = _uiState.value.preferHardwareAcceleration
                        )
                        insp to modelService.getMemoryState()
                    }
                    val importedInspection = inspections.firstOrNull { it.slot == slot }
                    val valid = importedInspection?.isValidOnnx == true
                    _uiState.update {
                        it.copy(
                            modelInspections = inspections,
                            memoryServiceState = memState,
                            isInspectingModels = false,
                            statusBannerMessage = if (valid) {
                                "Installed & loaded ${slot.canonicalFileName} (${formatBytes(file.length())}) into memory"
                            } else {
                                null
                            },
                            errorBannerMessage = if (!valid) {
                                importedInspection?.verificationNote
                                    ?: "Selected file is not a valid ONNX model."
                            } else {
                                null
                            }
                        )
                    }
                    if (slot == ModelSlot.DETECTOR && valid) {
                        _uiState.value.sourceBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = true) }
                        _uiState.value.targetBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = false) }
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isInspectingModels = false,
                            statusBannerMessage = null,
                            errorBannerMessage = "Model import failed: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun importAllModelsFromFolder(treeUri: Uri) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isInspectingModels = true,
                    memoryServiceState = it.memoryServiceState.copy(isLoading = true),
                    statusBannerMessage = "Scanning folder & auto-importing .onnx models...",
                    errorBannerMessage = null
                )
            }
            val result = withContext(Dispatchers.IO) {
                OnnxProtobufInspector.importAllModelsFromTreeUri(
                    context = getApplication(),
                    treeUri = treeUri,
                    onlyMissing = false,
                    onProgress = { progressMsg ->
                        _uiState.update { s ->
                            s.copy(statusBannerMessage = progressMsg)
                        }
                    }
                )
            }
            result.fold(
                onSuccess = { importedSlots ->
                    val ctx = getApplication<Application>()
                    val (inspections, memState, sdPackTypes, lamaFile) = withContext(Dispatchers.IO) {
                        val insp = OnnxProtobufInspector.inspectAllModels(ctx, ortEnv)
                        val mem = modelService.loadModelsIntoMemory(
                            preferHardwareAccel = _uiState.value.preferHardwareAcceleration,
                            lowMemoryMode = _uiState.value.lowMemoryMode
                        )
                        val packs = SdInpaintPackManager.discoverInstalledPacks(ctx).map { it.packType }
                        val lama = OnnxProtobufInspector.resolveModelFile(ctx, ModelSlot.INPAINTING)
                        Quadruple(insp, mem, packs, lama)
                    }
                    val hasLama = lamaFile.exists() && lamaFile.length() > 1024L
                    val defaultOpt = SdInpaintPackManager.determineDefaultModelOption(ctx, if (hasLama) lamaFile else null)
                    val label = SdInpaintPackManager.formatDropdownLabel(ctx, defaultOpt, if (hasLama) lamaFile else null)

                    val importedNames = buildList {
                        importedSlots.forEach { s ->
                            add(OnnxProtobufInspector.resolveModelFile(ctx, s).name)
                        }
                        sdPackTypes.forEach { packType ->
                            add(packType.displayLabel)
                        }
                    }

                    _uiState.update {
                        it.copy(
                            modelInspections = inspections,
                            memoryServiceState = memState,
                            installedSdInpaintPacks = sdPackTypes,
                            isLamaInpaintReady = hasLama,
                            inpaintModelOption = defaultOpt,
                            inpaintModelLabel = label,
                            isInspectingModels = false,
                            statusBannerMessage = if (importedNames.isNotEmpty()) {
                                "Auto-imported & verified ${importedNames.size} model(s): ${importedNames.joinToString()}."
                            } else {
                                null
                            },
                            errorBannerMessage = if (importedNames.isEmpty()) {
                                "No matching .onnx models or SD 1.5 Inpaint folders (fp16 / int8) found in selected folder."
                            } else {
                                null
                            }
                        )
                    }
                    if (ModelSlot.DETECTOR in importedSlots) {
                        _uiState.value.sourceBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = true) }
                        _uiState.value.targetBitmap?.let { bmp -> detectFacesForBitmap(bmp, isSource = false) }
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isInspectingModels = false,
                            memoryServiceState = it.memoryServiceState.copy(isLoading = false),
                            statusBannerMessage = null,
                            errorBannerMessage = "Folder scan failed: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun onSourceImageSelected(uri: Uri) {
        viewModelScope.launch {
            val maxDim = _uiState.value.qualityLevel.maxDecodeDimensionPx
            _uiState.update {
                it.copy(
                    isDetectingSource = true,
                    errorBannerMessage = null,
                    statusBannerMessage = null,
                    headPreviewState = null
                )
            }
            val decoded = withContext(Dispatchers.IO) {
                ImageGalleryHelper.decodeUriToBitmap(getApplication(), uri, maxDim)
            }
            decoded.fold(
                onSuccess = { bitmap ->
                    _uiState.update {
                        it.copy(
                            sourceBitmap = bitmap,
                            selectedSourceFaceIndex = 0,
                            swapResult = null
                        )
                    }
                    detectFacesForBitmap(bitmap, isSource = true)
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isDetectingSource = false,
                            errorBannerMessage = "Unsupported or unreadable Source image: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun onTargetImageSelected(uri: Uri) {
        viewModelScope.launch {
            val maxDim = _uiState.value.qualityLevel.maxDecodeDimensionPx
            _uiState.update {
                it.copy(
                    isDetectingTarget = true,
                    errorBannerMessage = null,
                    statusBannerMessage = null,
                    headPreviewState = null
                )
            }
            val decoded = withContext(Dispatchers.IO) {
                ImageGalleryHelper.decodeUriToBitmap(getApplication(), uri, maxDim)
            }
            decoded.fold(
                onSuccess = { bitmap ->
                    _uiState.update {
                        it.copy(
                            targetBitmap = bitmap,
                            selectedTargetFaceIndex = 0,
                            swapResult = null
                        )
                    }
                    detectFacesForBitmap(bitmap, isSource = false)
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isDetectingTarget = false,
                            errorBannerMessage = "Unsupported or unreadable Target image: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    /**
     * Explicit `[ Detect Head ]` action for Mode 2 (Head Replacement): re-runs head/face/landmark
     * and cranial volume detection on both loaded Source and Target photos.
     */
    fun detectHeadsOnLoadedPhotos() {
        val src = _uiState.value.sourceBitmap
        val tgt = _uiState.value.targetBitmap
        if (src == null && tgt == null) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please select a Source Head photo and/or Target Photo first.")
            }
            return
        }
        src?.let { detectFacesForBitmap(it, isSource = true) }
        tgt?.let { detectFacesForBitmap(it, isSource = false) }
        _uiState.update {
            it.copy(
                statusBannerMessage = "Detected head pose, 5-point landmarks, and cranial/hair/neck bounds."
            )
        }
    }

    /**
     * Explicit `[ Preview ]` action for Mode 2 (Head Replacement): runs Stage 2 (Cranial Alignment)
     * and Stage 4 (Head/Hair/Neck Segmentation) to show the user a live preview of the extracted
     * source head + hair + neck mask alongside the aligned target head region before full replacement.
     */
    fun generateHeadReplacementPreview() {
        val state = _uiState.value
        val srcBmp = state.sourceBitmap
        val tgtBmp = state.targetBitmap
        if (srcBmp == null || state.sourceFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "Select a Source Head photo with a detectable face/head first.")
            }
            return
        }
        if (tgtBmp == null || state.targetFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "Select a Target photo with a detectable face/head first.")
            }
            return
        }

        val srcFace = state.sourceFaces.getOrElse(state.selectedSourceFaceIndex) { state.sourceFaces.first() }
        val tgtFace = state.targetFaces.getOrElse(state.selectedTargetFaceIndex) { state.targetFaces.first() }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isGeneratingHeadPreview = true,
                    errorBannerMessage = null,
                    statusBannerMessage = "Generating Head/Hair/Neck Segmentation & Alignment Preview..."
                )
            }
            runCatching {
                withContext(Dispatchers.Default) {
                    val previewSize = 256
                    val srcPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
                        srcFace,
                        srcBmp.width,
                        srcBmp.height
                    )
                    val tgtPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
                        tgtFace,
                        tgtBmp.width,
                        tgtBmp.height
                    )
                    val srcMat = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(
                        srcFace.landmarks5,
                        previewSize,
                        srcPose.yawRatio
                    )
                    val tgtMat = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(
                        tgtFace.landmarks5,
                        previewSize,
                        srcPose.yawRatio
                    )
                    val alignedSrc = FaceAlignment.warpAffineCrop(srcBmp, srcMat, previewSize)
                    val alignedTgt = FaceAlignment.warpAffineCrop(tgtBmp, tgtMat, previewSize)

                    val masks = modelService.segmentHeadAndHair(
                        alignedHeadCrop = alignedSrc,
                        preferHardwareAccel = state.preferHardwareAcceleration
                    )
                    val segPreviewBmp = HeadSegmentationAndInpainting.createHeadSegmentationPreviewBitmap(
                        alignedSrc,
                        masks
                    )
                    alignedSrc.recycle()

                    HeadAlignmentPreviewState(
                        segmentedSourceHeadBitmap = segPreviewBmp,
                        alignedTargetHeadBitmap = alignedTgt,
                        sourceRollDeg = srcPose.rollDegrees,
                        targetRollDeg = tgtPose.rollDegrees,
                        segmentationLabel = masks.segmentationSourceLabel
                    )
                }
            }.fold(
                onSuccess = { preview ->
                    _uiState.update {
                        it.copy(
                            isGeneratingHeadPreview = false,
                            headPreviewState = preview,
                            memoryServiceState = modelService.getMemoryState(),
                            statusBannerMessage = "Head/Hair/Neck preview ready (${preview.segmentationLabel})."
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isGeneratingHeadPreview = false,
                            errorBannerMessage = "Preview generation failed: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    private fun detectFacesForBitmap(bitmap: Bitmap, isSource: Boolean) {
        viewModelScope.launch {
            if (isSource) {
                _uiState.update { it.copy(isDetectingSource = true) }
            } else {
                _uiState.update { it.copy(isDetectingTarget = true) }
            }

            val detResult = withContext(Dispatchers.Default) {
                runCatching {
                    modelService.detectFaces(bitmap)
                }
            }

            detResult.fold(
                onSuccess = { faces ->
                    val roleLabel = if (isSource) "Source" else "Target"
                    val minSide = kotlin.math.min(bitmap.width, bitmap.height)
                    val lowResMsg = if (isSource && minSide < 180) {
                        "Low-resolution Source photo (${bitmap.width}×${bitmap.height}). Higher resolution portrait recommended for sharper identity."
                    } else if (!isSource && minSide < 240) {
                        "Low-resolution Target photo (${bitmap.width}×${bitmap.height}). Output sharpness may be limited."
                    } else null

                    val poseOrOcclusionWarning = if (faces.isNotEmpty()) {
                        val primary = faces.first()
                        val pose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
                            primary,
                            bitmap.width,
                            bitmap.height
                        )
                        when {
                            kotlin.math.abs(pose.rollDegrees) > 38f || kotlin.math.abs(pose.yawRatio) > 0.62f ->
                                "Extreme head angle detected in $roleLabel photo (roll ${"%.0f".format(pose.rollDegrees)}°). Alignment adjusted automatically."
                            primary.score < 0.58f ->
                                "Face in $roleLabel photo appears partially hidden or low-contrast (confidence ${"%.0f".format(primary.score * 100)}%)."
                            faces.size > 1 ->
                                "Detected ${faces.size} faces in $roleLabel photo (Face 1..Face ${faces.size}). Tap a face card below to choose who to swap."
                            else -> null
                        }
                    } else null

                    val warning = if (faces.isEmpty()) {
                        "No face detected in the $roleLabel photo. Please select a clearer, well-lit portrait."
                    } else null

                    val infoMsg = lowResMsg ?: poseOrOcclusionWarning

                    _uiState.update { state ->
                        if (isSource) {
                            state.copy(
                                sourceFaces = faces,
                                selectedSourceFaceIndex = 0,
                                isDetectingSource = false,
                                memoryServiceState = modelService.getMemoryState(),
                                errorBannerMessage = warning,
                                statusBannerMessage = infoMsg ?: state.statusBannerMessage
                            )
                        } else {
                            state.copy(
                                targetFaces = faces,
                                selectedTargetFaceIndex = 0,
                                replaceAllTargetFaces = false,
                                isDetectingTarget = false,
                                memoryServiceState = modelService.getMemoryState(),
                                errorBannerMessage = warning,
                                statusBannerMessage = infoMsg ?: state.statusBannerMessage
                            )
                        }
                    }
                },
                onFailure = { err ->
                    _uiState.update { state ->
                        if (isSource) {
                            state.copy(
                                sourceFaces = emptyList(),
                                isDetectingSource = false,
                                errorBannerMessage = "Detection failed on source photo: ${err.message}"
                            )
                        } else {
                            state.copy(
                                targetFaces = emptyList(),
                                isDetectingTarget = false,
                                errorBannerMessage = "Detection failed on target photo: ${err.message}"
                            )
                        }
                    }
                }
            )
        }
    }

    fun runActiveModePipeline() {
        val state = _uiState.value
        val srcBitmap = state.sourceBitmap
        val tgtBitmap = state.targetBitmap

        if (srcBitmap == null) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please select a Source Photo (Donor Face) first.")
            }
            return
        }
        if (state.sourceFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "No face detected in source photo. Please choose a clearer portrait.")
            }
            return
        }
        if (tgtBitmap == null) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please select a Target Photo (Scene Photo) first.")
            }
            return
        }
        if (state.targetFaces.isEmpty()) {
            _uiState.update {
                it.copy(errorBannerMessage = "No face detected in target photo. Please choose a clearer portrait.")
            }
            return
        }
        if (!state.consentConfirmed) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please confirm ethical consent (tick the checkbox above the swap button).")
            }
            return
        }

        val selectedSourceFace = state.sourceFaces.getOrElse(state.selectedSourceFaceIndex) {
            state.sourceFaces.first()
        }
        val targetsToSwap = if (state.replaceAllTargetFaces) {
            state.targetFaces
        } else {
            listOf(
                state.targetFaces.getOrElse(state.selectedTargetFaceIndex) {
                    state.targetFaces.first()
                }
            )
        }

        if (state.isSwapping) return
        val appCtx = getApplication<Application>()
        ProcessingForegroundService.startOrUpdate(
            context = appCtx,
            title = "${state.studioMode.title} Running",
            detail = "Preparing offline ONNX Runtime...",
            progressPercent = 5
        )
        BackgroundProcessingCoordinator.updateSwap {
            it.copy(isRunning = true, stageTitle = "Stage 1: Detection", detailMessage = "Preparing ONNX...", errorMessage = null, statusMessage = null)
        }
        BackgroundProcessingCoordinator.scope.launch {
            _uiState.update {
                it.copy(
                    isSwapping = true,
                    errorBannerMessage = null,
                    statusBannerMessage = null,
                    showOriginalInComparison = false,
                    swapProgress = SwapStageProgress(
                        stepIndex = 1,
                        totalSteps = if (state.studioMode == StudioMode.HEAD_REPLACEMENT) 6 else 5,
                        stageTitle = "Stage 1: Detection",
                        detailMessage = "Preparing in-memory ONNX Runtime sessions...",
                        progressFraction = 0.05f
                    )
                )
            }

            try {
                val execResult = withContext(Dispatchers.Default) {
                    when (state.studioMode) {
                        StudioMode.FACE_SWAP -> modelService.swapFaces(
                            sourceBitmap = srcBitmap,
                            sourceFace = selectedSourceFace,
                            targetBitmap = tgtBitmap,
                            targetFacesToReplace = targetsToSwap,
                            enableColorTransfer = state.enableColorTransfer,
                            enableProvenanceWatermark = state.enableProvenanceWatermark,
                            allowTwoModelFallback = state.allowTwoModelFallbackForTesting,
                            preferHardwareAccel = state.preferHardwareAcceleration,
                            skinToneMode = state.skinToneMode,
                            faceReactionMode = state.faceReactionMode,
                            enableOcclusionProtection = state.enableOcclusionProtection,
                            portraitBlurStrength = state.portraitBlurStrength,
                            blendStrength = state.blendStrength,
                            enhancementStrength = state.enhancementStrength,
                            offsetX = state.faceOffsetX,
                            offsetY = state.faceOffsetY,
                            scaleAdjust = state.faceScaleAdjust,
                            identityPreservationMode = false,
                            onProgress = { progress ->
                                _uiState.update { s -> s.copy(swapProgress = progress) }
                                ProcessingForegroundService.startOrUpdate(
                                    context = appCtx,
                                    title = "${state.studioMode.title} • ${progress.stageTitle}",
                                    detail = progress.detailMessage,
                                    progressPercent = (progress.progressFraction * 100f).toInt().coerceIn(5, 99)
                                )
                            }
                        )

                        StudioMode.IDENTITY_PRESERVED -> modelService.swapFaces(
                            sourceBitmap = srcBitmap,
                            sourceFace = selectedSourceFace,
                            targetBitmap = tgtBitmap,
                            targetFacesToReplace = targetsToSwap,
                            enableColorTransfer = state.enableColorTransfer,
                            enableProvenanceWatermark = state.enableProvenanceWatermark,
                            allowTwoModelFallback = state.allowTwoModelFallbackForTesting,
                            preferHardwareAccel = state.preferHardwareAcceleration,
                            skinToneMode = state.skinToneMode,
                            faceReactionMode = state.faceReactionMode,
                            enableOcclusionProtection = state.enableOcclusionProtection,
                            portraitBlurStrength = state.portraitBlurStrength,
                            blendStrength = state.blendStrength,
                            enhancementStrength = 0.0f, // No synthetic GAN enhancement
                            offsetX = state.faceOffsetX,
                            offsetY = state.faceOffsetY,
                            scaleAdjust = state.faceScaleAdjust,
                            identityPreservationMode = true,
                            onProgress = { progress ->
                                _uiState.update { s -> s.copy(swapProgress = progress) }
                                ProcessingForegroundService.startOrUpdate(
                                    context = appCtx,
                                    title = "${state.studioMode.title} • ${progress.stageTitle}",
                                    detail = progress.detailMessage,
                                    progressPercent = (progress.progressFraction * 100f).toInt().coerceIn(5, 99)
                                )
                            }
                        )

                        StudioMode.HEAD_REPLACEMENT -> modelService.replaceFullHead(
                            sourceBitmap = srcBitmap,
                            sourceFace = selectedSourceFace,
                            targetBitmap = tgtBitmap,
                            targetFacesToReplace = targetsToSwap,
                            enableColorTransfer = state.enableColorTransfer,
                            enableProvenanceWatermark = state.enableProvenanceWatermark,
                            allowTwoModelFallback = state.allowTwoModelFallbackForTesting,
                            lowMemoryMode = state.lowMemoryMode || (state.qualityLevel == ProcessingQualityLevel.FAST),
                            preferHardwareAccel = state.preferHardwareAcceleration,
                            skinToneMode = state.skinToneMode,
                            faceReactionMode = state.faceReactionMode,
                            portraitBlurStrength = state.portraitBlurStrength,
                            blendStrength = state.blendStrength,
                            enhancementStrength = state.enhancementStrength,
                            offsetX = state.faceOffsetX,
                            offsetY = state.faceOffsetY,
                            scaleAdjust = state.faceScaleAdjust,
                            onProgress = { progress ->
                                _uiState.update { s -> s.copy(swapProgress = progress) }
                                ProcessingForegroundService.startOrUpdate(
                                    context = appCtx,
                                    title = "${state.studioMode.title} • ${progress.stageTitle}",
                                    detail = progress.detailMessage,
                                    progressPercent = (progress.progressFraction * 100f).toInt().coerceIn(5, 99)
                                )
                            }
                        )
                    }
                }

                val historyImgPath = withContext(Dispatchers.IO) {
                    ImageGalleryHelper.saveBitmapToHistoryCache(
                        getApplication(),
                        execResult.outputBitmap
                    )
                }

                val logId = auditRepository.insert(
                    SwapAuditLog(
                        sourceResolution = "${srcBitmap.width}x${srcBitmap.height}",
                        targetResolution = "${tgtBitmap.width}x${tgtBitmap.height}",
                        swappedFacesCount = execResult.swappedFacesCount,
                        pipelineSummary = "[${state.studioMode.title} • ${state.qualityLevel.title}] ${execResult.pipelineSummary}",
                        detectionMs = execResult.detectionMs,
                        embeddingMs = execResult.embeddingMs,
                        inswapperMs = execResult.inswapperMs,
                        blendingMs = execResult.blendingMs,
                        totalMs = execResult.totalMs,
                        colorTransferEnabled = state.enableColorTransfer,
                        watermarkEnabled = state.enableProvenanceWatermark,
                        resultImagePath = historyImgPath
                    )
                )

                val doneMsg = "${state.studioMode.title} completed in ${execResult.totalMs} ms (${execResult.swappedFacesCount} target(s) replaced)."
                BackgroundProcessingCoordinator.updateSwap {
                    it.copy(isRunning = false, result = execResult, auditLogId = logId, statusMessage = doneMsg, errorMessage = null)
                }
                _uiState.update {
                    it.copy(
                        isSwapping = false,
                        swapProgress = null,
                        swapResult = execResult,
                        lastAuditLogId = logId,
                        memoryServiceState = modelService.getMemoryState(),
                        statusBannerMessage = doneMsg
                    )
                }
                ProcessingForegroundService.finishAndNotify(
                    context = appCtx,
                    showCompletionNotification = true,
                    completedTitle = "${state.studioMode.title} Ready",
                    completedBody = doneMsg
                )
            } catch (oom: OutOfMemoryError) {
                modelService.releaseAllSessions()
                System.gc()
                val errMsg = "Insufficient RAM detected. Switched automatically to FAST / Low-Memory Tiled Mode — please tap Process again."
                BackgroundProcessingCoordinator.updateSwap {
                    it.copy(isRunning = false, errorMessage = errMsg)
                }
                _uiState.update {
                    it.copy(
                        isSwapping = false,
                        swapProgress = null,
                        lowMemoryMode = true,
                        qualityLevel = ProcessingQualityLevel.FAST,
                        memoryServiceState = modelService.getMemoryState(),
                        errorBannerMessage = errMsg
                    )
                }
                ProcessingForegroundService.finishAndNotify(
                    context = appCtx,
                    showCompletionNotification = true,
                    completedTitle = "${state.studioMode.title} Paused (Low RAM)",
                    completedBody = errMsg
                )
            } catch (t: Throwable) {
                val errMsg = t.message ?: "Processing failure: ${t.javaClass.simpleName}"
                BackgroundProcessingCoordinator.updateSwap {
                    it.copy(isRunning = false, errorMessage = errMsg)
                }
                _uiState.update {
                    it.copy(
                        isSwapping = false,
                        swapProgress = null,
                        errorBannerMessage = errMsg
                    )
                }
                ProcessingForegroundService.finishAndNotify(
                    context = appCtx,
                    showCompletionNotification = true,
                    completedTitle = "${state.studioMode.title} Failed",
                    completedBody = errMsg
                )
            }
        }
    }

    fun runVisualValidationSuite() {
        val state = _uiState.value
        if (state.isRunningVisualValidation) return
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isRunningVisualValidation = true,
                    errorBannerMessage = null,
                    statusBannerMessage = "Running Real Visual Validation across A (inswapper_128), B (hyperswap_1a_256), C (hyperswap_1b_256), D (hyperswap_1c_256) & 3 Upper-Lip Stages..."
                )
            }
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val (srcPair, tgtPair) = if (
                        state.sourceBitmap != null && state.sourceFaces.isNotEmpty() &&
                        state.targetBitmap != null && state.targetFaces.isNotEmpty()
                    ) {
                        val sFace = state.sourceFaces.getOrElse(state.selectedSourceFaceIndex) { state.sourceFaces.first() }
                        val tFace = state.targetFaces.getOrElse(state.selectedTargetFaceIndex) { state.targetFaces.first() }
                        (state.sourceBitmap to sFace) to (state.targetBitmap to tFace)
                    } else {
                        VisualValidationBenchmark.createRealisticSourceAndTargetPortraits()
                    }
                    VisualValidationBenchmark.runCompleteVisualValidation(
                        context = getApplication(),
                        ortEnv = ortEnv,
                        sourceBitmap = srcPair.first,
                        sourceFace = srcPair.second,
                        targetBitmap = tgtPair.first,
                        targetFace = tgtPair.second
                    )
                }
            }
            result.fold(
                onSuccess = { suite ->
                    _uiState.update {
                        it.copy(
                            visualValidationSuite = suite,
                            isRunningVisualValidation = false,
                            statusBannerMessage = "Visual Validation Complete: Winning Model = ${suite.winningCandidate.code} (${suite.winningCandidate.canonicalFileName}) • Zero Upper-Lip Moustache Verified."
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isRunningVisualValidation = false,
                            errorBannerMessage = "Visual Validation error: ${err.message}"
                        )
                    }
                }
            )
        }
    }

    fun saveVisualValidationSheetToGallery(save3StageSheet: Boolean = false) {
        val suite = _uiState.value.visualValidationSuite ?: return
        val bmp = if (save3StageSheet) suite.stage3PanelSheet else suite.comparison7PanelSheet
        val label = if (save3StageSheet) "3-Stage Upper-Lip Sheet" else "7-Panel A/B/C/D Comparison Sheet"
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) {
                ImageGalleryHelper.saveBitmapToGallery(
                    context = getApplication(),
                    bitmap = bmp,
                    isHd = true,
                    targetMaxDimension = 0
                )
            }
            saved.fold(
                onSuccess = { path ->
                    _uiState.update {
                        it.copy(
                            statusBannerMessage = "Saved $label to Gallery: $path",
                            errorBannerMessage = null
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(errorBannerMessage = "Could not save $label: ${err.message}")
                    }
                }
            )
        }
    }

    fun saveResultToGallery(isHd: Boolean = true) {
        val state = _uiState.value
        val result = state.swapResult ?: return
        val logId = state.lastAuditLogId
        val targetDim = if (isHd) state.outputResolution.maxExportDimensionPx else 0
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) {
                ImageGalleryHelper.saveBitmapToGallery(
                    context = getApplication(),
                    bitmap = result.outputBitmap,
                    isHd = isHd,
                    targetMaxDimension = targetDim
                )
            }
            saved.fold(
                onSuccess = { path ->
                    if (logId != null) {
                        auditRepository.markSavedToGallery(logId, path)
                    }
                    _uiState.update {
                        it.copy(
                            statusBannerMessage = "Saved ${if (isHd) "HD PNG" else "Standard JPG"} to Gallery: $path",
                            errorBannerMessage = null
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(errorBannerMessage = "Could not save to Gallery: ${err.message}")
                    }
                }
            )
        }
    }

    fun shareResultImage() {
        val result = _uiState.value.swapResult ?: return
        viewModelScope.launch {
            val shared = withContext(Dispatchers.IO) {
                ImageGalleryHelper.shareBitmap(getApplication(), result.outputBitmap)
            }
            shared.fold(
                onSuccess = { msg ->
                    _uiState.update {
                        it.copy(statusBannerMessage = msg, errorBannerMessage = null)
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(errorBannerMessage = "Share failed: ${err.message}")
                    }
                }
            )
        }
    }

    fun rotateResultBitmap90() {
        val currentResult = _uiState.value.swapResult ?: return
        val rotated = ImageGalleryHelper.rotateBitmap90(currentResult.outputBitmap)
        _uiState.update {
            it.copy(
                swapResult = currentResult.copy(outputBitmap = rotated),
                statusBannerMessage = "Rotated output image 90° (${rotated.width}×${rotated.height})."
            )
        }
    }

    fun cropResultBitmapAspect(aspectW: Int, aspectH: Int) {
        val currentResult = _uiState.value.swapResult ?: return
        val cropped = ImageGalleryHelper.cropBitmapToRatio(currentResult.outputBitmap, aspectW, aspectH)
        _uiState.update {
            it.copy(
                swapResult = currentResult.copy(outputBitmap = cropped),
                statusBannerMessage = "Cropped output image to $aspectW:$aspectH (${cropped.width}×${cropped.height})."
            )
        }
    }

    fun onInpaintSourceImageSelected(uri: Uri) {
        viewModelScope.launch {
            val maxDim = _uiState.value.qualityLevel.maxDecodeDimensionPx
            _uiState.update {
                it.copy(errorBannerMessage = null, statusBannerMessage = null)
            }
            val decoded = withContext(Dispatchers.IO) {
                ImageGalleryHelper.decodeUriToBitmap(getApplication(), uri, maxDim)
            }
            decoded.fold(
                onSuccess = { bitmap ->
                    _uiState.update {
                        it.copy(
                            inpaintSourceBitmap = bitmap,
                            inpaintMaskBitmap = null,
                            inpaintDepthMap = null,
                            inpaintAccessoryLayers = null,
                            selectedAccessoryTargets = emptySet(),
                            segformerAutoSelectedForAccessory = false,
                            accessoryMaskNotice = null,
                            inpaintResultBitmap = null,
                            inpaintElapsedSec = null,
                            statusBannerMessage = "Loaded Inpaint Source (${bitmap.width}×${bitmap.height}) from folder."
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(errorBannerMessage = "Could not open image from folder: ${err.message}")
                    }
                }
            )
        }
    }

    fun setInpaintMaskBitmap(mask: Bitmap?) {
        _uiState.update {
            it.copy(
                inpaintMaskBitmap = mask,
                errorBannerMessage = null,
                statusBannerMessage = if (mask != null) "Mask applied. Tap Create to run Inpaint." else null
            )
        }
    }

    fun setInpaintPrompt(prompt: String) {
        _uiState.update { it.copy(inpaintPrompt = prompt) }
    }

    fun setInpaintFastMode(fastMode: Boolean) {
        _uiState.update { it.copy(inpaintFastMode = fastMode) }
    }

    fun setInpaintStrength(strength: Float) {
        _uiState.update { it.copy(inpaintStrength = strength.coerceIn(0.10f, 1.00f)) }
    }

    fun setInpaintCfgScale(cfgScale: Float) {
        _uiState.update { it.copy(inpaintCfgScale = cfgScale.coerceIn(1.0f, 15.0f)) }
    }

    fun setInpaintDepthGuardEnabled(enabled: Boolean) {
        _uiState.update { it.copy(inpaintDepthGuardEnabled = enabled) }
        if (enabled && _uiState.value.inpaintDepthMap == null && _uiState.value.inpaintSourceBitmap != null) {
            computeInpaintDepthMap()
        }
    }

    fun computeInpaintDepthMap() {
        val state = _uiState.value
        val srcBmp = state.inpaintSourceBitmap ?: return
        if (state.isComputingInpaintDepth) return
        if (!DepthEstimationEngine.hasInstalledDepthModel(getApplication())) {
            _uiState.update {
                it.copy(
                    errorBannerMessage = "Install Depth Anything V2 or sentis-MiDaS ONNX in Models tab (1-Tap Auto-Import) to use Auto Depth Mask."
                )
            }
            return
        }
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isComputingInpaintDepth = true,
                    errorBannerMessage = null,
                    statusBannerMessage = "Analyzing 3D depth map for Auto-Mask..."
                )
            }
            val depthRes = withContext(Dispatchers.Default) {
                modelService.releaseAllSessions()
                DepthEstimationEngine.estimateDepth(
                    context = getApplication(),
                    ortEnv = ortEnv,
                    sourceBitmap = srcBmp,
                    preferHardwareAccel = state.preferHardwareAcceleration
                )
            }
            _uiState.update {
                if (depthRes != null) {
                    it.copy(
                        isComputingInpaintDepth = false,
                        inpaintDepthMap = depthRes,
                        statusBannerMessage = "Depth map ready (${depthRes.modelUsedLabel}, ${depthRes.elapsedMs}ms)."
                    )
                } else {
                    it.copy(
                        isComputingInpaintDepth = false,
                        errorBannerMessage = "Could not run Depth model on this image. Check Depth ONNX file in Models tab."
                    )
                }
            }
        }
    }

    private fun resolveAutoModelAndSegformerForAccessories(
        cur: FaceSwapUiState,
        targets: Set<AccessoryTarget>
    ): Triple<InpaintModelOption, String, Boolean> {
        val needsHeadParserAndSd = AccessoryTarget.CAP in targets || AccessoryTarget.GLASSES in targets
        if (!needsHeadParserAndSd) {
            return Triple(cur.inpaintModelOption, cur.inpaintModelLabel, false)
        }
        val nextOption = when {
            SdInpaintPackType.SD15_FP16 in cur.installedSdInpaintPacks -> InpaintModelOption.SD15_FP16
            SdInpaintPackType.REALVIS_INT8 in cur.installedSdInpaintPacks -> InpaintModelOption.REALVIS_INT8
            else -> cur.inpaintModelOption
        }
        val ctx = getApplication<Application>()
        val lamaFile = OnnxProtobufInspector.resolveModelFile(ctx, ModelSlot.INPAINTING)
        val nextLabel = if (nextOption != cur.inpaintModelOption) {
            SdInpaintPackManager.formatDropdownLabel(ctx, nextOption, if (lamaFile.exists()) lamaFile else null)
        } else {
            cur.inpaintModelLabel
        }
        return Triple(nextOption, nextLabel, true)
    }

    private fun ensureSegformerLoadedIfNeeded() {
        val state = _uiState.value
        if (!state.isSegmentationOnnxReady || state.memoryServiceState.segformerLoaded) return
        viewModelScope.launch {
            val memState = withContext(Dispatchers.IO) {
                modelService.loadModelSlotIntoMemory(
                    slot = ModelSlot.SEGMENTATION,
                    preferHardwareAccel = _uiState.value.preferHardwareAcceleration
                )
                modelService.getMemoryState()
            }
            _uiState.update { it.copy(memoryServiceState = memState) }
        }
    }

    fun updateAccessorySelectionAndExpand(targets: Set<AccessoryTarget>, expandPx: Float) {
        val clamped = expandPx.coerceIn(0f, 18f)
        val cur = _uiState.value
        val (autoOpt, autoLabel, autoSegformer) = resolveAutoModelAndSegformerForAccessories(cur, targets)
        _uiState.update {
            it.copy(
                selectedAccessoryTargets = targets,
                inpaintAccessoryExpandPx = clamped,
                inpaintModelOption = autoOpt,
                inpaintModelLabel = autoLabel,
                segformerAutoSelectedForAccessory = autoSegformer
            )
        }
        if (autoSegformer) {
            ensureSegformerLoadedIfNeeded()
        }
    }

    fun setInpaintAccessoryExpandPx(expandPx: Float) {
        val clamped = expandPx.coerceIn(0f, 18f)
        val cur = _uiState.value
        val layers = cur.inpaintAccessoryLayers
        val srcBmp = cur.inpaintSourceBitmap
        val targets = cur.selectedAccessoryTargets
        val updatedMask = if (layers != null && srcBmp != null && targets.isNotEmpty()) {
            layers.renderMaskBitmap(srcBmp.width, srcBmp.height, targets, clamped)
        } else {
            cur.inpaintMaskBitmap
        }
        _uiState.update {
            it.copy(
                inpaintAccessoryExpandPx = clamped,
                inpaintMaskBitmap = updatedMask
            )
        }
    }

    fun applyMaskWithOptionalPrompt(mask: Bitmap?, suggestedPrompt: String?) {
        _uiState.update { cur ->
            val nextPrompt = if (!suggestedPrompt.isNullOrBlank()) suggestedPrompt else cur.inpaintPrompt
            cur.copy(
                inpaintMaskBitmap = mask,
                inpaintPrompt = nextPrompt,
                errorBannerMessage = null,
                statusBannerMessage = if (mask != null) {
                    if (!suggestedPrompt.isNullOrBlank()) {
                        "Auto-mask & removal prompt ready. Tap Create to inpaint."
                    } else {
                        "Mask applied. Tap Create to run Inpaint."
                    }
                } else {
                    null
                }
            )
        }
    }

    fun computeInpaintAccessoryLayers() {
        val state = _uiState.value
        val srcBmp = state.inpaintSourceBitmap ?: return
        if (state.isComputingAccessoryMask) return
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isComputingAccessoryMask = true,
                    accessoryMaskNotice = "Detecting Cap / Glasses / Jewels...",
                    errorBannerMessage = null
                )
            }
            val layers = withContext(Dispatchers.Default) {
                AccessoryMaskEngine.extractAccessoryLayers(
                    context = getApplication(),
                    ortEnv = ortEnv,
                    sourceBitmap = srcBmp,
                    depthMap = state.inpaintDepthMap,
                    preferHardwareAccel = state.preferHardwareAcceleration
                )
            }
            _uiState.update { cur ->
                val notice = if (layers.detectedFacesCount <= 0) {
                    "No face detected in photo. Draw mask manually or use Auto BG."
                } else {
                    "Ready (${layers.engineLabel})"
                }
                val newMask = if (layers.detectedFacesCount > 0 && cur.selectedAccessoryTargets.isNotEmpty()) {
                    layers.renderMaskBitmap(
                        srcBmp.width,
                        srcBmp.height,
                        cur.selectedAccessoryTargets,
                        cur.inpaintAccessoryExpandPx
                    )
                } else {
                    cur.inpaintMaskBitmap
                }
                cur.copy(
                    isComputingAccessoryMask = false,
                    inpaintAccessoryLayers = layers,
                    inpaintMaskBitmap = newMask,
                    accessoryMaskNotice = notice
                )
            }
        }
    }

    fun toggleQuickAccessoryRemoval(target: AccessoryTarget) {
        val cur = _uiState.value
        val srcBmp = cur.inpaintSourceBitmap
        if (srcBmp == null) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please select a Source Image from device folder first.")
            }
            return
        }
        val nextTargets = if (target in cur.selectedAccessoryTargets) {
            cur.selectedAccessoryTargets - target
        } else {
            cur.selectedAccessoryTargets + target
        }
        val autoPrompt = if (nextTargets.isNotEmpty()) {
            AccessoryMaskEngine.buildCombinedRemovalPrompt(nextTargets)
        } else {
            ""
        }
        val (autoOpt, autoLabel, autoSegformer) = resolveAutoModelAndSegformerForAccessories(cur, nextTargets)
        val cachedLayers = cur.inpaintAccessoryLayers
        if (cachedLayers != null) {
            val maskBmp = if (nextTargets.isNotEmpty() && cachedLayers.detectedFacesCount > 0) {
                cachedLayers.renderMaskBitmap(srcBmp.width, srcBmp.height, nextTargets, cur.inpaintAccessoryExpandPx)
            } else {
                null
            }
            _uiState.update {
                it.copy(
                    selectedAccessoryTargets = nextTargets,
                    inpaintMaskBitmap = maskBmp,
                    inpaintPrompt = if (nextTargets.isNotEmpty()) autoPrompt else it.inpaintPrompt,
                    inpaintModelOption = autoOpt,
                    inpaintModelLabel = autoLabel,
                    segformerAutoSelectedForAccessory = autoSegformer,
                    errorBannerMessage = if (nextTargets.isNotEmpty() && cachedLayers.detectedFacesCount <= 0) {
                        "No face detected in photo for 1-Tap accessory mask. Tap Draw Mask to paint manually."
                    } else null,
                    statusBannerMessage = if (maskBmp != null) {
                        "Auto-mask ready (${nextTargets.joinToString(" + ") { t -> t.shortLabel }} • ${autoLabel}). Tap Create to remove."
                    } else null
                )
            }
            if (autoSegformer) ensureSegformerLoadedIfNeeded()
            return
        }

        _uiState.update {
            it.copy(
                selectedAccessoryTargets = nextTargets,
                inpaintPrompt = if (nextTargets.isNotEmpty()) autoPrompt else it.inpaintPrompt,
                inpaintModelOption = autoOpt,
                inpaintModelLabel = autoLabel,
                segformerAutoSelectedForAccessory = autoSegformer,
                isComputingAccessoryMask = nextTargets.isNotEmpty(),
                errorBannerMessage = null,
                statusBannerMessage = if (nextTargets.isNotEmpty()) {
                    "Detecting ${nextTargets.joinToString(" + ") { it.shortLabel }} on Source photo..."
                } else null
            )
        }
        if (autoSegformer) ensureSegformerLoadedIfNeeded()
        if (nextTargets.isEmpty()) {
            _uiState.update { it.copy(inpaintMaskBitmap = null) }
            return
        }

        viewModelScope.launch {
            val layers = withContext(Dispatchers.Default) {
                AccessoryMaskEngine.extractAccessoryLayers(
                    context = getApplication(),
                    ortEnv = ortEnv,
                    sourceBitmap = srcBmp,
                    depthMap = _uiState.value.inpaintDepthMap,
                    preferHardwareAccel = _uiState.value.preferHardwareAcceleration
                )
            }
            _uiState.update { latest ->
                val activeTargets = latest.selectedAccessoryTargets
                val maskBmp = if (activeTargets.isNotEmpty() && layers.detectedFacesCount > 0) {
                    layers.renderMaskBitmap(srcBmp.width, srcBmp.height, activeTargets, latest.inpaintAccessoryExpandPx)
                } else {
                    null
                }
                latest.copy(
                    isComputingAccessoryMask = false,
                    inpaintAccessoryLayers = layers,
                    inpaintMaskBitmap = maskBmp,
                    accessoryMaskNotice = if (layers.detectedFacesCount <= 0) {
                        "No face detected in photo. Use Draw Mask."
                    } else {
                        "Ready (${layers.engineLabel})"
                    },
                    errorBannerMessage = if (activeTargets.isNotEmpty() && layers.detectedFacesCount <= 0) {
                        "No face detected in photo for 1-Tap accessory mask. Tap Draw Mask to paint manually."
                    } else null,
                    statusBannerMessage = if (maskBmp != null) {
                        "Auto-mask ready (${layers.engineLabel}). Tap Draw Mask to refine or Create to remove!"
                    } else null
                )
            }
        }
    }

    fun selectInpaintModelOption(option: InpaintModelOption) {
        val ctx = getApplication<Application>()
        val lamaFile = OnnxProtobufInspector.resolveModelFile(ctx, ModelSlot.INPAINTING)
        val label = SdInpaintPackManager.formatDropdownLabel(
            ctx,
            option,
            if (lamaFile.exists()) lamaFile else null
        )
        _uiState.update {
            it.copy(
                inpaintModelOption = option,
                inpaintModelLabel = label,
                errorBannerMessage = null
            )
        }
    }

    fun reuseInpaintResultAsSource() {
        val resultBmp = _uiState.value.inpaintResultBitmap ?: return
        _uiState.update {
            it.copy(
                inpaintSourceBitmap = resultBmp,
                inpaintMaskBitmap = null,
                inpaintDepthMap = null,
                inpaintAccessoryLayers = null,
                selectedAccessoryTargets = emptySet(),
                segformerAutoSelectedForAccessory = false,
                accessoryMaskNotice = null,
                inpaintResultBitmap = null,
                inpaintElapsedSec = null,
                statusBannerMessage = "Result loaded as Source Image. Draw a new mask to edit further."
            )
        }
    }

    fun runInpaint() {
        val state = _uiState.value
        val srcBmp = state.inpaintSourceBitmap
        val maskBmp = state.inpaintMaskBitmap
        if (srcBmp == null) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please select a Source Image from device folder first.")
            }
            return
        }
        if (maskBmp == null) {
            _uiState.update {
                it.copy(errorBannerMessage = "Please tap 'Draw Mask' and paint the region to inpaint first.")
            }
            return
        }

        if (state.isInpainting) return
        val appCtx = getApplication<Application>()
        ProcessingForegroundService.startOrUpdate(
            context = appCtx,
            title = "Inpaint Image Generating",
            detail = "Running offline AI inpainting (safe during screen lock & app switch)...",
            progressPercent = 5
        )
        BackgroundProcessingCoordinator.updateInpaint {
            it.copy(isRunning = true, progressText = "Preparing offline inpainting engine...", errorMessage = null, statusMessage = null)
        }
        BackgroundProcessingCoordinator.scope.launch {
            _uiState.update {
                it.copy(
                    isInpainting = true,
                    inpaintProgressText = "Preparing offline inpainting engine...",
                    errorBannerMessage = null,
                    statusBannerMessage = null
                )
            }
            try {
                val res = withContext(Dispatchers.Default) {
                    val ctx = getApplication<Application>()
                    val installedPacks = SdInpaintPackManager.discoverInstalledPacks(ctx)
                    val primaryPack = SdInpaintPackManager.resolvePackForOption(ctx, state.inpaintModelOption)
                    val secondaryPack = if (state.inpaintModelOption == InpaintModelOption.AUTO) {
                        installedPacks.firstOrNull { it.packType != primaryPack?.packType }
                    } else {
                        null
                    }
                    if (primaryPack != null) {
                        // Free FaceSwap ONNX sessions before running multi-stage SD 1.5 diffusion
                        modelService.releaseAllSessions()
                    }
                    val activeDepthMap = if (state.inpaintDepthGuardEnabled && state.inpaintDepthMap == null &&
                        DepthEstimationEngine.hasInstalledDepthModel(ctx)
                    ) {
                        DepthEstimationEngine.estimateDepth(
                            context = ctx,
                            ortEnv = ortEnv,
                            sourceBitmap = srcBmp,
                            preferHardwareAccel = state.preferHardwareAcceleration,
                            onProgress = { msg -> _uiState.update { s -> s.copy(inpaintProgressText = msg) } }
                        )
                    } else {
                        state.inpaintDepthMap
                    }
                    val lamaFile = OnnxProtobufInspector.resolveModelFile(ctx, ModelSlot.INPAINTING)
                    InteractiveInpaintEngine.inpaint(
                        ortEnv = ortEnv,
                        sourceBitmap = srcBmp,
                        maskBitmap = maskBmp,
                        prompt = state.inpaintPrompt,
                        fastMode = state.inpaintFastMode,
                        lamaModelFile = if (lamaFile.exists()) lamaFile else null,
                        preferHardwareAccel = state.preferHardwareAcceleration,
                        selectedOption = state.inpaintModelOption,
                        primarySdPack = primaryPack,
                        secondarySdPack = secondaryPack,
                        depthMap = activeDepthMap,
                        enableDepthGuard = state.inpaintDepthGuardEnabled,
                        strength = state.inpaintStrength,
                        guidanceScale = state.inpaintCfgScale,
                        onProgress = { progressMsg ->
                            _uiState.update { s -> s.copy(inpaintProgressText = progressMsg) }
                            BackgroundProcessingCoordinator.updateInpaint { s -> s.copy(progressText = progressMsg) }
                            val stepMatch = Regex("Step\\s+(\\d+)/(\\d+)", RegexOption.IGNORE_CASE).find(progressMsg)
                            val pct = if (stepMatch != null) {
                                val curS = stepMatch.groupValues[1].toIntOrNull() ?: 1
                                val totS = stepMatch.groupValues[2].toIntOrNull()?.coerceAtLeast(1) ?: 8
                                (15 + (curS * 78) / totS).coerceIn(15, 95)
                            } else -1
                            ProcessingForegroundService.startOrUpdate(
                                context = appCtx,
                                title = "Inpaint Generating (Background Active)",
                                detail = progressMsg,
                                progressPercent = pct
                            )
                        }
                    )
                }

                val historyPath = withContext(Dispatchers.IO) {
                    ImageGalleryHelper.saveBitmapToHistoryCache(getApplication(), res.outputBitmap)
                }

                auditRepository.insert(
                    SwapAuditLog(
                        sourceResolution = "${srcBmp.width}x${srcBmp.height}",
                        targetResolution = "${res.outputBitmap.width}x${res.outputBitmap.height}",
                        swappedFacesCount = 1,
                        pipelineSummary = "[INPAINT • ${res.engineUsed}]" +
                            if (state.inpaintPrompt.isNotBlank()) " Prompt: ${state.inpaintPrompt}" else "",
                        detectionMs = 0L,
                        embeddingMs = 0L,
                        inswapperMs = res.elapsedMs,
                        blendingMs = 0L,
                        totalMs = res.elapsedMs,
                        colorTransferEnabled = false,
                        watermarkEnabled = false,
                        resultImagePath = historyPath
                    )
                )

                val sec = res.elapsedMs / 1000f
                val doneMsg = "Inpaint finished in ${"%.2f".format(sec)}s (${res.engineUsed})."
                BackgroundProcessingCoordinator.updateInpaint {
                    it.copy(
                        isRunning = false,
                        progressText = null,
                        resultBitmap = res.outputBitmap,
                        elapsedSec = sec,
                        engineLabel = res.engineUsed,
                        statusMessage = doneMsg,
                        errorMessage = null
                    )
                }
                _uiState.update {
                    it.copy(
                        isInpainting = false,
                        inpaintProgressText = null,
                        inpaintResultBitmap = res.outputBitmap,
                        inpaintElapsedSec = sec,
                        inpaintEngineLabel = res.engineUsed,
                        memoryServiceState = modelService.getMemoryState(),
                        statusBannerMessage = doneMsg
                    )
                }
                ProcessingForegroundService.finishAndNotify(
                    context = appCtx,
                    showCompletionNotification = true,
                    completedTitle = "Inpaint Complete! Tap to View",
                    completedBody = doneMsg
                )
            } catch (t: Throwable) {
                val errMsg = t.message ?: "Inpaint failed: ${t.javaClass.simpleName}"
                BackgroundProcessingCoordinator.updateInpaint {
                    it.copy(isRunning = false, progressText = null, errorMessage = errMsg)
                }
                _uiState.update {
                    it.copy(
                        isInpainting = false,
                        inpaintProgressText = null,
                        errorBannerMessage = errMsg
                    )
                }
                ProcessingForegroundService.finishAndNotify(
                    context = appCtx,
                    showCompletionNotification = true,
                    completedTitle = "Inpaint Failed",
                    completedBody = errMsg
                )
            }
        }
    }

    fun saveBitmapToFolderUri(uri: Uri, bitmap: Bitmap) {
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) {
                ImageGalleryHelper.saveBitmapToUri(getApplication(), uri, bitmap)
            }
            saved.fold(
                onSuccess = { msg ->
                    _uiState.update {
                        it.copy(statusBannerMessage = msg, errorBannerMessage = null)
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(errorBannerMessage = "Folder save failed: ${err.message}")
                    }
                }
            )
        }
    }

    fun saveAnyBitmapToGallery(bitmap: Bitmap, isHd: Boolean = true, label: String = "Image") {
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) {
                ImageGalleryHelper.saveBitmapToGallery(
                    context = getApplication(),
                    bitmap = bitmap,
                    isHd = isHd,
                    targetMaxDimension = 0
                )
            }
            saved.fold(
                onSuccess = { path ->
                    _uiState.update {
                        it.copy(
                            statusBannerMessage = "Saved $label to Gallery: $path",
                            errorBannerMessage = null
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(errorBannerMessage = "Could not save $label: ${err.message}")
                    }
                }
            )
        }
    }

    fun deleteAuditLog(id: Int) {
        viewModelScope.launch {
            auditRepository.deleteById(id)
        }
    }

    fun clearAllAuditLogs() {
        viewModelScope.launch {
            auditRepository.clearAll()
        }
    }

    override fun onCleared() {
        super.onCleared()
        if (!BackgroundProcessingCoordinator.isAnyJobRunning()) {
            runCatching { modelService.close() }
        }
    }

    private fun formatBytes(bytes: Long): String {
        val mb = bytes / (1024.0 * 1024.0)
        return if (mb >= 1.0) "%.1f MB".format(mb) else "${bytes / 1024} KB"
    }
}

private data class Quadruple<A, B, C, D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D
)
