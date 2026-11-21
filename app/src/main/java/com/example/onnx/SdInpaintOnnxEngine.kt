package com.example.onnx

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.Locale
import java.util.Random
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Full 100% Offline Stable Diffusion 1.5 Inpainting ONNX Runtime Engine.
 *
 * Supports both:
 *  - Priority 1: `stable-diffusion-v1-5-inpainting-onnx-fp16` (FLOAT16 / mixed precision)
 *  - Priority 2: `realvis51-inpaint-int8-onnx` (INT8 weights + FLOAT32 / FLOAT16 I/O)
 *
 * Executes components sequentially (`text_encoder` -> `vae_encoder` -> `unet` denoising loop ->
 * `vae_decoder`) closing each session right after use so Android RAM never spikes to OOM/lmkd kill.
 */
object SdInpaintOnnxEngine {

    private const val TAG = "INPAINT"
    private const val BASE_SD_RES = 512
    private const val LATENT_CHANNELS = 4
    private const val VAE_SCALE_FACTOR = 0.18215f
    const val DEFAULT_STRENGTH = 0.70f
    const val DEFAULT_GUIDANCE_SCALE = 7.0f
    const val DEFAULT_NEGATIVE_PROMPT =
        "blurry, low quality, deformed hands, extra fingers"

    class InsufficientNativeRamException(message: String) : RuntimeException(message)

    internal data class DeviceRamStatus(
        val availMb: Long,
        val totalMb: Long,
        val isLowMemory: Boolean
    )

    internal data class SdExecutionProfile(
        val maxDimensionPx: Int,
        val isLowRamConstrained: Boolean,
        val intraOpThreads: Int,
        val isRuntimeCrashLatched: Boolean,
        val is384UnsafeLatched: Boolean,
        val minRamBeforeVaeEncoderMb: Long = 600L,
        val minRamBeforeUnetMb: Long = 650L,
        val minRamBeforeVaeDecoderMbFor512: Long = 850L,
        val minRamBeforeVaeDecoderMbFor384: Long = 520L
    )

    internal fun queryDeviceRamStatus(context: Context?): DeviceRamStatus? {
        val ctx = context?.applicationContext ?: context ?: return null
        return runCatching {
            val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return null
            val info = ActivityManager.MemoryInfo()
            am.getMemoryInfo(info)
            val availMb = if (info.availMem > 0L) info.availMem / (1024L * 1024L) else Long.MAX_VALUE
            val totalMb = if (info.totalMem > 0L) info.totalMem / (1024L * 1024L) else Long.MAX_VALUE
            DeviceRamStatus(
                availMb = availMb,
                totalMb = totalMb,
                isLowMemory = info.lowMemory || (am.isLowRamDevice)
            )
        }.getOrNull()
    }

    internal fun queryAvailableRamInfo(context: Context?): Pair<Long, Boolean>? {
        val status = queryDeviceRamStatus(context) ?: return null
        return status.availMb to status.isLowMemory
    }

    internal fun resolveExecutionProfile(
        ramStatus: DeviceRamStatus?,
        fastMode: Boolean,
        maxPreferredResolutionPx: Int,
        isRuntimeCrashLatched: Boolean,
        is384UnsafeLatched: Boolean = false
    ): SdExecutionProfile {
        val is4GbOrLessDevice = ramStatus != null && ramStatus.totalMb in 1L..4600L
        val isBorderlineFreeRam = ramStatus != null && ramStatus.availMb < 1650L
        val isConstrained = isRuntimeCrashLatched ||
            is384UnsafeLatched ||
            (ramStatus?.isLowMemory == true) ||
            is4GbOrLessDevice ||
            isBorderlineFreeRam

        val maxDimPx = when {
            isConstrained -> min(maxPreferredResolutionPx, 384)
            fastMode -> min(maxPreferredResolutionPx, 384)
            else -> maxPreferredResolutionPx
        }
        val cores = Runtime.getRuntime().availableProcessors()
        val threads = if (isConstrained) {
            2
        } else if (cores >= 6) {
            min(cores - 1, 6)
        } else {
            cores.coerceIn(2, 4)
        }
        return SdExecutionProfile(
            maxDimensionPx = maxDimPx,
            isLowRamConstrained = isConstrained,
            intraOpThreads = threads,
            isRuntimeCrashLatched = isRuntimeCrashLatched,
            is384UnsafeLatched = is384UnsafeLatched
        )
    }

    internal data class TargetDimensions(
        val width: Int,
        val height: Int
    ) {
        val latentW: Int get() = width / 8
        val latentH: Int get() = height / 8
    }

    internal data class DenoisePlan(
        val baseSteps: Int,
        val denoiseSteps: Int,
        val startStepIndex: Int,
        val strength: Float
    )

    private data class CachedSessionEntry(
        val session: OrtSession,
        val epLabel: String,
        val filePath: String,
        val lastModified: Long
    )

    private val sessionLock = Any()
    private var cachedPackKey: String? = null
    private val cachedComponentSessions = mutableMapOf<String, CachedSessionEntry>()

    fun clearSessionCache() {
        synchronized(sessionLock) {
            for ((_, entry) in cachedComponentSessions) {
                runCatching { entry.session.close() }
            }
            cachedComponentSessions.clear()
            cachedPackKey = null
        }
    }

    private fun closeComponentSession(componentName: String) {
        synchronized(sessionLock) {
            cachedComponentSessions.remove(componentName)?.let { entry ->
                runCatching { entry.session.close() }
            }
        }
    }

    private val vaeSpatialShapeCache = java.util.concurrent.ConcurrentHashMap<String, TensorInfo>()

    private fun getOrCreatePackSession(
        ortEnv: OrtEnvironment,
        pack: InstalledSdInpaintPack,
        componentName: String,
        modelFile: java.io.File,
        preferHardwareAccel: Boolean,
        intraOpThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
        preferBasicOpt: Boolean = false,
        enableMemoryPattern: Boolean = false
    ): Pair<OrtSession, String> {
        val threads = intraOpThreads.coerceIn(1, 6)
        val packKey = "${pack.packDir.absolutePath}|hw=$preferHardwareAccel|t=$threads|basic=$preferBasicOpt|mp=$enableMemoryPattern"
        val lastMod = modelFile.lastModified()
        synchronized(sessionLock) {
            if (cachedPackKey != null && cachedPackKey != packKey) {
                for ((_, entry) in cachedComponentSessions) {
                    runCatching { entry.session.close() }
                }
                cachedComponentSessions.clear()
            }
            cachedPackKey = packKey

            val existing = cachedComponentSessions[componentName]
            if (existing != null &&
                existing.filePath == modelFile.absolutePath &&
                existing.lastModified == lastMod
            ) {
                return existing.session to existing.epLabel
            }
            if (existing != null) {
                runCatching { existing.session.close() }
                cachedComponentSessions.remove(componentName)
            }

            // Bypass NNAPI for SD 1.5 ONNX components because Android vendor NNAPI HALs often hang,
            // produce NaN/black pixels, or SIGSEGV on large Diffusion UNet/VAE graphs.
            val created = try {
                OnnxProtobufInspector.createSessionWithFallback(
                    ortEnv = ortEnv,
                    modelFile = modelFile,
                    preferHardwareAcceleration = false,
                    intraOpThreads = threads,
                    preferBasicOpt = preferBasicOpt,
                    enableMemoryPattern = enableMemoryPattern
                )
            } catch (oom: OutOfMemoryError) {
                // Free other cached SD components first, keep only what we're loading now
                for ((k, entry) in cachedComponentSessions.entries.toList()) {
                    if (k != componentName) {
                        runCatching { entry.session.close() }
                        cachedComponentSessions.remove(k)
                    }
                }
                OnnxProtobufInspector.clearCachedSessions()
                System.gc()
                OnnxProtobufInspector.createSessionWithFallback(
                    ortEnv = ortEnv,
                    modelFile = modelFile,
                    preferHardwareAcceleration = false,
                    intraOpThreads = min(threads, 2),
                    preferBasicOpt = true,
                    enableMemoryPattern = false
                )
            }

            cachedComponentSessions[componentName] = CachedSessionEntry(
                session = created.first,
                epLabel = created.second,
                filePath = modelFile.absolutePath,
                lastModified = lastMod
            )
            return created
        }
    }

    internal fun computeAspectMatchedTargetResolution(
        origW: Int,
        origH: Int,
        supportsDynamicShapes: Boolean,
        maxDimensionPx: Int = 768
    ): TargetDimensions {
        if (!supportsDynamicShapes) return TargetDimensions(BASE_SD_RES, BASE_SD_RES)
        if (maxDimensionPx <= 384) return TargetDimensions(384, 384)
        if (maxDimensionPx <= BASE_SD_RES) return TargetDimensions(BASE_SD_RES, BASE_SD_RES)
        val aspect = origW.coerceAtLeast(1).toFloat() / origH.coerceAtLeast(1).toFloat()
        return when {
            aspect <= 0.80f -> TargetDimensions(512, 768)
            aspect >= 1.25f -> TargetDimensions(768, 512)
            else -> TargetDimensions(512, 512)
        }
    }

    internal fun supportsSpatialShape(
        tensorInfo: TensorInfo?,
        expectedH: Int,
        expectedW: Int
    ): Boolean {
        val shape = tensorInfo?.shape ?: return false
        if (shape.size < 4) return false
        val hDim = shape[2]
        val wDim = shape[3]
        val hOk = hDim <= 0L || hDim == expectedH.toLong()
        val wOk = wDim <= 0L || wDim == expectedW.toLong()
        return hOk && wOk
    }

    internal fun selectCompatibleDims(
        candidateDims: TargetDimensions,
        vaeEncInfo: TensorInfo?,
        vaeDecInfo: TensorInfo?
    ): TargetDimensions {
        val encSupportsCand = vaeEncInfo == null || supportsSpatialShape(vaeEncInfo, candidateDims.height, candidateDims.width)
        val decSupportsCand = vaeDecInfo == null || supportsSpatialShape(vaeDecInfo, candidateDims.latentH, candidateDims.latentW)
        if (encSupportsCand && decSupportsCand) return candidateDims
        val fallbackDims = TargetDimensions(BASE_SD_RES, BASE_SD_RES)
        val encSupports512 = vaeEncInfo == null || supportsSpatialShape(vaeEncInfo, fallbackDims.height, fallbackDims.width)
        val decSupports512 = vaeDecInfo == null || supportsSpatialShape(vaeDecInfo, fallbackDims.latentH, fallbackDims.latentW)
        if (encSupports512 && decSupports512) return fallbackDims
        throw IllegalStateException(
            "Incompatible VAE spatial shapes in pack: encoder=${vaeEncInfo?.shape?.contentToString()} decoder=${vaeDecInfo?.shape?.contentToString()}."
        )
    }

    private data class DenoisedLatentStageResult(
        val roi: AspectRoi,
        val targetDims: TargetDimensions,
        val denoisedLatent: FloatArray,
        val unetEpLabel: String
    )

    internal fun computeDenoisePlan(fastMode: Boolean, strength: Float): DenoisePlan {
        val baseSteps = if (fastMode) 10 else 20
        val clampedStrength = strength.coerceIn(0.10f, 1.0f)
        val denoiseSteps = (baseSteps * clampedStrength).roundToInt().coerceIn(2, baseSteps)
        val startStepIndex = (baseSteps - denoiseSteps).coerceIn(0, baseSteps - 1)
        return DenoisePlan(
            baseSteps = baseSteps,
            denoiseSteps = denoiseSteps,
            startStepIndex = startStepIndex,
            strength = clampedStrength
        )
    }

    internal fun shouldRunUncondCfgPass(
        localStep: Int,
        totalSteps: Int,
        guidanceScale: Float,
        fastMode: Boolean,
        hasCachedUncond: Boolean
    ): Boolean {
        if (guidanceScale <= 1.05f) return false
        if (!fastMode || !hasCachedUncond || totalSteps <= 3) return true
        val fullCfgSteps = kotlin.math.ceil(totalSteps * 0.70).toInt().coerceAtLeast(2)
        return localStep < fullCfgSteps
    }

    private fun logDebug(message: String) {
        runCatching { Log.d(TAG, message) }
    }

    fun inpaint(
        ortEnv: OrtEnvironment,
        pack: InstalledSdInpaintPack,
        sourceBitmap: Bitmap,
        maskBitmap: Bitmap,
        prompt: String,
        negativePrompt: String = DEFAULT_NEGATIVE_PROMPT,
        fastMode: Boolean,
        preferHardwareAccel: Boolean = false,
        strength: Float = DEFAULT_STRENGTH,
        guidanceScale: Float = DEFAULT_GUIDANCE_SCALE,
        skipCompositeDebug: Boolean = false,
        seed: Long = System.currentTimeMillis(),
        releaseEachStageAfterUse: Boolean = true,
        context: Context? = null,
        maxPreferredResolutionPx: Int = 768,
        isCancelled: () -> Boolean = { false },
        onProgress: (String) -> Unit = {}
    ): InpaintExecutionResult {

        val t0 = System.currentTimeMillis()
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height
        val plan = computeDenoisePlan(fastMode = fastMode, strength = strength)
        val clampedCfg = guidanceScale.coerceIn(1.0f, 15.0f)
        OnnxProtobufInspector.promoteStuckCrashBreadcrumbIfAny(context)
        val initialRamStatus = queryDeviceRamStatus(context)
        val isRuntimeCrashLatched = OnnxProtobufInspector.isRuntimeLowRamLatched(context)
        val is384UnsafeLatched = OnnxProtobufInspector.isRuntime384UnsafeLatched(context)
        if (is384UnsafeLatched) {
            clearSessionCache()
            OnnxProtobufInspector.clearCachedSessions()
            System.gc()
            throw InsufficientNativeRamException(
                "Previous native force-stop occurred even at Low-RAM 384×384. Free device RAM, switch to LaMa/Fast CV, or reset Safe Mode in Settings."
            )
        }
        val execProfile = resolveExecutionProfile(
            ramStatus = initialRamStatus,
            fastMode = fastMode,
            maxPreferredResolutionPx = maxPreferredResolutionPx,
            isRuntimeCrashLatched = isRuntimeCrashLatched,
            is384UnsafeLatched = is384UnsafeLatched
        )
        val effectiveMaxDimPx = execProfile.maxDimensionPx
        if (execProfile.isLowRamConstrained) {
            OnnxProtobufInspector.clearCachedSessions()
        }
        logDebug(
            "ramProfile availMb=${initialRamStatus?.availMb} totalMb=${initialRamStatus?.totalMb} " +
                "latched=$isRuntimeCrashLatched maxDim=$effectiveMaxDimPx threads=${execProfile.intraOpThreads}"
        )

        // Align mask to source resolution
        val alignedMask = if (maskBitmap.width == origW && maskBitmap.height == origH) {
            maskBitmap
        } else {
            Bitmap.createScaledBitmap(maskBitmap, origW, origH, true)
        }

        val totalOrig = origW * origH
        val maskPx = IntArray(totalOrig)
        alignedMask.getPixels(maskPx, 0, origW, 0, 0, origW, origH)
        if (alignedMask !== maskBitmap) alignedMask.recycle()

        val fullMaskAlpha = FloatArray(totalOrig)
        var minX = origW
        var minY = origH
        var maxX = -1
        var maxY = -1
        var paintedCount = 0

        for (y in 0 until origH) {
            val rowOff = y * origW
            for (x in 0 until origW) {
                val c = maskPx[rowOff + x]
                val a = (c ushr 24) and 0xFF
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val s = max(a, max(r, max(g, b))) / 255f
                if (s > 0.12f) {
                    fullMaskAlpha[rowOff + x] = s.coerceIn(0f, 1f)
                    paintedCount++
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }
        logDebug(
            "source=${origW}x${origH} mask=${maskBitmap.width}x${maskBitmap.height} " +
                "nonZero=$paintedCount bounds=[$minX,$minY..$maxX,$maxY] " +
                "strength=${plan.strength} cfg=$clampedCfg steps=${plan.denoiseSteps}/${plan.baseSteps}"
        )
        require(paintedCount >= 8) {
            "Please paint a mask area first using 'Draw Mask'."
        }

        fun checkCancel() {
            if (isCancelled()) {
                throw java.util.concurrent.CancellationException("Diffusion cancelled by user.")
            }
        }

        checkCancel()
        fun encodeAndDenoiseToLatent(): DenoisedLatentStageResult {
            val normalizedPrompt = normalizePromptForClip(prompt)
            val effectivePrompt = normalizedPrompt.ifEmpty { "clean natural texture, high quality photo" }
            val effectiveNegative = normalizePromptForClip(negativePrompt).ifEmpty { DEFAULT_NEGATIVE_PROMPT }
            onProgress("1/4 Encoding positive & negative prompts (${pack.packType.displayLabel})...")

            // 1. CLIP Text Encoder -> uncond & cond embeddings [77, 768] (close immediately after use to save RAM)
            val tokenizer = ClipBpeTokenizer.loadFromDirectory(pack.tokenizerDir)
            logDebug(
                "tokenizerLoaded=${tokenizer.isLoaded()} vocabSize=${tokenizer.vocabSize} " +
                    "tokDir=${pack.tokenizerDir?.absolutePath} neg=\"$effectiveNegative\""
            )
            if (normalizedPrompt.isNotEmpty() && !tokenizer.isLoaded()) {
                throw IllegalStateException(
                    "CLIP tokenizer files (vocab.json + merges.txt or tokenizer.json) are missing in ${pack.packType.displayLabel}. Re-import the SD Inpaint folder."
                )
            }
            val condIds = tokenizer.tokenizeTo77(effectivePrompt)
            val uncondIds = tokenizer.tokenizeTo77(effectiveNegative)
            val (textEncSession, _) = getOrCreatePackSession(
                ortEnv = ortEnv,
                pack = pack,
                componentName = "text_encoder",
                modelFile = pack.textEncoderOnnx,
                preferHardwareAccel = preferHardwareAccel,
                intraOpThreads = execProfile.intraOpThreads,
                preferBasicOpt = fastMode
            )
            val (uncondEmbed, condEmbed) = try {
                OnnxProtobufInspector.runWithRuntimeCrashBreadcrumb(
                    context = context,
                    stageName = "sd_text_encoder",
                    resolutionPx = effectiveMaxDimPx
                ) {
                    runTextEncoder(
                        ortEnv = ortEnv,
                        session = textEncSession,
                        uncondIds = uncondIds,
                        condIds = condIds
                    )
                }
            } finally {
                if (releaseEachStageAfterUse) {
                    closeComponentSession("text_encoder")
                }
            }
            checkCancel()

            queryDeviceRamStatus(context)?.let { ram ->
                if (ram.isLowMemory || ram.availMb < execProfile.minRamBeforeVaeEncoderMb) {
                    clearSessionCache()
                    OnnxProtobufInspector.clearCachedSessions()
                    System.gc()
                    throw InsufficientNativeRamException(
                        "Insufficient free RAM before VAE encoding (${ram.availMb}MB available)."
                    )
                }
            }

            // Pre-check VAE Decoder spatial shape using cached metadata first to avoid double session loading
            val candidateDims = computeAspectMatchedTargetResolution(
                origW = origW,
                origH = origH,
                supportsDynamicShapes = true,
                maxDimensionPx = effectiveMaxDimPx
            )
            val vaeDecCacheKey = "${pack.vaeDecoderOnnx.absolutePath}|${pack.vaeDecoderOnnx.length()}|${pack.vaeDecoderOnnx.lastModified()}"
            val vaeDecProbeInfo: TensorInfo? = vaeSpatialShapeCache[vaeDecCacheKey] ?: try {
                val (probeDecSession, _) = getOrCreatePackSession(
                    ortEnv = ortEnv,
                    pack = pack,
                    componentName = "vae_decoder",
                    modelFile = pack.vaeDecoderOnnx,
                    preferHardwareAccel = preferHardwareAccel,
                    intraOpThreads = execProfile.intraOpThreads,
                    preferBasicOpt = fastMode
                )
                (probeDecSession.inputInfo[probeDecSession.inputNames.first()]?.info as? TensorInfo)?.also { info ->
                    vaeSpatialShapeCache[vaeDecCacheKey] = info
                }
            } finally {
                if (releaseEachStageAfterUse) {
                    closeComponentSession("vae_decoder")
                }
            }
            checkCancel()

            // 2. Sequential VAE Encoder (closed before loading heavy UNet)
            onProgress("2/4 Loading VAE Encoder...")
            val (vaeEncSession, _) = getOrCreatePackSession(
                ortEnv = ortEnv,
                pack = pack,
                componentName = "vae_encoder",
                modelFile = pack.vaeEncoderOnnx,
                preferHardwareAccel = preferHardwareAccel,
                intraOpThreads = execProfile.intraOpThreads,
                preferBasicOpt = fastMode
            )
            val vaeEncInfo = vaeEncSession.inputInfo[vaeEncSession.inputNames.first()]?.info as? TensorInfo
            var targetDims = selectCompatibleDims(candidateDims, vaeEncInfo, vaeDecProbeInfo)
            val downgradedToStatic512 = max(targetDims.width, targetDims.height) >= BASE_SD_RES &&
                candidateDims.width < BASE_SD_RES
            if (downgradedToStatic512 &&
                (execProfile.isRuntimeCrashLatched ||
                    (initialRamStatus != null && initialRamStatus.availMb < 1400L))
            ) {
                if (releaseEachStageAfterUse) closeComponentSession("vae_encoder")
                clearSessionCache()
                OnnxProtobufInspector.clearCachedSessions()
                System.gc()
                throw InsufficientNativeRamException(
                    "Low-RAM protection: installed SD 1.5 model only supports fixed 512×512 (384×384 not supported), which risks a native force-stop at ${initialRamStatus?.availMb ?: 0}MB free RAM."
                )
            }

            fun encodeLatentsAt(
                activeVaeEncSession: OrtSession,
                dims: TargetDimensions
            ): Triple<AspectRoi, FloatArray, Pair<FloatArray, FloatArray>> {
                val r = computeAspectRoi(
                    imgW = origW,
                    imgH = origH,
                    minX = minX,
                    minY = minY,
                    maxX = maxX,
                    maxY = maxY,
                    targetW = dims.width,
                    targetH = dims.height,
                    fastMode = fastMode
                )
                val (cropBmpTarget, rawMaskTarget) = extractTargetCropAndMask(
                    source = sourceBitmap,
                    fullMaskAlpha = fullMaskAlpha,
                    origW = origW,
                    origH = origH,
                    roi = r,
                    targetW = dims.width,
                    targetH = dims.height
                )
                val maskTarget = dilateBinaryMask(rawMaskTarget, dims.width, dims.height, radiusPx = 2)
                onProgress("2/4 Encoding ${dims.width}×${dims.height} (${dims.latentW}×${dims.latentH} latent)...")
                return OnnxProtobufInspector.runWithRuntimeCrashBreadcrumb(
                    context = context,
                    stageName = "sd_vae_encoder",
                    resolutionPx = max(dims.width, dims.height)
                ) {
                    val mLatent = runVaeEncoder(
                        ortEnv = ortEnv,
                        session = activeVaeEncSession,
                        cropBmp = cropBmpTarget,
                        maskTarget = maskTarget,
                        targetW = dims.width,
                        targetH = dims.height,
                        applyZeroMask = true
                    )
                    val iLatent = runVaeEncoder(
                        ortEnv = ortEnv,
                        session = activeVaeEncSession,
                        cropBmp = cropBmpTarget,
                        maskTarget = maskTarget,
                        targetW = dims.width,
                        targetH = dims.height,
                        applyZeroMask = false
                    )
                    cropBmpTarget.recycle()
                    val latMask = downsampleMaskToLatent(
                        mask = maskTarget,
                        targetW = dims.width,
                        targetH = dims.height,
                        latentW = dims.latentW,
                        latentH = dims.latentH
                    )
                    Triple(r, latMask, mLatent to iLatent)
                }
            }

            var encodedTriple = try {
                encodeLatentsAt(vaeEncSession, targetDims)
            } finally {
                if (releaseEachStageAfterUse) {
                    closeComponentSession("vae_encoder")
                }
            }
            checkCancel()

            queryDeviceRamStatus(context)?.let { ram ->
                if (ram.isLowMemory || ram.availMb < execProfile.minRamBeforeUnetMb) {
                    clearSessionCache()
                    OnnxProtobufInspector.clearCachedSessions()
                    System.gc()
                    throw InsufficientNativeRamException(
                        "Insufficient free RAM before UNet denoising (${ram.availMb}MB available)."
                    )
                }
            }

            if (execProfile.isLowRamConstrained || releaseEachStageAfterUse) {
                OnnxProtobufInspector.clearCachedSessions()
            }

            // 3. Load UNet alone in RAM and verify spatial dimensions before denoising
            onProgress("3/4 Loading UNet (${pack.packType.displayLabel})...")
            val useMemoryPatternForUnet = !execProfile.isLowRamConstrained
            var (unetSession, unetEpLabel) = getOrCreatePackSession(
                ortEnv = ortEnv,
                pack = pack,
                componentName = "unet",
                modelFile = pack.unetOnnx,
                preferHardwareAccel = preferHardwareAccel,
                intraOpThreads = execProfile.intraOpThreads,
                preferBasicOpt = fastMode,
                enableMemoryPattern = useMemoryPatternForUnet
            )
            val unetSampleName = unetSession.inputNames.firstOrNull { it.contains("sample", ignoreCase = true) }
                ?: unetSession.inputNames.first()
            val unetInfo = unetSession.inputInfo[unetSampleName]?.info as? TensorInfo
            val unetSupportsCurrent = supportsSpatialShape(unetInfo, targetDims.latentH, targetDims.latentW)
            if (!unetSupportsCurrent && (targetDims.width != BASE_SD_RES || targetDims.height != BASE_SD_RES)) {
                val fallback512 = TargetDimensions(BASE_SD_RES, BASE_SD_RES)
                if (vaeDecProbeInfo != null &&
                    !supportsSpatialShape(vaeDecProbeInfo, fallback512.latentH, fallback512.latentW)
                ) {
                    if (releaseEachStageAfterUse) closeComponentSession("unet")
                    throw IllegalStateException(
                        "UNet and VAE Decoder do not share a compatible spatial resolution."
                    )
                }
                val ramBeforeRetry512 = queryDeviceRamStatus(context)
                if (execProfile.isRuntimeCrashLatched ||
                    (execProfile.isLowRamConstrained && ramBeforeRetry512 != null && ramBeforeRetry512.availMb < 1400L)
                ) {
                    if (releaseEachStageAfterUse) closeComponentSession("unet")
                    clearSessionCache()
                    OnnxProtobufInspector.clearCachedSessions()
                    System.gc()
                    throw InsufficientNativeRamException(
                        "Low-RAM protection: UNet requires fixed 512×512 (384×384 not supported), which risks a native force-stop."
                    )
                }
                targetDims = fallback512
                if (releaseEachStageAfterUse) {
                    closeComponentSession("unet")
                }
                val (reEncSession, _) = getOrCreatePackSession(
                    ortEnv = ortEnv,
                    pack = pack,
                    componentName = "vae_encoder",
                    modelFile = pack.vaeEncoderOnnx,
                    preferHardwareAccel = preferHardwareAccel,
                    intraOpThreads = execProfile.intraOpThreads,
                    preferBasicOpt = fastMode
                )
                encodedTriple = try {
                    encodeLatentsAt(reEncSession, targetDims)
                } finally {
                    if (releaseEachStageAfterUse) {
                        closeComponentSession("vae_encoder")
                    }
                }
                val reloadedUnet = getOrCreatePackSession(
                    ortEnv = ortEnv,
                    pack = pack,
                    componentName = "unet",
                    modelFile = pack.unetOnnx,
                    preferHardwareAccel = preferHardwareAccel,
                    intraOpThreads = execProfile.intraOpThreads,
                    preferBasicOpt = fastMode,
                    enableMemoryPattern = useMemoryPatternForUnet
                )
                unetSession = reloadedUnet.first
                unetEpLabel = reloadedUnet.second
            }

            val roi = encodedTriple.first
            val maskLatent = encodedTriple.second
            val maskedLatent = encodedTriple.third.first
            val initLatent = encodedTriple.third.second

            logDebug(
                "targetDims=${targetDims.width}x${targetDims.height} latent=${targetDims.latentW}x${targetDims.latentH} " +
                    "ep=$unetEpLabel sequential=$releaseEachStageAfterUse roi=[${roi.left},${roi.top},${roi.width}x${roi.height}]"
            )

            // 4. UNet Denoising Loop (Euler Discrete Scheduler) + Runtime Native Crash Breadcrumb
            val denoisedLatent = try {
                OnnxProtobufInspector.runWithRuntimeCrashBreadcrumb(
                    context = context,
                    stageName = "sd_unet",
                    resolutionPx = max(targetDims.width, targetDims.height)
                ) {
                    runUnetDenoisingLoop(
                        ortEnv = ortEnv,
                        session = unetSession,
                        uncondEmbed = uncondEmbed,
                        condEmbed = condEmbed,
                        maskedLatent = maskedLatent,
                        initLatent = initLatent,
                        maskLatent = maskLatent,
                        latentW = targetDims.latentW,
                        latentH = targetDims.latentH,
                        plan = plan,
                        guidanceScale = clampedCfg,
                        fastMode = fastMode,
                        seed = seed,
                        isCancelled = isCancelled,
                        onStep = { step, total ->
                            onProgress("3/4 Diffusion step $step/$total (${targetDims.width}×${targetDims.height} • $unetEpLabel)...")
                        }
                    )
                }
            } finally {
                if (releaseEachStageAfterUse) {
                    closeComponentSession("unet")
                    System.gc()
                }
            }
            return DenoisedLatentStageResult(
                roi = roi,
                targetDims = targetDims,
                denoisedLatent = denoisedLatent,
                unetEpLabel = unetEpLabel
            )
        }

        val stageResult = encodeAndDenoiseToLatent()
        val roi = stageResult.roi
        val targetDims = stageResult.targetDims
        val denoisedLatent = stageResult.denoisedLatent
        val unetEpLabel = stageResult.unetEpLabel
        if (releaseEachStageAfterUse) {
            // Ensure prompt embeddings, UNet tensors, and cached FaceSwap sessions are reclaimed before VAE Decode (+700MB spike)
            OnnxProtobufInspector.clearCachedSessions()
            System.gc()
        }
        checkCancel()

        val requiredPreVaeDecodeRamMb = if (max(targetDims.width, targetDims.height) >= BASE_SD_RES) {
            execProfile.minRamBeforeVaeDecoderMbFor512
        } else {
            execProfile.minRamBeforeVaeDecoderMbFor384
        }
        queryDeviceRamStatus(context)?.let { ram ->
            if (ram.isLowMemory || ram.availMb < requiredPreVaeDecodeRamMb) {
                clearSessionCache()
                OnnxProtobufInspector.clearCachedSessions()
                System.gc()
                throw InsufficientNativeRamException(
                    "Insufficient free RAM before 4/4 VAE Decoding at ${targetDims.width}×${targetDims.height} (${ram.availMb}MB available, need >= ${requiredPreVaeDecodeRamMb}MB)."
                )
            }
        }

        // 5. Sequential VAE Decoder -> targetDims RGB Bitmap (closed immediately after decoding)
        onProgress("4/4 Decoding ${targetDims.width}×${targetDims.height} to ${origW}×${origH}...")
        val (vaeDecSession, _) = getOrCreatePackSession(
            ortEnv = ortEnv,
            pack = pack,
            componentName = "vae_decoder",
            modelFile = pack.vaeDecoderOnnx,
            preferHardwareAccel = preferHardwareAccel,
            intraOpThreads = execProfile.intraOpThreads,
            preferBasicOpt = fastMode
        )
        val decodedCrop = try {
            OnnxProtobufInspector.runWithRuntimeCrashBreadcrumb(
                context = context,
                stageName = "sd_vae_decoder",
                resolutionPx = max(targetDims.width, targetDims.height)
            ) {
                runVaeDecoder(
                    ortEnv = ortEnv,
                    session = vaeDecSession,
                    latent = denoisedLatent,
                    targetW = targetDims.width,
                    targetH = targetDims.height
                )
            }
        } finally {
            if (releaseEachStageAfterUse) {
                closeComponentSession("vae_decoder")
                System.gc()
            }
        }

        // 7. Composite back onto exact origW x origH sourceBitmap without aspect stretching
        val resultBitmap = if (skipCompositeDebug) {
            pasteCropDirectDebug(sourceBitmap = sourceBitmap, generatedCrop = decodedCrop, roi = roi)
        } else {
            compositeCropIntoSource(
                sourceBitmap = sourceBitmap,
                fullMaskAlpha = fullMaskAlpha,
                generatedCrop = decodedCrop,
                roi = roi,
                featherRadius = if (fastMode) 4 else 6
            )
        }
        decodedCrop.recycle()

        val elapsed = (System.currentTimeMillis() - t0).coerceAtLeast(1L)
        val badgeStr = String.format(
            Locale.US,
            "%s [%dx%d • %s] (%d/%d steps • Str %.1f • CFG %.1f)",
            pack.packType.displayLabel,
            targetDims.width,
            targetDims.height,
            unetEpLabel,
            plan.denoiseSteps,
            plan.baseSteps,
            plan.strength * 10f,
            clampedCfg
        )
        logDebug("completed in ${elapsed}ms -> $badgeStr")
        return InpaintExecutionResult(
            outputBitmap = resultBitmap,
            elapsedMs = elapsed,
            engineUsed = badgeStr
        )
    }

    private data class AspectRoi(
        val left: Int,
        val top: Int,
        val width: Int,
        val height: Int
    )

    private fun computeAspectRoi(
        imgW: Int,
        imgH: Int,
        minX: Int,
        minY: Int,
        maxX: Int,
        maxY: Int,
        targetW: Int,
        targetH: Int,
        fastMode: Boolean = false
    ): AspectRoi {
        val targetAspect = targetW.toFloat() / targetH.toFloat()
        val maskW = (maxX - minX + 1).coerceAtLeast(1)
        val maskH = (maxY - minY + 1).coerceAtLeast(1)
        val cx = (minX + maxX) / 2
        val cy = (minY + maxY) / 2

        // For small/medium portraits or masks covering most of the frame, keep full image in aspect-locked ROI
        if (imgW <= targetW && imgH <= targetH || maskW >= imgW * 0.55f || maskH >= imgH * 0.55f) {
            val imgAspect = imgW.toFloat() / imgH.coerceAtLeast(1)
            return if (imgAspect <= targetAspect) {
                val roiH = imgH
                val roiW = (roiH * targetAspect).roundToInt().coerceAtLeast(imgW)
                AspectRoi((imgW - roiW) / 2, 0, roiW, roiH)
            } else {
                val roiW = imgW
                val roiH = (roiW / targetAspect).roundToInt().coerceAtLeast(imgH)
                AspectRoi(0, (imgH - roiH) / 2, roiW, roiH)
            }
        }

        // Localized mask on a high-res photo: tightly frame mask (even tighter in Fast Mode) with context matching targetAspect
        val contextScale = if (fastMode) 1.85f else 1.65f
        val minImgShare = if (fastMode) 0.38f else 0.55f
        var desiredW = max((maskW * contextScale).roundToInt(), (imgW * minImgShare).roundToInt())
        var desiredH = max((maskH * contextScale).roundToInt(), (imgH * minImgShare).roundToInt())
        if (desiredW / desiredH.coerceAtLeast(1).toFloat() < targetAspect) {
            desiredW = (desiredH * targetAspect).roundToInt()
        } else {
            desiredH = (desiredW / targetAspect).roundToInt()
        }

        val left = if (desiredW <= imgW) {
            (cx - desiredW / 2).coerceIn(0, imgW - desiredW)
        } else {
            (imgW - desiredW) / 2
        }
        val top = if (desiredH <= imgH) {
            (cy - desiredH / 2).coerceIn(0, imgH - desiredH)
        } else {
            (imgH - desiredH) / 2
        }
        return AspectRoi(left, top, desiredW, desiredH)
    }

    private fun extractTargetCropAndMask(
        source: Bitmap,
        fullMaskAlpha: FloatArray,
        origW: Int,
        origH: Int,
        roi: AspectRoi,
        targetW: Int,
        targetH: Int
    ): Pair<Bitmap, FloatArray> {
        val roiBmp = Bitmap.createBitmap(roi.width, roi.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(roiBmp)
        canvas.drawColor(Color.rgb(128, 128, 128))
        val srcLeft = max(0, roi.left)
        val srcTop = max(0, roi.top)
        val srcRight = min(origW, roi.left + roi.width)
        val srcBottom = min(origH, roi.top + roi.height)
        if (srcRight > srcLeft && srcBottom > srcTop) {
            val dstLeft = srcLeft - roi.left
            val dstTop = srcTop - roi.top
            val dstRight = srcRight - roi.left
            val dstBottom = srcBottom - roi.top
            val paint = Paint(Paint.FILTER_BITMAP_FLAG)
            // Extend outer columns/rows cleanly so edge pixels mirror naturally instead of flat gray bars
            if (dstLeft > 0) {
                canvas.drawBitmap(
                    source,
                    Rect(srcLeft, srcTop, min(origW, srcLeft + 1), srcBottom),
                    Rect(0, dstTop, dstLeft, dstBottom),
                    paint
                )
            }
            if (dstRight < roi.width) {
                canvas.drawBitmap(
                    source,
                    Rect(max(0, srcRight - 1), srcTop, srcRight, srcBottom),
                    Rect(dstRight, dstTop, roi.width, dstBottom),
                    paint
                )
            }
            if (dstTop > 0) {
                canvas.drawBitmap(
                    source,
                    Rect(srcLeft, srcTop, srcRight, min(origH, srcTop + 1)),
                    Rect(dstLeft, 0, dstRight, dstTop),
                    paint
                )
            }
            if (dstBottom < roi.height) {
                canvas.drawBitmap(
                    source,
                    Rect(srcLeft, max(0, srcBottom - 1), srcRight, srcBottom),
                    Rect(dstLeft, dstBottom, dstRight, roi.height),
                    paint
                )
            }
            canvas.drawBitmap(source, Rect(srcLeft, srcTop, srcRight, srcBottom), Rect(dstLeft, dstTop, dstRight, dstBottom), paint)
        }
        val cropTarget = if (roiBmp.width == targetW && roiBmp.height == targetH) {
            roiBmp
        } else {
            val scaled = Bitmap.createScaledBitmap(roiBmp, targetW, targetH, true)
            roiBmp.recycle()
            scaled
        }

        val maskTarget = FloatArray(targetW * targetH)
        val stepScaleX = roi.width / targetW.toFloat()
        val stepScaleY = roi.height / targetH.toFloat()
        for (y in 0 until targetH) {
            val origY0 = roi.top + (y * stepScaleY).toInt()
            val origY1 = max(origY0, roi.top + ((y + 1) * stepScaleY).toInt() - 1)
            if (origY1 < 0 || origY0 >= origH) continue
            val clampedY0 = origY0.coerceIn(0, origH - 1)
            val clampedY1 = origY1.coerceIn(0, origH - 1)
            val dstRow = y * targetW
            for (x in 0 until targetW) {
                val origX0 = roi.left + (x * stepScaleX).toInt()
                val origX1 = max(origX0, roi.left + ((x + 1) * stepScaleX).toInt() - 1)
                if (origX1 < 0 || origX0 >= origW) continue
                val clampedX0 = origX0.coerceIn(0, origW - 1)
                val clampedX1 = origX1.coerceIn(0, origW - 1)
                var maxVal = 0f
                for (oy in clampedY0..clampedY1) {
                    val srcRow = oy * origW
                    for (ox in clampedX0..clampedX1) {
                        val v = fullMaskAlpha[srcRow + ox]
                        if (v > maxVal) maxVal = v
                    }
                }
                maskTarget[dstRow + x] = if (maxVal >= 0.20f) 1.0f else 0.0f
            }
        }
        return cropTarget to maskTarget
    }

    internal fun dilateBinaryMask(
        mask: FloatArray,
        width: Int,
        height: Int,
        radiusPx: Int
    ): FloatArray {
        if (radiusPx <= 0 || mask.isEmpty()) return mask.copyOf()
        val horiz = FloatArray(width * height)
        for (y in 0 until height) {
            val rowOff = y * width
            for (x in 0 until width) {
                if (mask[rowOff + x] >= 0.5f) {
                    val x0 = max(0, x - radiusPx)
                    val x1 = min(width - 1, x + radiusPx)
                    for (nx in x0..x1) {
                        horiz[rowOff + nx] = 1.0f
                    }
                }
            }
        }
        val out = FloatArray(width * height)
        for (y in 0 until height) {
            val rowOff = y * width
            for (x in 0 until width) {
                if (horiz[rowOff + x] >= 0.5f) {
                    val y0 = max(0, y - radiusPx)
                    val y1 = min(height - 1, y + radiusPx)
                    for (ny in y0..y1) {
                        out[ny * width + x] = 1.0f
                    }
                }
            }
        }
        return out
    }

    internal fun downsampleMaskToLatent(
        mask: FloatArray,
        targetW: Int,
        targetH: Int,
        latentW: Int,
        latentH: Int
    ): FloatArray {
        val out = FloatArray(latentW * latentH)
        val blockX = (targetW / latentW).coerceAtLeast(1)
        val blockY = (targetH / latentH).coerceAtLeast(1)
        val blockArea = (blockX * blockY).toFloat()
        for (ly in 0 until latentH) {
            for (lx in 0 until latentW) {
                var sum = 0f
                val y0 = ly * blockY
                val x0 = lx * blockX
                for (dy in 0 until blockY) {
                    val row = (y0 + dy) * targetW + x0
                    for (dx in 0 until blockX) {
                        sum += mask[row + dx]
                    }
                }
                out[ly * latentW + lx] = if ((sum / blockArea) >= 0.30f) 1.0f else 0.0f
            }
        }
        return out
    }

    private fun extractClipHiddenStateTensor(res: OrtSession.Result): OnnxTensor {
        for (i in 0 until res.size()) {
            val t = res[i] as? OnnxTensor ?: continue
            val shape = t.info?.shape ?: continue
            if (shape.size == 3 && shape[2] == 768L) {
                return t
            }
        }
        return res[0] as OnnxTensor
    }

    private fun runTextEncoder(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        uncondIds: LongArray,
        condIds: LongArray
    ): Pair<FloatArray, FloatArray> {
        val inputName = session.inputNames.first()
        val inputType = (session.inputInfo[inputName]?.info as? TensorInfo)?.type ?: OnnxJavaType.INT32
        val shape = longArrayOf(1L, ClipBpeTokenizer.MAX_LENGTH.toLong())

        val uncondOut = createIdsTensor(ortEnv, uncondIds, shape, inputType).use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                readTensorAsFloats(extractClipHiddenStateTensor(res))
            }
        }
        val condOut = createIdsTensor(ortEnv, condIds, shape, inputType).use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                readTensorAsFloats(extractClipHiddenStateTensor(res))
            }
        }
        return uncondOut to condOut
    }

    private fun runVaeEncoder(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        cropBmp: Bitmap,
        maskTarget: FloatArray,
        targetW: Int,
        targetH: Int,
        applyZeroMask: Boolean
    ): FloatArray {
        val hw = targetW * targetH
        val latentHW = (targetW / 8) * (targetH / 8)
        val px = IntArray(hw)
        cropBmp.getPixels(px, 0, targetW, 0, 0, targetW, targetH)

        val nchw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            val c = px[i]
            val keep = if (applyZeroMask && maskTarget[i] >= 0.5f) 0.0f else 1.0f
            val r = (((c ushr 16) and 0xFF) / 127.5f - 1.0f) * keep
            val g = (((c ushr 8) and 0xFF) / 127.5f - 1.0f) * keep
            val b = ((c and 0xFF) / 127.5f - 1.0f) * keep
            nchw[i] = r
            nchw[hw + i] = g
            nchw[2 * hw + i] = b
        }

        val inputName = session.inputNames.first()
        val inputType = (session.inputInfo[inputName]?.info as? TensorInfo)?.type ?: OnnxJavaType.FLOAT
        val shape = longArrayOf(1L, 3L, targetH.toLong(), targetW.toLong())

        return createFloatOrHalfTensor(ortEnv, nchw, shape, inputType).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { res ->
                val latent = readTensorAsFloats(res[0] as OnnxTensor)
                val expectedSize = LATENT_CHANNELS * latentHW
                val out = FloatArray(expectedSize)
                for (i in 0 until minOf(expectedSize, latent.size)) {
                    out[i] = latent[i] * VAE_SCALE_FACTOR
                }
                out
            }
        }
    }

    private fun runUnetDenoisingLoop(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        uncondEmbed: FloatArray,
        condEmbed: FloatArray,
        maskedLatent: FloatArray,
        initLatent: FloatArray,
        maskLatent: FloatArray,
        latentW: Int,
        latentH: Int,
        plan: DenoisePlan,
        guidanceScale: Float,
        fastMode: Boolean = true,
        seed: Long,
        isCancelled: () -> Boolean = { false },
        onStep: (Int, Int) -> Unit
    ): FloatArray {
        val hwLatent = latentW * latentH
        val latentSize = LATENT_CHANNELS * hwLatent

        val schedule = buildEulerSchedule(plan.baseSteps)
        val rng = Random(seed)
        val startSigma = schedule.sigmas[plan.startStepIndex]
        val latents = if (plan.startStepIndex == 0) {
            FloatArray(latentSize) { sampleGaussian(rng) * startSigma }
        } else {
            FloatArray(latentSize) { i -> initLatent[i] + sampleGaussian(rng) * startSigma }
        }

        val names = session.inputNames.toList()
        val sampleName = names.firstOrNull { it.contains("sample", ignoreCase = true) } ?: names[0]
        val timestepName = names.firstOrNull { it.contains("timestep", ignoreCase = true) } ?: names.getOrElse(1) { names[0] }
        val encoderName = names.firstOrNull {
            it.contains("encoder", ignoreCase = true) || it.contains("hidden", ignoreCase = true)
        } ?: names.getOrElse(2) { names.last() }

        val sampleInfo = session.inputInfo[sampleName]?.info as? TensorInfo
        val timestepInfo = session.inputInfo[timestepName]?.info as? TensorInfo
        val encoderInfo = session.inputInfo[encoderName]?.info as? TensorInfo

        val sampleType = sampleInfo?.type ?: OnnxJavaType.FLOAT
        val timestepType = timestepInfo?.type ?: OnnxJavaType.FLOAT
        val encoderType = encoderInfo?.type ?: OnnxJavaType.FLOAT
        val sampleChannels = sampleInfo?.shape?.getOrNull(1)?.toInt()?.takeIf { it > 0 } ?: 9
        val forcedBatch2 = sampleInfo?.shape?.getOrNull(0) == 2L || encoderInfo?.shape?.getOrNull(0) == 2L
        val batchSize = if (forcedBatch2) 2 else 1
        val timestepShape = resolveTimestepShape(timestepInfo?.shape, batchSize = batchSize)
        val embedTokenCount = (uncondEmbed.size / 768).coerceAtLeast(1).toLong()
        logDebug(
            "UNet channels=$sampleChannels batchSize=$batchSize latent=${latentW}x${latentH} " +
                "startIdx=${plan.startStepIndex}/${plan.baseSteps} denoiseSteps=${plan.denoiseSteps} " +
                "sampleType=$sampleType timestepType=$timestepType encoderType=$encoderType fastMode=$fastMode"
        )

        val sampleStride = sampleChannels * hwLatent
        // Reusable direct ByteBuffer for FP16 sample inputs to avoid per-step native direct buffer allocations
        val reusableSampleHalfBuf: ByteBuffer? = if (sampleType == OnnxJavaType.FLOAT16) {
            ByteBuffer.allocateDirect(batchSize * sampleStride * 2).order(ByteOrder.nativeOrder())
        } else {
            null
        }
        val reusableSampleFloatBuf: FloatBuffer? = if (sampleType != OnnxJavaType.FLOAT16) {
            FloatBuffer.allocate(batchSize * sampleStride)
        } else {
            null
        }

        if (batchSize == 2) {
            val bothEmbeds = FloatArray(uncondEmbed.size + condEmbed.size)
            System.arraycopy(uncondEmbed, 0, bothEmbeds, 0, uncondEmbed.size)
            System.arraycopy(condEmbed, 0, bothEmbeds, uncondEmbed.size, condEmbed.size)
            val encoderShape2 = longArrayOf(2L, embedTokenCount, 768L)
            val sampleShape2 = longArrayOf(2L, sampleChannels.toLong(), latentH.toLong(), latentW.toLong())
            val sampleInput2 = FloatArray(2 * sampleStride)

            createFloatOrHalfTensor(ortEnv, bothEmbeds, encoderShape2, encoderType).use { bothEncTensor ->
                for ((localStep, stepIdx) in (plan.startStepIndex until plan.baseSteps).withIndex()) {
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Diffusion cancelled by user.")
                    onStep(localStep + 1, plan.denoiseSteps)
                    val sigma = schedule.sigmas[stepIdx]
                    val nextSigma = schedule.sigmas[stepIdx + 1]
                    val timestep = schedule.timesteps[stepIdx]
                    val invScale = 1.0f / sqrt(sigma * sigma + 1.0f)

                    if (sampleChannels >= 9) {
                        for (i in 0 until latentSize) sampleInput2[i] = latents[i] * invScale
                        for (i in 0 until hwLatent) sampleInput2[latentSize + i] = maskLatent[i]
                        val offset = latentSize + hwLatent
                        for (i in 0 until latentSize) sampleInput2[offset + i] = maskedLatent[i]
                    } else {
                        for (i in 0 until latentSize) sampleInput2[i] = latents[i] * invScale
                    }
                    System.arraycopy(sampleInput2, 0, sampleInput2, sampleStride, sampleStride)

                    createFloatOrHalfTensor(
                        ortEnv,
                        sampleInput2,
                        sampleShape2,
                        sampleType,
                        reusableHalfBuf = reusableSampleHalfBuf,
                        reusableFloatBuf = reusableSampleFloatBuf
                    ).use { sTensor ->
                        createTimestepTensor(ortEnv, timestep, timestepShape, timestepType).use { tTensor ->
                            val bothPred = session.run(
                                mapOf(
                                    sampleName to sTensor,
                                    timestepName to tTensor,
                                    encoderName to bothEncTensor
                                )
                            ).use { res ->
                                readTensorAsFloats(res[0] as OnnxTensor)
                            }

                            val dt = nextSigma - sigma
                            for (i in 0 until latentSize) {
                                val u = bothPred[i]
                                val c = bothPred[latentSize + i]
                                val noisePred = if (u.isFinite() && c.isFinite()) {
                                    u + guidanceScale * (c - u)
                                } else {
                                    0f
                                }
                                val updated = latents[i] + noisePred * dt
                                latents[i] = if (updated.isFinite()) updated.coerceIn(-25f, 25f) else latents[i]
                            }

                            if (sampleChannels < 9) {
                                for (c in 0 until LATENT_CHANNELS) {
                                    val cOff = c * hwLatent
                                    for (p in 0 until hwLatent) {
                                        if (maskLatent[p] < 0.5f) {
                                            latents[cOff + p] = initLatent[cOff + p] + sampleGaussian(rng) * nextSigma
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            val encoderShape = longArrayOf(1L, embedTokenCount, 768L)
            val sampleShape = longArrayOf(1L, sampleChannels.toLong(), latentH.toLong(), latentW.toLong())
            val sampleInput = FloatArray(sampleStride)
            val needsAnyCfg = guidanceScale > 1.05f
            var lastUncondPred: FloatArray? = null

            createFloatOrHalfTensor(ortEnv, condEmbed, encoderShape, encoderType).use { condTensor ->
                val uncondTensor: OnnxTensor? = if (needsAnyCfg) {
                    createFloatOrHalfTensor(ortEnv, uncondEmbed, encoderShape, encoderType)
                } else {
                    null
                }
                try {
                    for ((localStep, stepIdx) in (plan.startStepIndex until plan.baseSteps).withIndex()) {
                        if (isCancelled()) throw java.util.concurrent.CancellationException("Diffusion cancelled by user.")
                        onStep(localStep + 1, plan.denoiseSteps)
                        val sigma = schedule.sigmas[stepIdx]
                        val nextSigma = schedule.sigmas[stepIdx + 1]
                        val timestep = schedule.timesteps[stepIdx]
                        val invScale = 1.0f / sqrt(sigma * sigma + 1.0f)

                        if (sampleChannels >= 9) {
                            for (i in 0 until latentSize) {
                                sampleInput[i] = latents[i] * invScale
                            }
                            for (i in 0 until hwLatent) {
                                sampleInput[latentSize + i] = maskLatent[i]
                            }
                            val offset = latentSize + hwLatent
                            for (i in 0 until latentSize) {
                                sampleInput[offset + i] = maskedLatent[i]
                            }
                        } else {
                            for (i in 0 until latentSize) {
                                sampleInput[i] = latents[i] * invScale
                            }
                        }

                        createFloatOrHalfTensor(
                            ortEnv,
                            sampleInput,
                            sampleShape,
                            sampleType,
                            reusableHalfBuf = reusableSampleHalfBuf,
                            reusableFloatBuf = reusableSampleFloatBuf
                        ).use { sTensor ->
                            createTimestepTensor(ortEnv, timestep, timestepShape, timestepType).use { tTensor ->
                                val runUncond = uncondTensor != null && shouldRunUncondCfgPass(
                                    localStep = localStep,
                                    totalSteps = plan.denoiseSteps,
                                    guidanceScale = guidanceScale,
                                    fastMode = fastMode,
                                    hasCachedUncond = lastUncondPred != null
                                )
                                val uncondPred: FloatArray? = if (runUncond && uncondTensor != null) {
                                    session.run(
                                        mapOf(
                                            sampleName to sTensor,
                                            timestepName to tTensor,
                                            encoderName to uncondTensor
                                        )
                                    ).use { res ->
                                        readTensorAsFloats(res[0] as OnnxTensor)
                                    }.also { lastUncondPred = it }
                                } else {
                                    lastUncondPred
                                }

                                if (isCancelled()) throw java.util.concurrent.CancellationException("Diffusion cancelled by user.")

                                val condPred = session.run(
                                    mapOf(
                                        sampleName to sTensor,
                                        timestepName to tTensor,
                                        encoderName to condTensor
                                    )
                                ).use { res ->
                                    readTensorAsFloats(res[0] as OnnxTensor)
                                }

                                val dt = nextSigma - sigma
                                if (!needsAnyCfg || uncondPred == null) {
                                    for (i in 0 until latentSize) {
                                        val c = condPred[i]
                                        val noisePred = if (c.isFinite()) c else 0f
                                        val updated = latents[i] + noisePred * dt
                                        latents[i] = if (updated.isFinite()) updated.coerceIn(-25f, 25f) else latents[i]
                                    }
                                } else {
                                    for (i in 0 until latentSize) {
                                        val u = uncondPred[i]
                                        val c = condPred[i]
                                        val noisePred = if (u.isFinite() && c.isFinite()) {
                                            u + guidanceScale * (c - u)
                                        } else {
                                            0f
                                        }
                                        val updated = latents[i] + noisePred * dt
                                        latents[i] = if (updated.isFinite()) updated.coerceIn(-25f, 25f) else latents[i]
                                    }
                                }

                                if (sampleChannels < 9) {
                                    for (c in 0 until LATENT_CHANNELS) {
                                        val cOff = c * hwLatent
                                        for (p in 0 until hwLatent) {
                                            if (maskLatent[p] < 0.5f) {
                                                latents[cOff + p] = initLatent[cOff + p] + sampleGaussian(rng) * nextSigma
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } finally {
                    runCatching { uncondTensor?.close() }
                }
            }
        }
        return latents
    }

    private fun runVaeDecoder(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        latent: FloatArray,
        targetW: Int,
        targetH: Int
    ): Bitmap {
        val scaledLatent = FloatArray(latent.size) { i -> latent[i] / VAE_SCALE_FACTOR }
        val latentW = targetW / 8
        val latentH = targetH / 8
        val inputName = session.inputNames.first()
        val inputType = (session.inputInfo[inputName]?.info as? TensorInfo)?.type ?: OnnxJavaType.FLOAT
        val shape = longArrayOf(1L, LATENT_CHANNELS.toLong(), latentH.toLong(), latentW.toLong())

        val hw = targetW * targetH
        val pixels = IntArray(hw)
        createFloatOrHalfTensor(ortEnv, scaledLatent, shape, inputType).use { lTensor ->
            session.run(mapOf(inputName to lTensor)).use { res ->
                val rgbFloats = readTensorAsFloats(res[0] as OnnxTensor)
                if (rgbFloats.size < 3 * hw) {
                    throw IllegalStateException(
                        "VAE decoder output size mismatch: expected >= ${3 * hw}, got ${rgbFloats.size} for ${targetW}x${targetH}."
                    )
                }
                for (i in 0 until hw) {
                    val r = ((rgbFloats[i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                    val g = ((rgbFloats[hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                    val b = ((rgbFloats[2 * hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                    pixels[i] = Color.rgb(r, g, b)
                }
            }
        }
        return Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888).apply {
            setPixels(pixels, 0, targetW, 0, 0, targetW, targetH)
        }
    }

    private fun pasteCropDirectDebug(
        sourceBitmap: Bitmap,
        generatedCrop: Bitmap,
        roi: AspectRoi
    ): Bitmap {
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height
        val cropFull = if (roi.width == generatedCrop.width && roi.height == generatedCrop.height) {
            generatedCrop
        } else {
            Bitmap.createScaledBitmap(generatedCrop, roi.width, roi.height, true)
        }
        val out = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val intersectLeft = max(0, roi.left)
        val intersectTop = max(0, roi.top)
        val intersectRight = min(origW, roi.left + roi.width)
        val intersectBottom = min(origH, roi.top + roi.height)
        if (intersectRight > intersectLeft && intersectBottom > intersectTop) {
            val srcRect = Rect(
                intersectLeft - roi.left,
                intersectTop - roi.top,
                intersectRight - roi.left,
                intersectBottom - roi.top
            )
            val dstRect = Rect(intersectLeft, intersectTop, intersectRight, intersectBottom)
            canvas.drawBitmap(cropFull, srcRect, dstRect, Paint(Paint.FILTER_BITMAP_FLAG))
        }
        if (cropFull !== generatedCrop) cropFull.recycle()
        return out
    }

    private fun compositeCropIntoSource(
        sourceBitmap: Bitmap,
        fullMaskAlpha: FloatArray,
        generatedCrop: Bitmap,
        roi: AspectRoi,
        featherRadius: Int
    ): Bitmap {
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height

        val cropFull = if (roi.width == generatedCrop.width && roi.height == generatedCrop.height) {
            generatedCrop
        } else {
            Bitmap.createScaledBitmap(generatedCrop, roi.width, roi.height, true)
        }

        val fullGenerated = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val genCanvas = Canvas(fullGenerated)
        val intersectLeft = max(0, roi.left)
        val intersectTop = max(0, roi.top)
        val intersectRight = min(origW, roi.left + roi.width)
        val intersectBottom = min(origH, roi.top + roi.height)
        if (intersectRight > intersectLeft && intersectBottom > intersectTop) {
            val srcRect = Rect(
                intersectLeft - roi.left,
                intersectTop - roi.top,
                intersectRight - roi.left,
                intersectBottom - roi.top
            )
            val dstRect = Rect(intersectLeft, intersectTop, intersectRight, intersectBottom)
            genCanvas.drawBitmap(cropFull, srcRect, dstRect, Paint(Paint.FILTER_BITMAP_FLAG))
        }
        if (cropFull !== generatedCrop) cropFull.recycle()

        val total = origW * origH
        val srcPx = IntArray(total)
        val genPx = IntArray(total)
        sourceBitmap.getPixels(srcPx, 0, origW, 0, 0, origW, origH)
        fullGenerated.getPixels(genPx, 0, origW, 0, 0, origW, origH)
        fullGenerated.recycle()

        val feathered = featherMask(fullMaskAlpha, origW, origH, featherRadius)
        val outPx = IntArray(total)
        for (i in 0 until total) {
            val m = feathered[i]
            if (m <= 0.005f) {
                outPx[i] = srcPx[i]
            } else if (m >= 0.995f) {
                outPx[i] = genPx[i]
            } else {
                val s = srcPx[i]
                val p = genPx[i]
                val r = (((s ushr 16) and 0xFF) * (1f - m) + ((p ushr 16) and 0xFF) * m).roundToInt().coerceIn(0, 255)
                val g = (((s ushr 8) and 0xFF) * (1f - m) + ((p ushr 8) and 0xFF) * m).roundToInt().coerceIn(0, 255)
                val b = ((s and 0xFF) * (1f - m) + (p and 0xFF) * m).roundToInt().coerceIn(0, 255)
                outPx[i] = Color.rgb(r, g, b)
            }
        }
        return Bitmap.createBitmap(origW, origH, Bitmap.Config.ARGB_8888).apply {
            setPixels(outPx, 0, origW, 0, 0, origW, origH)
        }
    }

    internal fun normalizePromptForClip(rawPrompt: String): String {
        val trimmed = rawPrompt.trim()
        if (trimmed.isEmpty()) return ""
        return trimmed.replace(
            Regex("\\b(?:sharee|shari|saari|sari)s?\\b", RegexOption.IGNORE_CASE)
        ) { match ->
            if (match.value.endsWith("s", ignoreCase = true) && !match.value.equals("sari", ignoreCase = true)) {
                "sarees"
            } else {
                "saree"
            }
        }
    }

    internal fun featherMask(mask: FloatArray, w: Int, h: Int, radius: Int): FloatArray {
        if (radius <= 0) {
            return FloatArray(mask.size) { i -> mask[i].coerceIn(0f, 1f) }
        }
        val temp = FloatArray(w * h)
        val out = FloatArray(w * h)
        val denom = (2 * radius + 1).toFloat()
        for (y in 0 until h) {
            val rowOff = y * w
            for (x in 0 until w) {
                var sum = 0f
                for (dx in -radius..radius) {
                    sum += mask[rowOff + (x + dx).coerceIn(0, w - 1)]
                }
                temp[rowOff + x] = sum / denom
            }
        }
        for (y in 0 until h) {
            val rowOff = y * w
            for (x in 0 until w) {
                var sum = 0f
                for (dy in -radius..radius) {
                    sum += temp[(y + dy).coerceIn(0, h - 1) * w + x]
                }
                val orig = mask[rowOff + x].coerceIn(0f, 1f)
                // Strictly prevent outward bleed outside painted mask while feathering inner boundary
                out[rowOff + x] = if (orig <= 0.05f) {
                    0f
                } else {
                    min(orig, sum / denom).coerceIn(0f, 1f)
                }
            }
        }
        return out
    }

    internal data class EulerSchedule(
        val timesteps: FloatArray,
        val sigmas: FloatArray
    )

    internal fun buildEulerSchedule(numSteps: Int): EulerSchedule {
        val trainSteps = 1000
        val betaStartSqrt = sqrt(0.00085)
        val betaEndSqrt = sqrt(0.012)
        val alphasCumprod = DoubleArray(trainSteps)
        var cum = 1.0
        for (i in 0 until trainSteps) {
            val bSqrt = betaStartSqrt + (betaEndSqrt - betaStartSqrt) * (i.toDouble() / (trainSteps - 1))
            val alpha = 1.0 - bSqrt * bSqrt
            cum *= alpha
            alphasCumprod[i] = cum
        }

        val timesteps = FloatArray(numSteps)
        val sigmas = FloatArray(numSteps + 1)
        for (i in 0 until numSteps) {
            val tFloat = if (numSteps == 1) 999.0 else 999.0 * (1.0 - i.toDouble() / (numSteps - 1))
            timesteps[i] = tFloat.toFloat()
            val idxLow = tFloat.toInt().coerceIn(0, trainSteps - 1)
            val idxHigh = (idxLow + 1).coerceIn(0, trainSteps - 1)
            val frac = tFloat - idxLow
            val ab = alphasCumprod[idxLow] * (1.0 - frac) + alphasCumprod[idxHigh] * frac
            sigmas[i] = sqrt((1.0 - ab) / ab.coerceAtLeast(1e-8)).toFloat()
        }
        sigmas[numSteps] = 0.0f
        return EulerSchedule(timesteps, sigmas)
    }

    private fun sampleGaussian(rng: Random): Float {
        val u1 = rng.nextDouble().coerceIn(1e-7, 1.0 - 1e-7)
        val u2 = rng.nextDouble()
        return (sqrt(-2.0 * ln(u1)) * cos(2.0 * Math.PI * u2)).toFloat()
    }

    internal fun resolveTimestepShape(modelShape: LongArray?, batchSize: Int): LongArray {
        if (modelShape == null || modelShape.isEmpty()) return longArrayOf()
        val dim0 = modelShape[0]
        if (dim0 > 0L) return longArrayOf(dim0)
        return longArrayOf(batchSize.toLong())
    }

    private fun createIdsTensor(
        env: OrtEnvironment,
        ids: LongArray,
        shape: LongArray,
        type: OnnxJavaType
    ): OnnxTensor {
        return when (type) {
            OnnxJavaType.INT32 -> {
                val intArr = IntArray(ids.size) { ids[it].toInt() }
                OnnxTensor.createTensor(env, IntBuffer.wrap(intArr), shape)
            }
            else -> OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
        }
    }

    private fun createTimestepTensor(
        env: OrtEnvironment,
        value: Float,
        shape: LongArray,
        type: OnnxJavaType
    ): OnnxTensor {
        val count = if (shape.isEmpty()) 1 else shape.fold(1) { acc, dim -> acc * dim.toInt().coerceAtLeast(1) }
        val floats = FloatArray(count) { value }
        return when (type) {
            OnnxJavaType.INT64 -> OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(count) { value.roundToInt().toLong() }),
                shape
            )
            OnnxJavaType.INT32 -> OnnxTensor.createTensor(
                env,
                IntBuffer.wrap(IntArray(count) { value.roundToInt() }),
                shape
            )
            else -> createFloatOrHalfTensor(env, floats, shape, type)
        }
    }

    private fun createFloatOrHalfTensor(
        env: OrtEnvironment,
        data: FloatArray,
        shape: LongArray,
        type: OnnxJavaType,
        reusableHalfBuf: ByteBuffer? = null,
        reusableFloatBuf: FloatBuffer? = null
    ): OnnxTensor {
        if (type == OnnxJavaType.FLOAT16) {
            val neededBytes = data.size * 2
            val byteBuf = if (reusableHalfBuf != null && reusableHalfBuf.capacity() >= neededBytes) {
                reusableHalfBuf.clear()
                reusableHalfBuf.limit(neededBytes)
                reusableHalfBuf
            } else {
                ByteBuffer.allocateDirect(neededBytes).order(ByteOrder.nativeOrder())
            }
            val shortView = byteBuf.asShortBuffer()
            for (i in data.indices) {
                shortView.put(floatToHalfBits(data[i]))
            }
            byteBuf.rewind()
            return OnnxTensor.createTensor(env, byteBuf, shape, OnnxJavaType.FLOAT16)
        }
        if (reusableFloatBuf != null && reusableFloatBuf.capacity() >= data.size) {
            reusableFloatBuf.clear()
            reusableFloatBuf.put(data)
            reusableFloatBuf.flip()
            return OnnxTensor.createTensor(env, reusableFloatBuf, shape)
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
    }

    private fun readTensorAsFloats(tensor: OnnxTensor): FloatArray {
        return when (tensor.info.type) {
            OnnxJavaType.FLOAT16 -> {
                val shortBuf = tensor.shortBuffer
                val shorts = ShortArray(shortBuf.remaining())
                shortBuf.get(shorts)
                FloatArray(shorts.size) { i ->
                    val v = halfBitsToFloat(shorts[i])
                    if (v.isFinite()) v else 0f
                }
            }
            else -> {
                val fb = tensor.floatBuffer
                val floats = FloatArray(fb.remaining())
                fb.get(floats)
                for (i in floats.indices) {
                    if (!floats[i].isFinite()) floats[i] = 0f
                }
                floats
            }
        }
    }

    /**
     * IEEE-754 binary16 (FP16) encoder compatible with API 24+ and JVM unit tests.
     */
    internal fun floatToHalfBits(value: Float): Short {
        val bits = java.lang.Float.floatToIntBits(value)
        val sign = (bits ushr 16) and 0x8000
        val exp = (bits ushr 23) and 0xFF
        val mant = bits and 0x7FFFFF

        return when {
            exp == 0xFF -> {
                // NaN or Infinity
                (sign or 0x7C00 or if (mant != 0) 0x0200 else 0).toShort()
            }
            exp > 142 -> {
                // Overflow -> clamp to max finite FP16 (65504.0) to avoid NaN diffusion spikes
                (sign or 0x7BFF).toShort()
            }
            exp < 103 -> {
                // Too small -> zero
                sign.toShort()
            }
            exp < 113 -> {
                // Subnormal FP16
                val m = mant or 0x800000
                val shift = 113 - exp + 13
                (sign or (m ushr shift)).toShort()
            }
            else -> {
                val newExp = (exp - 112) shl 10
                val newMant = mant ushr 13
                (sign or newExp or newMant).toShort()
            }
        }
    }

    /**
     * IEEE-754 binary16 (FP16) decoder compatible with API 24+ and JVM unit tests.
     */
    internal fun halfBitsToFloat(halfBits: Short): Float {
        val h = halfBits.toInt() and 0xFFFF
        val sign = (h and 0x8000) shl 16
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x03FF

        val outBits = when (exp) {
            0 -> {
                if (mant == 0) {
                    sign
                } else {
                    // Subnormal
                    var m = mant
                    var e = 127 - 14
                    while ((m and 0x0400) == 0) {
                        m = m shl 1
                        e--
                    }
                    m = m and 0x03FF
                    sign or (e shl 23) or (m shl 13)
                }
            }
            31 -> {
                sign or 0x7F800000 or (mant shl 13)
            }
            else -> {
                val newExp = exp + (127 - 15)
                sign or (newExp shl 23) or (mant shl 13)
            }
        }
        return java.lang.Float.intBitsToFloat(outBits)
    }
}
