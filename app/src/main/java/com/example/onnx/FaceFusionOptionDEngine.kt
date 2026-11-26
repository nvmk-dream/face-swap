package com.example.onnx

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import java.io.File
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.CancellationException
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

enum class OptionDSwapperChoice(
    val id: String,
    val title: String,
    val shortBadge: String
) {
    AUTO("auto", "Auto", "Best Installed"),
    HYPERSWAP("hyperswap", "HyperSwap 256", "256px"),
    SIMSWAP_512("simswap_512", "SimSwap 512", "512px"),
    RESWAPPER_256("reswapper_256", "ReSwapper 256", "256px"),
    INSWAPPER_128("inswapper_128", "InSwapper 128", "128px")
}

enum class OptionDEnhancerChoice(
    val id: String,
    val title: String,
    val shortBadge: String
) {
    AUTO("auto", "Auto", "Best Installed"),
    CODEFORMER("codeformer", "CodeFormer (w)", "w Slider"),
    GFPGAN("gfpgan", "GFPGAN 1.4", "512px"),
    GPEN_BFR("gpen_bfr", "GPEN-BFR", "256–1024"),
    RESTOREFORMER("restoreformer", "RestoreFormer++", "512px")
}

data class OptionDFaceFusionConfig(
    val swapperChoice: OptionDSwapperChoice = OptionDSwapperChoice.AUTO,
    val enhancerChoice: OptionDEnhancerChoice = OptionDEnhancerChoice.AUTO,
    val codeformerFidelity: Float = 0.60f,
    val pixelBoostEnabled: Boolean = false
)

/**
 * FaceFusion (Offline Kotlin Port) scoped exclusively to Option D (`OPTION_D_CRANIOFACIAL_FIT`).
 * Options A, B, and B+C do NOT call into this engine.
 *
 * Implements:
 *  1. Multi-variant model selection (`SimSwap 512/256`, `ReSwapper 256`, `HyperSwap 256`,
 *     `BlendSwap/Ghost 256`, `InSwapper 128`) with ImageNet / [-1..1] / [0..1], NCHW/NHWC, FP16/FP32.
 *  2. FaceFusion `Pixel Boost` sub-pixel interleaved phase shifting (`implodePixelBoost` / `explodePixelBoost`).
 *  3. `CodeFormer.onnx` (`w` scalar/1D DOUBLE/FLOAT/FLOAT16 input) + `GPEN-BFR 256/512/1024` +
 *     `RestoreFormer++` + `GFPGAN 1.4`.
 *  4. `face_occluder.onnx` / `dfl_xseg.onnx` / `xseg_1/2/3.onnx` 256x256 foreground occlusion gate.
 */
object FaceFusionOptionDEngine {

    internal enum class InputNormMode {
        SIGNED_MINUS1_TO_1,
        UNSIGNED_0_TO_1,
        IMAGENET
    }

    internal enum class ChannelLayout {
        RGB,
        BGR
    }

    internal data class SwapperInputSpec(
        val channelLayout: ChannelLayout,
        val normMode: InputNormMode
    ) {
        fun label(): String = "${channelLayout.name}_${normMode.name}"
    }

    data class OptionDSwapCropResult(
        val cropBitmap: Bitmap,
        val forwardMatrix128: FloatArray,
        val activeSwapFile: File,
        val modelCropSize: Int,
        val effectiveOutputSize: Int,
        val usedPixelBoost: Boolean,
        val usedNeuralSwap: Boolean,
        val diagnosticSummary: String
    )

    data class OptionDEnhancerResult(
        val restored512: Bitmap,
        val usedEnhancerFileName: String,
        val isCodeFormer: Boolean,
        val appliedFidelityW: Float
    )

    private val verifiedOptionDSpecCache = ConcurrentHashMap<String, SwapperInputSpec>()

    fun listInstalledSwappers(context: Context): List<File> =
        OnnxProtobufInspector.listSlotVariantFiles(context, ModelSlot.SWAPPER)

    fun listInstalledEnhancers(context: Context): List<File> =
        OnnxProtobufInspector.listSlotVariantFiles(context, ModelSlot.ENHANCEMENT)

    fun resolveOccluderFile(context: Context): File? {
        val primary = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.FACE_OCCLUDER)
        if (primary.exists() && primary.length() > 1024L) return primary
        return OnnxProtobufInspector.listSlotVariantFiles(context, ModelSlot.FACE_OCCLUDER).firstOrNull()
    }

    fun resolveSimSwapEmbeddingConverterFile(context: Context): File? {
        val root = OnnxProtobufInspector.getModelsRootDir(context)
        val searchDirs = listOf(
            File(root, ModelSlot.RECOGNIZER.subdirectory),
            File(root, ModelSlot.SWAPPER.subdirectory),
            root
        )
        for (dir in searchDirs) {
            if (!dir.exists() || !dir.isDirectory) continue
            val exact = File(dir, "crossface_simswap.onnx")
            if (exact.isFile && exact.length() > 1024L) return exact
            val fuzzy = dir.listFiles()?.firstOrNull { f ->
                val l = f.name.lowercase()
                f.isFile && f.length() > 1024L && l.endsWith(".onnx") &&
                    l.contains("crossface") && l.contains("simswap")
            }
            if (fuzzy != null) return fuzzy
        }
        return null
    }

    fun isFileMatchingSwapperChoice(file: File, choice: OptionDSwapperChoice): Boolean {
        val lower = file.name.lowercase()
        return when (choice) {
            OptionDSwapperChoice.AUTO -> true
            OptionDSwapperChoice.HYPERSWAP -> lower.contains("hyperswap")
            OptionDSwapperChoice.SIMSWAP_512 -> lower.contains("simswap")
            OptionDSwapperChoice.RESWAPPER_256 ->
                lower.contains("reswapper") || lower.contains("blendswap") || lower.contains("ghost_") || lower.contains("uniface")
            OptionDSwapperChoice.INSWAPPER_128 -> lower.contains("inswapper")
        }
    }

    fun isFileMatchingEnhancerChoice(file: File, choice: OptionDEnhancerChoice): Boolean {
        val lower = file.name.lowercase()
        return when (choice) {
            OptionDEnhancerChoice.AUTO -> true
            OptionDEnhancerChoice.CODEFORMER -> lower.contains("codeformer")
            OptionDEnhancerChoice.GFPGAN -> lower.contains("gfpgan")
            OptionDEnhancerChoice.GPEN_BFR -> lower.contains("gpen")
            OptionDEnhancerChoice.RESTOREFORMER -> lower.contains("restoreformer")
        }
    }

    fun resolveSwapperFile(context: Context, choice: OptionDSwapperChoice): File {
        val installed = listInstalledSwappers(context)
        if (choice != OptionDSwapperChoice.AUTO) {
            val matched = installed
                .filter { isFileMatchingSwapperChoice(it, choice) }
                .maxByOrNull { it.length() }
            if (matched != null) return matched
        }
        // AUTO in Option D prioritizes SimSwap 512 > HyperSwap 256 > ReSwapper 256 > InSwapper 128
        val autoOrder = installed.maxByOrNull { file ->
            val l = file.name.lowercase()
            val completeBonus = if (file.length() >= 8L * 1024L * 1024L) 1000 else 0
            val modelScore = when {
                l.contains("simswap_unofficial_512") || l.contains("simswap_512") -> 95
                l.contains("hyperswap_1b") -> 90
                l.contains("hyperswap_1c") -> 88
                l.contains("hyperswap_1a") -> 86
                l.contains("hyperswap") -> 84
                l.contains("reswapper_256") -> 82
                l.contains("simswap_256") -> 80
                l.contains("blendswap") || l.contains("ghost_") || l.contains("uniface") -> 75
                l.contains("inswapper_128_fp16") -> 65
                l.contains("inswapper") -> 60
                else -> 50
            }
            completeBonus + modelScore
        }
        return autoOrder ?: OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SWAPPER)
    }

    fun resolveEnhancerFile(context: Context, choice: OptionDEnhancerChoice): File? {
        val installed = listInstalledEnhancers(context)
        if (choice != OptionDEnhancerChoice.AUTO) {
            val matched = installed
                .filter { isFileMatchingEnhancerChoice(it, choice) }
                .maxByOrNull { it.length() }
            if (matched != null) return matched
        }
        // AUTO in Option D prioritizes CodeFormer (when installed) > GPEN-BFR 512/1024 > GFPGAN 1.4 > RestoreFormer++
        return installed.maxByOrNull { file ->
            val l = file.name.lowercase()
            when {
                l.contains("codeformer") -> 95
                l.contains("gpen_bfr_512") || l.contains("gpen_bfr_1024") -> 90
                l.contains("gfpgan_1.4") || l.contains("gfpgan") -> 85
                l.contains("restoreformer") -> 80
                l.contains("gpen") -> 75
                else -> 50
            }
        }
    }

    /**
     * FaceFusion `implode_pixel_boost`: splits a `(boostSize x boostSize)` image into `factor * factor`
     * phase-shifted `(modelSize x modelSize)` sub-crops with zero quality loss.
     */
    fun implodePixelBoost(
        pixels: IntArray,
        boostSize: Int,
        modelSize: Int
    ): List<IntArray> {
        require(boostSize >= modelSize && modelSize > 0 && boostSize % modelSize == 0) {
            "boostSize ($boostSize) must be an integer multiple of modelSize ($modelSize)."
        }
        require(pixels.size >= boostSize * boostSize) {
            "Expected at least ${boostSize * boostSize} pixels, got ${pixels.size}."
        }
        val factor = boostSize / modelSize
        val tileLen = modelSize * modelSize
        val tiles = ArrayList<IntArray>(factor * factor)
        for (ry in 0 until factor) {
            for (rx in 0 until factor) {
                val tile = IntArray(tileLen)
                for (my in 0 until modelSize) {
                    val syOffset = (my * factor + ry) * boostSize + rx
                    val dstRow = my * modelSize
                    for (mx in 0 until modelSize) {
                        tile[dstRow + mx] = pixels[syOffset + mx * factor]
                    }
                }
                tiles.add(tile)
            }
        }
        return tiles
    }

    /**
     * FaceFusion `explode_pixel_boost`: interleaves `factor * factor` swapped `(modelSize x modelSize)`
     * sub-crops back into a crisp `(boostSize x boostSize)` image.
     */
    fun explodePixelBoost(
        tiles: List<IntArray>,
        boostSize: Int,
        modelSize: Int
    ): IntArray {
        require(boostSize >= modelSize && modelSize > 0 && boostSize % modelSize == 0) {
            "boostSize ($boostSize) must be an integer multiple of modelSize ($modelSize)."
        }
        val factor = boostSize / modelSize
        require(tiles.size == factor * factor) {
            "Expected ${factor * factor} tiles, got ${tiles.size}."
        }
        val out = IntArray(boostSize * boostSize)
        var tileIdx = 0
        for (ry in 0 until factor) {
            for (rx in 0 until factor) {
                val tile = tiles[tileIdx++]
                for (my in 0 until modelSize) {
                    val dstRow = (my * factor + ry) * boostSize + rx
                    val srcRow = my * modelSize
                    for (mx in 0 until modelSize) {
                        out[dstRow + mx * factor] = tile[srcRow + mx]
                    }
                }
            }
        }
        return out
    }

    private fun isSimSwapFile(file: File): Boolean =
        file.name.lowercase().contains("simswap")

    private fun isEmapProjectionRequired(file: File): Boolean {
        val lower = file.name.lowercase()
        return lower.contains("inswapper")
    }

    fun runOptionDSwapCrop(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        sourceFace: DetectedFace,
        targetBitmap: Bitmap,
        targetFace: DetectedFace,
        sourceEmbedding: SourceEmbeddingResult,
        config: OptionDFaceFusionConfig,
        preferHardwareAccel: Boolean = false,
        preloadedSwapSession: OrtSession? = null,
        isCancelled: () -> Boolean = { false }
    ): OptionDSwapCropResult {
        if (isCancelled()) throw CancellationException("Face swap cancelled by user.")

        val swapFile = resolveSwapperFile(context, config.swapperChoice)
        var modelCropSize = inferCropSizeFromFileName(swapFile)

        val validatedPreloadedSession = preloadedSwapSession?.takeIf { sess ->
            val spatialDim = sess.inputInfo.values.firstNotNullOfOrNull { nodeInfo ->
                val tInfo = nodeInfo.info as? TensorInfo
                if (tInfo != null && tInfo.shape.size == 4) {
                    val nhwc = tInfo.shape[3] == 3L && tInfo.shape[1] != 3L
                    (if (nhwc) tInfo.shape[1] else tInfo.shape[2]).toInt()
                } else null
            }
            spatialDim == null || spatialDim <= 0 || spatialDim == modelCropSize
        }

        val swapSession = validatedPreloadedSession ?: if (swapFile.exists() && swapFile.length() > 1024L && ortEnv != null) {
            OnnxProtobufInspector.getOrCreateCachedSession(
                ortEnv = ortEnv,
                file = swapFile,
                preferHardwareAcceleration = preferHardwareAccel
            )
        } else null

        var targetInputName = "target"
        var sourceInputName = "source"
        var targetOutputName: String? = null
        var isTargetNhwc = false
        var targetInputType = OnnxJavaType.FLOAT
        var sourceInputType = OnnxJavaType.FLOAT

        if (swapSession != null) {
            for ((name, nodeInfo) in swapSession.inputInfo) {
                val tInfo = nodeInfo.info as? TensorInfo ?: continue
                if (tInfo.shape.size == 4) {
                    targetInputName = name
                    targetInputType = tInfo.type
                    val nhwc = tInfo.shape[3] == 3L && tInfo.shape[1] != 3L
                    isTargetNhwc = nhwc
                    val hDim = (if (nhwc) tInfo.shape[1] else tInfo.shape[2]).toInt()
                    if (hDim in listOf(128, 256, 512)) {
                        modelCropSize = hDim
                    }
                } else if (tInfo.shape.size == 2) {
                    sourceInputName = name
                    sourceInputType = tInfo.type
                }
            }
            for ((outName, nodeInfo) in swapSession.outputInfo) {
                val tInfo = nodeInfo.info as? TensorInfo ?: continue
                if (tInfo.shape.size == 4) {
                    val hasThree = tInfo.shape[1] == 3L || tInfo.shape[3] == 3L
                    if (hasThree || targetOutputName == null) {
                        targetOutputName = outName
                    }
                }
            }
        }

        val simSwapAlign = isSimSwapFile(swapFile)
        val shouldUsePixelBoost = config.pixelBoostEnabled &&
            swapSession != null &&
            ortEnv != null &&
            modelCropSize < 512
        val boostSize = when {
            !shouldUsePixelBoost -> modelCropSize
            modelCropSize == 256 -> 512
            modelCropSize == 128 -> 256
            else -> modelCropSize
        }

        val alignMatrix = if (simSwapAlign) {
            FaceAlignment.estimateNormArcFaceV1(targetFace.landmarks5, boostSize)
        } else {
            FaceAlignment.estimateNorm(targetFace.landmarks5, boostSize)
        }
        val forwardMatrix128 = FloatArray(6) { i -> alignMatrix[i] * (128f / boostSize.toFloat()) }
        val alignedTarget = FaceAlignment.warpAffineCrop(targetBitmap, alignMatrix, boostSize)

        if (swapSession == null || ortEnv == null) {
            val poseAlignedSource = FaceAlignment.warpSourceToTargetPose(
                sourceBitmap = sourceBitmap,
                sourceLandmarks5 = sourceFace.landmarks5,
                targetLandmarks5 = targetFace.landmarks5,
                dstSize = boostSize
            )
            val fallbackCrop = VisualValidationBenchmark.synthesizeRawSwapCandidateCrop(
                candidate = InSwapperEngine.DEFAULT_PRODUCTION_CANDIDATE,
                srcCrop = poseAlignedSource,
                tgtCrop = alignedTarget,
                cropSize = boostSize
            )
            poseAlignedSource.recycle()
            alignedTarget.recycle()
            return OptionDSwapCropResult(
                cropBitmap = fallbackCrop,
                forwardMatrix128 = forwardMatrix128,
                activeSwapFile = swapFile,
                modelCropSize = modelCropSize,
                effectiveOutputSize = boostSize,
                usedPixelBoost = false,
                usedNeuralSwap = false,
                diagnosticSummary = "optD_fallback(${boostSize}px)"
            )
        }

        var usedCrossfaceConverter = false
        val sourceVec = when {
            isEmapProjectionRequired(swapFile) -> {
                InSwapperEngine.resolveSwapperSourceVector(
                    context = context,
                    activeSwapFile = swapFile,
                    isHyperSwap256 = false,
                    sourceEmbedding = sourceEmbedding
                )
            }
            simSwapAlign -> {
                val crossfaceFile = resolveSimSwapEmbeddingConverterFile(context)
                val converted = if (crossfaceFile != null) {
                    runSimSwapEmbeddingConverter(
                        ortEnv = ortEnv,
                        converterFile = crossfaceFile,
                        unnormedEmbedding512 = sourceEmbedding.unnormedEmbedding512,
                        preferHardwareAccel = preferHardwareAccel
                    )
                } else null
                if (converted != null) {
                    usedCrossfaceConverter = true
                    converted
                } else {
                    sourceEmbedding.rawNormedEmbedding512
                }
            }
            else -> sourceEmbedding.rawNormedEmbedding512
        }
        val crossfaceTag = if (simSwapAlign) {
            if (usedCrossfaceConverter) " + crossface_simswap" else " (direct L2)"
        } else ""

        val sourceTensor = if (sourceInputType == OnnxJavaType.FLOAT16) {
            val sHalf = ShortArray(512) { idx -> DepthEstimationEngine.floatToFloat16(sourceVec[idx]) }
            OnnxTensor.createTensor(ortEnv, ShortBuffer.wrap(sHalf), longArrayOf(1L, 512L), OnnxJavaType.FLOAT16)
        } else {
            OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(sourceVec), longArrayOf(1L, 512L))
        }

        try {
            val boostPixels = IntArray(boostSize * boostSize)
            alignedTarget.getPixels(boostPixels, 0, boostSize, 0, 0, boostSize, boostSize)
            alignedTarget.recycle()

            val cacheKey = "${swapFile.absolutePath}_${swapFile.length()}_${swapFile.lastModified()}"
            val defaultSpec = defaultSpecForSwapper(swapFile)
            var activeSpec = verifiedOptionDSpecCache[cacheKey] ?: defaultSpec
            val graphHint = OnnxProtobufInspector.inspectSwapperGraphSpec(swapFile)
                ?.outputRangeHint ?: OnnxProtobufInspector.SwapperOutputRangeHint.AUTO

            val outPixels: IntArray
            val diagText: String
            if (shouldUsePixelBoost) {
                val tiles = implodePixelBoost(boostPixels, boostSize, modelCropSize)
                val outTiles = ArrayList<IntArray>(tiles.size)
                var lastDiag = ""
                for (tileIdx in tiles.indices) {
                    if (isCancelled()) throw CancellationException("Face swap cancelled during Pixel Boost.")
                    val tilePx = tiles[tileIdx]
                    val (swappedTilePx, chosenSpec, tileDiag) = if (tileIdx == 0) {
                        runSingleTileWithSelfHealing(
                            ortEnv = ortEnv,
                            session = swapSession,
                            targetInputName = targetInputName,
                            sourceInputName = sourceInputName,
                            preferredOutputName = targetOutputName,
                            isTargetNhwc = isTargetNhwc,
                            targetInputType = targetInputType,
                            targetPixels = tilePx,
                            cropSize = modelCropSize,
                            initialSpec = activeSpec,
                            isHyperOrSimSwap = !isEmapProjectionRequired(swapFile),
                            outputRangeHint = graphHint,
                            sourceTensor = sourceTensor
                        )
                    } else {
                        val (px, st) = runSwapperPass(
                            ortEnv = ortEnv,
                            session = swapSession,
                            targetInputName = targetInputName,
                            sourceInputName = sourceInputName,
                            preferredOutputName = targetOutputName,
                            isTargetNhwc = isTargetNhwc,
                            targetInputType = targetInputType,
                            targetPixels = tilePx,
                            cropSize = modelCropSize,
                            spec = activeSpec,
                            outputRangeHint = graphHint,
                            sourceTensor = sourceTensor
                        )
                        Triple(px, activeSpec, formatStatsDiag(activeSpec, st))
                    }
                    activeSpec = chosenSpec
                    verifiedOptionDSpecCache[cacheKey] = chosenSpec
                    lastDiag = tileDiag
                    outTiles.add(swappedTilePx)
                }
                outPixels = explodePixelBoost(outTiles, boostSize, modelCropSize)
                diagText = "boost=${modelCropSize}->${boostSize} (${outTiles.size}x) | $lastDiag"
            } else {
                val (swappedPx, chosenSpec, singleDiag) = runSingleTileWithSelfHealing(
                    ortEnv = ortEnv,
                    session = swapSession,
                    targetInputName = targetInputName,
                    sourceInputName = sourceInputName,
                    preferredOutputName = targetOutputName,
                    isTargetNhwc = isTargetNhwc,
                    targetInputType = targetInputType,
                    targetPixels = boostPixels,
                    cropSize = modelCropSize,
                    initialSpec = activeSpec,
                    isHyperOrSimSwap = !isEmapProjectionRequired(swapFile),
                    outputRangeHint = graphHint,
                    sourceTensor = sourceTensor
                )
                verifiedOptionDSpecCache[cacheKey] = chosenSpec
                outPixels = swappedPx
                diagText = singleDiag
            }

            val outBmp = Bitmap.createBitmap(boostSize, boostSize, Bitmap.Config.ARGB_8888)
            outBmp.setPixels(outPixels, 0, boostSize, 0, 0, boostSize, boostSize)
            return OptionDSwapCropResult(
                cropBitmap = outBmp,
                forwardMatrix128 = forwardMatrix128,
                activeSwapFile = swapFile,
                modelCropSize = modelCropSize,
                effectiveOutputSize = boostSize,
                usedPixelBoost = shouldUsePixelBoost,
                usedNeuralSwap = true,
                diagnosticSummary = "$diagText$crossfaceTag"
            )
        } finally {
            runCatching { sourceTensor.close() }
        }
    }

    private fun inferCropSizeFromFileName(file: File): Int {
        val l = file.name.lowercase()
        return when {
            l.contains("512") -> 512
            l.contains("128") -> 128
            else -> 256
        }
    }

    private fun runSimSwapEmbeddingConverter(
        ortEnv: OrtEnvironment,
        converterFile: File,
        unnormedEmbedding512: FloatArray,
        preferHardwareAccel: Boolean
    ): FloatArray? {
        if (!converterFile.exists() || converterFile.length() <= 1024L || unnormedEmbedding512.size != 512) {
            return null
        }
        val session = OnnxProtobufInspector.getOrCreateCachedSession(
            ortEnv = ortEnv,
            file = converterFile,
            preferHardwareAcceleration = preferHardwareAccel
        ) ?: return null
        return runCatching {
            val inEntry = session.inputInfo.entries.firstOrNull() ?: return@runCatching null
            val inInfo = inEntry.value.info as? TensorInfo
            val shape = if (inInfo != null && inInfo.shape.size == 1) longArrayOf(512L) else longArrayOf(1L, 512L)
            val inTensor = if (inInfo?.type == OnnxJavaType.FLOAT16) {
                val shorts = ShortArray(512) { idx -> DepthEstimationEngine.floatToFloat16(unnormedEmbedding512[idx]) }
                OnnxTensor.createTensor(ortEnv, ShortBuffer.wrap(shorts), shape, OnnxJavaType.FLOAT16)
            } else {
                OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(unnormedEmbedding512), shape)
            }
            inTensor.use { tensor ->
                session.run(mapOf(inEntry.key to tensor)).use { res ->
                    val outT = res[0] as OnnxTensor
                    val outVec = FloatArray(512)
                    if (outT.info.type == OnnxJavaType.FLOAT16) {
                        val sb = outT.shortBuffer
                        val cnt = min(sb.remaining(), 512)
                        for (i in 0 until cnt) outVec[i] = DepthEstimationEngine.float16ToFloat(sb.get())
                    } else {
                        val fb = outT.floatBuffer
                        fb.get(outVec, 0, min(fb.remaining(), 512))
                    }
                    val norm = ArcFaceRecognizer.computeL2Norm(outVec)
                    if (!norm.isFinite() || norm < 1e-4f) return@runCatching null
                    FloatArray(512) { idx -> outVec[idx] / norm }
                }
            }
        }.getOrNull()
    }

    internal fun defaultSpecForSwapper(file: File): SwapperInputSpec {
        val l = file.name.lowercase()
        return when {
            // FaceFusion core.py: simswap_unofficial_512 / simswap_512 uses mean=[0,0,0], std=[1,1,1] ([0..1] RGB),
            // whereas simswap_256 uses ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
            l.contains("simswap") && l.contains("512") ->
                SwapperInputSpec(ChannelLayout.RGB, InputNormMode.UNSIGNED_0_TO_1)
            l.contains("simswap") ->
                SwapperInputSpec(ChannelLayout.RGB, InputNormMode.IMAGENET)
            l.contains("blendswap") ->
                SwapperInputSpec(ChannelLayout.RGB, InputNormMode.UNSIGNED_0_TO_1)
            l.contains("ghost_") ->
                SwapperInputSpec(ChannelLayout.BGR, InputNormMode.SIGNED_MINUS1_TO_1)
            l.contains("reswapper") -> SwapperInputSpec(ChannelLayout.RGB, InputNormMode.SIGNED_MINUS1_TO_1)
            l.contains("inswapper") -> SwapperInputSpec(ChannelLayout.RGB, InputNormMode.UNSIGNED_0_TO_1)
            else -> SwapperInputSpec(ChannelLayout.RGB, InputNormMode.SIGNED_MINUS1_TO_1)
        }
    }

    internal fun shouldRetryCorruptedOrSolarizedOutput(stats: InSwapperEngine.RawOutputStats): Boolean {
        return stats.neonRedFraction > 0.015f ||
            stats.collapsedBlackFraction > 0.02f ||
            stats.nanFraction > 0.005f ||
            stats.lumaCorrelation < 0.42f ||
            stats.centerFaceDiff > 48.0f
    }

    internal fun computeSpecQualityCost(stats: InSwapperEngine.RawOutputStats): Float {
        val corrPenalty = if (stats.lumaCorrelation < 0.50f) (0.50f - stats.lumaCorrelation) * 4.5f else 0f
        val colorBreakdownPenalty = if (stats.centerFaceDiff > 44.0f) (stats.centerFaceDiff - 44.0f) / 18.0f else 0f
        val rangeSpan = stats.maxVal - stats.minVal
        val rangePenalty = if (rangeSpan < 0.40f) (0.40f - rangeSpan) * 2.0f else 0f
        return stats.artifactScore + corrPenalty + colorBreakdownPenalty + rangePenalty
    }

    private fun runSingleTileWithSelfHealing(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        targetInputName: String,
        sourceInputName: String,
        preferredOutputName: String?,
        isTargetNhwc: Boolean,
        targetInputType: OnnxJavaType,
        targetPixels: IntArray,
        cropSize: Int,
        initialSpec: SwapperInputSpec,
        isHyperOrSimSwap: Boolean,
        outputRangeHint: OnnxProtobufInspector.SwapperOutputRangeHint,
        sourceTensor: OnnxTensor
    ): Triple<IntArray, SwapperInputSpec, String> {
        var (bestPx, bestStats) = runSwapperPass(
            ortEnv, session, targetInputName, sourceInputName, preferredOutputName,
            isTargetNhwc, targetInputType, targetPixels, cropSize, initialSpec, outputRangeHint, sourceTensor
        )
        var bestSpec = initialSpec

        val hasCorruption = shouldRetryCorruptedOrSolarizedOutput(bestStats)
        val hasRangeCollapse = (bestStats.maxVal - bestStats.minVal) < 0.40f ||
            bestStats.gtOneFraction > 0.25f
        val isPassthrough = InSwapperEngine.shouldRetryNearIdentityPassthrough(
            isHyperSwapModel = isHyperOrSimSwap,
            hasCorruption = hasCorruption,
            lumaCorrelation = bestStats.lumaCorrelation,
            centerFaceDiff = bestStats.centerFaceDiff
        )

        if (hasCorruption || (isHyperOrSimSwap && (isPassthrough || hasRangeCollapse))) {
            val candidates = listOf(
                SwapperInputSpec(ChannelLayout.RGB, InputNormMode.UNSIGNED_0_TO_1),
                SwapperInputSpec(ChannelLayout.RGB, InputNormMode.SIGNED_MINUS1_TO_1),
                SwapperInputSpec(ChannelLayout.RGB, InputNormMode.IMAGENET),
                SwapperInputSpec(ChannelLayout.BGR, InputNormMode.UNSIGNED_0_TO_1),
                SwapperInputSpec(ChannelLayout.BGR, InputNormMode.SIGNED_MINUS1_TO_1),
                SwapperInputSpec(ChannelLayout.BGR, InputNormMode.IMAGENET)
            ).filter { it != initialSpec }

            for (cand in candidates) {
                val (candPx, candStats) = runSwapperPass(
                    ortEnv, session, targetInputName, sourceInputName, preferredOutputName,
                    isTargetNhwc, targetInputType, targetPixels, cropSize, cand, outputRangeHint, sourceTensor
                )
                if (hasCorruption || hasRangeCollapse) {
                    val candCost = computeSpecQualityCost(candStats)
                    val bestCost = computeSpecQualityCost(bestStats)
                    val candRange = candStats.maxVal - candStats.minVal
                    val bestRange = bestStats.maxVal - bestStats.minVal
                    val betterRange = !hasCorruption && hasRangeCollapse &&
                        candCost <= 0.15f && candRange > bestRange + 0.25f
                    if (candCost + 0.02f < bestCost || betterRange) {
                        bestPx = candPx
                        bestStats = candStats
                        bestSpec = cand
                        val isCleanAndCorrelated = !shouldRetryCorruptedOrSolarizedOutput(bestStats) &&
                            bestStats.lumaCorrelation >= 0.62f &&
                            (bestStats.maxVal - bestStats.minVal) >= 0.45f
                        if (isCleanAndCorrelated) break
                    }
                } else {
                    val candClean = !shouldRetryCorruptedOrSolarizedOutput(candStats) &&
                        candStats.lumaCorrelation in 0.48f..0.985f &&
                        (candStats.maxVal - candStats.minVal) >= 0.45f
                    if (candClean && candStats.centerFaceDiff > bestStats.centerFaceDiff * 1.45f + 3.0f) {
                        bestPx = candPx
                        bestStats = candStats
                        bestSpec = cand
                        break
                    }
                }
            }
        }
        return Triple(bestPx, bestSpec, formatStatsDiag(bestSpec, bestStats))
    }

    private fun runSwapperPass(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        targetInputName: String,
        sourceInputName: String,
        preferredOutputName: String?,
        isTargetNhwc: Boolean,
        targetInputType: OnnxJavaType,
        targetPixels: IntArray,
        cropSize: Int,
        spec: SwapperInputSpec,
        outputRangeHint: OnnxProtobufInspector.SwapperOutputRangeHint,
        sourceTensor: OnnxTensor
    ): Pair<IntArray, InSwapperEngine.RawOutputStats> {
        val hw = cropSize * cropSize
        val inFloats = FloatArray(3 * hw)
        for (i in 0 until hw) {
            val c = targetPixels[i]
            val r01 = ((c ushr 16) and 0xFF) / 255.0f
            val g01 = ((c ushr 8) and 0xFF) / 255.0f
            val b01 = (c and 0xFF) / 255.0f

            val v0 = if (spec.channelLayout == ChannelLayout.RGB) r01 else b01
            val v1 = g01
            val v2 = if (spec.channelLayout == ChannelLayout.RGB) b01 else r01

            val n0: Float
            val n1: Float
            val n2: Float
            when (spec.normMode) {
                InputNormMode.IMAGENET -> {
                    val mean0 = if (spec.channelLayout == ChannelLayout.RGB) 0.485f else 0.406f
                    val mean2 = if (spec.channelLayout == ChannelLayout.RGB) 0.406f else 0.485f
                    val std0 = if (spec.channelLayout == ChannelLayout.RGB) 0.229f else 0.225f
                    val std2 = if (spec.channelLayout == ChannelLayout.RGB) 0.225f else 0.229f
                    n0 = (v0 - mean0) / std0
                    n1 = (v1 - 0.456f) / 0.224f
                    n2 = (v2 - mean2) / std2
                }
                InputNormMode.SIGNED_MINUS1_TO_1 -> {
                    n0 = (v0 - 0.5f) / 0.5f
                    n1 = (v1 - 0.5f) / 0.5f
                    n2 = (v2 - 0.5f) / 0.5f
                }
                InputNormMode.UNSIGNED_0_TO_1 -> {
                    n0 = v0
                    n1 = v1
                    n2 = v2
                }
            }
            if (isTargetNhwc) {
                val base = i * 3
                inFloats[base] = n0
                inFloats[base + 1] = n1
                inFloats[base + 2] = n2
            } else {
                inFloats[i] = n0
                inFloats[hw + i] = n1
                inFloats[2 * hw + i] = n2
            }
        }

        val shape = if (isTargetNhwc) {
            longArrayOf(1L, cropSize.toLong(), cropSize.toLong(), 3L)
        } else {
            longArrayOf(1L, 3L, cropSize.toLong(), cropSize.toLong())
        }
        val targetTensor = if (targetInputType == OnnxJavaType.FLOAT16) {
            val shorts = ShortArray(3 * hw) { idx -> DepthEstimationEngine.floatToFloat16(inFloats[idx]) }
            OnnxTensor.createTensor(ortEnv, ShortBuffer.wrap(shorts), shape, OnnxJavaType.FLOAT16)
        } else {
            OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inFloats), shape)
        }

        targetTensor.use { tt ->
            session.run(mapOf(targetInputName to tt, sourceInputName to sourceTensor)).use { res ->
                val outVal = if (preferredOutputName != null) {
                    res.get(preferredOutputName).orElse(res[0])
                } else res[0]
                val outT = outVal as OnnxTensor
                val outInfo = outT.info
                val outShape = outInfo.shape
                val isOutNhwc = outShape.size == 4 && outShape[3] == 3L && outShape[1] != 3L
                val rawOut = FloatArray(3 * hw)
                if (outInfo.type == OnnxJavaType.FLOAT16) {
                    val sb = outT.shortBuffer
                    val n = min(sb.remaining(), rawOut.size)
                    for (i in 0 until n) rawOut[i] = DepthEstimationEngine.float16ToFloat(sb.get())
                } else {
                    val fb = outT.floatBuffer
                    fb.get(rawOut, 0, min(fb.remaining(), rawOut.size))
                }

                // Convert NHWC output -> NCHW if needed so shared RawOutputStats/decoder works identically
                val nchwFloats = if (isOutNhwc) {
                    FloatArray(3 * hw).also { dst ->
                        for (i in 0 until hw) {
                            val b = i * 3
                            dst[i] = rawOut[b]
                            dst[hw + i] = rawOut[b + 1]
                            dst[2 * hw + i] = rawOut[b + 2]
                        }
                    }
                } else rawOut

                val isSigned = InSwapperEngine.inferSignedOutputMode(nchwFloats, outputRangeHint)
                val channelOrder = if (spec.channelLayout == ChannelLayout.RGB) {
                    InSwapperEngine.SwapChannelOrder.RGB
                } else {
                    InSwapperEngine.SwapChannelOrder.BGR
                }
                val outPx = InSwapperEngine.decodeRawOutputToPixels(
                    outFloats = nchwFloats,
                    hw = hw,
                    channelOrder = channelOrder,
                    isSignedOutput = isSigned
                )
                val stats = InSwapperEngine.analyzeRawSwapOutput(
                    outFloats = nchwFloats,
                    outPixels = outPx,
                    targetPixels = targetPixels,
                    isSignedOutput = isSigned
                )
                return outPx to stats
            }
        }
    }

    private fun formatStatsDiag(spec: SwapperInputSpec, stats: InSwapperEngine.RawOutputStats): String =
        "spec=${spec.label()} out=${if (stats.isSignedOutput) "m1p1" else "0p1"} " +
            "range=${"%.2f".format(stats.minVal)}..${"%.2f".format(stats.maxVal)} " +
            "corr=${"%.2f".format(stats.lumaCorrelation)} cdiff=${"%.1f".format(stats.centerFaceDiff)}"

    /**
     * Runs any FaceFusion-supported ONNX enhancer (`CodeFormer.onnx` with optional `w` fidelity input,
     * `GFPGAN 1.4`, `GPEN-BFR 256/512/1024`, or `RestoreFormer++`) strictly inside Option D.
     */
    fun runOptionDEnhancer512(
        ortEnv: OrtEnvironment?,
        enhancerFile: File?,
        crop128To512: Bitmap,
        codeformerFidelity: Float = 0.60f,
        preferHardwareAccel: Boolean = false,
        preloadedSession: OrtSession? = null
    ): OptionDEnhancerResult? {
        if (ortEnv == null) return null
        val session = preloadedSession ?: run {
            if (enhancerFile == null || !enhancerFile.exists() || enhancerFile.length() <= 1024L) return null
            OnnxProtobufInspector.getOrCreateCachedSession(
                ortEnv = ortEnv,
                file = enhancerFile,
                preferHardwareAcceleration = preferHardwareAccel
            )
        } ?: return null

        val fidelityW = codeformerFidelity.coerceIn(0f, 1f)
        return runCatching {
            val entries = session.inputInfo.entries
            val imageEntry = entries.firstOrNull { (_, node) ->
                (node.info as? TensorInfo)?.shape?.size == 4
            } ?: entries.firstOrNull() ?: return@runCatching null

            val imageInfo = imageEntry.value.info as? TensorInfo ?: return@runCatching null
            val inShape = imageInfo.shape
            val isNhwcIn = inShape.size == 4 && inShape[3] == 3L && inShape[1] != 3L
            val rawH = if (inShape.size == 4) (if (isNhwcIn) inShape[1] else inShape[2]) else 512L
            val inferredSize = enhancerFile?.name?.lowercase()?.let { l ->
                when {
                    l.contains("1024") -> 1024
                    l.contains("256") -> 256
                    else -> 512
                }
            } ?: 512
            val modelSize = if (rawH > 0L) rawH.toInt().coerceIn(128, 1024) else inferredSize
            val hw = modelSize * modelSize

            val pixels = if (modelSize == 512) {
                FaceBlender.upscaleBitmapBicubicTo512(crop128To512)
            } else {
                val scaled = Bitmap.createScaledBitmap(crop128To512, modelSize, modelSize, true)
                IntArray(hw).also {
                    scaled.getPixels(it, 0, modelSize, 0, 0, modelSize, modelSize)
                    scaled.recycle()
                }
            }

            val floatInput = FloatArray(3 * hw)
            if (isNhwcIn) {
                for (i in 0 until hw) {
                    val c = pixels[i]
                    val b = i * 3
                    floatInput[b] = (((c ushr 16) and 0xFF) / 255.0f - 0.5f) / 0.5f
                    floatInput[b + 1] = (((c ushr 8) and 0xFF) / 255.0f - 0.5f) / 0.5f
                    floatInput[b + 2] = ((c and 0xFF) / 255.0f - 0.5f) / 0.5f
                }
            } else {
                for (i in 0 until hw) {
                    val c = pixels[i]
                    floatInput[i] = (((c ushr 16) and 0xFF) / 255.0f - 0.5f) / 0.5f
                    floatInput[hw + i] = (((c ushr 8) and 0xFF) / 255.0f - 0.5f) / 0.5f
                    floatInput[2 * hw + i] = ((c and 0xFF) / 255.0f - 0.5f) / 0.5f
                }
            }

            val imgShape = if (isNhwcIn) {
                longArrayOf(1L, modelSize.toLong(), modelSize.toLong(), 3L)
            } else {
                longArrayOf(1L, 3L, modelSize.toLong(), modelSize.toLong())
            }
            val imgTensor = if (imageInfo.type == OnnxJavaType.FLOAT16) {
                val shorts = ShortArray(3 * hw) { idx -> DepthEstimationEngine.floatToFloat16(floatInput[idx]) }
                OnnxTensor.createTensor(ortEnv, ShortBuffer.wrap(shorts), imgShape, OnnxJavaType.FLOAT16)
            } else {
                OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(floatInput), imgShape)
            }

            // Optional 2nd CodeFormer weight/fidelity scalar `w` (DOUBLE / FLOAT / FLOAT16, shape [], [1], or [1,1])
            val weightEntry = entries.firstOrNull { (name, node) ->
                if (name == imageEntry.key) return@firstOrNull false
                val tInfo = node.info as? TensorInfo ?: return@firstOrNull false
                val lower = name.lowercase()
                lower == "w" || lower.contains("weight") || lower.contains("fidelity") ||
                    tInfo.shape.isEmpty() || (tInfo.shape.size <= 2 && tInfo.shape.all { it <= 1L })
            }

            val weightTensor: OnnxTensor? = if (weightEntry != null) {
                val wInfo = weightEntry.value.info as TensorInfo
                val wShape = when {
                    wInfo.shape.isEmpty() -> longArrayOf()
                    else -> LongArray(wInfo.shape.size) { 1L }
                }
                when (wInfo.type) {
                    OnnxJavaType.DOUBLE -> OnnxTensor.createTensor(
                        ortEnv,
                        DoubleBuffer.wrap(doubleArrayOf(fidelityW.toDouble())),
                        wShape
                    )
                    OnnxJavaType.FLOAT16 -> OnnxTensor.createTensor(
                        ortEnv,
                        ShortBuffer.wrap(shortArrayOf(DepthEstimationEngine.floatToFloat16(fidelityW))),
                        wShape,
                        OnnxJavaType.FLOAT16
                    )
                    else -> OnnxTensor.createTensor(
                        ortEnv,
                        FloatBuffer.wrap(floatArrayOf(fidelityW)),
                        wShape
                    )
                }
            } else null

            try {
                val inputMap = LinkedHashMap<String, OnnxTensor>(2)
                inputMap[imageEntry.key] = imgTensor
                if (weightEntry != null && weightTensor != null) {
                    inputMap[weightEntry.key] = weightTensor
                }

                val outPixels = IntArray(hw)
                var nanFloats = 0
                var introducedNeonRed = 0
                var introducedBlack = 0

                session.run(inputMap).use { res ->
                    val outT = res[0] as OnnxTensor
                    val outInfo = outT.info
                    val outShape = outInfo.shape
                    val isNhwcOut = outShape.size == 4 && outShape[3] == 3L && outShape[1] != 3L
                    val outFloats = FloatArray(3 * hw)
                    if (outInfo.type == OnnxJavaType.FLOAT16) {
                        val sb = outT.shortBuffer
                        val cnt = min(sb.remaining(), outFloats.size)
                        for (i in 0 until cnt) outFloats[i] = DepthEstimationEngine.float16ToFloat(sb.get())
                    } else {
                        val fb = outT.floatBuffer
                        fb.get(outFloats, 0, min(fb.remaining(), outFloats.size))
                    }

                    var minV = Float.MAX_VALUE
                    var maxV = -Float.MAX_VALUE
                    for (v in outFloats) {
                        if (!v.isFinite()) {
                            nanFloats++
                        } else {
                            if (v < minV) minV = v
                            if (v > maxV) maxV = v
                        }
                    }
                    if (nanFloats > hw / 50) return@runCatching null
                    val isUnsigned01 = minV >= -0.05f && maxV <= 1.25f
                    val is255 = maxV > 2.5f

                    for (i in 0 until hw) {
                        val frRaw = if (isNhwcOut) outFloats[i * 3] else outFloats[i]
                        val fgRaw = if (isNhwcOut) outFloats[i * 3 + 1] else outFloats[hw + i]
                        val fbRaw = if (isNhwcOut) outFloats[i * 3 + 2] else outFloats[2 * hw + i]
                        val fr = if (frRaw.isFinite()) frRaw else 0f
                        val fg = if (fgRaw.isFinite()) fgRaw else 0f
                        val fb = if (fbRaw.isFinite()) fbRaw else 0f
                        val r: Int
                        val g: Int
                        val b: Int
                        when {
                            is255 -> {
                                r = fr.toInt().coerceIn(0, 255)
                                g = fg.toInt().coerceIn(0, 255)
                                b = fb.toInt().coerceIn(0, 255)
                            }
                            isUnsigned01 -> {
                                r = (fr.coerceIn(0f, 1f) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                g = (fg.coerceIn(0f, 1f) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                b = (fb.coerceIn(0f, 1f) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                            }
                            else -> {
                                r = (((fr * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                g = (((fg * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                b = (((fb * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                            }
                        }
                        outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                        val inC = pixels[i]
                        val inR = (inC ushr 16) and 0xFF
                        val inG = (inC ushr 8) and 0xFF
                        val inB = inC and 0xFF
                        val inLum = 0.299f * inR + 0.587f * inG + 0.114f * inB
                        if ((r - g >= 72) && (r - b >= 72) && (inR - inG < 48)) introducedNeonRed++
                        if (r <= 8 && g <= 8 && b <= 8 && inLum > 48f) introducedBlack++
                    }
                }
                if ((introducedNeonRed + introducedBlack).toFloat() / hw.toFloat() > 0.025f) {
                    return@runCatching null
                }

                val restoredRaw = Bitmap.createBitmap(modelSize, modelSize, Bitmap.Config.ARGB_8888)
                restoredRaw.setPixels(outPixels, 0, modelSize, 0, 0, modelSize, modelSize)
                val restored512 = if (modelSize == 512) {
                    restoredRaw
                } else {
                    Bitmap.createScaledBitmap(restoredRaw, 512, 512, true).also { restoredRaw.recycle() }
                }
                OptionDEnhancerResult(
                    restored512 = restored512,
                    usedEnhancerFileName = enhancerFile?.name ?: "enhancer.onnx",
                    isCodeFormer = weightEntry != null || (enhancerFile?.name?.lowercase()?.contains("codeformer") == true),
                    appliedFidelityW = fidelityW
                )
            } finally {
                runCatching { imgTensor.close() }
                runCatching { weightTensor?.close() }
            }
        }.getOrNull()
    }

    /**
     * Runs `face_occluder.onnx`, `dfl_xseg.onnx`, or `xseg_1/2/3.onnx` (`256x256` or `512x512`,
     * NHWC or NCHW, FP16 or FP32) and produces a feathered `512x512` foreground face visibility gate
     * (`1.0` = visible face skin, `0.0` = occluded by hand/hair/sunglasses/accessory).
     */
    fun computeXsegOcclusionGate512(
        ortEnv: OrtEnvironment?,
        alignedTarget512: Bitmap,
        occluderFile: File?,
        preferHardwareAccel: Boolean = false
    ): FloatArray? {
        if (ortEnv == null || occluderFile == null || !occluderFile.exists() || occluderFile.length() <= 1024L) {
            return null
        }
        val session = OnnxProtobufInspector.getOrCreateCachedSession(
            ortEnv = ortEnv,
            file = occluderFile,
            preferHardwareAcceleration = preferHardwareAccel
        ) ?: return null

        return runCatching {
            val inEntry = session.inputInfo.entries.firstOrNull() ?: return@runCatching null
            val inInfo = inEntry.value.info as? TensorInfo ?: return@runCatching null
            val inShape = inInfo.shape
            val isNhwc = inShape.size == 4 && inShape[3] == 3L && inShape[1] != 3L
            val rawH = if (inShape.size == 4) (if (isNhwc) inShape[1] else inShape[2]) else 256L
            val modelSize = if (rawH > 0L) rawH.toInt().coerceIn(128, 512) else 256
            val hw = modelSize * modelSize

            val scaled = if (alignedTarget512.width == modelSize && alignedTarget512.height == modelSize) {
                alignedTarget512
            } else {
                Bitmap.createScaledBitmap(alignedTarget512, modelSize, modelSize, true)
            }
            val pixels = IntArray(hw)
            scaled.getPixels(pixels, 0, modelSize, 0, 0, modelSize, modelSize)
            if (scaled !== alignedTarget512) scaled.recycle()

            val inFloats = FloatArray(3 * hw)
            if (isNhwc) {
                for (i in 0 until hw) {
                    val c = pixels[i]
                    val b = i * 3
                    inFloats[b] = ((c ushr 16) and 0xFF) / 255.0f
                    inFloats[b + 1] = ((c ushr 8) and 0xFF) / 255.0f
                    inFloats[b + 2] = (c and 0xFF) / 255.0f
                }
            } else {
                for (i in 0 until hw) {
                    val c = pixels[i]
                    inFloats[i] = ((c ushr 16) and 0xFF) / 255.0f
                    inFloats[hw + i] = ((c ushr 8) and 0xFF) / 255.0f
                    inFloats[2 * hw + i] = (c and 0xFF) / 255.0f
                }
            }
            val shape = if (isNhwc) {
                longArrayOf(1L, modelSize.toLong(), modelSize.toLong(), 3L)
            } else {
                longArrayOf(1L, 3L, modelSize.toLong(), modelSize.toLong())
            }
            val inTensor = if (inInfo.type == OnnxJavaType.FLOAT16) {
                val shorts = ShortArray(3 * hw) { idx -> DepthEstimationEngine.floatToFloat16(inFloats[idx]) }
                OnnxTensor.createTensor(ortEnv, ShortBuffer.wrap(shorts), shape, OnnxJavaType.FLOAT16)
            } else {
                OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inFloats), shape)
            }

            val modelGate = FloatArray(hw)
            inTensor.use { tt ->
                session.run(mapOf(inEntry.key to tt)).use { res ->
                    val outT = res[0] as OnnxTensor
                    val outInfo = outT.info
                    val outShape = outInfo.shape

                    val totalElems = outInfo.shape.fold(1L) { acc, v -> acc * max(1L, v) }.toInt().coerceAtLeast(hw)
                    val rawOut = FloatArray(totalElems)
                    if (outInfo.type == OnnxJavaType.FLOAT16) {
                        val sb = outT.shortBuffer
                        val cnt = min(sb.remaining(), rawOut.size)
                        for (i in 0 until cnt) rawOut[i] = DepthEstimationEngine.float16ToFloat(sb.get())
                    } else {
                        val fb = outT.floatBuffer
                        fb.get(rawOut, 0, min(fb.remaining(), rawOut.size))
                    }

                    // Determine layout: [1, H, W, C], [1, C, H, W], or [1, H, W]
                    val outH: Int
                    val outW: Int
                    val channels: Int
                    val isOutNhwc: Boolean
                    when (outShape.size) {
                        4 -> {
                            val chLast = outShape[3] in 1L..2L && outShape[1] > 4L
                            isOutNhwc = chLast
                            channels = (if (chLast) outShape[3] else outShape[1]).toInt().coerceIn(1, 2)
                            outH = (if (chLast) outShape[1] else outShape[2]).toInt().coerceIn(16, 512)
                            outW = (if (chLast) outShape[2] else outShape[3]).toInt().coerceIn(16, 512)
                        }
                        3 -> {
                            isOutNhwc = false
                            channels = 1
                            outH = outShape[1].toInt().coerceIn(16, 512)
                            outW = outShape[2].toInt().coerceIn(16, 512)
                        }
                        else -> return@runCatching null
                    }
                    val plane = outH * outW
                    var minV = Float.MAX_VALUE
                    var maxV = -Float.MAX_VALUE
                    val limit = min(rawOut.size, plane * channels)
                    for (i in 0 until limit) {
                        val v = rawOut[i]
                        if (v.isFinite()) {
                            if (v < minV) minV = v
                            if (v > maxV) maxV = v
                        }
                    }
                    val needsSigmoid = minV < -0.05f || maxV > 1.05f

                    for (my in 0 until modelSize) {
                        val sy = (my * outH / modelSize).coerceIn(0, outH - 1)
                        val dstRow = my * modelSize
                        for (mx in 0 until modelSize) {
                            val sx = (mx * outW / modelSize).coerceIn(0, outW - 1)
                            val spatialIdx = sy * outW + sx
                            val prob = if (channels == 2) {
                                val c0 = if (isOutNhwc) rawOut[spatialIdx * 2] else rawOut[spatialIdx]
                                val c1 = if (isOutNhwc) rawOut[spatialIdx * 2 + 1] else rawOut[plane + spatialIdx]
                                val e0 = exp((c0 - max(c0, c1)).toDouble())
                                val e1 = exp((c1 - max(c0, c1)).toDouble())
                                (e1 / (e0 + e1)).toFloat()
                            } else {
                                val v = if (isOutNhwc) rawOut[spatialIdx] else rawOut[spatialIdx]
                                val safeV = if (v.isFinite()) v else 1.0f
                                if (needsSigmoid) {
                                    (1.0 / (1.0 + exp((-safeV).coerceIn(-20f, 20f).toDouble()))).toFloat()
                                } else {
                                    safeV.coerceIn(0f, 1f)
                                }
                            }
                            modelGate[dstRow + mx] = prob.coerceIn(0f, 1f)
                        }
                    }
                }
            }

            // Upscale to 512x512 bilinearly and apply a 5x5 feather blur for smooth occlusion borders
            val target512 = 512
            val upscaled512 = FloatArray(target512 * target512)
            for (y in 0 until target512) {
                val fy = (y.toFloat() / (target512 - 1)) * (modelSize - 1)
                val y0 = fy.toInt().coerceIn(0, modelSize - 1)
                val y1 = (y0 + 1).coerceAtMost(modelSize - 1)
                val wy = fy - y0
                val row = y * target512
                for (x in 0 until target512) {
                    val fx = (x.toFloat() / (target512 - 1)) * (modelSize - 1)
                    val x0 = fx.toInt().coerceIn(0, modelSize - 1)
                    val x1 = (x0 + 1).coerceAtMost(modelSize - 1)
                    val wx = fx - x0
                    val v00 = modelGate[y0 * modelSize + x0]
                    val v10 = modelGate[y0 * modelSize + x1]
                    val v01 = modelGate[y1 * modelSize + x0]
                    val v11 = modelGate[y1 * modelSize + x1]
                    val top = v00 + (v10 - v00) * wx
                    val bot = v01 + (v11 - v01) * wx
                    upscaled512[row + x] = (top + (bot - top) * wy).coerceIn(0f, 1f)
                }
            }

            val smoothed = FloatArray(target512 * target512)
            for (y in 0 until target512) {
                val y0 = (y - 2).coerceAtLeast(0)
                val y1 = (y + 2).coerceAtMost(target512 - 1)
                val row = y * target512
                for (x in 0 until target512) {
                    val x0 = (x - 2).coerceAtLeast(0)
                    val x1 = (x + 2).coerceAtMost(target512 - 1)
                    var sum = 0f
                    var cnt = 0
                    for (ny in y0..y1) {
                        val nRow = ny * target512
                        for (nx in x0..x1) {
                            sum += upscaled512[nRow + nx]
                            cnt++
                        }
                    }
                    smoothed[row + x] = (sum / cnt.coerceAtLeast(1)).coerceIn(0f, 1f)
                }
            }
            smoothed
        }.getOrNull()
    }
}
