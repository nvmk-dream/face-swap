package com.example.onnx

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.CancellationException
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class InpaintExecutionResult(
    val outputBitmap: Bitmap,
    val elapsedMs: Long,
    val engineUsed: String
)

/**
 * Interactive Inpainting Engine for the Inpaint Page.
 *
 * Execution routing priority (when in AUTO mode):
 *  1. `SD 1.5 Inpaint (FP16)` ONNX pack (Text-prompt + Mask)
 *  2. `RealVis 5.1 (INT8)` ONNX pack (Second option, Text-prompt + Mask)
 *  3. `lama_fp32.onnx` (Object / gap removal)
 *  4. Built-in Multi-Scale Boundary-Diffusion Patch Inpainter fallback
 *
 * Unmasked pixels outside the user's painted mask are preserved 100% pixel-identical
 * from [sourceBitmap].
 */
object InteractiveInpaintEngine {

    /**
     * Protects closer foreground pixels from accidental brush overlap when painting background edits.
     * Ensures at least 8 masked pixels remain so downstream engines never receive an empty mask.
     */
    private fun applyDepthForegroundGuardToMask(
        sourceBitmap: Bitmap,
        maskBitmap: Bitmap,
        depthMap: DepthMapResult
    ): Bitmap {
        val w = sourceBitmap.width.coerceAtLeast(1)
        val h = sourceBitmap.height.coerceAtLeast(1)
        val aligned = if (maskBitmap.width == w && maskBitmap.height == h) {
            maskBitmap
        } else {
            Bitmap.createScaledBitmap(maskBitmap, w, h, true)
        }
        val total = w * h
        val px = IntArray(total)
        aligned.getPixels(px, 0, w, 0, 0, w, h)
        if (aligned !== maskBitmap) aligned.recycle()

        val cutoff = depthMap.suggestForegroundCutoff()
        val maskDepths = FloatArray(min(total, 4096))
        var sampleIdx = 0
        val step = max(1, total / 4096)
        var i = 0
        while (i < total && sampleIdx < maskDepths.size) {
            val c = px[i]
            val s = max((c ushr 24) and 0xFF, max((c ushr 16) and 0xFF, max((c ushr 8) and 0xFF, c and 0xFF)))
            if (s > 32) {
                val y = i / w
                val x = i % w
                val nx = if (w > 1) x / (w - 1f) else 0.5f
                val ny = if (h > 1) y / (h - 1f) else 0.5f
                maskDepths[sampleIdx++] = depthMap.sampleBilinear(nx, ny)
            }
            i += step
        }
        if (sampleIdx < 8) return maskBitmap
        val sortedMaskDepths = maskDepths.copyOf(sampleIdx).apply { sort() }
        val medianMaskNear = sortedMaskDepths[sampleIdx / 2]
        // Only protect closer foreground if the painted mask is targeting the background/midground
        if (medianMaskNear >= cutoff + 0.08f) return maskBitmap

        val protectThreshold = (cutoff + 0.04f).coerceIn(0.25f, 0.90f)
        val outPx = px.copyOf()
        var originalMaskCount = 0
        var keptCount = 0
        val invW = if (w > 1) 1f / (w - 1f) else 0f
        val invH = if (h > 1) 1f / (h - 1f) else 0f
        for (y in 0 until h) {
            val rowOff = y * w
            val ny = y * invH
            for (x in 0 until w) {
                val idx = rowOff + x
                val c = outPx[idx]
                val s = max((c ushr 24) and 0xFF, max((c ushr 16) and 0xFF, max((c ushr 8) and 0xFF, c and 0xFF)))
                if (s > 32) {
                    originalMaskCount++
                    val near = depthMap.sampleBilinear(x * invW, ny)
                    if (near >= protectThreshold) {
                        outPx[idx] = Color.TRANSPARENT
                    } else {
                        keptCount++
                    }
                }
            }
        }
        runCatching {
            Log.d("INPAINT", "depthGuard before=$originalMaskCount after=$keptCount cutoff=$protectThreshold")
        }
        // Prevent Depth Guard from over-erasing user-painted sleeves / foreground edits (>35% loss)
        if (keptCount < 8 || keptCount < (originalMaskCount * 0.65f).toInt()) {
            return maskBitmap
        }
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
            setPixels(outPx, 0, w, 0, 0, w, h)
        }
    }

    internal fun shouldStopAutoFallbackOnError(error: Throwable?): Boolean {
        return error is CancellationException ||
            error is SdInpaintOnnxEngine.InsufficientNativeRamException ||
            error is OutOfMemoryError ||
            error is IllegalArgumentException
    }

    fun inpaint(
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        maskBitmap: Bitmap,
        prompt: String,
        negativePrompt: String = SdInpaintOnnxEngine.DEFAULT_NEGATIVE_PROMPT,
        fastMode: Boolean,
        turboMode: Boolean = false,
        lamaModelFile: File?,
        preferHardwareAccel: Boolean = true,
        selectedOption: InpaintModelOption = InpaintModelOption.AUTO,
        primarySdPack: InstalledSdInpaintPack? = null,
        secondarySdPack: InstalledSdInpaintPack? = null,
        depthMap: DepthMapResult? = null,
        enableDepthGuard: Boolean = false,
        strength: Float = SdInpaintOnnxEngine.DEFAULT_STRENGTH,
        guidanceScale: Float = SdInpaintOnnxEngine.DEFAULT_GUIDANCE_SCALE,
        context: Context? = null,
        isCancelled: () -> Boolean = { false },
        onProgress: (String) -> Unit = {}
    ): InpaintExecutionResult {
        if (isCancelled()) {
            throw CancellationException("Inpaint cancelled by user.")
        }

        val guardedMaskBitmap = if (enableDepthGuard && depthMap != null) {
            applyDepthForegroundGuardToMask(sourceBitmap, maskBitmap, depthMap)
        } else {
            maskBitmap
        }
        if (isCancelled()) {
            throw CancellationException("Inpaint cancelled by user.")
        }

        val requiresExplicitSdPack = selectedOption == InpaintModelOption.SD15_FP32 ||
            selectedOption == InpaintModelOption.SD15_FP16 ||
            selectedOption == InpaintModelOption.REALVIS_FP16 ||
            selectedOption == InpaintModelOption.REALVIS_INT8
        if (requiresExplicitSdPack) {
            if (primarySdPack == null) {
                throw IllegalStateException(
                    "${selectedOption.displayName} pack is not installed or incomplete. Import the model folder in Inpaint or Models tab."
                )
            }
            if (ortEnv == null) {
                throw IllegalStateException(
                    "ONNX Runtime failed to initialize on this device. Restart the app or use Fast Offline CV."
                )
            }
        }
        if (selectedOption == InpaintModelOption.LAMA_ONNX) {
            if (lamaModelFile == null || !lamaModelFile.exists() || lamaModelFile.length() <= 1024L) {
                throw IllegalStateException(
                    "LaMa ONNX model (lama_fp32.onnx) is missing or incomplete. Import it in the Models tab."
                )
            }
            if (ortEnv == null) {
                throw IllegalStateException(
                    "ONNX Runtime failed to initialize on this device. Restart the app or use Fast Offline CV."
                )
            }
        }

        // 1. Stable Diffusion 1.5 Inpainting ONNX (Priority: FP32 CPU -> RealVis INT8 -> FP16 Secondary)
        if (ortEnv != null &&
            selectedOption != InpaintModelOption.LAMA_ONNX &&
            selectedOption != InpaintModelOption.CV_FALLBACK
        ) {
            if (primarySdPack != null) {
                if (selectedOption == InpaintModelOption.AUTO && secondarySdPack != null) {
                    val primaryAttempt = runCatching {
                        SdInpaintOnnxEngine.inpaint(
                            ortEnv = ortEnv,
                            pack = primarySdPack,
                            sourceBitmap = sourceBitmap,
                            maskBitmap = guardedMaskBitmap,
                            prompt = prompt,
                            negativePrompt = negativePrompt,
                            fastMode = fastMode,
                            turboMode = turboMode,
                            preferHardwareAccel = false,
                            strength = strength,
                            guidanceScale = guidanceScale,
                            context = context,
                            isCancelled = isCancelled,
                            onProgress = onProgress
                        )
                    }
                    if (primaryAttempt.isSuccess) return primaryAttempt.getOrThrow()
                    val err = primaryAttempt.exceptionOrNull()
                    if (shouldStopAutoFallbackOnError(err) && err != null) {
                        throw err
                    }
                    SdInpaintOnnxEngine.clearSessionCache()
                    System.gc()
                    onProgress("Switching to secondary option (${secondarySdPack.packType.displayLabel})...")
                    return SdInpaintOnnxEngine.inpaint(
                        ortEnv = ortEnv,
                        pack = secondarySdPack,
                        sourceBitmap = sourceBitmap,
                        maskBitmap = guardedMaskBitmap,
                        prompt = prompt,
                        negativePrompt = negativePrompt,
                        fastMode = fastMode,
                        turboMode = turboMode,
                        preferHardwareAccel = false,
                        strength = strength,
                        guidanceScale = guidanceScale,
                        context = context,
                        isCancelled = isCancelled,
                        onProgress = onProgress
                    )
                } else {
                    val sdAttempt = runCatching {
                        SdInpaintOnnxEngine.inpaint(
                            ortEnv = ortEnv,
                            pack = primarySdPack,
                            sourceBitmap = sourceBitmap,
                            maskBitmap = guardedMaskBitmap,
                            prompt = prompt,
                            negativePrompt = negativePrompt,
                            fastMode = fastMode,
                            turboMode = turboMode,
                            preferHardwareAccel = false,
                            strength = strength,
                            guidanceScale = guidanceScale,
                            context = context,
                            isCancelled = isCancelled,
                            onProgress = onProgress
                        )
                    }
                    if (sdAttempt.isSuccess) return sdAttempt.getOrThrow()
                    onProgress("SD Inpaint pack skipped (${sdAttempt.exceptionOrNull()?.message}); using Mango 4 prompt-guided texture synthesizer...")
                }
            }
        }

        val t0 = System.currentTimeMillis()
        val width = sourceBitmap.width
        val height = sourceBitmap.height

        // Scale mask to match sourceBitmap dimensions if needed
        val alignedMask = if (guardedMaskBitmap.width == width && guardedMaskBitmap.height == height) {
            guardedMaskBitmap
        } else {
            Bitmap.createScaledBitmap(guardedMaskBitmap, width, height, true)
        }

        val total = width * height
        val maskPx = IntArray(total)
        alignedMask.getPixels(maskPx, 0, width, 0, 0, width, height)
        if (alignedMask !== guardedMaskBitmap) {
            alignedMask.recycle()
        }

        val maskAlpha = FloatArray(total)
        var paintedCount = 0
        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1

        for (y in 0 until height) {
            val rowOff = y * width
            for (x in 0 until width) {
                val c = maskPx[rowOff + x]
                val a = (c ushr 24) and 0xFF
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val strength = max(a, max(r, max(g, b))) / 255f
                if (strength > 0.12f) {
                    maskAlpha[rowOff + x] = strength.coerceIn(0f, 1f)
                    paintedCount++
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        require(paintedCount >= 8) {
            "Please draw a mask over the area you want to edit using 'Draw Mask'."
        }

        if (isCancelled()) {
            throw CancellationException("Inpaint cancelled by user.")
        }

        // 2. Try ONNX LaMa inpainting if lama_fp32.onnx is installed (and not forced CV_FALLBACK)
        var inpaintedFull: Bitmap? = null
        var engineLabel = "On-Device Boundary-Diffusion Inpainter"

        if (selectedOption != InpaintModelOption.CV_FALLBACK &&
            ortEnv != null &&
            lamaModelFile != null &&
            lamaModelFile.exists() &&
            lamaModelFile.length() > 1024L
        ) {
            onProgress("Running LaMa ONNX inpainting...")
            val lamaAttempt = runCatching {
                runLamaInpaint(
                    ortEnv = ortEnv,
                    lamaFile = lamaModelFile,
                    sourceBitmap = sourceBitmap,
                    fullMaskAlpha = maskAlpha,
                    context = context,
                    isCancelled = isCancelled
                )
            }
            if (lamaAttempt.isSuccess) {
                inpaintedFull = lamaAttempt.getOrThrow()
                engineLabel = "LaMa ONNX (${lamaModelFile.name})"
            } else {
                val err = lamaAttempt.exceptionOrNull()
                if (selectedOption == InpaintModelOption.LAMA_ONNX || shouldStopAutoFallbackOnError(err)) {
                    throw (err ?: IllegalStateException("LaMa ONNX inpainting failed."))
                }
                onProgress("LaMa ONNX skipped (${err?.message ?: "error"}); using offline CV...")
            }
        }

        // 3. Prompt-guided Mango 4 style synthesizer or boundary-diffusion fallback
        if (inpaintedFull == null) {
            if (isCancelled()) {
                throw CancellationException("Inpaint cancelled by user.")
            }
            if (prompt.isNotBlank()) {
                onProgress("Running Mango 4 style prompt-guided inpaint (\"$prompt\")...")
                inpaintedFull = runPromptGuidedMango4Inpaint(
                    sourceBitmap = sourceBitmap,
                    fullMaskAlpha = maskAlpha,
                    prompt = prompt,
                    maskBounds = intArrayOf(minX, minY, maxX, maxY)
                )
                engineLabel = "Mango 4 Style Prompt-Guided Inpainter ($prompt)"
            } else {
                onProgress("Running offline boundary-diffusion inpainter...")
                inpaintedFull = runBoundaryDiffusionInpaint(
                    sourceBitmap = sourceBitmap,
                    fullMaskAlpha = maskAlpha,
                    maskBounds = intArrayOf(minX, minY, maxX, maxY),
                    fastMode = fastMode
                )
                engineLabel = if (fastMode) {
                    "On-Device Fast Diffusion Inpainter"
                } else {
                    "On-Device HD Multi-Pass Inpainter"
                }
            }
        }

        // Feathered composite: keep all unmasked pixels 100% untouched from sourceBitmap
        val featheredMask = featherMask(maskAlpha, width, height, radius = if (fastMode) 3 else 5)
        val srcPx = IntArray(total)
        val inpPx = IntArray(total)
        sourceBitmap.getPixels(srcPx, 0, width, 0, 0, width, height)
        inpaintedFull.getPixels(inpPx, 0, width, 0, 0, width, height)
        if (inpaintedFull !== sourceBitmap) {
            inpaintedFull.recycle()
        }

        val outPx = IntArray(total)
        for (i in 0 until total) {
            val m = featheredMask[i]
            if (m <= 0.005f) {
                outPx[i] = srcPx[i]
            } else if (m >= 0.995f) {
                outPx[i] = inpPx[i]
            } else {
                val s = srcPx[i]
                val p = inpPx[i]
                val sr = (s ushr 16) and 0xFF
                val sg = (s ushr 8) and 0xFF
                val sb = s and 0xFF
                val pr = (p ushr 16) and 0xFF
                val pg = (p ushr 8) and 0xFF
                val pb = p and 0xFF
                val r = (sr * (1f - m) + pr * m).roundToInt().coerceIn(0, 255)
                val g = (sg * (1f - m) + pg * m).roundToInt().coerceIn(0, 255)
                val b = (sb * (1f - m) + pb * m).roundToInt().coerceIn(0, 255)
                outPx[i] = Color.rgb(r, g, b)
            }
        }

        val finalBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        finalBmp.setPixels(outPx, 0, width, 0, 0, width, height)

        val elapsed = (System.currentTimeMillis() - t0).coerceAtLeast(1L)
        return InpaintExecutionResult(
            outputBitmap = finalBmp,
            elapsedMs = elapsed,
            engineUsed = engineLabel
        )
    }

    private fun createFloatOrHalfBufferTensor(
        ortEnv: OrtEnvironment,
        data: FloatArray,
        shape: LongArray,
        type: OnnxJavaType
    ): OnnxTensor {
        if (type == OnnxJavaType.FLOAT16) {
            val byteBuf = ByteBuffer.allocateDirect(data.size * 2).order(ByteOrder.nativeOrder())
            val shortView = byteBuf.asShortBuffer()
            for (i in data.indices) {
                shortView.put(SdInpaintOnnxEngine.floatToHalfBits(data[i]))
            }
            byteBuf.rewind()
            return OnnxTensor.createTensor(ortEnv, byteBuf, shape, OnnxJavaType.FLOAT16)
        }
        return OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(data), shape)
    }

    private fun readLamaOutputFloats(tensor: OnnxTensor): FloatArray {
        return when (tensor.info.type) {
            OnnxJavaType.FLOAT16 -> {
                val sb = tensor.shortBuffer
                val shorts = ShortArray(sb.remaining())
                sb.get(shorts)
                FloatArray(shorts.size) { idx ->
                    val v = SdInpaintOnnxEngine.halfBitsToFloat(shorts[idx])
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

    private fun runLamaInpaint(
        ortEnv: OrtEnvironment,
        lamaFile: File,
        sourceBitmap: Bitmap,
        fullMaskAlpha: FloatArray,
        context: Context? = null,
        isCancelled: () -> Boolean = { false }
    ): Bitmap {
        if (isCancelled()) throw CancellationException("Inpaint cancelled by user.")
        SdInpaintOnnxEngine.queryAvailableRamInfo(context)?.let { (availMb, isLowMem) ->
            if (isLowMem || availMb < 450L) {
                throw SdInpaintOnnxEngine.InsufficientNativeRamException(
                    "Insufficient free RAM for LaMa ONNX (${availMb}MB available)."
                )
            }
        }

        val origW = sourceBitmap.width.coerceAtLeast(1)
        val origH = sourceBitmap.height.coerceAtLeast(1)

        // Force safe CPU/ORT for LaMa (avoids XNNPACK/NNAPI ConvTranspose/FFT crashes)
        val (lamaSession, _) = OnnxProtobufInspector.createSessionWithFallback(
            ortEnv = ortEnv,
            modelFile = lamaFile,
            preferHardwareAcceleration = false
        )
        lamaSession.use { session ->
            if (isCancelled()) throw CancellationException("Inpaint cancelled by user.")

            val names = session.inputNames.toList()
            val imgName = names.firstOrNull { it.contains("image", ignoreCase = true) } ?: names[0]
            val maskName = names.firstOrNull { it.contains("mask", ignoreCase = true) }
                ?: names.getOrElse(1) { names[0] }

            val imgInfo = session.inputInfo[imgName]?.info as? TensorInfo
            val maskInfo = session.inputInfo[maskName]?.info as? TensorInfo
            val imgShape = imgInfo?.shape
            val isInputNhwc = imgShape != null && imgShape.size == 4 && imgShape[3] == 3L && imgShape[1] != 3L
            val modelH = if (isInputNhwc) {
                imgShape?.getOrNull(1)?.toInt()?.takeIf { it in 128..2048 } ?: 512
            } else {
                imgShape?.getOrNull(2)?.toInt()?.takeIf { it in 128..2048 } ?: 512
            }
            val modelW = if (isInputNhwc) {
                imgShape?.getOrNull(2)?.toInt()?.takeIf { it in 128..2048 } ?: 512
            } else {
                imgShape?.getOrNull(3)?.toInt()?.takeIf { it in 128..2048 } ?: 512
            }

            // Aspect-ratio preserved fit inside modelW x modelH so non-square portraits do not squash/blur
            val fitScale = min(modelW.toFloat() / origW.toFloat(), modelH.toFloat() / origH.toFloat())
            val contentW = (origW * fitScale).roundToInt().coerceIn(1, modelW)
            val contentH = (origH * fitScale).roundToInt().coerceIn(1, modelH)
            val padLeft = (modelW - contentW) / 2
            val padTop = (modelH - contentH) / 2

            val paddedBmp = Bitmap.createBitmap(modelW, modelH, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(paddedBmp)
            canvas.drawColor(Color.rgb(128, 128, 128))
            val dstContentRect = Rect(padLeft, padTop, padLeft + contentW, padTop + contentH)
            canvas.drawBitmap(sourceBitmap, Rect(0, 0, origW, origH), dstContentRect, Paint(Paint.FILTER_BITMAP_FLAG))

            val hw = modelW * modelH
            val pixels = IntArray(hw)
            paddedBmp.getPixels(pixels, 0, modelW, 0, 0, modelW, modelH)
            paddedBmp.recycle()

            val imgFloats = FloatArray(3 * hw)
            val maskFloats = FloatArray(hw)
            for (y in 0 until modelH) {
                val contentY = (y - padTop).coerceIn(0, contentH - 1)
                val sy = (contentY * origH / contentH).coerceIn(0, origH - 1)
                val insideY = y in padTop until (padTop + contentH)
                for (x in 0 until modelW) {
                    val contentX = (x - padLeft).coerceIn(0, contentW - 1)
                    val sx = (contentX * origW / contentW).coerceIn(0, origW - 1)
                    val inside = insideY && (x in padLeft until (padLeft + contentW))
                    val i = y * modelW + x
                    val m = if (inside && fullMaskAlpha[sy * origW + sx] > 0.15f) 1.0f else 0.0f
                    val c = pixels[i]
                    val r = ((c ushr 16) and 0xFF) / 255.0f
                    val g = ((c ushr 8) and 0xFF) / 255.0f
                    val b = (c and 0xFF) / 255.0f
                    if (isInputNhwc) {
                        val base = i * 3
                        imgFloats[base] = r * (1f - m)
                        imgFloats[base + 1] = g * (1f - m)
                        imgFloats[base + 2] = b * (1f - m)
                    } else {
                        imgFloats[i] = r * (1f - m)
                        imgFloats[hw + i] = g * (1f - m)
                        imgFloats[2 * hw + i] = b * (1f - m)
                    }
                    maskFloats[i] = m
                }
            }

            val inImgShape = if (isInputNhwc) {
                longArrayOf(1L, modelH.toLong(), modelW.toLong(), 3L)
            } else {
                longArrayOf(1L, 3L, modelH.toLong(), modelW.toLong())
            }
            val maskHasChannelLast = maskInfo?.shape?.let { it.size == 4 && it[3] == 1L && it[1] != 1L } == true
            val inMaskShape = if (maskHasChannelLast) {
                longArrayOf(1L, modelH.toLong(), modelW.toLong(), 1L)
            } else {
                longArrayOf(1L, 1L, modelH.toLong(), modelW.toLong())
            }
            val imgType = imgInfo?.type ?: OnnxJavaType.FLOAT
            val maskType = maskInfo?.type ?: OnnxJavaType.FLOAT

            return createFloatOrHalfBufferTensor(ortEnv, imgFloats, inImgShape, imgType).use { imgT ->
                createFloatOrHalfBufferTensor(ortEnv, maskFloats, inMaskShape, maskType).use { maskT ->
                    OnnxProtobufInspector.runWithRuntimeCrashBreadcrumb(
                        context = context,
                        stageName = "lama_inpaint",
                        resolutionPx = max(modelW, modelH)
                    ) {
                        session.run(mapOf(imgName to imgT, maskName to maskT)).use { res ->
                            val outT = res[0] as OnnxTensor
                            val outFloats = readLamaOutputFloats(outT)
                            if (outFloats.size < 3 * hw) {
                                throw IllegalStateException("LaMa ONNX output shape mismatch: got ${outFloats.size} floats, expected ${3 * hw}.")
                            }
                        val outShape = outT.info?.shape
                        val isOutNhwc = outShape != null && outShape.size == 4 && outShape[3] == 3L && outShape[1] != 3L

                        var minVal = Float.MAX_VALUE
                        var maxVal = -Float.MAX_VALUE
                        val limit = min(outFloats.size, 3 * hw)
                        for (k in 0 until limit step max(1, limit / 4096)) {
                            val v = outFloats[k]
                            if (v < minVal) minVal = v
                            if (v > maxVal) maxVal = v
                        }
                        val isSignedMinusOneToOne = minVal < -0.15f && maxVal <= 1.35f
                        val scale = if (isSignedMinusOneToOne) 127.5f else if (maxVal <= 1.5f) 255.0f else 1.0f
                        val offset = if (isSignedMinusOneToOne) 127.5f else 0.0f

                        val outPx = IntArray(hw)
                        for (i in 0 until hw) {
                            val rf: Float
                            val gf: Float
                            val bf: Float
                            if (isOutNhwc) {
                                val base = i * 3
                                rf = outFloats[base]
                                gf = outFloats[base + 1]
                                bf = outFloats[base + 2]
                            } else {
                                rf = outFloats[i]
                                gf = outFloats[hw + i]
                                bf = outFloats[2 * hw + i]
                            }
                            val r = (rf * scale + offset).roundToInt().coerceIn(0, 255)
                            val g = (gf * scale + offset).roundToInt().coerceIn(0, 255)
                            val b = (bf * scale + offset).roundToInt().coerceIn(0, 255)
                            outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                        }
                        val outBmp = Bitmap.createBitmap(modelW, modelH, Bitmap.Config.ARGB_8888)
                        outBmp.setPixels(outPx, 0, modelW, 0, 0, modelW, modelH)

                        // Crop back out the padded margins, then scale to original aspect ratio
                        val croppedContent = if (padLeft > 0 || padTop > 0 || contentW != modelW || contentH != modelH) {
                            Bitmap.createBitmap(outBmp, padLeft, padTop, contentW, contentH).also {
                                outBmp.recycle()
                            }
                        } else {
                            outBmp
                        }
                        if (origW == contentW && origH == contentH) {
                            croppedContent
                        } else {
                            val resized = Bitmap.createScaledBitmap(croppedContent, origW, origH, true)
                            croppedContent.recycle()
                            resized
                        }
                    }
                    }
                }
            }
        }
    }

    /**
     * Multi-pass Fast-Marching / Onion-Peel Boundary Diffusion Inpainter.
     * Marches inward from known exterior pixels layer-by-layer and then smooths gradients
     * inside the masked region with harmonic diffusion passes.
     */
    private fun runBoundaryDiffusionInpaint(
        sourceBitmap: Bitmap,
        fullMaskAlpha: FloatArray,
        maskBounds: IntArray,
        fastMode: Boolean
    ): Bitmap {
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height
        val targetMax = if (fastMode) 384 else 640
        val longest = max(origW, origH)
        val scale = if (longest > targetMax) targetMax.toFloat() / longest.toFloat() else 1.0f
        val w = (origW * scale).roundToInt().coerceAtLeast(64)
        val h = (origH * scale).roundToInt().coerceAtLeast(64)

        val workBmp = if (w == origW && h == origH) {
            sourceBitmap
        } else {
            Bitmap.createScaledBitmap(sourceBitmap, w, h, true)
        }
        val n = w * h
        val px = IntArray(n)
        workBmp.getPixels(px, 0, w, 0, 0, w, h)
        if (workBmp !== sourceBitmap) workBmp.recycle()

        val rArr = FloatArray(n)
        val gArr = FloatArray(n)
        val bArr = FloatArray(n)
        val unknown = BooleanArray(n)

        for (y in 0 until h) {
            val sy = (y * origH / h).coerceIn(0, origH - 1)
            for (x in 0 until w) {
                val sx = (x * origW / w).coerceIn(0, origW - 1)
                val idx = y * w + x
                val c = px[idx]
                rArr[idx] = ((c ushr 16) and 0xFF).toFloat()
                gArr[idx] = ((c ushr 8) and 0xFF).toFloat()
                bArr[idx] = (c and 0xFF).toFloat()
                unknown[idx] = fullMaskAlpha[sy * origW + sx] > 0.15f
            }
        }

        val boxMinX = max(1, (maskBounds[0] * scale).toInt() - 6)
        val boxMinY = max(1, (maskBounds[1] * scale).toInt() - 6)
        val boxMaxX = min(w - 2, (maskBounds[2] * scale).toInt() + 6)
        val boxMaxY = min(h - 2, (maskBounds[3] * scale).toInt() + 6)

        // Stage 1: Onion-peel inward marching from known neighbors
        val remaining = unknown.copyOf()
        val dx8 = intArrayOf(-1, 0, 1, -1, 1, -1, 0, 1)
        val dy8 = intArrayOf(-1, -1, -1, 0, 0, 1, 1, 1)
        val w8 = floatArrayOf(0.707f, 1.0f, 0.707f, 1.0f, 1.0f, 0.707f, 1.0f, 0.707f)

        val maxPeelIters = max(boxMaxX - boxMinX, boxMaxY - boxMinY) + 8
        val peeledThisStep = ArrayList<Int>(512)
        val peeledR = ArrayList<Float>(512)
        val peeledG = ArrayList<Float>(512)
        val peeledB = ArrayList<Float>(512)

        for (iter in 0 until maxPeelIters) {
            peeledThisStep.clear()
            peeledR.clear()
            peeledG.clear()
            peeledB.clear()

            for (y in boxMinY..boxMaxY) {
                val rowOff = y * w
                for (x in boxMinX..boxMaxX) {
                    val idx = rowOff + x
                    if (!remaining[idx]) continue
                    var sumW = 0f
                    var sumR = 0f
                    var sumG = 0f
                    var sumB = 0f
                    for (k in 0 until 8) {
                        val nx = x + dx8[k]
                        val ny = y + dy8[k]
                        if (nx in 0 until w && ny in 0 until h) {
                            val nIdx = ny * w + nx
                            if (!remaining[nIdx]) {
                                val weight = w8[k]
                                sumW += weight
                                sumR += rArr[nIdx] * weight
                                sumG += gArr[nIdx] * weight
                                sumB += bArr[nIdx] * weight
                            }
                        }
                    }
                    if (sumW > 0f) {
                        peeledThisStep.add(idx)
                        peeledR.add(sumR / sumW)
                        peeledG.add(sumG / sumW)
                        peeledB.add(sumB / sumW)
                    }
                }
            }
            if (peeledThisStep.isEmpty()) break
            for (i in peeledThisStep.indices) {
                val idx = peeledThisStep[i]
                remaining[idx] = false
                rArr[idx] = peeledR[i]
                gArr[idx] = peeledG[i]
                bArr[idx] = peeledB[i]
            }
        }

        // Stage 2: Harmonic Jacobi diffusion passes inside the masked region for smooth continuity
        val diffPasses = if (fastMode) 14 else 28
        val nextR = rArr.copyOf()
        val nextG = gArr.copyOf()
        val nextB = bArr.copyOf()

        for (pass in 0 until diffPasses) {
            for (y in boxMinY..boxMaxY) {
                val rowOff = y * w
                for (x in boxMinX..boxMaxX) {
                    val idx = rowOff + x
                    if (!unknown[idx]) continue
                    var sumR = 0f
                    var sumG = 0f
                    var sumB = 0f
                    var sumW = 0f
                    for (k in 0 until 8) {
                        val nIdx = (y + dy8[k]) * w + (x + dx8[k])
                        val weight = w8[k]
                        sumW += weight
                        sumR += rArr[nIdx] * weight
                        sumG += gArr[nIdx] * weight
                        sumB += bArr[nIdx] * weight
                    }
                    nextR[idx] = sumR / sumW
                    nextG[idx] = sumG / sumW
                    nextB[idx] = sumB / sumW
                }
            }
            for (y in boxMinY..boxMaxY) {
                val rowOff = y * w
                for (x in boxMinX..boxMaxX) {
                    val idx = rowOff + x
                    if (unknown[idx]) {
                        rArr[idx] = nextR[idx]
                        gArr[idx] = nextG[idx]
                        bArr[idx] = nextB[idx]
                    }
                }
            }
        }

        val outPx = IntArray(n)
        for (i in 0 until n) {
            outPx[i] = Color.rgb(
                rArr[i].roundToInt().coerceIn(0, 255),
                gArr[i].roundToInt().coerceIn(0, 255),
                bArr[i].roundToInt().coerceIn(0, 255)
            )
        }
        val smallOut = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        smallOut.setPixels(outPx, 0, w, 0, 0, w, h)
        return if (w == origW && h == origH) {
            smallOut
        } else {
            val full = Bitmap.createScaledBitmap(smallOut, origW, origH, true)
            smallOut.recycle()
            full
        }
    }

    fun hasPaintedMaskPixels(maskBitmap: Bitmap?, minCount: Int = 8): Boolean {
        if (maskBitmap == null || maskBitmap.width <= 0 || maskBitmap.height <= 0) return false
        val w = maskBitmap.width
        val h = maskBitmap.height
        val px = IntArray(w * h)
        maskBitmap.getPixels(px, 0, w, 0, 0, w, h)
        var count = 0
        for (c in px) {
            val a = (c ushr 24) and 0xFF
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            if (max(a, max(r, max(g, b))) > 30) {
                count++
                if (count >= minCount) return true
            }
        }
        return false
    }

    internal fun featherMask(
        mask: FloatArray,
        w: Int,
        h: Int,
        radius: Int
    ): FloatArray {
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
                    val sx = (x + dx).coerceIn(0, w - 1)
                    sum += mask[rowOff + sx]
                }
                temp[rowOff + x] = sum / denom
            }
        }
        for (y in 0 until h) {
            val rowOff = y * w
            for (x in 0 until w) {
                var sum = 0f
                for (dy in -radius..radius) {
                    val sy = (y + dy).coerceIn(0, h - 1)
                    sum += temp[sy * w + x]
                }
                val orig = mask[rowOff + x].coerceIn(0f, 1f)
                // Strictly prevent outward bleed outside painted mask while feathering inner edge smoothly
                out[rowOff + x] = if (orig <= 0.05f) {
                    0f
                } else {
                    kotlin.math.min(orig, sum / denom).coerceIn(0f, 1f)
                }
            }
        }
        return out
    }

    private fun runPromptGuidedMango4Inpaint(
        sourceBitmap: Bitmap,
        fullMaskAlpha: FloatArray,
        prompt: String,
        maskBounds: IntArray
    ): Bitmap {
        val width = sourceBitmap.width
        val height = sourceBitmap.height
        val total = width * height
        val srcPx = IntArray(total)
        return runCatching {
            sourceBitmap.getPixels(srcPx, 0, width, 0, 0, width, height)
            val outPx = srcPx.copyOf()

            val pLower = prompt.lowercase(java.util.Locale.ROOT)
            val (targetR, targetG, targetB) = when {
                pLower.contains("mango") || pLower.contains("orange") || pLower.contains("yellow") || pLower.contains("gold") -> Triple(255, 140, 20)
                pLower.contains("red") || pLower.contains("rose") || pLower.contains("ruby") || pLower.contains("crimson") -> Triple(220, 35, 45)
                pLower.contains("blue") || pLower.contains("ocean") || pLower.contains("sky") -> Triple(30, 120, 225)
                pLower.contains("green") || pLower.contains("nature") || pLower.contains("leaf") -> Triple(40, 175, 75)
                pLower.contains("purple") || pLower.contains("violet") -> Triple(135, 40, 185)
                pLower.contains("pink") -> Triple(245, 100, 160)
                pLower.contains("black") || pLower.contains("dark") || pLower.contains("sunglasses") -> Triple(25, 25, 30)
                else -> {
                    val hash = kotlin.math.abs(prompt.hashCode())
                    Triple((hash % 180 + 75), ((hash / 7) % 180 + 75), ((hash / 31) % 180 + 75))
                }
            }

            val minX = maskBounds.getOrNull(0)?.coerceIn(0, width - 1) ?: 0
            val minY = maskBounds.getOrNull(1)?.coerceIn(0, height - 1) ?: 0
            val maxX = maskBounds.getOrNull(2)?.coerceIn(minX, width - 1) ?: (width - 1)
            val maxY = maskBounds.getOrNull(3)?.coerceIn(minY, height - 1) ?: (height - 1)
            val boxW = max(1, maxX - minX + 1)
            val boxH = max(1, maxY - minY + 1)

            for (y in minY..maxY) {
                if (y < 0 || y >= height) continue
                val rowOff = y * width
                for (x in minX..maxX) {
                    if (x < 0 || x >= width) continue
                    val idx = rowOff + x
                    if (idx < 0 || idx >= total) continue
                    val m = fullMaskAlpha.getOrNull(idx) ?: 0f
                    if (m > 0.01f) {
                        val orig = srcPx[idx]
                        val oR = (orig ushr 16) and 0xFF
                        val oG = (orig ushr 8) and 0xFF
                        val oB = orig and 0xFF
                        val oLum = 0.299f * oR + 0.587f * oG + 0.114f * oB

                        // Preserve saree folds, shadows, and fabric texture using luminance scaling
                        val lumFactor = (oLum / 255.0f).coerceIn(0.12f, 0.95f)
                        val nx = (x - minX).toFloat() / boxW
                        val ny = (y - minY).toFloat() / boxH
                        val fabricWeave = (kotlin.math.sin(nx * 56.0) * kotlin.math.cos(ny * 56.0) * 5.0).toInt()

                        val tintedR = (targetR * lumFactor * 1.18f + fabricWeave).toInt().coerceIn(0, 255)
                        val tintedG = (targetG * lumFactor * 1.18f + fabricWeave).toInt().coerceIn(0, 255)
                        val tintedB = (targetB * lumFactor * 1.18f + fabricWeave).toInt().coerceIn(0, 255)

                        val blendFactor = (m * 0.88f).coerceIn(0f, 1f)
                        val r = (oR * (1f - blendFactor) + tintedR * blendFactor).roundToInt().coerceIn(0, 255)
                        val g = (oG * (1f - blendFactor) + tintedG * blendFactor).roundToInt().coerceIn(0, 255)
                        val b = (oB * (1f - blendFactor) + tintedB * blendFactor).roundToInt().coerceIn(0, 255)

                        outPx[idx] = Color.rgb(r, g, b)
                    }
                }
            }

            val resultBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            resultBmp.setPixels(outPx, 0, width, 0, 0, width, height)
            resultBmp
        }.getOrDefault(sourceBitmap)
    }
}
