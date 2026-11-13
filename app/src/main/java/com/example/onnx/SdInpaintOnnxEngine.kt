package com.example.onnx

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
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
    private const val SD_RES = 512
    private const val LATENT_RES = 64
    private const val LATENT_CHANNELS = 4
    private const val VAE_SCALE_FACTOR = 0.18215f
    const val DEFAULT_STRENGTH = 0.70f
    const val DEFAULT_GUIDANCE_SCALE = 7.0f
    private const val DEFAULT_NEGATIVE_PROMPT =
        "lowres, bad anatomy, blurry, distorted, deformed, watermark, text"

    internal data class DenoisePlan(
        val baseSteps: Int,
        val denoiseSteps: Int,
        val startStepIndex: Int,
        val strength: Float
    )

    internal fun computeDenoisePlan(fastMode: Boolean, strength: Float): DenoisePlan {
        val baseSteps = if (fastMode) 16 else 24
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

    private fun logDebug(message: String) {
        runCatching { Log.d(TAG, message) }
    }

    fun inpaint(
        ortEnv: OrtEnvironment,
        pack: InstalledSdInpaintPack,
        sourceBitmap: Bitmap,
        maskBitmap: Bitmap,
        prompt: String,
        fastMode: Boolean,
        preferHardwareAccel: Boolean = false,
        strength: Float = DEFAULT_STRENGTH,
        guidanceScale: Float = DEFAULT_GUIDANCE_SCALE,
        skipCompositeDebug: Boolean = false,
        seed: Long = System.currentTimeMillis(),
        onProgress: (String) -> Unit = {}
    ): InpaintExecutionResult {

        val t0 = System.currentTimeMillis()
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height
        val plan = computeDenoisePlan(fastMode = fastMode, strength = strength)
        val clampedCfg = guidanceScale.coerceIn(1.0f, 15.0f)

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

        // Release any active FaceSwap sessions so SD 1.5 components get full native RAM
        OnnxProtobufInspector.clearCachedSessions()
        System.gc()

        // 1. Compute context-aware square ROI aligned around the painted mask + dilate 512px mask by 5px
        val roi = computeSquareRoi(origW, origH, minX, minY, maxX, maxY)
        val (cropBmp512, rawMask512) = extract512CropAndMask(sourceBitmap, fullMaskAlpha, origW, origH, roi)
        val mask512 = dilateBinaryMask(rawMask512, SD_RES, SD_RES, radiusPx = 5)
        val rawMaskMean = rawMask512.sum() / rawMask512.size.toFloat()
        val dilatedMaskMean = mask512.sum() / mask512.size.toFloat()

        val effectivePrompt = prompt.trim().ifEmpty { "clean natural texture, high quality photo" }
        onProgress("1/4 Encoding prompt (${pack.packType.displayLabel})...")

        // 2. CLIP Text Encoder -> uncond & cond embeddings [77, 768]
        val tokenizer = ClipBpeTokenizer.loadFromDirectory(pack.tokenizerDir)
        val condIds = tokenizer.tokenizeTo77(effectivePrompt)
        val uncondIds = tokenizer.tokenizeTo77(DEFAULT_NEGATIVE_PROMPT)
        logDebug(
            "tokenizerLoaded=${tokenizer.isLoaded()} vocabSize=${tokenizer.vocabSize} " +
                "tokDir=${pack.tokenizerDir?.absolutePath} condIds=${condIds.take(10).joinToString()}"
        )
        val (uncondEmbed, condEmbed) = runTextEncoder(
            ortEnv = ortEnv,
            textEncoderFile = pack.textEncoderOnnx,
            uncondIds = uncondIds,
            condIds = condIds,
            preferHardwareAccel = preferHardwareAccel
        )
        System.gc()

        // 3. VAE Encoder -> masked image latent [4, 64, 64] & original latent [4, 64, 64]
        onProgress("2/4 Encoding image into 64×64 latent space...")
        val maskedLatent = runVaeEncoder(
            ortEnv = ortEnv,
            vaeEncoderFile = pack.vaeEncoderOnnx,
            cropBmp512 = cropBmp512,
            mask512 = mask512,
            applyZeroMask = true,
            preferHardwareAccel = preferHardwareAccel
        )
        val initLatent = runVaeEncoder(
            ortEnv = ortEnv,
            vaeEncoderFile = pack.vaeEncoderOnnx,
            cropBmp512 = cropBmp512,
            mask512 = mask512,
            applyZeroMask = false,
            preferHardwareAccel = preferHardwareAccel
        )
        cropBmp512.recycle()
        System.gc()

        // Downsample dilated 512x512 mask -> 64x64 latent mask
        val mask64 = downsampleMask512To64(mask512)
        val latentMaskMean = mask64.sum() / mask64.size.toFloat()
        logDebug(
            "mask mean full=$rawMaskMean dilated512=$dilatedMaskMean latent=$latentMaskMean " +
                "roi=[${roi.left},${roi.top},size=${roi.size}]"
        )

        // 4. UNet Denoising Loop (Euler Discrete Scheduler)
        val denoisedLatent = runUnetDenoisingLoop(
            ortEnv = ortEnv,
            unetFile = pack.unetOnnx,
            uncondEmbed = uncondEmbed,
            condEmbed = condEmbed,
            maskedLatent = maskedLatent,
            initLatent = initLatent,
            mask64 = mask64,
            plan = plan,
            guidanceScale = clampedCfg,
            seed = seed,
            preferHardwareAccel = preferHardwareAccel,
            onStep = { step, total ->
                onProgress("3/4 Diffusion denoising step $step/$total...")
            }
        )
        System.gc()

        // 5. VAE Decoder -> 512x512 RGB Bitmap
        onProgress("4/4 Decoding latent to HD image...")
        val decodedCrop512 = runVaeDecoder(
            ortEnv = ortEnv,
            vaeDecoderFile = pack.vaeDecoderOnnx,
            latent = denoisedLatent,
            preferHardwareAccel = preferHardwareAccel
        )
        System.gc()

        // 6. Composite strictly inside masked region back onto full-resolution sourceBitmap
        val resultBitmap = if (skipCompositeDebug) {
            pasteCropDirectDebug(sourceBitmap = sourceBitmap, crop512 = decodedCrop512, roi = roi)
        } else {
            compositeCropIntoSource(
                sourceBitmap = sourceBitmap,
                fullMaskAlpha = fullMaskAlpha,
                crop512 = decodedCrop512,
                roi = roi,
                featherRadius = if (fastMode) 4 else 6
            )
        }
        decodedCrop512.recycle()

        val elapsed = (System.currentTimeMillis() - t0).coerceAtLeast(1L)
        val badgeStr = String.format(
            Locale.US,
            "%s (%d/%d steps • Str %.1f • CFG %.1f)",
            pack.packType.displayLabel,
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

    private data class SquareRoi(
        val left: Int,
        val top: Int,
        val size: Int
    )

    private fun computeSquareRoi(
        imgW: Int,
        imgH: Int,
        minX: Int,
        minY: Int,
        maxX: Int,
        maxY: Int
    ): SquareRoi {
        val maskW = (maxX - minX + 1).coerceAtLeast(1)
        val maskH = (maxY - minY + 1).coerceAtLeast(1)
        val maxImgSide = max(imgW, imgH)
        val minImgSide = min(imgW, imgH)

        val desiredSize = max((max(maskW, maskH) * 1.65f).roundToInt(), (minImgSide * 0.78f).roundToInt())
            .coerceAtLeast(SD_RES.coerceAtMost(maxImgSide))
        val cx = (minX + maxX) / 2
        val cy = (minY + maxY) / 2

        return if (desiredSize <= minImgSide) {
            val left = (cx - desiredSize / 2).coerceIn(0, imgW - desiredSize)
            val top = (cy - desiredSize / 2).coerceIn(0, imgH - desiredSize)
            SquareRoi(left, top, desiredSize)
        } else {
            // Cover entire mask with square canvas centered on the image
            val side = max(maxImgSide, desiredSize).coerceAtMost((maxImgSide * 1.15f).roundToInt())
            val left = (imgW - side) / 2
            val top = (imgH - side) / 2
            SquareRoi(left, top, side)
        }
    }

    private fun extract512CropAndMask(
        source: Bitmap,
        fullMaskAlpha: FloatArray,
        origW: Int,
        origH: Int,
        roi: SquareRoi
    ): Pair<Bitmap, FloatArray> {
        // Render ROI onto square bitmap with neutral border replication
        val roiBmp = Bitmap.createBitmap(roi.size, roi.size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(roiBmp)
        canvas.drawColor(Color.rgb(128, 128, 128))
        val srcLeft = max(0, roi.left)
        val srcTop = max(0, roi.top)
        val srcRight = min(origW, roi.left + roi.size)
        val srcBottom = min(origH, roi.top + roi.size)
        if (srcRight > srcLeft && srcBottom > srcTop) {
            val dstRect = Rect(
                srcLeft - roi.left,
                srcTop - roi.top,
                srcRight - roi.left,
                srcBottom - roi.top
            )
            canvas.drawBitmap(source, Rect(srcLeft, srcTop, srcRight, srcBottom), dstRect, Paint(Paint.FILTER_BITMAP_FLAG))
        }
        val crop512 = if (roiBmp.width == SD_RES && roiBmp.height == SD_RES) {
            roiBmp
        } else {
            val scaled = Bitmap.createScaledBitmap(roiBmp, SD_RES, SD_RES, true)
            roiBmp.recycle()
            scaled
        }

        val mask512 = FloatArray(SD_RES * SD_RES)
        val stepScale = roi.size / SD_RES.toFloat()
        for (y in 0 until SD_RES) {
            val origY0 = roi.top + (y * stepScale).toInt()
            val origY1 = max(origY0, roi.top + ((y + 1) * stepScale).toInt() - 1)
            val clampedY0 = origY0.coerceIn(0, origH - 1)
            val clampedY1 = origY1.coerceIn(0, origH - 1)
            if (origY1 < 0 || origY0 >= origH) continue
            val dstRow = y * SD_RES
            for (x in 0 until SD_RES) {
                val origX0 = roi.left + (x * stepScale).toInt()
                val origX1 = max(origX0, roi.left + ((x + 1) * stepScale).toInt() - 1)
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
                mask512[dstRow + x] = if (maxVal >= 0.20f) 1.0f else 0.0f
            }
        }
        return crop512 to mask512
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

    internal fun downsampleMask512To64(mask512: FloatArray): FloatArray {
        val out = FloatArray(LATENT_RES * LATENT_RES)
        val block = SD_RES / LATENT_RES // 8
        for (ly in 0 until LATENT_RES) {
            for (lx in 0 until LATENT_RES) {
                var sum = 0f
                val y0 = ly * block
                val x0 = lx * block
                for (dy in 0 until block) {
                    val row = (y0 + dy) * SD_RES + x0
                    for (dx in 0 until block) {
                        sum += mask512[row + dx]
                    }
                }
                out[ly * LATENT_RES + lx] = if ((sum / (block * block)) >= 0.12f) 1.0f else 0.0f
            }
        }
        return out
    }

    private fun runTextEncoder(
        ortEnv: OrtEnvironment,
        textEncoderFile: java.io.File,
        uncondIds: LongArray,
        condIds: LongArray,
        preferHardwareAccel: Boolean
    ): Pair<FloatArray, FloatArray> {
        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(textEncoderFile.absolutePath, opts).use { session ->
                val inputName = session.inputNames.first()
                val inputType = (session.inputInfo[inputName]?.info as? TensorInfo)?.type ?: OnnxJavaType.INT32
                val shape = longArrayOf(1L, ClipBpeTokenizer.MAX_LENGTH.toLong())

                val uncondOut = createIdsTensor(ortEnv, uncondIds, shape, inputType).use { t ->
                    session.run(mapOf(inputName to t)).use { res ->
                        readTensorAsFloats(res[0] as OnnxTensor)
                    }
                }
                val condOut = createIdsTensor(ortEnv, condIds, shape, inputType).use { t ->
                    session.run(mapOf(inputName to t)).use { res ->
                        readTensorAsFloats(res[0] as OnnxTensor)
                    }
                }
                return uncondOut to condOut
            }
        }
    }

    private fun runVaeEncoder(
        ortEnv: OrtEnvironment,
        vaeEncoderFile: java.io.File,
        cropBmp512: Bitmap,
        mask512: FloatArray,
        applyZeroMask: Boolean,
        preferHardwareAccel: Boolean
    ): FloatArray {
        val hw = SD_RES * SD_RES
        val px = IntArray(hw)
        cropBmp512.getPixels(px, 0, SD_RES, 0, 0, SD_RES, SD_RES)

        val nchw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            val c = px[i]
            val keep = if (applyZeroMask && mask512[i] >= 0.5f) 0.0f else 1.0f
            val r = (((c ushr 16) and 0xFF) / 127.5f - 1.0f) * keep
            val g = (((c ushr 8) and 0xFF) / 127.5f - 1.0f) * keep
            val b = ((c and 0xFF) / 127.5f - 1.0f) * keep
            nchw[i] = r
            nchw[hw + i] = g
            nchw[2 * hw + i] = b
        }

        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(vaeEncoderFile.absolutePath, opts).use { session ->
                val inputName = session.inputNames.first()
                val inputType = (session.inputInfo[inputName]?.info as? TensorInfo)?.type ?: OnnxJavaType.FLOAT
                val shape = longArrayOf(1L, 3L, SD_RES.toLong(), SD_RES.toLong())

                return createFloatOrHalfTensor(ortEnv, nchw, shape, inputType).use { inputTensor ->
                    session.run(mapOf(inputName to inputTensor)).use { res ->
                        val latent = readTensorAsFloats(res[0] as OnnxTensor)
                        // Some Optimum vae_encoder exports return mean/std 8-channel or 4-channel latent_sample
                        val expectedSize = LATENT_CHANNELS * LATENT_RES * LATENT_RES
                        val out = FloatArray(expectedSize)
                        for (i in 0 until minOf(expectedSize, latent.size)) {
                            out[i] = latent[i] * VAE_SCALE_FACTOR
                        }
                        out
                    }
                }
            }
        }
    }

    private fun runUnetDenoisingLoop(
        ortEnv: OrtEnvironment,
        unetFile: java.io.File,
        uncondEmbed: FloatArray,
        condEmbed: FloatArray,
        maskedLatent: FloatArray,
        initLatent: FloatArray,
        mask64: FloatArray,
        plan: DenoisePlan,
        guidanceScale: Float,
        seed: Long,
        preferHardwareAccel: Boolean,
        onStep: (Int, Int) -> Unit
    ): FloatArray {
        val latentSize = LATENT_CHANNELS * LATENT_RES * LATENT_RES
        val hw64 = LATENT_RES * LATENT_RES

        // Build SD 1.5 scaled_linear Euler Discrete schedule
        val schedule = buildEulerSchedule(plan.baseSteps)
        val rng = Random(seed)
        val startSigma = schedule.sigmas[plan.startStepIndex]
        val latents = if (plan.startStepIndex == 0) {
            FloatArray(latentSize) { sampleGaussian(rng) * startSigma }
        } else {
            FloatArray(latentSize) { i -> initLatent[i] + sampleGaussian(rng) * startSigma }
        }

        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(unetFile.absolutePath, opts).use { session ->
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
                    "UNet channels=$sampleChannels batchSize=$batchSize " +
                        "startIdx=${plan.startStepIndex}/${plan.baseSteps} denoiseSteps=${plan.denoiseSteps} " +
                        "sampleType=$sampleType timestepType=$timestepType encoderType=$encoderType"
                )

                val sampleStride = sampleChannels * hw64
                if (batchSize == 2) {
                    val bothEmbeds = FloatArray(uncondEmbed.size + condEmbed.size)
                    System.arraycopy(uncondEmbed, 0, bothEmbeds, 0, uncondEmbed.size)
                    System.arraycopy(condEmbed, 0, bothEmbeds, uncondEmbed.size, condEmbed.size)
                    val encoderShape2 = longArrayOf(2L, embedTokenCount, 768L)
                    val sampleShape2 = longArrayOf(2L, sampleChannels.toLong(), LATENT_RES.toLong(), LATENT_RES.toLong())
                    val sampleInput2 = FloatArray(2 * sampleStride)

                    createFloatOrHalfTensor(ortEnv, bothEmbeds, encoderShape2, encoderType).use { bothEncTensor ->
                        for ((localStep, stepIdx) in (plan.startStepIndex until plan.baseSteps).withIndex()) {
                            onStep(localStep + 1, plan.denoiseSteps)
                            val sigma = schedule.sigmas[stepIdx]
                            val nextSigma = schedule.sigmas[stepIdx + 1]
                            val timestep = schedule.timesteps[stepIdx]
                            val invScale = 1.0f / sqrt(sigma * sigma + 1.0f)

                            if (sampleChannels >= 9) {
                                for (i in 0 until latentSize) sampleInput2[i] = latents[i] * invScale
                                for (i in 0 until hw64) sampleInput2[latentSize + i] = mask64[i]
                                val offset = latentSize + hw64
                                for (i in 0 until latentSize) sampleInput2[offset + i] = maskedLatent[i]
                            } else {
                                for (i in 0 until latentSize) sampleInput2[i] = latents[i] * invScale
                            }
                            System.arraycopy(sampleInput2, 0, sampleInput2, sampleStride, sampleStride)

                            createFloatOrHalfTensor(ortEnv, sampleInput2, sampleShape2, sampleType).use { sTensor ->
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
                                            val cOff = c * hw64
                                            for (p in 0 until hw64) {
                                                if (mask64[p] < 0.5f) {
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
                    val sampleShape = longArrayOf(1L, sampleChannels.toLong(), LATENT_RES.toLong(), LATENT_RES.toLong())
                    val sampleInput = FloatArray(sampleStride)

                    createFloatOrHalfTensor(ortEnv, uncondEmbed, encoderShape, encoderType).use { uncondTensor ->
                        createFloatOrHalfTensor(ortEnv, condEmbed, encoderShape, encoderType).use { condTensor ->
                            for ((localStep, stepIdx) in (plan.startStepIndex until plan.baseSteps).withIndex()) {
                                onStep(localStep + 1, plan.denoiseSteps)
                                val sigma = schedule.sigmas[stepIdx]
                                val nextSigma = schedule.sigmas[stepIdx + 1]
                                val timestep = schedule.timesteps[stepIdx]
                                val invScale = 1.0f / sqrt(sigma * sigma + 1.0f)

                                if (sampleChannels >= 9) {
                                    // Standard 9-channel SD 1.5 Inpainting UNet: [scaled_latent(4), mask(1), masked_latent(4)]
                                    for (i in 0 until latentSize) {
                                        sampleInput[i] = latents[i] * invScale
                                    }
                                    for (i in 0 until hw64) {
                                        sampleInput[latentSize + i] = mask64[i]
                                    }
                                    val offset = latentSize + hw64
                                    for (i in 0 until latentSize) {
                                        sampleInput[offset + i] = maskedLatent[i]
                                    }
                                } else {
                                    // 4-channel UNet fallback (Blended Latent Diffusion)
                                    for (i in 0 until latentSize) {
                                        sampleInput[i] = latents[i] * invScale
                                    }
                                }

                                createFloatOrHalfTensor(ortEnv, sampleInput, sampleShape, sampleType).use { sTensor ->
                                    createTimestepTensor(ortEnv, timestep, timestepShape, timestepType).use { tTensor ->
                                        val uncondPred = session.run(
                                            mapOf(
                                                sampleName to sTensor,
                                                timestepName to tTensor,
                                                encoderName to uncondTensor
                                            )
                                        ).use { res ->
                                            readTensorAsFloats(res[0] as OnnxTensor)
                                        }

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

                                        // If 4-channel fallback UNet, preserve unmasked background latents
                                        if (sampleChannels < 9) {
                                            for (c in 0 until LATENT_CHANNELS) {
                                                val cOff = c * hw64
                                                for (p in 0 until hw64) {
                                                    val m = mask64[p]
                                                    if (m < 0.5f) {
                                                        val noisyInit = initLatent[cOff + p] + sampleGaussian(rng) * nextSigma
                                                        latents[cOff + p] = noisyInit
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return latents
    }

    private fun runVaeDecoder(
        ortEnv: OrtEnvironment,
        vaeDecoderFile: java.io.File,
        latent: FloatArray,
        preferHardwareAccel: Boolean
    ): Bitmap {
        val scaledLatent = FloatArray(latent.size) { i -> latent[i] / VAE_SCALE_FACTOR }
        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(vaeDecoderFile.absolutePath, opts).use { session ->
                val inputName = session.inputNames.first()
                val inputType = (session.inputInfo[inputName]?.info as? TensorInfo)?.type ?: OnnxJavaType.FLOAT
                val shape = longArrayOf(1L, LATENT_CHANNELS.toLong(), LATENT_RES.toLong(), LATENT_RES.toLong())

                return createFloatOrHalfTensor(ortEnv, scaledLatent, shape, inputType).use { lTensor ->
                    session.run(mapOf(inputName to lTensor)).use { res ->
                        val rgbFloats = readTensorAsFloats(res[0] as OnnxTensor)
                        val hw = SD_RES * SD_RES
                        val pixels = IntArray(hw)
                        for (i in 0 until hw) {
                            val r = ((rgbFloats[i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                            val g = ((rgbFloats[hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                            val b = ((rgbFloats[2 * hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                            pixels[i] = Color.rgb(r, g, b)
                        }
                        Bitmap.createBitmap(SD_RES, SD_RES, Bitmap.Config.ARGB_8888).apply {
                            setPixels(pixels, 0, SD_RES, 0, 0, SD_RES, SD_RES)
                        }
                    }
                }
            }
        }
    }

    private fun pasteCropDirectDebug(
        sourceBitmap: Bitmap,
        crop512: Bitmap,
        roi: SquareRoi
    ): Bitmap {
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height
        val cropFull = if (roi.size == SD_RES) {
            crop512
        } else {
            Bitmap.createScaledBitmap(crop512, roi.size, roi.size, true)
        }
        val out = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val intersectLeft = max(0, roi.left)
        val intersectTop = max(0, roi.top)
        val intersectRight = min(origW, roi.left + roi.size)
        val intersectBottom = min(origH, roi.top + roi.size)
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
        if (cropFull !== crop512) cropFull.recycle()
        return out
    }

    private fun compositeCropIntoSource(
        sourceBitmap: Bitmap,
        fullMaskAlpha: FloatArray,
        crop512: Bitmap,
        roi: SquareRoi,
        featherRadius: Int
    ): Bitmap {
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height

        val cropFull = if (roi.size == SD_RES) {
            crop512
        } else {
            Bitmap.createScaledBitmap(crop512, roi.size, roi.size, true)
        }

        val fullGenerated = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val genCanvas = Canvas(fullGenerated)
        val intersectLeft = max(0, roi.left)
        val intersectTop = max(0, roi.top)
        val intersectRight = min(origW, roi.left + roi.size)
        val intersectBottom = min(origH, roi.top + roi.size)
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
        if (cropFull !== crop512) cropFull.recycle()

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

    private fun featherMask(mask: FloatArray, w: Int, h: Int, radius: Int): FloatArray {
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
                out[rowOff + x] = max(mask[rowOff + x], sum / denom).coerceIn(0f, 1f)
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

    private fun resolveTimestepShape(modelShape: LongArray?, batchSize: Int): LongArray {
        if (modelShape == null || modelShape.isEmpty()) return longArrayOf()
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
        type: OnnxJavaType
    ): OnnxTensor {
        if (type == OnnxJavaType.FLOAT16) {
            val byteBuf = ByteBuffer.allocateDirect(data.size * 2).order(ByteOrder.nativeOrder())
            val shortView = byteBuf.asShortBuffer()
            for (i in data.indices) {
                shortView.put(floatToHalfBits(data[i]))
            }
            byteBuf.rewind()
            return OnnxTensor.createTensor(env, byteBuf, shape, OnnxJavaType.FLOAT16)
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
