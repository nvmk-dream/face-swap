package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min

/**
 * Automatic Body/Neck Age-to-Head Age Harmony Engine.
 *
 * Models supported (both optional, lazy-loaded via Stage-by-Stage RAM Guard):
 *  1. `mivolo_v2_age.onnx` (`Sam-Apostel/mivolo-v2-age-onnx`)
 *     - Input: `faces_bodies` FLOAT[1, 6, 384, 384] (Ch 0..2 Face RGB, Ch 3..5 Body RGB ImageNet-normalized)
 *     - Output: `age` FLOAT[1, 1]
 *  2. `face_reaging.onnx` (`Glat0s/face_reaging-onnx` FRAN U-Net, plus 4-ch & multi-input variants)
 *     - Default Input: `input` FLOAT[1, 5, 512, 512] (RGB [0..1] + src_age/100 + tgt_age/100)
 *     - Output: `output` FLOAT[1, 3, 512, 512] signed residual RGB delta added to input, or full RGB fallback
 *  3. Built-in CV Fallback (`FaceAgingStyleEngine`) if `face_reaging.onnx` is not installed.
 */
object AgeHarmonyEngine {

    const val MIVOLO_SIZE = 384
    const val REAGING_SIZE = 512

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    data class AgeCropRect(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float
    ) {
        fun toRectF(): RectF = RectF(left, top, right, bottom)
        val width: Float get() = (right - left).coerceAtLeast(1f)
        val height: Float get() = (bottom - top).coerceAtLeast(1f)
    }

    data class AgeHarmonyConfig(
        val autoBodyAgeMatch: Boolean = true,
        val manualAgingFactor: Float = 0.0f
    )

    data class AgeMatchPlan(
        val sourceAge: Float,
        val targetBodyAge: Float,
        val finalTargetAge: Float,
        val ageDeltaYears: Float,
        val effectiveSliderFactor: Float,
        val shouldApplyReAging: Boolean,
        val gfpganWeightScale: Float,
        val usedMivoloModel: Boolean,
        val summaryLabel: String
    )

    /**
     * Derives an anatomical neck/shoulders/torso ROI below the detected face so MiVOLO v2 can judge
     * Target body maturity even without a separate person-box detector.
     * Returns `null` for close-up portraits where less than `0.45 * faceHeight` is visible below chin.
     */
    fun computeBodyCropRect(
        face: DetectedFace,
        imageWidth: Int,
        imageHeight: Int
    ): AgeCropRect? {
        if (imageWidth <= 8 || imageHeight <= 8) return null
        val fw = face.boundingBox.width().coerceAtLeast(16f)
        val fh = face.boundingBox.height().coerceAtLeast(16f)
        val belowChin = (imageHeight.toFloat() - face.boundingBox.bottom).coerceAtLeast(0f)
        if (belowChin < fh * 0.45f) return null

        val cx = face.boundingBox.centerX()
        val left = (cx - fw * 1.45f).coerceIn(0f, (imageWidth - 2).toFloat())
        val right = (cx + fw * 1.45f).coerceIn(left + 8f, imageWidth.toFloat())
        val top = (face.boundingBox.top - fh * 0.25f).coerceIn(0f, (imageHeight - 2).toFloat())
        val bottom = min(imageHeight.toFloat(), face.boundingBox.bottom + fh * 3.8f)
            .coerceAtLeast(top + 8f)
        return AgeCropRect(left = left, top = top, right = right, bottom = bottom)
    }

    /**
     * Builds a neck-and-shoulders-below-forehead crop for body-only age anchoring so heavily retouched
     * or youthful faces still respect neck skin, posture, and torso maturity.
     */
    fun computeBelowForeheadBodyCropRect(
        face: DetectedFace,
        imageWidth: Int,
        imageHeight: Int
    ): AgeCropRect? {
        if (imageWidth <= 8 || imageHeight <= 8) return null
        val fw = face.boundingBox.width().coerceAtLeast(16f)
        val fh = face.boundingBox.height().coerceAtLeast(16f)
        val belowChin = (imageHeight.toFloat() - face.boundingBox.bottom).coerceAtLeast(0f)
        if (belowChin < fh * 0.90f) return null

        val cx = face.boundingBox.centerX()
        val left = (cx - fw * 1.55f).coerceIn(0f, (imageWidth - 2).toFloat())
        val right = (cx + fw * 1.55f).coerceIn(left + 8f, imageWidth.toFloat())
        val top = (face.boundingBox.bottom - fh * 0.15f).coerceIn(0f, (imageHeight - 2).toFloat())
        val bottom = min(imageHeight.toFloat(), face.boundingBox.bottom + fh * 4.2f)
            .coerceAtLeast(top + 8f)
        return AgeCropRect(left = left, top = top, right = right, bottom = bottom)
    }

    /**
     * Letterboxes [cropRect] from [bitmap] into `dstSize x dstSize` preserving aspect ratio,
     * converts RGB to `[0, 1]`, applies ImageNet mean/std normalization, and writes NCHW channels
     * into [outFloatArray] starting at `channelOffset * dstSize * dstSize`.
     * When [cropRect] is `null`, writes normalized black padding `(0 - mean) / std`.
     */
    fun letterboxRgbAndNormalizeImageNet(
        bitmap: Bitmap?,
        cropRect: AgeCropRect?,
        outFloatArray: FloatArray,
        channelOffset: Int,
        dstSize: Int = MIVOLO_SIZE
    ) {
        val hw = dstSize * dstSize
        val ch0 = channelOffset * hw
        val ch1 = (channelOffset + 1) * hw
        val ch2 = (channelOffset + 2) * hw

        val padR = (0f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        val padG = (0f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        val padB = (0f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        for (i in 0 until hw) {
            outFloatArray[ch0 + i] = padR
            outFloatArray[ch1 + i] = padG
            outFloatArray[ch2 + i] = padB
        }

        if (bitmap == null || cropRect == null) return
        val x0 = cropRect.left.toInt().coerceIn(0, (bitmap.width - 1).coerceAtLeast(0))
        val y0 = cropRect.top.toInt().coerceIn(0, (bitmap.height - 1).coerceAtLeast(0))
        val x1 = cropRect.right.toInt().coerceIn(x0 + 1, bitmap.width)
        val y1 = cropRect.bottom.toInt().coerceIn(y0 + 1, bitmap.height)
        val cw = (x1 - x0).coerceAtLeast(1)
        val ch = (y1 - y0).coerceAtLeast(1)

        val scale = min(dstSize.toFloat() / cw.toFloat(), dstSize.toFloat() / ch.toFloat())
        val scaledW = (cw * scale).toInt().coerceIn(1, dstSize)
        val scaledH = (ch * scale).toInt().coerceIn(1, dstSize)
        val offX = (dstSize - scaledW) / 2
        val offY = (dstSize - scaledH) / 2

        val cropped = Bitmap.createBitmap(bitmap, x0, y0, cw, ch)
        val resized = Bitmap.createScaledBitmap(cropped, scaledW, scaledH, true)
        if (cropped !== bitmap) cropped.recycle()

        val px = IntArray(scaledW * scaledH)
        resized.getPixels(px, 0, scaledW, 0, 0, scaledW, scaledH)
        resized.recycle()

        for (y in 0 until scaledH) {
            val dstRow = (offY + y) * dstSize + offX
            val srcRow = y * scaledW
            for (x in 0 until scaledW) {
                val c = px[srcRow + x]
                val r = ((c ushr 16) and 0xFF) / 255.0f
                val g = ((c ushr 8) and 0xFF) / 255.0f
                val b = (c and 0xFF) / 255.0f
                val dIdx = dstRow + x
                outFloatArray[ch0 + dIdx] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                outFloatArray[ch1 + dIdx] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                outFloatArray[ch2 + dIdx] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            }
        }
    }

    /**
     * Resolves target age, age delta, GFPGAN anti-oversmoothing scale, and user-facing summary
     * from estimated ages + `autoBodyAgeMatch` toggle + manual `faceAgingFactor`.
     */
    fun resolveAgeMatchPlan(
        sourceAge: Float,
        targetBodyAge: Float,
        usedMivoloModel: Boolean,
        config: AgeHarmonyConfig
    ): AgeMatchPlan {
        val safeSrc = sourceAge.coerceIn(4f, 92f)
        val safeBody = targetBodyAge.coerceIn(4f, 92f)
        val manual = config.manualAgingFactor.coerceIn(-1f, 1f)

        val finalTargetAge: Float
        val ageDelta: Float
        val shouldApply: Boolean
        val effectiveSlider: Float

        if (config.autoBodyAgeMatch) {
            val manualShiftYears = manual * 28f
            finalTargetAge = (safeBody + manualShiftYears).coerceIn(5f, 92f)
            ageDelta = finalTargetAge - safeSrc
            shouldApply = (usedMivoloModel && abs(ageDelta) >= 2.5f) || abs(manual) >= 0.05f
            effectiveSlider = (ageDelta / 32f).coerceIn(-1f, 1f)
        } else {
            val manualShiftYears = manual * 32f
            finalTargetAge = (safeSrc + manualShiftYears).coerceIn(5f, 92f)
            ageDelta = manualShiftYears
            shouldApply = abs(manual) >= 0.05f
            effectiveSlider = manual
        }

        // Prevent GFPGAN from over-smoothing skin wrinkles when aging up to 38+ years
        val gfpScale = when {
            finalTargetAge >= 55f && ageDelta >= 4f -> 0.50f
            finalTargetAge >= 38f && ageDelta >= 4f -> 0.72f
            else -> 1.0f
        }

        val summary = when {
            !shouldApply && usedMivoloModel ->
                "Age Harmony: Src ~${safeSrc.toInt()}y ≈ Body ~${safeBody.toInt()}y (Already Matched)"
            !shouldApply && config.autoBodyAgeMatch && !usedMivoloModel ->
                "Age Harmony: Auto Ready (Optional MiVOLO ONNX not installed)"
            !shouldApply ->
                "Age Harmony: Off"
            config.autoBodyAgeMatch && usedMivoloModel ->
                "Age Harmony: Src ~${safeSrc.toInt()}y → Body ~${safeBody.toInt()}y (Target ${finalTargetAge.toInt()}y)"
            else -> {
                val sign = if (effectiveSlider >= 0f) "+" else ""
                "Manual Age Shift: ${sign}${(ageDelta).toInt()}y (Target ~${finalTargetAge.toInt()}y)"
            }
        }

        return AgeMatchPlan(
            sourceAge = safeSrc,
            targetBodyAge = safeBody,
            finalTargetAge = finalTargetAge,
            ageDeltaYears = ageDelta,
            effectiveSliderFactor = effectiveSlider,
            shouldApplyReAging = shouldApply,
            gfpganWeightScale = gfpScale,
            usedMivoloModel = usedMivoloModel,
            summaryLabel = summary
        )
    }

    /**
     * Packs an ARGB pixel buffer into NCHW FloatArray for `Glat0s/face_reaging-onnx` and variants:
     *  - `channelCount == 5`: RGB in `[0, 1]` + constant plane `sourceAge / 100f` + `targetAge / 100f`
     *  - `channelCount == 4`: RGB in `[0, 1]` + constant plane `(targetAge - sourceAge) / 100f`
     *  - `channelCount == 3`: RGB in `[0, 1]` (ages passed via separate scalar inputs)
     */
    fun packReagingInputTensor(
        pixels: IntArray,
        size: Int,
        channelCount: Int,
        sourceAge: Float,
        targetAge: Float
    ): FloatArray {
        val hw = size * size
        val ch = channelCount.coerceIn(3, 5)
        val out = FloatArray(ch * hw)
        val srcNorm = (sourceAge / 100f).coerceIn(0.01f, 0.99f)
        val tgtNorm = (targetAge / 100f).coerceIn(0.01f, 0.99f)
        val deltaNorm = ((targetAge - sourceAge) / 100f).coerceIn(-0.90f, 0.90f)

        for (i in 0 until hw) {
            val c = pixels[i]
            out[i] = ((c ushr 16) and 0xFF) / 255.0f
            out[hw + i] = ((c ushr 8) and 0xFF) / 255.0f
            out[2 * hw + i] = (c and 0xFF) / 255.0f
            if (ch == 5) {
                out[3 * hw + i] = srcNorm
                out[4 * hw + i] = tgtNorm
            } else if (ch == 4) {
                out[3 * hw + i] = deltaNorm
            }
        }
        return out
    }

    /**
     * Applies FRAN re-aging output `[1, 3, H, W]` onto [pixels] using automatic detection of:
     *  - Residual delta output (e.g. `Glat0s/face_reaging-onnx`, centered around 0.0 -> `rgb + delta`)
     *  - Direct normalized RGB output (`[-1, 1]` or `[0, 1]`)
     * Respects optional per-pixel [maskWeights] (`0.0f..1.0f`) and smooth border feathering.
     */
    fun blendReagingOutput(
        pixels: IntArray,
        size: Int,
        outputFloats: FloatArray,
        maskWeights: FloatArray? = null,
        strength: Float = 1.0f
    ) {
        val hw = size * size
        if (pixels.size < hw || outputFloats.size < 3 * hw) return

        var minVal = Float.POSITIVE_INFINITY
        var maxVal = Float.NEGATIVE_INFINITY
        var absSum = 0.0
        val sampleStep = max(1, (3 * hw) / 2048)
        var sampleCount = 0
        var idx = 0
        while (idx < 3 * hw) {
            val v = outputFloats[idx]
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
            absSum += abs(v)
            sampleCount++
            idx += sampleStep
        }
        val absMean = (absSum / sampleCount.coerceAtLeast(1)).toFloat()
        // Glat0s FRAN outputs signed residual delta centered near 0 (usually in [-0.45, +0.45], absMean < 0.22)
        val isResidualDelta = (minVal < -0.015f && maxVal < 0.65f) || absMean < 0.22f
        val isSignedDirectRgb = !isResidualDelta && minVal < -0.05f

        val clampedStrength = strength.coerceIn(0f, 1.25f)
        val margin = size * 0.06f

        for (y in 0 until size) {
            val row = y * size
            val fy = when {
                y < margin -> (y / margin).coerceIn(0f, 1f)
                y > size - 1 - margin -> ((size - 1 - y) / margin).coerceIn(0f, 1f)
                else -> 1f
            }
            for (x in 0 until size) {
                val fx = when {
                    x < margin -> (x / margin).coerceIn(0f, 1f)
                    x > size - 1 - margin -> ((size - 1 - x) / margin).coerceIn(0f, 1f)
                    else -> 1f
                }
                val i = row + x
                val w = ((maskWeights?.getOrNull(i) ?: 1f) * min(fx, fy) * clampedStrength).coerceIn(0f, 1.25f)
                if (w <= 0.002f) continue

                val c = pixels[i]
                val inR = ((c ushr 16) and 0xFF) / 255.0f
                val inG = ((c ushr 8) and 0xFF) / 255.0f
                val inB = (c and 0xFF) / 255.0f

                val oR = outputFloats[i]
                val oG = outputFloats[hw + i]
                val oB = outputFloats[2 * hw + i]

                val outR: Float
                val outG: Float
                val outB: Float
                when {
                    isResidualDelta -> {
                        outR = (inR + oR * w).coerceIn(0f, 1f)
                        outG = (inG + oG * w).coerceIn(0f, 1f)
                        outB = (inB + oB * w).coerceIn(0f, 1f)
                    }
                    isSignedDirectRgb -> {
                        val tR = (oR * 0.5f + 0.5f).coerceIn(0f, 1f)
                        val tG = (oG * 0.5f + 0.5f).coerceIn(0f, 1f)
                        val tB = (oB * 0.5f + 0.5f).coerceIn(0f, 1f)
                        val blend = w.coerceIn(0f, 1f)
                        outR = inR * (1f - blend) + tR * blend
                        outG = inG * (1f - blend) + tG * blend
                        outB = inB * (1f - blend) + tB * blend
                    }
                    else -> {
                        val tR = oR.coerceIn(0f, 1f)
                        val tG = oG.coerceIn(0f, 1f)
                        val tB = oB.coerceIn(0f, 1f)
                        val blend = w.coerceIn(0f, 1f)
                        outR = inR * (1f - blend) + tR * blend
                        outG = inG * (1f - blend) + tG * blend
                        outB = inB * (1f - blend) + tB * blend
                    }
                }

                val rInt = (outR * 255f + 0.5f).toInt().coerceIn(0, 255)
                val gInt = (outG * 255f + 0.5f).toInt().coerceIn(0, 255)
                val bInt = (outB * 255f + 0.5f).toInt().coerceIn(0, 255)
                pixels[i] = (0xFF shl 24) or (rInt shl 16) or (gInt shl 8) or bInt
            }
        }
    }

    /**
     * Runs `mivolo_v2_age.onnx` on face + anatomical body crop.
     * When [anchorBodyOnly] is `true` and enough below-chin torso is visible, runs a second pass
     * with black face + below-forehead body crop so retouched/smooth faces still reflect body maturity.
     */
    fun estimateAgeWithMivolo(
        ortEnv: OrtEnvironment?,
        mivoloFile: File?,
        bitmap: Bitmap,
        face: DetectedFace,
        anchorBodyOnly: Boolean = false,
        preferHardwareAccel: Boolean = false
    ): Float? {
        if (ortEnv == null || mivoloFile == null || !mivoloFile.exists() || mivoloFile.length() <= 1024L) {
            return null
        }
        val session = OnnxProtobufInspector.getOrCreateCachedSession(
            ortEnv = ortEnv,
            file = mivoloFile,
            preferHardwareAcceleration = preferHardwareAccel
        ) ?: return null

        val inputName = session.inputNames.firstOrNull() ?: return null
        val tInfo = session.inputInfo[inputName]?.info as? TensorInfo
        val channels = (tInfo?.shape?.getOrNull(1)?.toInt() ?: 6).coerceIn(3, 6)
        val inputSize = (tInfo?.shape?.getOrNull(2)?.toInt()?.takeIf { it in 128..512 }) ?: MIVOLO_SIZE

        val fw = face.boundingBox.width().coerceAtLeast(16f)
        val fh = face.boundingBox.height().coerceAtLeast(16f)
        val facePadX = fw * 0.20f
        val facePadY = fh * 0.25f
        val faceRect = AgeCropRect(
            left = (face.boundingBox.left - facePadX).coerceIn(0f, (bitmap.width - 2).toFloat()),
            top = (face.boundingBox.top - facePadY).coerceIn(0f, (bitmap.height - 2).toFloat()),
            right = (face.boundingBox.right + facePadX).coerceIn(2f, bitmap.width.toFloat()),
            bottom = (face.boundingBox.bottom + facePadY).coerceIn(2f, bitmap.height.toFloat())
        )
        val bodyRect = computeBodyCropRect(face, bitmap.width, bitmap.height)

        fun runSingleMivoloPass(faceCrop: AgeCropRect?, bodyCrop: AgeCropRect?): Float? {
            val hw = inputSize * inputSize
            val floatArr = FloatArray(channels * hw)
            letterboxRgbAndNormalizeImageNet(bitmap, faceCrop, floatArr, channelOffset = 0, dstSize = inputSize)
            if (channels >= 6) {
                letterboxRgbAndNormalizeImageNet(bitmap, bodyCrop, floatArr, channelOffset = 3, dstSize = inputSize)
            }
            val buf = FloatBuffer.wrap(floatArr)
            val shape = longArrayOf(1L, channels.toLong(), inputSize.toLong(), inputSize.toLong())
            return runCatching {
                OnnxTensor.createTensor(ortEnv, buf, shape).use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { result ->
                        val outT = result[0] as? OnnxTensor ?: return@use null
                        val outFb = outT.floatBuffer
                        if (outFb.remaining() > 0) outFb.get(0).coerceIn(1f, 95f) else null
                    }
                }
            }.getOrNull()
        }

        val holisticAge = runSingleMivoloPass(faceRect, bodyRect) ?: return null
        if (!anchorBodyOnly || channels < 6) return holisticAge

        val belowForeheadBody = computeBelowForeheadBodyCropRect(face, bitmap.width, bitmap.height)
            ?: return holisticAge
        val bodyOnlyAge = runSingleMivoloPass(faceCrop = null, bodyCrop = belowForeheadBody)
            ?: return holisticAge

        return (0.55f * holisticAge + 0.45f * bodyOnlyAge).coerceIn(1f, 95f)
    }

    /**
     * Resolves an [AgeMatchPlan] for a Source face and Target face/body pair.
     * Intended to run BEFORE heavy Swapper/GFPGAN/SegFormer sessions are loaded so native RAM stays minimal.
     */
    fun resolveAgePlan(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        sourceFace: DetectedFace,
        targetBitmap: Bitmap,
        targetFace: DetectedFace,
        config: AgeHarmonyConfig,
        preferHardwareAccel: Boolean = false
    ): AgeMatchPlan {
        val mivoloFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.AGE_ESTIMATOR)
        val hasMivolo = mivoloFile.exists() && mivoloFile.length() > 1024L && ortEnv != null

        var sourceAge: Float? = null
        var targetBodyAge: Float? = null
        var usedMivolo = false

        if (hasMivolo && (config.autoBodyAgeMatch || abs(config.manualAgingFactor) >= 0.05f)) {
            sourceAge = estimateAgeWithMivolo(
                ortEnv = ortEnv,
                mivoloFile = mivoloFile,
                bitmap = sourceBitmap,
                face = sourceFace,
                anchorBodyOnly = false,
                preferHardwareAccel = preferHardwareAccel
            )
            targetBodyAge = estimateAgeWithMivolo(
                ortEnv = ortEnv,
                mivoloFile = mivoloFile,
                bitmap = targetBitmap,
                face = targetFace,
                anchorBodyOnly = true,
                preferHardwareAccel = preferHardwareAccel
            )
            usedMivolo = (sourceAge != null && targetBodyAge != null)
        }

        return resolveAgeMatchPlan(
            sourceAge = sourceAge ?: 30f,
            targetBodyAge = targetBodyAge ?: 30f,
            usedMivoloModel = usedMivolo,
            config = config
        )
    }

    /**
     * Applies neural re-aging (`face_reaging.onnx`) if installed (auto-detecting 5-ch, 4-ch,
     * and 3-ch+scalar ONNX graph inputs + residual vs direct RGB outputs), or gracefully falls back
     * to [FaceAgingStyleEngine] using the resolved age delta.
     */
    fun reageCrop512(
        ortEnv: OrtEnvironment?,
        reagingFile: File?,
        pixels512: IntArray,
        geom512: FaceBlender.WarpedFaceGeometry?,
        agePlan: AgeMatchPlan,
        maskWeights512: FloatArray? = null,
        preferHardwareAccel: Boolean = false
    ): Boolean {
        if (!agePlan.shouldApplyReAging) return false
        val hw512 = REAGING_SIZE * REAGING_SIZE
        if (pixels512.size < hw512) return false

        // Protect ocular sockets (sclera, pupils, eyelashes) and glabella so age lines don't muddy eyes
        val combinedMask = FloatArray(hw512) { idx ->
            val baseMask = maskWeights512?.getOrNull(idx) ?: 1.0f
            if (baseMask <= 0.002f) {
                0f
            } else if (geom512 != null) {
                val x = (idx % REAGING_SIZE).toFloat()
                val y = (idx / REAGING_SIZE).toFloat()
                val eyeProt = computeEyeProtectionWeight512(x, y, geom512)
                (baseMask * (1f - eyeProt * 0.90f)).coerceIn(0f, 1f)
            } else {
                baseMask.coerceIn(0f, 1f)
            }
        }

        val hasReagingOnnx = ortEnv != null && reagingFile != null &&
            reagingFile.exists() && reagingFile.length() > 1024L

        if (hasReagingOnnx) {
            val session = OnnxProtobufInspector.getOrCreateCachedSession(
                ortEnv = ortEnv,
                file = reagingFile!!,
                preferHardwareAcceleration = preferHardwareAccel
            )
            if (session != null) {
                val ranOnnx = runCatching {
                    runDynamicReagingSession(
                        ortEnv = ortEnv!!,
                        session = session,
                        pixels512 = pixels512,
                        sourceAge = agePlan.sourceAge,
                        targetAge = agePlan.finalTargetAge,
                        combinedMask512 = combinedMask
                    )
                }.getOrDefault(false)
                if (ranOnnx) return true
            }
        }

        // CV Fallback: Enhanced FaceAgingStyleEngine respecting combinedMask
        if (geom512 != null && abs(agePlan.effectiveSliderFactor) >= 0.05f) {
            FaceAgingStyleEngine.applyFaceAgingAndStyle512(
                pixels512 = pixels512,
                geom512 = geom512,
                params = FaceAgingStyleEngine.AgingParameters(
                    ageFactor = agePlan.effectiveSliderFactor,
                    wrinkleIntensity = 0.65f,
                    nasolabialFoldEnhancement = true
                ),
                maskWeights = combinedMask
            )
            return true
        }
        return false
    }

    private fun runDynamicReagingSession(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        pixels512: IntArray,
        sourceAge: Float,
        targetAge: Float,
        combinedMask512: FloatArray
    ): Boolean {
        val inputsInfo = session.inputInfo
        var imageInputName: String? = null
        var imageChannels = 5
        var modelSize = REAGING_SIZE

        for ((name, node) in inputsInfo) {
            val ti = node.info as? TensorInfo ?: continue
            if (ti.shape.size == 4) {
                imageInputName = name
                val ch = ti.shape[1].toInt()
                if (ch in 3..5) imageChannels = ch
                val h = ti.shape[2].toInt()
                if (h == 256 || h == 512) modelSize = h
                break
            }
        }
        val activeImgInput = imageInputName ?: inputsInfo.keys.firstOrNull() ?: return false

        val workingPixels: IntArray
        val workingMask: FloatArray
        if (modelSize != REAGING_SIZE) {
            val srcBmp = Bitmap.createBitmap(REAGING_SIZE, REAGING_SIZE, Bitmap.Config.ARGB_8888)
            srcBmp.setPixels(pixels512, 0, REAGING_SIZE, 0, 0, REAGING_SIZE, REAGING_SIZE)
            val scaledBmp = Bitmap.createScaledBitmap(srcBmp, modelSize, modelSize, true)
            srcBmp.recycle()
            workingPixels = IntArray(modelSize * modelSize)
            scaledBmp.getPixels(workingPixels, 0, modelSize, 0, 0, modelSize, modelSize)
            scaledBmp.recycle()
            workingMask = FloatArray(modelSize * modelSize) { idx ->
                val sy = (idx / modelSize) * REAGING_SIZE / modelSize
                val sx = (idx % modelSize) * REAGING_SIZE / modelSize
                combinedMask512[(sy.coerceIn(0, REAGING_SIZE - 1) * REAGING_SIZE) + sx.coerceIn(0, REAGING_SIZE - 1)]
            }
        } else {
            workingPixels = pixels512
            workingMask = combinedMask512
        }

        val imgTensorFloats = packReagingInputTensor(
            pixels = workingPixels,
            size = modelSize,
            channelCount = imageChannels,
            sourceAge = sourceAge,
            targetAge = targetAge
        )

        val createdTensors = mutableListOf<OnnxTensor>()
        return try {
            val feedMap = LinkedHashMap<String, OnnxTensor>()
            val imgTensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(imgTensorFloats),
                longArrayOf(1L, imageChannels.toLong(), modelSize.toLong(), modelSize.toLong())
            )
            createdTensors.add(imgTensor)
            feedMap[activeImgInput] = imgTensor

            // Populate any auxiliary scalar / 1D age inputs if the exported ONNX graph uses separate age inputs
            for ((name, node) in inputsInfo) {
                if (name == activeImgInput) continue
                val ti = node.info as? TensorInfo ?: continue
                val lower = name.lowercase()
                val ageScalarVal = when {
                    lower.contains("src") || lower.contains("source") -> sourceAge / 100f
                    lower.contains("delta") || lower.contains("diff") -> (targetAge - sourceAge) / 100f
                    else -> targetAge / 100f
                }
                val auxShape = if (ti.shape.isEmpty()) {
                    longArrayOf()
                } else {
                    LongArray(ti.shape.size) { d -> ti.shape[d].let { if (it <= 0L) 1L else it } }
                }
                val auxCount = auxShape.fold(1) { acc, v -> acc * v.toInt() }.coerceAtLeast(1)
                val auxArr = FloatArray(auxCount) { ageScalarVal }
                val auxTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(auxArr), auxShape)
                createdTensors.add(auxTensor)
                feedMap[name] = auxTensor
            }

            session.run(feedMap).use { results ->
                val outT = results[0] as? OnnxTensor ?: return false
                val hwModel = modelSize * modelSize
                val outFloats = FloatArray(3 * hwModel)
                val fb = outT.floatBuffer
                if (fb.remaining() < 3 * hwModel) return false
                fb.get(outFloats)
                blendReagingOutput(
                    pixels = workingPixels,
                    size = modelSize,
                    outputFloats = outFloats,
                    maskWeights = workingMask,
                    strength = 1.0f
                )
            }

            if (modelSize != REAGING_SIZE) {
                val agedSmall = Bitmap.createBitmap(modelSize, modelSize, Bitmap.Config.ARGB_8888)
                agedSmall.setPixels(workingPixels, 0, modelSize, 0, 0, modelSize, modelSize)
                val upscaled = Bitmap.createScaledBitmap(agedSmall, REAGING_SIZE, REAGING_SIZE, true)
                agedSmall.recycle()
                upscaled.getPixels(pixels512, 0, REAGING_SIZE, 0, 0, REAGING_SIZE, REAGING_SIZE)
                upscaled.recycle()
            }
            true
        } finally {
            for (t in createdTensors) {
                runCatching { t.close() }
            }
        }
    }

    private fun computeEyeProtectionWeight512(
        x: Float,
        y: Float,
        geom: FaceBlender.WarpedFaceGeometry
    ): Float {
        val eyeRx = geom.eyeDist * 0.48f
        val eyeRy = geom.eyeDist * 0.30f

        fun ellipSq(cx: Float, cy: Float): Float {
            val dx = x - cx
            val dy = y - cy
            val u = (dx * geom.cosA + dy * geom.sinA) / eyeRx
            val v = (-dx * geom.sinA + dy * geom.cosA) / eyeRy
            return u * u + v * v
        }

        val minSq = min(ellipSq(geom.leftEye.x, geom.leftEye.y), ellipSq(geom.rightEye.x, geom.rightEye.y))
        return when {
            minSq <= 0.55f -> 1.0f
            minSq >= 1.0f -> 0.0f
            else -> {
                val t = (minSq - 0.55f) / 0.45f
                0.5f * (1.0f + cos(Math.PI * t).toFloat())
            }
        }
    }
}
