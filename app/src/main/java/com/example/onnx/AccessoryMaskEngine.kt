package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

enum class AccessoryTarget(
    val id: String,
    val shortLabel: String,
    val removalButtonLabel: String,
    val removalPromptPhrase: String
) {
    CAP(
        id = "cap",
        shortLabel = "Cap",
        removalButtonLabel = "Remove Cap",
        removalPromptPhrase = "natural hair and forehead, bare head, clean realistic skin, no cap, no hat"
    ),
    GLASSES(
        id = "glasses",
        shortLabel = "Glasses",
        removalButtonLabel = "Remove Glasses",
        removalPromptPhrase = "clear natural open eyes, realistic eyelashes and eyebrows, no sunglasses, no glasses"
    ),
    JEWELS(
        id = "jewels",
        shortLabel = "Jewels",
        removalButtonLabel = "Remove Jewels",
        removalPromptPhrase = "clean bare neck, ears and skin, natural skin texture, no necklace, no earrings, no jewelry"
    )
}

data class AccessoryMaskLayers(
    val width: Int,
    val height: Int,
    val capBaseMask: FloatArray,
    val glassesBaseMask: FloatArray,
    val jewelsBaseMask: FloatArray,
    val detectedFacesCount: Int,
    val usedNeuralParser: Boolean,
    val engineLabel: String
) {
    fun hasAnyMaskFor(targets: Set<AccessoryTarget>): Boolean {
        if (targets.isEmpty() || detectedFacesCount <= 0) return false
        for (i in 0 until (width * height)) {
            if (AccessoryTarget.CAP in targets && capBaseMask[i] > 0.18f) return true
            if (AccessoryTarget.GLASSES in targets && glassesBaseMask[i] > 0.18f) return true
            if (AccessoryTarget.JEWELS in targets && jewelsBaseMask[i] > 0.18f) return true
        }
        return false
    }

    /**
     * Renders a cyan (`#00E5FF`) ARGB_8888 mask bitmap at `(targetWidth, targetHeight)` for the
     * selected accessory targets with real-time morphological expansion (`expandPx`).
     */
    fun renderMaskBitmap(
        targetWidth: Int,
        targetHeight: Int,
        selectedTargets: Set<AccessoryTarget>,
        expandPx: Float = 6f
    ): Bitmap {
        val tw = targetWidth.coerceAtLeast(1)
        val th = targetHeight.coerceAtLeast(1)
        val total = width * height
        val combined = FloatArray(total)

        val useCap = AccessoryTarget.CAP in selectedTargets
        val useGlasses = AccessoryTarget.GLASSES in selectedTargets
        val useJewels = AccessoryTarget.JEWELS in selectedTargets

        for (i in 0 until total) {
            var v = 0f
            if (useCap) v = max(v, capBaseMask[i])
            if (useGlasses) v = max(v, glassesBaseMask[i])
            if (useJewels) v = max(v, jewelsBaseMask[i])
            combined[i] = v
        }

        val radius = expandPx.roundToInt().coerceIn(0, 18)
        val expanded = if (radius > 0) {
            dilateMaskSeparable(combined, width, height, radius)
        } else {
            combined
        }

        val workPx = IntArray(total)
        for (i in 0 until total) {
            val a = (expanded[i].coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
            workPx[i] = if (a > 16) Color.argb(a, 0, 229, 255) else Color.TRANSPARENT
        }

        val workBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            setPixels(workPx, 0, width, 0, 0, width, height)
        }
        if (width == tw && height == th) return workBmp

        val outBmp = Bitmap.createScaledBitmap(workBmp, tw, th, true)
        if (outBmp !== workBmp) workBmp.recycle()
        return outBmp
    }

    private fun dilateMaskSeparable(src: FloatArray, w: Int, h: Int, radius: Int): FloatArray {
        val horiz = FloatArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val x0 = (x - radius).coerceAtLeast(0)
                val x1 = (x + radius).coerceAtMost(w - 1)
                var best = 0f
                for (nx in x0..x1) {
                    val v = src[row + nx]
                    if (v > best) best = v
                }
                horiz[row + x] = best
            }
        }
        val out = FloatArray(w * h)
        for (y in 0 until h) {
            val y0 = (y - radius).coerceAtLeast(0)
            val y1 = (y + radius).coerceAtMost(h - 1)
            val row = y * w
            for (x in 0 until w) {
                var best = 0f
                for (ny in y0..y1) {
                    val v = horiz[ny * w + x]
                    if (v > best) best = v
                }
                out[row + x] = best
            }
        }
        return out
    }
}

/**
 * Automated Face-Accessory Mask Engine for Inpaint:
 * Generates 1-tap pixel masks for Head Cap (hat), Cooling Glass (sunglasses/eyeglasses),
 * and Jewels (necklaces & earrings) using `segformer_B5_ce.onnx` (CelebAMask-HQ 19-class)
 * with Biometric Landmark + Depth fallback when neural face-parsing is unavailable or partial.
 */
object AccessoryMaskEngine {

    fun buildCombinedRemovalPrompt(targets: Set<AccessoryTarget>): String {
        if (targets.isEmpty()) return ""
        val ordered = listOf(AccessoryTarget.CAP, AccessoryTarget.GLASSES, AccessoryTarget.JEWELS)
            .filter { it in targets }
        return ordered.joinToString(", ") { it.removalPromptPhrase } +
            ", photorealistic skin texture, seamless natural lighting"
    }

    fun extractAccessoryLayers(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        depthMap: DepthMapResult? = null,
        preferHardwareAccel: Boolean = false
    ): AccessoryMaskLayers {
        val maxSide = 512
        val srcW = sourceBitmap.width.coerceAtLeast(1)
        val srcH = sourceBitmap.height.coerceAtLeast(1)
        val scale = (maxSide.toFloat() / max(srcW, srcH).toFloat()).coerceAtMost(1f)
        val workW = (srcW * scale).roundToInt().coerceIn(64, maxSide)
        val workH = (srcH * scale).roundToInt().coerceIn(64, maxSide)
        val total = workW * workH

        val capMask = FloatArray(total)
        val glassesMask = FloatArray(total)
        val jewelsMask = FloatArray(total)

        val faces = detectFacesForSource(context, ortEnv, sourceBitmap, preferHardwareAccel)
        if (faces.isEmpty()) {
            return AccessoryMaskLayers(
                width = workW,
                height = workH,
                capBaseMask = capMask,
                glassesBaseMask = glassesMask,
                jewelsBaseMask = jewelsMask,
                detectedFacesCount = 0,
                usedNeuralParser = false,
                engineLabel = "No face detected"
            )
        }

        val segFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SEGMENTATION)
        val hasSegOnnx = ortEnv != null && segFile.exists() && segFile.length() > 1024L
        var usedNeural = false

        val sx = workW.toFloat() / srcW.toFloat()
        val sy = workH.toFloat() / srcH.toFloat()

        for (face in faces) {
            val kps = face.landmarks5
            if (kps.size < 5) continue

            val leftEyeX = kps[0].x * sx
            val leftEyeY = kps[0].y * sy
            val rightEyeX = kps[1].x * sx
            val rightEyeY = kps[1].y * sy
            val noseX = kps[2].x * sx
            val noseY = kps[2].y * sy
            val leftMouthX = kps[3].x * sx
            val leftMouthY = kps[3].y * sy
            val rightMouthX = kps[4].x * sx
            val rightMouthY = kps[4].y * sy

            val eyeMidX = (leftEyeX + rightEyeX) * 0.5f
            val eyeMidY = (leftEyeY + rightEyeY) * 0.5f
            val mouthMidY = (leftMouthY + rightMouthY) * 0.5f
            val eyeDist = hypot((rightEyeX - leftEyeX).toDouble(), (rightEyeY - leftEyeY).toDouble())
                .toFloat()
                .coerceAtLeast(10f)

            // Expand cranial + upper-chest square crop so hats AND necklaces stay inside crop
            val cropCenterX = eyeMidX / sx
            val cropCenterY = ((eyeMidY + mouthMidY) * 0.5f + eyeDist * 0.15f) / sy
            val srcEyeDist = eyeDist / ((sx + sy) * 0.5f)
            val halfCropSrc = (srcEyeDist * 2.95f).coerceAtLeast(64f)
            val cropRect = RectF(
                (cropCenterX - halfCropSrc).coerceAtLeast(0f),
                (cropCenterY - halfCropSrc * 1.08f).coerceAtLeast(0f),
                (cropCenterX + halfCropSrc).coerceAtMost(srcW.toFloat()),
                (cropCenterY + halfCropSrc * 1.12f).coerceAtMost(srcH.toFloat())
            )

            var capPixelsFromNeural = 0
            var glassesPixelsFromNeural = 0
            var jewelsPixelsFromNeural = 0

            if (hasSegOnnx && cropRect.width() >= 24f && cropRect.height() >= 24f) {
                val parsed = runSegformerOnCrop(
                    ortEnv = ortEnv!!,
                    segFile = segFile,
                    sourceBitmap = sourceBitmap,
                    cropRect = cropRect,
                    preferHardwareAccel = preferHardwareAccel
                )
                if (parsed != null) {
                    usedNeural = true
                    val (classMap, mapSize) = parsed
                    val invCropW = 1f / cropRect.width().coerceAtLeast(1f)
                    val invCropH = 1f / cropRect.height().coerceAtLeast(1f)

                    val xStart = (cropRect.left * sx).toInt().coerceIn(0, workW - 1)
                    val xEnd = (cropRect.right * sx).toInt().coerceIn(0, workW - 1)
                    val yStart = (cropRect.top * sy).toInt().coerceIn(0, workH - 1)
                    val yEnd = (cropRect.bottom * sy).toInt().coerceIn(0, workH - 1)

                    for (wy in yStart..yEnd) {
                        val srcPy = wy / sy
                        val vNorm = ((srcPy - cropRect.top) * invCropH).coerceIn(0f, 0.999f)
                        val cy = (vNorm * mapSize).toInt().coerceIn(0, mapSize - 1)
                        val row = wy * workW
                        val cRow = cy * mapSize
                        for (wx in xStart..xEnd) {
                            val srcPx = wx / sx
                            val uNorm = ((srcPx - cropRect.left) * invCropW).coerceIn(0f, 0.999f)
                            val cx = (uNorm * mapSize).toInt().coerceIn(0, mapSize - 1)
                            val cls = classMap[cRow + cx]
                            val idx = row + wx
                            when (cls) {
                                18 -> { // hat / cap
                                    capMask[idx] = 1.0f
                                    capPixelsFromNeural++
                                }
                                3 -> { // eye_g (glasses / sunglasses)
                                    glassesMask[idx] = 1.0f
                                    glassesPixelsFromNeural++
                                }
                                15 -> { // neck_l (necklace)
                                    jewelsMask[idx] = 1.0f
                                    jewelsPixelsFromNeural++
                                }
                                9 -> { // ear_r (earring / right ear jewelry in some CelebAMask mappings)
                                    if (wy > eyeMidY + eyeDist * 0.05f) {
                                        jewelsMask[idx] = max(jewelsMask[idx], 0.85f)
                                        jewelsPixelsFromNeural++
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Fill any missing/weak accessory channel with smart Biometric + Depth region masks
            // so tapping Cap / Glasses / Jewels ALWAYS provides an accurate starting mask.
            if (capPixelsFromNeural < 35) {
                paintBiometricCapRegion(
                    outMask = capMask,
                    w = workW,
                    h = workH,
                    eyeMidX = eyeMidX,
                    eyeMidY = eyeMidY,
                    eyeDist = eyeDist,
                    depthMap = depthMap
                )
            }
            if (glassesPixelsFromNeural < 25) {
                paintBiometricGlassesRegion(
                    outMask = glassesMask,
                    w = workW,
                    h = workH,
                    leftEyeX = leftEyeX,
                    leftEyeY = leftEyeY,
                    rightEyeX = rightEyeX,
                    rightEyeY = rightEyeY,
                    noseX = noseX,
                    noseY = noseY,
                    eyeDist = eyeDist
                )
            }
            if (jewelsPixelsFromNeural < 30) {
                paintBiometricJewelsRegion(
                    outMask = jewelsMask,
                    w = workW,
                    h = workH,
                    eyeMidX = eyeMidX,
                    eyeMidY = eyeMidY,
                    mouthMidY = mouthMidY,
                    eyeDist = eyeDist,
                    depthMap = depthMap
                )
            }
        }

        val label = if (usedNeural) {
            "SegFormer Face Parser + Biometric Landmarks"
        } else if (depthMap != null) {
            "Biometric Landmarks + ${depthMap.modelUsedLabel}"
        } else {
            "Biometric Face Landmarks"
        }

        return AccessoryMaskLayers(
            width = workW,
            height = workH,
            capBaseMask = capMask,
            glassesBaseMask = glassesMask,
            jewelsBaseMask = jewelsMask,
            detectedFacesCount = faces.size,
            usedNeuralParser = usedNeural,
            engineLabel = label
        )
    }

    private fun detectFacesForSource(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        preferHardwareAccel: Boolean
    ): List<DetectedFace> {
        val detFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.DETECTOR)
        if (ortEnv != null && detFile.exists() && detFile.length() > 1024L) {
            val onnxFaces = runCatching {
                ScrfdFaceDetector.detectFacesOnnx(
                    ortEnv = ortEnv,
                    detModelFile = detFile,
                    bitmap = sourceBitmap
                )
            }.getOrNull().orEmpty()
            if (onnxFaces.isNotEmpty()) return onnxFaces
        }
        return runCatching {
            ScrfdFaceDetector.detectFacesAndroidPreviewFallback(sourceBitmap)
        }.getOrDefault(emptyList())
    }

    private fun runSegformerOnCrop(
        ortEnv: OrtEnvironment,
        segFile: File,
        sourceBitmap: Bitmap,
        cropRect: RectF,
        preferHardwareAccel: Boolean
    ): Pair<IntArray, Int>? {
        val modelSize = 512
        val cropBmp = Bitmap.createBitmap(modelSize, modelSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(cropBmp)
        val srcRect = Rect(
            cropRect.left.toInt().coerceIn(0, sourceBitmap.width - 1),
            cropRect.top.toInt().coerceIn(0, sourceBitmap.height - 1),
            cropRect.right.toInt().coerceIn(1, sourceBitmap.width),
            cropRect.bottom.toInt().coerceIn(1, sourceBitmap.height)
        )
        canvas.drawBitmap(
            sourceBitmap,
            srcRect,
            Rect(0, 0, modelSize, modelSize),
            Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
        )

        val hw = modelSize * modelSize
        val pixels = IntArray(hw)
        cropBmp.getPixels(pixels, 0, modelSize, 0, 0, modelSize, modelSize)
        cropBmp.recycle()

        val floatBuffer = FloatBuffer.allocate(3 * hw)
        for (i in 0 until hw) {
            val c = pixels[i]
            val r = ((c ushr 16) and 0xFF) / 255.0f
            val g = ((c ushr 8) and 0xFF) / 255.0f
            val b = (c and 0xFF) / 255.0f
            floatBuffer.put(i, (r - 0.485f) / 0.229f)
            floatBuffer.put(hw + i, (g - 0.456f) / 0.224f)
            floatBuffer.put(2 * hw + i, (b - 0.406f) / 0.225f)
        }
        floatBuffer.rewind()

        return runCatching {
            val session: OrtSession = requireNotNull(
                OnnxProtobufInspector.getOrCreateCachedSession(
                    ortEnv = ortEnv,
                    file = segFile,
                    preferHardwareAcceleration = preferHardwareAccel
                )
            )
            val inputName = session.inputNames.first()
            val shape = longArrayOf(1L, 3L, modelSize.toLong(), modelSize.toLong())
            OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { inTensor ->
                session.run(mapOf(inputName to inTensor)).use { result ->
                    val outTensor = result[0] as OnnxTensor
                    val outShape = outTensor.info.shape
                    if (outShape.size != 4 || outShape[1] < 14L) return null
                    val numClasses = outShape[1].toInt()
                    val outH = outShape[2].toInt()
                    val outW = outShape[3].toInt()
                    val planeSize = outH * outW
                    val outFloats = FloatArray(numClasses * planeSize)
                    outTensor.floatBuffer.get(outFloats)

                    val classMap = IntArray(hw)
                    for (y in 0 until modelSize) {
                        val sy = (y * outH / modelSize).coerceIn(0, outH - 1)
                        val row = y * modelSize
                        for (x in 0 until modelSize) {
                            val sx = (x * outW / modelSize).coerceIn(0, outW - 1)
                            val spatialIdx = sy * outW + sx
                            var bestClass = 0
                            var bestLogit = -Float.MAX_VALUE
                            for (cls in 0 until numClasses) {
                                val logit = outFloats[cls * planeSize + spatialIdx]
                                if (logit > bestLogit) {
                                    bestLogit = logit
                                    bestClass = cls
                                }
                            }
                            classMap[row + x] = bestClass
                        }
                    }
                    classMap to modelSize
                }
            }
        }.getOrNull()
    }

    private fun paintBiometricCapRegion(
        outMask: FloatArray,
        w: Int,
        h: Int,
        eyeMidX: Float,
        eyeMidY: Float,
        eyeDist: Float,
        depthMap: DepthMapResult?
    ) {
        // Cranial cap & visor region above eyebrows
        val centerX = eyeMidX
        val centerY = eyeMidY - eyeDist * 1.28f
        val rx = eyeDist * 1.65f
        val ry = eyeDist * 1.05f
        val minForeheadY = eyeMidY - eyeDist * 0.42f

        val subjectDepth = depthMap?.sampleBilinear(
            (eyeMidX / (w - 1f).coerceAtLeast(1f)).coerceIn(0f, 1f),
            (eyeMidY / (h - 1f).coerceAtLeast(1f)).coerceIn(0f, 1f)
        )

        val x0 = (centerX - rx).toInt().coerceIn(0, w - 1)
        val x1 = (centerX + rx).toInt().coerceIn(0, w - 1)
        val y0 = (centerY - ry).toInt().coerceIn(0, h - 1)
        val y1 = minForeheadY.toInt().coerceIn(0, h - 1)
        val invW = 1f / (w - 1f).coerceAtLeast(1f)
        val invH = 1f / (h - 1f).coerceAtLeast(1f)

        for (y in y0..y1) {
            val dy = (y - centerY) / ry
            val row = y * w
            for (x in x0..x1) {
                val dx = (x - centerX) / rx
                val d = dx * dx + dy * dy
                if (d <= 1.0f) {
                    var weight = ((1.0f - d) * 3.5f).coerceIn(0f, 1f)
                    if (depthMap != null && subjectDepth != null) {
                        val near = depthMap.sampleBilinear(x * invW, y * invH)
                        // Reject distant background behind cap crown
                        if (near < subjectDepth - 0.22f) {
                            weight *= ((near - (subjectDepth - 0.30f)) / 0.08f).coerceIn(0f, 1f)
                        }
                    }
                    if (weight > outMask[row + x]) {
                        outMask[row + x] = weight
                    }
                }
            }
        }
    }

    private fun paintBiometricGlassesRegion(
        outMask: FloatArray,
        w: Int,
        h: Int,
        leftEyeX: Float,
        leftEyeY: Float,
        rightEyeX: Float,
        rightEyeY: Float,
        noseX: Float,
        noseY: Float,
        eyeDist: Float
    ) {
        val eyeMidX = (leftEyeX + rightEyeX) * 0.5f
        val eyeMidY = (leftEyeY + rightEyeY) * 0.5f
        val bridgeY = eyeMidY * 0.86f + noseY * 0.14f
        val halfW = eyeDist * 1.28f
        val halfH = eyeDist * 0.44f

        val x0 = (eyeMidX - halfW).toInt().coerceIn(0, w - 1)
        val x1 = (eyeMidX + halfW).toInt().coerceIn(0, w - 1)
        val y0 = (bridgeY - halfH).toInt().coerceIn(0, h - 1)
        val y1 = (bridgeY + halfH).toInt().coerceIn(0, h - 1)

        val dxDir = (rightEyeX - leftEyeX) / eyeDist
        val dyDir = (rightEyeY - leftEyeY) / eyeDist

        for (y in y0..y1) {
            val row = y * w
            for (x in x0..x1) {
                val relX = x - eyeMidX
                val relY = y - bridgeY
                // Rotate along eye roll axis
                val u = relX * dxDir + relY * dyDir
                val v = -relX * dyDir + relY * dxDir
                val nu = abs(u) / halfW
                val nv = abs(v) / halfH
                // Rounded rectangular sunglasses mask
                val boxDist = max(nu, nv)
                val cornerDist = hypot(max(0f, nu - 0.65f) / 0.35f, max(0f, nv - 0.55f) / 0.45f)
                val d = max(boxDist, cornerDist)
                if (d <= 1.0f) {
                    val weight = ((1.0f - d) * 4.0f).coerceIn(0f, 1f)
                    if (weight > outMask[row + x]) {
                        outMask[row + x] = weight
                    }
                }
            }
        }
    }

    private fun paintBiometricJewelsRegion(
        outMask: FloatArray,
        w: Int,
        h: Int,
        eyeMidX: Float,
        eyeMidY: Float,
        mouthMidY: Float,
        eyeDist: Float,
        depthMap: DepthMapResult?
    ) {
        val subjectDepth = depthMap?.sampleBilinear(
            (eyeMidX / (w - 1f).coerceAtLeast(1f)).coerceIn(0f, 1f),
            (eyeMidY / (h - 1f).coerceAtLeast(1f)).coerceIn(0f, 1f)
        )
        val invW = 1f / (w - 1f).coerceAtLeast(1f)
        val invH = 1f / (h - 1f).coerceAtLeast(1f)

        // 1) Left and Right Ear-lobe earring drop ovals
        val earOffsetX = eyeDist * 1.32f
        val earCenterY = (eyeMidY + mouthMidY) * 0.5f + eyeDist * 0.22f
        val earRx = eyeDist * 0.30f
        val earRy = eyeDist * 0.48f
        val earCentersX = floatArrayOf(eyeMidX - earOffsetX, eyeMidX + earOffsetX)

        for (ecx in earCentersX) {
            val x0 = (ecx - earRx).toInt().coerceIn(0, w - 1)
            val x1 = (ecx + earRx).toInt().coerceIn(0, w - 1)
            val y0 = (earCenterY - earRy).toInt().coerceIn(0, h - 1)
            val y1 = (earCenterY + earRy).toInt().coerceIn(0, h - 1)
            for (y in y0..y1) {
                val dy = (y - earCenterY) / earRy
                val row = y * w
                for (x in x0..x1) {
                    val dx = (x - ecx) / earRx
                    val d = dx * dx + dy * dy
                    if (d <= 1.0f) {
                        var weight = ((1.0f - d) * 3.5f).coerceIn(0f, 1f)
                        if (depthMap != null && subjectDepth != null) {
                            val near = depthMap.sampleBilinear(x * invW, y * invH)
                            if (near < subjectDepth - 0.25f) weight *= 0.25f
                        }
                        if (weight > outMask[row + x]) outMask[row + x] = weight
                    }
                }
            }
        }

        // 2) Lower-neck / collarbone necklace U-collar band
        val neckCenterX = eyeMidX
        val neckTopY = mouthMidY + eyeDist * 0.72f
        val neckBottomY = mouthMidY + eyeDist * 2.15f
        val neckHalfW = eyeDist * 1.25f
        val nx0 = (neckCenterX - neckHalfW).toInt().coerceIn(0, w - 1)
        val nx1 = (neckCenterX + neckHalfW).toInt().coerceIn(0, w - 1)
        val ny0 = neckTopY.toInt().coerceIn(0, h - 1)
        val ny1 = neckBottomY.toInt().coerceIn(0, h - 1)
        val neckCenterY = (neckTopY + neckBottomY) * 0.5f
        val neckHalfH = ((neckBottomY - neckTopY) * 0.5f).coerceAtLeast(6f)

        for (y in ny0..ny1) {
            val dy = (y - neckCenterY) / neckHalfH
            val row = y * w
            for (x in nx0..nx1) {
                val dx = (x - neckCenterX) / neckHalfW
                val d = dx * dx + dy * dy
                if (d <= 1.0f) {
                    var weight = ((1.0f - d) * 3.2f).coerceIn(0f, 1f)
                    if (depthMap != null && subjectDepth != null) {
                        val near = depthMap.sampleBilinear(x * invW, y * invH)
                        if (near < subjectDepth - 0.25f) weight *= 0.25f
                    }
                    if (weight > outMask[row + x]) outMask[row + x] = weight
                }
            }
        }
    }
}
