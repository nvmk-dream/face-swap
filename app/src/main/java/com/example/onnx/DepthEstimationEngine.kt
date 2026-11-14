package com.example.onnx

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import java.io.File
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

enum class DepthModelFamily(
    val id: String,
    val displayLabel: String,
    val priorityOrder: Int
) {
    DEPTH_ANYTHING_V2(
        id = "depth_anything_v2",
        displayLabel = "Depth Anything V2 Small",
        priorityOrder = 1
    ),
    SENTIS_MIDAS(
        id = "sentis_midas",
        displayLabel = "sentis-MiDaS (DPT Next-ViT)",
        priorityOrder = 2
    )
}

data class InstalledDepthModel(
    val family: DepthModelFamily,
    val file: File,
    val displayName: String,
    val priorityScore: Int
)

/**
 * Normalized monocular depth map in `[0f, 1f]` where:
 *  - `1.0f` = closest foreground to the camera
 *  - `0.0f` = farthest background from the camera
 */
data class DepthMapResult(
    val width: Int,
    val height: Int,
    val nearMap: FloatArray,
    val modelUsedLabel: String,
    val elapsedMs: Long
) {
    fun sampleBilinear(nx: Float, ny: Float): Float {
        if (width <= 0 || height <= 0 || nearMap.isEmpty()) return 0.5f
        val fx = (nx.coerceIn(0f, 1f) * (width - 1)).coerceIn(0f, (width - 1).toFloat())
        val fy = (ny.coerceIn(0f, 1f) * (height - 1)).coerceIn(0f, (height - 1).toFloat())
        val x0 = fx.toInt().coerceIn(0, width - 1)
        val y0 = fy.toInt().coerceIn(0, height - 1)
        val x1 = (x0 + 1).coerceAtMost(width - 1)
        val y1 = (y0 + 1).coerceAtMost(height - 1)
        val dx = fx - x0
        val dy = fy - y0

        val v00 = nearMap[y0 * width + x0]
        val v10 = nearMap[y0 * width + x1]
        val v01 = nearMap[y1 * width + x0]
        val v11 = nearMap[y1 * width + x1]

        val top = v00 + (v10 - v00) * dx
        val bot = v01 + (v11 - v01) * dx
        return (top + (bot - top) * dy).coerceIn(0f, 1f)
    }

    /**
     * Suggests a foreground/background depth cutoff threshold in `[0.18f, 0.85f]`.
     */
    fun suggestForegroundCutoff(): Float {
        if (nearMap.isEmpty()) return 0.48f
        val bins = 64
        val hist = IntArray(bins)
        for (v in nearMap) {
            val b = (v * (bins - 1)).roundToInt().coerceIn(0, bins - 1)
            hist[b]++
        }
        val total = nearMap.size.toDouble()
        var sumAll = 0.0
        for (i in 0 until bins) sumAll += i * hist[i].toDouble()

        var sumBg = 0.0
        var wBg = 0.0
        var maxVar = -1.0
        var bestBin = bins / 2
        for (i in 0 until bins) {
            wBg += hist[i].toDouble()
            if (wBg <= 0.0) continue
            val wFg = total - wBg
            if (wFg <= 0.0) break
            sumBg += i * hist[i].toDouble()
            val mBg = sumBg / wBg
            val mFg = (sumAll - sumBg) / wFg
            val diff = mBg - mFg
            val varBetween = wBg * wFg * diff * diff
            if (varBetween > maxVar) {
                maxVar = varBetween
                bestBin = i
            }
        }
        return (bestBin.toFloat() / (bins - 1)).coerceIn(0.18f, 0.85f)
    }

    /**
     * Renders a cyan (`#00E5FF`) ARGB_8888 mask bitmap at `(targetWidth, targetHeight)`.
     * When [maskBackground] is true, pixels farther than [cutoff] (`near < cutoff`) are masked.
     * When [maskBackground] is false, foreground pixels (`near >= cutoff`) are masked.
     */
    fun renderAutoMaskBitmap(
        targetWidth: Int,
        targetHeight: Int,
        cutoff: Float,
        maskBackground: Boolean
    ): Bitmap {
        val w = targetWidth.coerceAtLeast(1)
        val h = targetHeight.coerceAtLeast(1)
        val px = IntArray(w * h)
        val invW = if (w > 1) 1f / (w - 1) else 0f
        val invH = if (h > 1) 1f / (h - 1) else 0f
        val softness = 0.035f
        val thr = cutoff.coerceIn(0.05f, 0.95f)

        for (y in 0 until h) {
            val ny = y * invH
            val rowOff = y * w
            for (x in 0 until w) {
                val nx = x * invW
                val near = sampleBilinear(nx, ny)
                val maskScore = if (maskBackground) {
                    ((thr + softness - near) / (2f * softness)).coerceIn(0f, 1f)
                } else {
                    ((near - (thr - softness)) / (2f * softness)).coerceIn(0f, 1f)
                }
                val alpha = (maskScore * 255f).roundToInt().coerceIn(0, 255)
                px[rowOff + x] = if (alpha > 16) {
                    Color.argb(alpha, 0, 229, 255)
                } else {
                    Color.TRANSPARENT
                }
            }
        }
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
            setPixels(px, 0, w, 0, 0, w, h)
        }
    }
}

/**
 * Universal On-Device Monocular Depth Estimation Engine.
 *
 * Priority 1: `Depth Anything V2 Small` (`model.onnx` / `model_fp16.onnx` / `model_int8.onnx` -> 518x518, ImageNet norm)
 * Priority 2: `sentis-MiDaS` (`dpt_next_vit_large_384.onnx` / `dpt_swin2_tiny_256.onnx` -> baked [0, 1] norm)
 *
 * Runs with short-lived sessions that close immediately after generating the normalized depth map
 * so Android native RAM remains available for Face Swap and SD 1.5 Inpainting.
 */
object DepthEstimationEngine {

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun classifyDepthFile(fileName: String, parentPathHint: String = ""): Pair<DepthModelFamily, Int>? {
        val lower = fileName.trim().lowercase()
        val pathLower = parentPathHint.trim().lowercase()
        if (!lower.endsWith(".onnx")) return null

        val isDepthAnythingExplicit = lower.contains("depth_anything") || lower.contains("depth-anything")
        val isDepthAnythingGeneric = (lower == "model.onnx" ||
            lower == "model_fp16.onnx" ||
            lower == "model_int8.onnx" ||
            lower == "model_uint8.onnx" ||
            lower == "model_quantized.onnx") &&
            (pathLower.contains("depth_anything") || pathLower.contains("depth-anything") || pathLower.contains("depth"))

        if (isDepthAnythingExplicit || isDepthAnythingGeneric) {
            val variantScore = when {
                lower == "model.onnx" || lower == "depth_anything_v2_small.onnx" || lower.contains("fp32") -> 290
                lower.contains("fp16") -> 275
                lower.contains("int8") || lower.contains("uint8") || lower.contains("quantized") -> 250
                else -> 240
            }
            return DepthModelFamily.DEPTH_ANYTHING_V2 to variantScore
        }

        val isMidas = lower.contains("dpt_next_vit") ||
            lower.contains("dpt_swin") ||
            lower.contains("dpt_beit") ||
            lower.contains("dpt_large") ||
            lower.contains("dpt_levit") ||
            lower.contains("midas") ||
            ((lower == "model.onnx" || lower == "model_fp16.onnx") && pathLower.contains("midas"))

        if (isMidas) {
            val variantScore = when {
                lower.contains("next_vit_large_384") -> 195
                lower.contains("swin2_tiny_256") -> 185
                lower.contains("swin2_base_384") -> 180
                lower.contains("beit_base_384") -> 175
                lower.contains("levit_224") -> 165
                lower.contains("midas_v21_384") -> 160
                lower.contains("large") -> 140
                else -> 150
            }
            return DepthModelFamily.SENTIS_MIDAS to variantScore
        }

        return null
    }

    fun canonicalImportedFileName(originalName: String, parentPathHint: String = ""): String {
        val lower = originalName.trim().lowercase()
        val classified = classifyDepthFile(originalName, parentPathHint)
        if (classified?.first == DepthModelFamily.DEPTH_ANYTHING_V2) {
            return when {
                lower.contains("fp16") -> "depth_anything_v2_fp16.onnx"
                lower.contains("int8") || lower.contains("uint8") || lower.contains("quantized") ->
                    "depth_anything_v2_int8.onnx"
                else -> "depth_anything_v2_small.onnx"
            }
        }
        return originalName.trim().ifEmpty { "dpt_next_vit_large_384.onnx" }
    }

    fun resolveDepthCandidates(context: Context): List<InstalledDepthModel> {
        val root = OnnxProtobufInspector.getModelsRootDir(context)
        val depthDir = File(root, ModelSlot.DEPTH.subdirectory)
        val searchDirs = buildList {
            if (depthDir.exists()) add(depthDir)
            OnnxProtobufInspector.SUBDIRECTORIES.forEach { sub ->
                if (sub != ModelSlot.DEPTH.subdirectory) {
                    val d = File(root, sub)
                    if (d.exists()) add(d)
                }
            }
            add(root)
        }

        val seenPaths = HashSet<String>()
        val found = mutableListOf<InstalledDepthModel>()
        for (dir in searchDirs) {
            dir.listFiles()?.forEach { file ->
                if (!file.isFile || file.length() <= 1024L) return@forEach
                if (!seenPaths.add(file.absolutePath)) return@forEach
                val classified = classifyDepthFile(file.name, dir.name) ?: return@forEach
                found.add(
                    InstalledDepthModel(
                        family = classified.first,
                        file = file,
                        displayName = file.name,
                        priorityScore = classified.second
                    )
                )
            }
        }
        return found.sortedWith(
            compareBy<InstalledDepthModel> { it.family.priorityOrder }
                .thenByDescending { it.priorityScore }
                .thenBy { it.file.name }
        )
    }

    fun resolvePrimaryDepthFile(context: Context): File {
        val first = resolveDepthCandidates(context).firstOrNull()?.file
        if (first != null) return first
        val root = OnnxProtobufInspector.getModelsRootDir(context)
        return File(File(root, ModelSlot.DEPTH.subdirectory), ModelSlot.DEPTH.canonicalFileName)
    }

    fun hasInstalledDepthModel(context: Context): Boolean =
        resolveDepthCandidates(context).isNotEmpty()

    fun estimateDepth(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        preferHardwareAccel: Boolean = false,
        onProgress: (String) -> Unit = {}
    ): DepthMapResult? {
        if (ortEnv == null) return null
        val candidates = resolveDepthCandidates(context)
        if (candidates.isEmpty()) return null

        for ((idx, candidate) in candidates.withIndex()) {
            onProgress(
                if (idx == 0) {
                    "Analyzing 3D depth (${candidate.family.displayLabel})..."
                } else {
                    "Fallback depth (${candidate.family.displayLabel})..."
                }
            )
            val attempt = runCatching {
                runSingleDepthModel(
                    ortEnv = ortEnv,
                    candidate = candidate,
                    sourceBitmap = sourceBitmap,
                    preferHardwareAccel = preferHardwareAccel
                )
            }
            if (attempt.isSuccess) {
                return attempt.getOrNull()
            }
        }
        return null
    }

    /**
     * Computes a `[512 x 512]` foreground-occlusion gate in `[0f, 1f]` aligned to the Face Swap 512 crop.
     * `1.0f` = normal face pixel (allow swap), `0.0f` = foreground object closer than the face
     * (hand, mic, cup, phone, glasses frame) -> preserve target pixels!
     */
    fun computeFaceOcclusionGate512(
        depthMap: DepthMapResult,
        targetWidth: Int,
        targetHeight: Int,
        invM512: FloatArray,
        leftEye: PointF,
        rightEye: PointF,
        nose: PointF,
        leftMouth: PointF,
        rightMouth: PointF
    ): FloatArray {
        val size = 512
        val total = size * size
        val gate = FloatArray(total) { 1.0f }
        if (targetWidth <= 1 || targetHeight <= 1 || depthMap.nearMap.isEmpty()) return gate

        fun sampleAtCrop(cx: Float, cy: Float): Float {
            val tx = invM512[0] * cx + invM512[1] * cy + invM512[2]
            val ty = invM512[3] * cx + invM512[4] * cy + invM512[5]
            return depthMap.sampleBilinear(tx / (targetWidth - 1f), ty / (targetHeight - 1f))
        }

        // Sample core upper-face reference plane (eyes, bridge, cheeks, nose) to estimate normal face depth
        val faceCenterX = (leftEye.x + rightEye.x + leftMouth.x + rightMouth.x) * 0.25f
        val faceCenterY = (leftEye.y + rightEye.y + leftMouth.y + rightMouth.y) * 0.25f
        val samples = FloatArray(7)
        samples[0] = sampleAtCrop(leftEye.x, leftEye.y)
        samples[1] = sampleAtCrop(rightEye.x, rightEye.y)
        samples[2] = sampleAtCrop(nose.x, nose.y)
        samples[3] = sampleAtCrop((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f)
        samples[4] = sampleAtCrop(leftMouth.x, leftMouth.y)
        samples[5] = sampleAtCrop(rightMouth.x, rightMouth.y)
        samples[6] = sampleAtCrop(faceCenterX, faceCenterY)
        samples.sort()

        // Use upper percentile (closest natural facial feature, e.g. nose/forehead) + safety margin
        // so normal nose protrusion is NEVER mistaken for an occluding hand/object.
        val faceNearCeiling = samples[5]
        val occluderStartDelta = 0.13f
        val occluderFullDelta = 0.24f

        val a0 = invM512[0]
        val a1 = invM512[1]
        val a2 = invM512[2]
        val b0 = invM512[3]
        val b1 = invM512[4]
        val b2 = invM512[5]
        val invW = 1f / (targetWidth - 1f)
        val invH = 1f / (targetHeight - 1f)

        for (y in 0 until size) {
            val rowOff = y * size
            val rowTx = a1 * y + a2
            val rowTy = b1 * y + b2
            for (x in 0 until size) {
                val tx = a0 * x + rowTx
                val ty = b0 * x + rowTy
                val near = depthMap.sampleBilinear(tx * invW, ty * invH)
                val delta = near - faceNearCeiling
                if (delta > occluderStartDelta) {
                    val occ = ((delta - occluderStartDelta) / (occluderFullDelta - occluderStartDelta))
                        .coerceIn(0f, 1f)
                    gate[rowOff + x] = (1f - occ).coerceIn(0f, 1f)
                }
            }
        }
        return gate
    }

    private fun runSingleDepthModel(
        ortEnv: OrtEnvironment,
        candidate: InstalledDepthModel,
        sourceBitmap: Bitmap,
        preferHardwareAccel: Boolean
    ): DepthMapResult {
        val t0 = System.currentTimeMillis()
        OnnxProtobufInspector.clearCachedSessions()
        System.gc()

        val (depthSession, _) = OnnxProtobufInspector.createSessionWithFallback(
            ortEnv = ortEnv,
            modelFile = candidate.file,
            preferHardwareAcceleration = preferHardwareAccel
        )
        depthSession.use { session ->
                val inputEntry = session.inputInfo.entries.firstOrNull()
                    ?: error("Depth model has no input tensor.")
                val inputName = inputEntry.key
                val inputInfo = inputEntry.value.info as? TensorInfo
                    ?: error("Depth input is not a tensor.")
                val shape = inputInfo.shape

                val isNhwc = shape.size == 4 && shape[3] == 3L && shape[1] != 3L
                val rawH = if (shape.size == 4) (if (isNhwc) shape[1] else shape[2]) else -1L
                val rawW = if (shape.size == 4) (if (isNhwc) shape[2] else shape[3]) else -1L

                val defaultSize = when {
                    candidate.family == DepthModelFamily.DEPTH_ANYTHING_V2 -> 518
                    candidate.file.name.lowercase().contains("256") -> 256
                    candidate.file.name.lowercase().contains("512") -> 512
                    candidate.file.name.lowercase().contains("224") -> 224
                    else -> 384
                }
                val inH = if (rawH > 0L) rawH.toInt() else defaultSize
                val inW = if (rawW > 0L) rawW.toInt() else defaultSize

                val useBakedZeroToOne = candidate.family == DepthModelFamily.SENTIS_MIDAS ||
                    inputName.equals("input_image", ignoreCase = true)

                val resized = if (sourceBitmap.width == inW && sourceBitmap.height == inH) {
                    sourceBitmap
                } else {
                    Bitmap.createScaledBitmap(sourceBitmap, inW, inH, true)
                }
                val pixels = IntArray(inW * inH)
                resized.getPixels(pixels, 0, inW, 0, 0, inW, inH)
                if (resized !== sourceBitmap) resized.recycle()

                val floatCount = 3 * inH * inW
                val inputFloats = FloatArray(floatCount)
                val hw = inH * inW
                if (isNhwc) {
                    for (i in 0 until hw) {
                        val c = pixels[i]
                        val r = ((c ushr 16) and 0xFF) / 255f
                        val g = ((c ushr 8) and 0xFF) / 255f
                        val b = (c and 0xFF) / 255f
                        val base = i * 3
                        if (useBakedZeroToOne) {
                            inputFloats[base] = r
                            inputFloats[base + 1] = g
                            inputFloats[base + 2] = b
                        } else {
                            inputFloats[base] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                            inputFloats[base + 1] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                            inputFloats[base + 2] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                        }
                    }
                } else {
                    for (i in 0 until hw) {
                        val c = pixels[i]
                        val r = ((c ushr 16) and 0xFF) / 255f
                        val g = ((c ushr 8) and 0xFF) / 255f
                        val b = (c and 0xFF) / 255f
                        if (useBakedZeroToOne) {
                            inputFloats[i] = r
                            inputFloats[hw + i] = g
                            inputFloats[2 * hw + i] = b
                        } else {
                            inputFloats[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                            inputFloats[hw + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                            inputFloats[2 * hw + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                        }
                    }
                }

                val tensorShape = if (isNhwc) {
                    longArrayOf(1L, inH.toLong(), inW.toLong(), 3L)
                } else {
                    longArrayOf(1L, 3L, inH.toLong(), inW.toLong())
                }

                val inputTensor = if (inputInfo.type == OnnxJavaType.FLOAT16) {
                    val shorts = ShortArray(floatCount) { idx -> floatToFloat16(inputFloats[idx]) }
                    OnnxTensor.createTensor(ortEnv, ShortBuffer.wrap(shorts), tensorShape, OnnxJavaType.FLOAT16)
                } else {
                    OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputFloats), tensorShape)
                }

                val (outW, outH, rawDepth) = inputTensor.use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { results ->
                        val outTensor = results[0] as? OnnxTensor
                            ?: error("Depth model output is not an OnnxTensor.")
                        val outInfo = outTensor.info
                        val outShape = outInfo.shape
                        val hOut = when (outShape.size) {
                            4 -> outShape[2].toInt().coerceAtLeast(1)
                            3 -> outShape[1].toInt().coerceAtLeast(1)
                            2 -> outShape[0].toInt().coerceAtLeast(1)
                            else -> inH
                        }
                        val wOut = when (outShape.size) {
                            4 -> outShape[3].toInt().coerceAtLeast(1)
                            3 -> outShape[2].toInt().coerceAtLeast(1)
                            2 -> outShape[1].toInt().coerceAtLeast(1)
                            else -> inW
                        }
                        val count = hOut * wOut
                        val arr = FloatArray(count)
                        if (outInfo.type == OnnxJavaType.FLOAT16) {
                            val sb = outTensor.shortBuffer
                            for (i in 0 until min(count, sb.remaining())) {
                                arr[i] = float16ToFloat(sb.get())
                            }
                        } else {
                            val fb = outTensor.floatBuffer
                            val copyLen = min(count, fb.remaining())
                            fb.get(arr, 0, copyLen)
                        }
                        Triple(wOut, hOut, arr)
                    }
                }

                val normalizedNear = normalizePercentileDepth(rawDepth)
                val elapsed = (System.currentTimeMillis() - t0).coerceAtLeast(1L)
                return DepthMapResult(
                    width = outW,
                    height = outH,
                    nearMap = normalizedNear,
                    modelUsedLabel = "${candidate.family.displayLabel} (${candidate.displayName})",
                    elapsedMs = elapsed
                )
        }
    }

    private fun normalizePercentileDepth(raw: FloatArray): FloatArray {
        val n = raw.size
        if (n == 0) return FloatArray(0)
        val finite = FloatArray(n)
        var validCount = 0
        for (v in raw) {
            if (v.isFinite()) {
                finite[validCount++] = v
            }
        }
        if (validCount == 0) return FloatArray(n) { 0.5f }
        val sorted = finite.copyOf(validCount)
        sorted.sort()
        val p2 = sorted[((validCount - 1) * 0.02f).roundToInt().coerceIn(0, validCount - 1)]
        val p98 = sorted[((validCount - 1) * 0.98f).roundToInt().coerceIn(0, validCount - 1)]
        val range = (p98 - p2).takeIf { abs(it) > 1e-6f } ?: 1f

        return FloatArray(n) { idx ->
            val v = raw[idx]
            if (!v.isFinite()) 0.5f else ((v - p2) / range).coerceIn(0f, 1f)
        }
    }

    private fun floatToFloat16(value: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(value)
        val sign = (bits ushr 16) and 0x8000
        var valBits = (bits and 0x7FFFFFFF) + 0x1000
        if (valBits >= 0x47800000) {
            return if ((bits and 0x7FFFFFFF) >= 0x47800000) {
                if (valBits < 0x7F800000) (sign or 0x7C00).toShort()
                else (sign or 0x7C00 or ((bits and 0x007FFFFF) ushr 13)).toShort()
            } else {
                (sign or 0x7BFF).toShort()
            }
        }
        if (valBits >= 0x38800000) {
            return (sign or ((valBits - 0x38000000) ushr 13)).toShort()
        }
        if (valBits < 0x33000000) return sign.toShort()
        valBits = (bits and 0x7FFFFFFF) ushr 23
        val mantissa = (bits and 0x007FFFFF) or 0x00800000
        return (sign or ((mantissa + (0x800000 ushr (valBits - 102))) ushr (126 - valBits))).toShort()
    }

    private fun float16ToFloat(h: Short): Float {
        val bits = h.toInt() and 0xFFFF
        val sign = (bits and 0x8000) shl 16
        val exp = (bits ushr 10) and 0x1F
        val mant = bits and 0x03FF
        val fBits = when (exp) {
            0 -> if (mant == 0) sign else {
                var e = 127 - 14
                var m = mant
                while ((m and 0x0400) == 0) {
                    m = m shl 1
                    e--
                }
                sign or (e shl 23) or ((m and 0x03FF) shl 13)
            }
            31 -> sign or 0x7F800000 or (mant shl 13)
            else -> sign or ((exp + (127 - 15)) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(fBits)
    }
}
