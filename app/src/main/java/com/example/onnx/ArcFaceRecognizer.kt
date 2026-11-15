package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.PointF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.sqrt

data class SourceEmbeddingResult(
    val rawNormedEmbedding512: FloatArray,
    val latentSourceVector512: FloatArray,
    val aligned112Crop: Bitmap,
    val usedArcFaceModel: Boolean,
    val usedEmbeddedEmap: Boolean,
    val providerSummary: String,
    val rawEmbeddingL2Norm: Float = 1.0f,
    val executionProvider: String = "CPU"
)

/**
 * Extracts the 512-dimensional face identity embedding required by `inswapper_128.onnx`.
 *
 * Architectural Requirement:
 *  - `inswapper_128.onnx` takes two inputs:
 *      1) `target`: float32[1, 3, 128, 128]
 *      2) `source`: float32[1, 512]
 *  - Because `det_10g.onnx` only outputs bounding boxes and 5-point keypoints,
 *    a 3rd model (`w600k_r50.onnx` or `w600k_mbf.onnx` from InsightFace Buffalo_L/S)
 *    is required to compute the raw 512-D ArcFace identity vector from a 112x112 aligned crop.
 *  - Next, the L2-normalized 512-D ArcFace vector `e_norm` is multiplied by the 512x512 `emap`
 *    matrix extracted from `inswapper_128.onnx`'s ONNX initializer graph and L2-normalized again:
 *    `latent = L2Normalize(e_norm @ emap)`.
 */
object ArcFaceRecognizer {

    private const val ARCFACE_SIZE = 112
    private const val EMBEDDING_DIM = 512

    fun extractSourceLatentEmbedding(
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        sourceLandmarks5: List<PointF>,
        arcFaceModelFile: File?,
        emap512x512: FloatArray?,
        allowTwoModelFallbackForTesting: Boolean,
        preloadedArcFaceSession: OrtSession? = null
    ): SourceEmbeddingResult {
        // 1. Align source face to 112x112 canonical ArcFace coordinate frame via 5-point Umeyama transform
        val m112 = FaceAlignment.estimateNorm(sourceLandmarks5, ARCFACE_SIZE)
        val aligned112 = FaceAlignment.warpAffineCrop(sourceBitmap, m112, ARCFACE_SIZE)

        val rawEmbedding: FloatArray
        val usedArcFace: Boolean

        if (ortEnv != null && preloadedArcFaceSession != null) {
            rawEmbedding = runArcFaceSession(ortEnv, preloadedArcFaceSession, aligned112)
            usedArcFace = true
        } else if (ortEnv != null && arcFaceModelFile != null && arcFaceModelFile.exists() && arcFaceModelFile.length() > 1024L) {
            rawEmbedding = runArcFaceOnnx(ortEnv, arcFaceModelFile, aligned112)
            usedArcFace = true
        } else if (allowTwoModelFallbackForTesting) {
            rawEmbedding = computeDeterministicSpatialDescriptor512(aligned112)
            usedArcFace = false
        } else {
            aligned112.recycle()
            throw IllegalStateException(
                "Missing 3rd required model (w600k_r50.onnx). " +
                    "inswapper_128.onnx requires a [1, 512] ArcFace identity embedding tensor for its 'source' input, " +
                    "which det_10g.onnx cannot produce. Please import w600k_r50.onnx in the ONNX Models tab, " +
                    "or enable '2-Model Testing Mode' to test the pipeline with a deterministic 512-D projection."
            )
        }

        var rawNorm = computeL2Norm(rawEmbedding)
        if (usedArcFace && (!rawNorm.isFinite() || rawNorm < 1e-4f)) {
            aligned112.recycle()
            throw IllegalStateException(
                "ArcFace model (${arcFaceModelFile?.name ?: "w600k_r50.onnx"}) returned a degenerate zero/NaN embedding (norm=$rawNorm). " +
                    "Verify the model file is not corrupted."
            )
        }
        val activeEp = when {
            preloadedArcFaceSession != null -> "In-Memory"
            usedArcFace && arcFaceModelFile != null ->
                OnnxProtobufInspector.getCachedExecutionProvider(arcFaceModelFile) ?: "CPU(ALL_OPT)"
            else -> "Fallback"
        }
        val normedEmbedding = l2Normalize(rawEmbedding)

        // 2. Project through 512x512 `emap` matrix extracted from `inswapper_128.onnx`
        val latentVector: FloatArray
        val usedEmap: Boolean
        if (emap512x512 != null && emap512x512.size == EMBEDDING_DIM * EMBEDDING_DIM) {
            val projected = FloatArray(EMBEDDING_DIM)
            for (col in 0 until EMBEDDING_DIM) {
                var sum = 0f
                for (row in 0 until EMBEDDING_DIM) {
                    sum += normedEmbedding[row] * emap512x512[row * EMBEDDING_DIM + col]
                }
                projected[col] = sum
            }
            latentVector = l2Normalize(projected)
            usedEmap = true
        } else {
            latentVector = normedEmbedding.copyOf()
            usedEmap = false
        }

        val summary = buildString {
            append(
                when {
                    preloadedArcFaceSession != null -> "w600k_r50.onnx (In-Memory Session 512-D)"
                    usedArcFace -> "w600k_r50.onnx (ArcFace 512-D, EP=$activeEp, norm=${"%.1f".format(rawNorm)})"
                    else -> "2-Model Spatial Descriptor (512-D)"
                }
            )
            append(if (usedEmap) " + inswapper emap[512x512]" else " (direct L2 norm)")
        }

        return SourceEmbeddingResult(
            rawNormedEmbedding512 = normedEmbedding,
            latentSourceVector512 = latentVector,
            aligned112Crop = aligned112,
            usedArcFaceModel = usedArcFace,
            usedEmbeddedEmap = usedEmap,
            providerSummary = summary,
            rawEmbeddingL2Norm = rawNorm,
            executionProvider = activeEp
        )
    }

    fun computeL2Norm(vec: FloatArray): Float {
        var sumSq = 0.0
        for (v in vec) {
            if (!v.isFinite()) return Float.NaN
            sumSq += (v * v).toDouble()
        }
        return sqrt(sumSq).toFloat()
    }

    fun runArcFaceSession(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        aligned112: Bitmap
    ): FloatArray {
        val hw = ARCFACE_SIZE * ARCFACE_SIZE
        val pixels = IntArray(hw)
        aligned112.getPixels(pixels, 0, ARCFACE_SIZE, 0, 0, ARCFACE_SIZE, ARCFACE_SIZE)

        // NCHW [1, 3, 112, 112], RGB normalized: (pixel - 127.5f) / 127.5f
        val floatBuffer = FloatBuffer.allocate(3 * hw)
        for (i in 0 until hw) {
            val c = pixels[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            floatBuffer.put(i, (r - 127.5f) / 127.5f)
            floatBuffer.put(hw + i, (g - 127.5f) / 127.5f)
            floatBuffer.put(2 * hw + i, (b - 127.5f) / 127.5f)
        }
        floatBuffer.rewind()

        val inputName = session.inputNames.first()
        val shape = longArrayOf(1L, 3L, ARCFACE_SIZE.toLong(), ARCFACE_SIZE.toLong())
        OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { results ->
                val outTensor = results[0] as OnnxTensor
                val fb = outTensor.floatBuffer
                val out = FloatArray(EMBEDDING_DIM)
                val count = minOf(fb.remaining(), EMBEDDING_DIM)
                fb.get(out, 0, count)
                return out
            }
        }
    }

    private fun runArcFaceOnnx(
        ortEnv: OrtEnvironment,
        arcFaceFile: File,
        aligned112: Bitmap
    ): FloatArray {
        val session = requireNotNull(
            OnnxProtobufInspector.getOrCreateCachedSession(
                ortEnv = ortEnv,
                file = arcFaceFile,
                preferHardwareAcceleration = false
            )
        ) { "Failed to load ArcFace session from ${arcFaceFile.absolutePath}" }
        return runArcFaceSession(ortEnv, session, aligned112)
    }

    /**
     * Deterministic 512-D multiscale spatial-chromatic facial descriptor used ONLY when the user
     * explicitly enables 2-Model Testing Mode prior to importing `w600k_r50.onnx`.
     */
    private fun computeDeterministicSpatialDescriptor512(aligned112: Bitmap): FloatArray {
        val hw = ARCFACE_SIZE * ARCFACE_SIZE
        val pixels = IntArray(hw)
        aligned112.getPixels(pixels, 0, ARCFACE_SIZE, 0, 0, ARCFACE_SIZE, ARCFACE_SIZE)
        val desc = FloatArray(EMBEDDING_DIM)

        // Sample 16x16 grid (256 cells) across the 112x112 aligned face crop (7x7 pixels per cell):
        // First 256 dims: normalized luminance & local horizontal gradient
        // Next 256 dims: normalized chrominance (R-G, B-Y) & vertical gradient
        val cellSize = 7
        var cellIdx = 0
        for (gy in 0 until 16) {
            for (gx in 0 until 16) {
                var sumL = 0f
                var sumRg = 0f
                var sumBy = 0f
                var gradX = 0f
                for (py in 0 until cellSize) {
                    val y = gy * cellSize + py
                    for (px in 0 until cellSize) {
                        val x = gx * cellSize + px
                        val c = pixels[y * ARCFACE_SIZE + x]
                        val r = ((c ushr 16) and 0xFF) / 255f - 0.5f
                        val g = ((c ushr 8) and 0xFF) / 255f - 0.5f
                        val b = (c and 0xFF) / 255f - 0.5f
                        val lum = 0.299f * r + 0.587f * g + 0.114f * b
                        sumL += lum
                        sumRg += (r - g)
                        sumBy += (b - 0.5f * (r + g))
                        if (px > 0) {
                            val prevC = pixels[y * ARCFACE_SIZE + (x - 1)]
                            val prevR = ((prevC ushr 16) and 0xFF) / 255f - 0.5f
                            gradX += (r - prevR)
                        }
                    }
                }
                val invArea = 1f / (cellSize * cellSize)
                desc[cellIdx] = (sumL + gradX) * invArea
                desc[256 + cellIdx] = (sumRg + sumBy) * invArea
                cellIdx++
            }
        }
        return l2Normalize(desc)
    }

    fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0.0
        for (v in vec) {
            sumSq += (v * v).toDouble()
        }
        val norm = sqrt(sumSq).toFloat().coerceAtLeast(1e-12f)
        return FloatArray(vec.size) { i -> vec[i] / norm }
    }
}
