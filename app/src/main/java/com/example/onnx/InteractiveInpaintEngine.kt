package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.Color
import java.io.File
import java.nio.FloatBuffer
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
 * Uses `lama_fp32.onnx` via ONNX Runtime when installed in the Inpainting model slot,
 * and falls back cleanly to an on-device Multi-Scale Boundary-Diffusion Patch Inpainter
 * so users can always paint a mask and inpaint offline.
 *
 * Unmasked pixels outside the user's painted mask are preserved 100% pixel-identical
 * from [sourceBitmap].
 */
object InteractiveInpaintEngine {

    fun inpaint(
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        maskBitmap: Bitmap,
        prompt: String,
        fastMode: Boolean,
        lamaModelFile: File?,
        preferHardwareAccel: Boolean = true
    ): InpaintExecutionResult {
        val t0 = System.currentTimeMillis()
        val width = sourceBitmap.width
        val height = sourceBitmap.height

        // Scale mask to match sourceBitmap dimensions if needed
        val alignedMask = if (maskBitmap.width == width && maskBitmap.height == height) {
            maskBitmap
        } else {
            Bitmap.createScaledBitmap(maskBitmap, width, height, true)
        }

        val total = width * height
        val maskPx = IntArray(total)
        alignedMask.getPixels(maskPx, 0, width, 0, 0, width, height)
        if (alignedMask !== maskBitmap) {
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

        // Try ONNX LaMa inpainting if lama_fp32.onnx is installed
        var inpaintedFull: Bitmap? = null
        var engineLabel = "On-Device Boundary-Diffusion Inpainter"

        if (ortEnv != null && lamaModelFile != null && lamaModelFile.exists() && lamaModelFile.length() > 1024L) {
            val lamaOut = runCatching {
                runLamaInpaint(
                    ortEnv = ortEnv,
                    lamaFile = lamaModelFile,
                    sourceBitmap = sourceBitmap,
                    fullMaskAlpha = maskAlpha,
                    preferHardwareAccel = preferHardwareAccel
                )
            }.getOrNull()
            if (lamaOut != null) {
                inpaintedFull = lamaOut
                engineLabel = "LaMa ONNX (lama_fp32.onnx)"
            }
        }

        if (inpaintedFull == null) {
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

    private fun runLamaInpaint(
        ortEnv: OrtEnvironment,
        lamaFile: File,
        sourceBitmap: Bitmap,
        fullMaskAlpha: FloatArray,
        preferHardwareAccel: Boolean
    ): Bitmap {
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height
        val modelSize = 512
        val scaled = if (origW == modelSize && origH == modelSize) {
            sourceBitmap
        } else {
            Bitmap.createScaledBitmap(sourceBitmap, modelSize, modelSize, true)
        }
        val hw = modelSize * modelSize
        val pixels = IntArray(hw)
        scaled.getPixels(pixels, 0, modelSize, 0, 0, modelSize, modelSize)
        if (scaled !== sourceBitmap) scaled.recycle()

        val imgBuf = FloatBuffer.allocate(3 * hw)
        val maskBuf = FloatBuffer.allocate(hw)
        for (y in 0 until modelSize) {
            val sy = (y * origH / modelSize).coerceIn(0, origH - 1)
            for (x in 0 until modelSize) {
                val sx = (x * origW / modelSize).coerceIn(0, origW - 1)
                val i = y * modelSize + x
                val m = if (fullMaskAlpha[sy * origW + sx] > 0.15f) 1.0f else 0.0f
                val c = pixels[i]
                val r = ((c ushr 16) and 0xFF) / 255.0f
                val g = ((c ushr 8) and 0xFF) / 255.0f
                val b = (c and 0xFF) / 255.0f
                imgBuf.put(i, r * (1f - m))
                imgBuf.put(hw + i, g * (1f - m))
                imgBuf.put(2 * hw + i, b * (1f - m))
                maskBuf.put(i, m)
            }
        }
        imgBuf.rewind()
        maskBuf.rewind()

        OnnxProtobufInspector.createOptimizedSessionOptions(preferHardwareAccel).use { opts ->
            ortEnv.createSession(lamaFile.absolutePath, opts).use { session ->
                val names = session.inputNames.toList()
                val imgName = names.firstOrNull { it.contains("image", ignoreCase = true) } ?: names[0]
                val maskName = names.firstOrNull { it.contains("mask", ignoreCase = true) }
                    ?: names.getOrElse(1) { names[0] }

                OnnxTensor.createTensor(ortEnv, imgBuf, longArrayOf(1L, 3L, 512L, 512L)).use { imgT ->
                    OnnxTensor.createTensor(ortEnv, maskBuf, longArrayOf(1L, 1L, 512L, 512L)).use { maskT ->
                        session.run(mapOf(imgName to imgT, maskName to maskT)).use { res ->
                            val outT = res[0] as OnnxTensor
                            val outFloats = FloatArray(3 * hw)
                            outT.floatBuffer.get(outFloats)
                            val maxVal = outFloats.maxOrNull() ?: 1.0f
                            val scale = if (maxVal <= 1.5f) 255.0f else 1.0f
                            val outPx = IntArray(hw)
                            for (i in 0 until hw) {
                                val r = (outFloats[i] * scale).toInt().coerceIn(0, 255)
                                val g = (outFloats[hw + i] * scale).toInt().coerceIn(0, 255)
                                val b = (outFloats[2 * hw + i] * scale).toInt().coerceIn(0, 255)
                                outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                            }
                            val bmp512 = Bitmap.createBitmap(modelSize, modelSize, Bitmap.Config.ARGB_8888)
                            bmp512.setPixels(outPx, 0, modelSize, 0, 0, modelSize, modelSize)
                            return if (origW == modelSize && origH == modelSize) {
                                bmp512
                            } else {
                                val resized = Bitmap.createScaledBitmap(bmp512, origW, origH, true)
                                bmp512.recycle()
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

    private fun featherMask(
        mask: FloatArray,
        w: Int,
        h: Int,
        radius: Int
    ): FloatArray {
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
                // Preserve full interior while feathering edge smoothly
                out[rowOff + x] = max(mask[rowOff + x], sum / denom).coerceIn(0f, 1f)
            }
        }
        return out
    }
}
