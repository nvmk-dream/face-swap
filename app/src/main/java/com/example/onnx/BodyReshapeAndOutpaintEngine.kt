package com.example.onnx

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

enum class OutpaintExtendPreset(
    val shortLabel: String,
    val fraction: Float,
    val suggestedPromptSuffix: String
) {
    NONE("Off", 0f, ""),
    KNEE("+40% Knee", 0.40f, "lower body, thighs and knees, matching outfit continuing down naturally, photorealistic, high detail"),
    FULL_BODY("+70% Full Feet", 0.70f, "full body standing portrait down to feet, natural legs and footwear on floor, matching outfit continuing seamlessly, photorealistic, sharp focus")
}

data class ExtendedOutpaintCanvas(
    val extendedBitmap: Bitmap,
    val autoMaskBitmap: Bitmap,
    val originalWidth: Int,
    val originalHeight: Int,
    val extraHeight: Int
)

object BodyReshapeAndOutpaintEngine {

    /**
     * 100% Offline Curvy Body Contour Reshape using smooth inverse-mapped hourglass curve.
     * - Protects upper head/face/shoulders completely above `waistYRatio - 0.14f`.
     * - Gently cinches waist around `waistYRatio` and curves hip/lower-torso outward around `hipYRatio`.
     * - Keeps left/right outer canvas edges anchored to prevent room/background wall bending.
     */
    fun applyCurvyBodyReshape(
        sourceBitmap: Bitmap,
        curvyStrength: Float,
        waistYRatio: Float = 0.44f,
        hipYRatio: Float = 0.64f,
        centerXRatio: Float = 0.50f
    ): Bitmap {
        val strength = curvyStrength.coerceIn(0f, 1f)
        val w = sourceBitmap.width
        val h = sourceBitmap.height
        if (w < 8 || h < 8 || strength <= 0.01f) {
            return sourceBitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val srcPx = IntArray(w * h)
        sourceBitmap.getPixels(srcPx, 0, w, 0, 0, w, h)
        val outPx = reshapePixels(
            srcPx = srcPx,
            w = w,
            h = h,
            curvyStrength = strength,
            waistYRatio = waistYRatio,
            hipYRatio = hipYRatio,
            centerXRatio = centerXRatio
        )

        val outBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(outPx, 0, w, 0, 0, w, h)
        return outBmp
    }

    internal fun reshapePixels(
        srcPx: IntArray,
        w: Int,
        h: Int,
        curvyStrength: Float,
        waistYRatio: Float = 0.44f,
        hipYRatio: Float = 0.64f,
        centerXRatio: Float = 0.50f
    ): IntArray {
        val strength = curvyStrength.coerceIn(0f, 1f)
        if (w < 8 || h < 8 || strength <= 0.01f) {
            return srcPx.copyOf()
        }
        val outPx = IntArray(w * h)

        val clampedCenterX = (centerXRatio.coerceIn(0.25f, 0.75f)) * w
        val halfW = max(1f, w * 0.5f)
        val safeHipY = hipYRatio.coerceIn(0.45f, 0.88f)
        val safeWaistY = min(waistYRatio.coerceIn(0.28f, 0.68f), safeHipY - 0.12f)
        val headProtectEndY = (safeWaistY - 0.14f).coerceAtLeast(0.18f)
        val maxShiftPx = w * 0.16f * strength

        // Precompute horizontal bell-curve profile per x
        val xFactor = FloatArray(w)
        for (x in 0 until w) {
            val dx = ((x - clampedCenterX) / halfW).coerceIn(-1.2f, 1.2f)
            val normDist = abs(dx).coerceIn(0f, 1f)
            val bell = dx * exp(-(dx / 0.50f).pow(2))
            val borderAnchor = (1f - normDist.pow(2.2f)).coerceIn(0f, 1f)
            xFactor[x] = bell * borderAnchor
        }

        for (y in 0 until h) {
            val yNorm = y.toFloat() / h
            val rowOff = y * w

            if (yNorm <= headProtectEndY) {
                System.arraycopy(srcPx, rowOff, outPx, rowOff, w)
                continue
            }

            // Smooth fade-in below headProtectEndY so neck/chest transitions smoothly
            val topFade = ((yNorm - headProtectEndY) / 0.10f).coerceIn(0f, 1f)
            val smoothTopFade = topFade * topFade * (3f - 2f * topFade)

            // Gentle waist contour + rich lower-hip curve
            val waistGauss = exp(-((yNorm - safeWaistY) / 0.075f).pow(2))
            val hipGauss = exp(-((yNorm - safeHipY) / 0.135f).pow(2))
            val rowDisplacementScale = smoothTopFade * ((hipGauss * 1.15f) - (waistGauss * 0.24f)) * maxShiftPx

            if (abs(rowDisplacementScale) < 0.05f) {
                System.arraycopy(srcPx, rowOff, outPx, rowOff, w)
                continue
            }

            for (x in 0 until w) {
                val shift = rowDisplacementScale * xFactor[x]
                val srcX = (x - shift).coerceIn(0f, (w - 1).toFloat())
                val x0 = srcX.toInt().coerceIn(0, w - 1)
                val x1 = (x0 + 1).coerceIn(0, w - 1)
                val fx = srcX - x0

                if (x0 == x1 || fx <= 0.001f) {
                    outPx[rowOff + x] = srcPx[rowOff + x0]
                } else {
                    val c0 = srcPx[rowOff + x0]
                    val c1 = srcPx[rowOff + x1]
                    val inv = 1f - fx
                    val a = (((c0 ushr 24) and 0xFF) * inv + ((c1 ushr 24) and 0xFF) * fx).roundToInt().coerceIn(0, 255)
                    val r = (((c0 ushr 16) and 0xFF) * inv + ((c1 ushr 16) and 0xFF) * fx).roundToInt().coerceIn(0, 255)
                    val g = (((c0 ushr 8) and 0xFF) * inv + ((c1 ushr 8) and 0xFF) * fx).roundToInt().coerceIn(0, 255)
                    val b = ((c0 and 0xFF) * inv + (c1 and 0xFF) * fx).roundToInt().coerceIn(0, 255)
                    outPx[rowOff + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        return outPx
    }

    /**
     * Expands a Half-Size (hip/waist cropped) source photo downwards to prepare a Full-Body
     * Outpaint canvas (`extendedBitmap`) and a seam-feathered `autoMaskBitmap`.
     *
     * - Top portion preserves original photo 100%.
     * - Overlap feather band (~6% above bottom edge of original photo) blends dress/body continuity.
     * - Newly extended bottom band uses a soft context-blurred downward gradient so VAE avoids hard edge lines.
     */
    fun extendCanvasForFullBodyOutpaint(
        sourceBitmap: Bitmap,
        extendFraction: Float
    ): ExtendedOutpaintCanvas {
        val origW = sourceBitmap.width
        val origH = sourceBitmap.height
        val safeFraction = extendFraction.coerceIn(0.10f, 1.25f)
        val extraH = (origH * safeFraction).roundToInt().coerceAtLeast(24)
        val newH = origH + extraH

        val srcPx = IntArray(origW * origH)
        sourceBitmap.getPixels(srcPx, 0, origW, 0, 0, origW, origH)

        val (extPx, maskPx) = extendPixels(srcPx, origW, origH, extendFraction)

        val extendedBmp = Bitmap.createBitmap(origW, newH, Bitmap.Config.ARGB_8888)
        extendedBmp.setPixels(extPx, 0, origW, 0, 0, origW, newH)

        val maskBmp = Bitmap.createBitmap(origW, newH, Bitmap.Config.ARGB_8888)
        maskBmp.setPixels(maskPx, 0, origW, 0, 0, origW, newH)

        return ExtendedOutpaintCanvas(
            extendedBitmap = extendedBmp,
            autoMaskBitmap = maskBmp,
            originalWidth = origW,
            originalHeight = origH,
            extraHeight = extraH
        )
    }

    internal fun extendPixels(
        srcPx: IntArray,
        origW: Int,
        origH: Int,
        extendFraction: Float
    ): Pair<IntArray, IntArray> {
        val safeFraction = extendFraction.coerceIn(0.10f, 1.25f)
        val extraH = (origH * safeFraction).roundToInt().coerceAtLeast(24)
        val newH = origH + extraH

        val extPx = IntArray(origW * newH)
        System.arraycopy(srcPx, 0, extPx, 0, origW * origH)

        // Average last 10 rows per column for smooth horizontal continuation seed
        val sampleRows = min(10, origH)
        val startRow = origH - sampleRows
        val edgeR = FloatArray(origW)
        val edgeG = FloatArray(origW)
        val edgeB = FloatArray(origW)
        var sumTotalR = 0.0
        var sumTotalG = 0.0
        var sumTotalB = 0.0

        for (x in 0 until origW) {
            var sr = 0f
            var sg = 0f
            var sb = 0f
            for (ry in startRow until origH) {
                val c = srcPx[ry * origW + x]
                sr += (c ushr 16) and 0xFF
                sg += (c ushr 8) and 0xFF
                sb += c and 0xFF
            }
            edgeR[x] = sr / sampleRows
            edgeG[x] = sg / sampleRows
            edgeB[x] = sb / sampleRows
            sumTotalR += edgeR[x]
            sumTotalG += edgeG[x]
            sumTotalB += edgeB[x]
        }

        val ambientR = (sumTotalR / origW).toFloat()
        val ambientG = (sumTotalG / origW).toFloat()
        val ambientB = (sumTotalB / origW).toFloat()

        // Horizontal blur on seed strip to avoid vertical streaking artifacts in latent space
        val blurRadius = max(4, origW / 24)
        val smoothR = FloatArray(origW)
        val smoothG = FloatArray(origW)
        val smoothB = FloatArray(origW)
        for (x in 0 until origW) {
            val left = max(0, x - blurRadius)
            val right = min(origW - 1, x + blurRadius)
            val count = (right - left + 1).toFloat()
            var ar = 0f
            var ag = 0f
            var ab = 0f
            for (kx in left..right) {
                ar += edgeR[kx]
                ag += edgeG[kx]
                ab += edgeB[kx]
            }
            smoothR[x] = ar / count
            smoothG[x] = ag / count
            smoothB[x] = ab / count
        }

        // Fill newly added bottom canvas area with context-aware gradient + subtle micro-grain
        for (dy in 0 until extraH) {
            val t = (dy + 1).toFloat() / extraH.toFloat()
            val ambientBlend = (t * 0.78f).coerceIn(0f, 0.85f)
            val floorShade = 1f - 0.10f * t
            val rowOff = (origH + dy) * origW
            for (x in 0 until origW) {
                val microNoise = (((x * 37 + dy * 17) and 7) - 3.5f) * (0.4f + t * 0.8f)
                val r = ((smoothR[x] * (1f - ambientBlend) + ambientR * ambientBlend) * floorShade + microNoise)
                    .roundToInt().coerceIn(0, 255)
                val g = ((smoothG[x] * (1f - ambientBlend) + ambientG * ambientBlend) * floorShade + microNoise)
                    .roundToInt().coerceIn(0, 255)
                val b = ((smoothB[x] * (1f - ambientBlend) + ambientB * ambientBlend) * floorShade + microNoise)
                    .roundToInt().coerceIn(0, 255)
                extPx[rowOff + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        // Generate auto-mask covering extended area + smooth feathered seam overlap into lower hip edge
        val maskPx = IntArray(origW * newH)
        val overlapPx = (origH * 0.06f).roundToInt().coerceIn(8, 28)
        val seamStartY = (origH - overlapPx).coerceAtLeast(0)

        for (y in seamStartY until origH) {
            val t = if (overlapPx > 0) (y - seamStartY + 1).toFloat() / overlapPx.toFloat() else 1f
            val smoothT = (0.5f - 0.5f * cos(Math.PI * t.coerceIn(0f, 1f))).toFloat()
            val alpha = (smoothT * 255f).roundToInt().coerceIn(0, 255)
            val maskColor = (alpha shl 24) or 0x00FFFFFF
            val rowOff = y * origW
            for (x in 0 until origW) {
                maskPx[rowOff + x] = maskColor
            }
        }
        val fullWhite = -0x1 // 0xFFFFFFFF
        for (y in origH until newH) {
            val rowOff = y * origW
            for (x in 0 until origW) {
                maskPx[rowOff + x] = fullWhite
            }
        }
        return extPx to maskPx
    }

    /**
     * Appends full-body / outpaint context terms safely without duplicating existing tags.
     */
    fun buildOutpaintPrompt(existingPrompt: String, preset: OutpaintExtendPreset): String {
        if (preset == OutpaintExtendPreset.NONE) return existingPrompt
        val clean = existingPrompt.trim().trimEnd(',')
        if (clean.isEmpty()) return preset.suggestedPromptSuffix
        val lower = clean.lowercase()
        return if (lower.contains("full body") || lower.contains("feet") || lower.contains("legs")) {
            clean
        } else {
            "$clean, ${preset.suggestedPromptSuffix}"
        }
    }
}
