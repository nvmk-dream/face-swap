package com.example.onnx

import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-precision Negative Artifact Suppression Engine:
 * 1. Protruding Tongue & Open Oral Cavity Preservation (eliminates split-tongue, forked-tongue,
 *    and mouth-cavity skin distortion when target face has tongue sticking out or open mouth).
 * 2. Eyelid, Squint & Ocular Makeup Protection (eliminates cloudy pupils, deformed double-crease eyelids,
 *    and makeup smudging on winking/squinting eyes).
 * 3. Configurable parameters for UI settings & pipeline automation.
 */
object NegativeArtifactSuppression {

    private const val HD_SIZE = 512

    data class SuppressionSettings(
        val enableTongueAndOralSuppression: Boolean = true,
        val enableOcularMakeupSuppression: Boolean = true,
        val enablePlasticSkinSuppression: Boolean = true,
        val tongueSensitivity: Float = 0.85f,
        val eyeMakeupPreservationStrength: Float = 0.80f,
        val facialAgingFactor: Float = 0.0f // -1.0 (Youthful) to +1.0 (Mature/Aging)
    )

    /**
     * Inspects target and swapped pixels at 512x512 resolution for protruding tongue,
     * parted lips, teeth, and dark inner oral cavity. When detected, restores 100% of the
     * target's authentic tongue texture, moisture, and dental contours, eliminating
     * neural split-tongue artifacts.
     */
    fun suppressOralAndTongueArtifacts512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        geom512: FaceBlender.WarpedFaceGeometry,
        sensitivity: Float = 0.85f
    ): Boolean {
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val mouthWidth = hypot(
            (geom512.rightMouth.x - geom512.leftMouth.x).toDouble(),
            (geom512.rightMouth.y - geom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(65f, 240f)

        // Tongue and open mouth can extend significantly downward toward the chin
        val rx = mouthWidth * 0.44f
        val ryUp = mouthWidth * 0.18f
        val ryDown = mouthWidth * 0.58f // Downward extension for protruding tongue
        val cosA = geom512.cosA
        val sinA = geom512.sinA

        // Scan oral and sub-oral bounding box
        val maxExt = max(rx, ryDown) + 8f
        val minX = (mouthMidX - maxExt).toInt().coerceIn(8, HD_SIZE - 9)
        val maxX = (mouthMidX + maxExt).toInt().coerceIn(8, HD_SIZE - 9)
        val minY = (mouthMidY - ryUp - 8f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxY = (mouthMidY + ryDown + 12f).toInt().coerceIn(8, HD_SIZE - 9)

        var tonguePixelCount = 0
        var teethPixelCount = 0
        var oralCavityCount = 0
        var totalSamples = 0

        for (y in minY..maxY) {
            val row = y * HD_SIZE
            val yf = y.toFloat()
            for (x in minX..maxX) {
                val xf = x.toFloat()
                val dx = xf - mouthMidX
                val dy = yf - mouthMidY
                val u = dx * cosA + dy * sinA
                val v = -dx * sinA + dy * cosA

                val uNorm = u / rx
                val vNorm = if (v >= 0f) v / ryDown else -v / ryUp
                if (uNorm * uNorm + vNorm * vNorm > 1.0f) continue
                totalSamples++

                val tc = tgtPx512[row + x]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB
                val tMax = max(tR, max(tG, tB)).coerceAtLeast(1)
                val tMin = min(tR, min(tG, tB))
                val tSat = (tMax - tMin).toFloat() / tMax.toFloat()

                // Tongue chromatic signature: reddish-pink, prominent R channel, healthy saturation
                val isTongueColor = v > 2f && tR > tG + 16 && tR > tB + 10 && tSat >= 0.16f && tSat <= 0.78f && tLum >= 60f && tLum <= 215f
                // Teeth chromatic signature: high luminance, neutral saturation
                val isTeethColor = tLum > 165f && tSat < 0.22f
                // Dark mouth interior
                val isDarkOralCavity = tLum < 46f && tSat < 0.35f

                if (isTongueColor) tonguePixelCount++
                if (isTeethColor) teethPixelCount++
                if (isDarkOralCavity) oralCavityCount++
            }
        }

        if (totalSamples < 20) return false
        val tongueRatio = tonguePixelCount.toFloat() / totalSamples.toFloat()
        val teethRatio = teethPixelCount.toFloat() / totalSamples.toFloat()
        val oralRatio = oralCavityCount.toFloat() / totalSamples.toFloat()

        val hasTongueOrOpenMouth = (tongueRatio > 0.04f * (1.1f - sensitivity)) ||
            (teethRatio > 0.08f) ||
            (tonguePixelCount >= 24) ||
            (oralRatio > 0.12f && teethRatio > 0.03f)

        if (!hasTongueOrOpenMouth) return false

        // Seamless preservation blend: reconstruct tongue & oral cavity with feathering
        for (y in minY..maxY) {
            val row = y * HD_SIZE
            val yf = y.toFloat()
            for (x in minX..maxX) {
                val xf = x.toFloat()
                val dx = xf - mouthMidX
                val dy = yf - mouthMidY
                val u = dx * cosA + dy * sinA
                val v = -dx * sinA + dy * cosA

                val uNorm = u / rx
                val vNorm = if (v >= 0f) v / ryDown else -v / ryUp
                val radialDistSq = uNorm * uNorm + vNorm * vNorm
                if (radialDistSq >= 1.0f) continue

                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB
                val tMax = max(tR, max(tG, tB)).coerceAtLeast(1)
                val tMin = min(tR, min(tG, tB))
                val tSat = (tMax - tMin).toFloat() / tMax.toFloat()

                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF

                // Determine pixel confidence in tongue / teeth / cavity
                val isTonguePixel = v > 2f && tR > tG + 14 && tR > tB + 8 && tSat >= 0.14f && tSat <= 0.82f && tLum >= 55f && tLum <= 225f
                val isTeethPixel = tLum > 160f && tSat < 0.24f
                val isDarkInterior = tLum < 52f && v < ryDown * 0.6f

                val featureConfidence = when {
                    isTonguePixel -> 1.0f
                    isTeethPixel -> 0.96f
                    isDarkInterior -> 0.92f
                    else -> {
                        val dist = sqrt(radialDistSq)
                        if (dist < 0.70f) 0.45f else 0.0f
                    }
                }
                if (featureConfidence <= 0.01f) continue

                val spatialEnvelope = 0.5f * (1.0f + cos(Math.PI * sqrt(radialDistSq)).toFloat())
                val blendWeight = (featureConfidence * spatialEnvelope * 0.98f).coerceIn(0f, 0.98f)

                val outR = (sR * (1f - blendWeight) + tR * blendWeight).toInt().coerceIn(0, 255)
                val outG = (sG * (1f - blendWeight) + tG * blendWeight).toInt().coerceIn(0, 255)
                val outB = (sB * (1f - blendWeight) + tB * blendWeight).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
        return true
    }

    /**
     * Suppresses eye makeup deformation, pupil cloudiness, and squint/winking anomalies.
     * Preserves target's eyeliner, glitter, and eyelash integrity while maintaining source identity.
     */
    fun suppressOcularAndEyelidArtifacts512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        geom512: FaceBlender.WarpedFaceGeometry,
        makeupPreservationStrength: Float = 0.80f
    ) {
        val eyeCenters: Array<PointF> = arrayOf(geom512.leftEye, geom512.rightEye)
        val rx = geom512.eyeDist * 0.32f
        val ryUpper = geom512.eyeDist * 0.22f
        val ryLower = geom512.eyeDist * 0.16f

        for (eyeCenter in eyeCenters) {
            val maxR = max(rx, ryUpper) + 4f
            val minX = (eyeCenter.x - maxR).toInt().coerceIn(8, HD_SIZE - 9)
            val maxX = (eyeCenter.x + maxR).toInt().coerceIn(8, HD_SIZE - 9)
            val minY = (eyeCenter.y - ryUpper - 4f).toInt().coerceIn(8, HD_SIZE - 9)
            val maxY = (eyeCenter.y + ryLower + 4f).toInt().coerceIn(8, HD_SIZE - 9)

            for (y in minY..maxY) {
                val row = y * HD_SIZE
                val yf = y.toFloat()
                for (x in minX..maxX) {
                    val xf = x.toFloat()
                    val dx = xf - eyeCenter.x
                    val dy = yf - eyeCenter.y
                    val u = dx * geom512.cosA + dy * geom512.sinA
                    val v = -dx * geom512.sinA + dy * geom512.cosA

                    val uNorm = u / rx
                    val vNorm = if (v < 0f) -v / ryUpper else v / ryLower
                    val dSq = uNorm * uNorm + vNorm * vNorm
                    if (dSq >= 1.0f) continue

                    val idx = row + x
                    val tc = tgtPx512[idx]
                    val tR = (tc ushr 16) and 0xFF
                    val tG = (tc ushr 8) and 0xFF
                    val tB = tc and 0xFF
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                    val sc = swapPx512[idx]
                    val sR = (sc ushr 16) and 0xFF
                    val sG = (sc ushr 8) and 0xFF
                    val sB = sc and 0xFF
                    val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB

                    val isEyelinerLoss = tLum < 65f && sLum > tLum + 28f
                    val isCloudyPupil = dSq < 0.40f && tLum < 55f && sLum > tLum + 38f
                    val isSquintFoldDeformation = v < -2f && abs(sLum - tLum) > 42f

                    if (isEyelinerLoss || isCloudyPupil || isSquintFoldDeformation) {
                        val env = (0.5f * (1.0f + cos(Math.PI * sqrt(dSq)))).toFloat()
                        val w = (env * makeupPreservationStrength * 0.82f).coerceIn(0f, 0.88f)
                        val outR = (sR * (1f - w) + tR * w).toInt().coerceIn(0, 255)
                        val outG = (sG * (1f - w) + tG * w).toInt().coerceIn(0, 255)
                        val outB = (sB * (1f - w) + tB * w).toInt().coerceIn(0, 255)
                        swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                    }
                }
            }
        }
    }
}
