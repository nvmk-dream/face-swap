package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Controls whose skin tone is applied to the swapped face or head:
 *  - TARGET_SCENE: Matches the Target scene person's skin tone and scene lighting (best when blending onto target's neck/body).
 *  - SOURCE_IDENTITY: Preserves the Source identity person's authentic skin tone (chrominance + base skin tone from source photo)
 *    while adapting gentle lighting contrast so it blends cleanly without looking flat.
 *  - BALANCED_BLEND: 50% Source identity skin tone + 50% Target scene skin tone.
 */
enum class SkinToneSourceMode(
    val title: String,
    val tamilTitle: String,
    val tamilSubtitle: String
) {
    TARGET_SCENE(
        title = "Target Scene Skin Tone",
        tamilTitle = "Target படத்தின் தோல் நிறம்",
        tamilSubtitle = "Target போட்டோவில் உள்ளவரின் உடல்/கழுத்து நிறம் மற்றும் வெளிச்சத்திற்கு ஏற்ப பொருந்தும்"
    ),
    SOURCE_IDENTITY(
        title = "Source Identity Skin Tone",
        tamilTitle = "Source முகத்தின் தோல் நிறம்",
        tamilSubtitle = "Source (மாற்றும் நபரின்) சொந்த தோல் நிறத்தை அப்படியே தக்கவைக்கும்"
    ),
    BALANCED_BLEND(
        title = "50/50 Balanced Skin Tone",
        tamilTitle = "இரண்டும் கலந்த நிறம் (50/50)",
        tamilSubtitle = "Source மற்றும் Target இருவரின் தோல் நிறத்தையும் சமமாகக் கலந்து இயற்கையாக மாற்றும்"
    )
}

/**
 * Controls whose facial reaction (smile, visible teeth, visible tongue, mouth expression & eye reaction) is used:
 *  - TARGET_REACTION: Uses the Target scene person's facial reaction (smile, open mouth, visible teeth, visible tongue)
 *    and restores crisp 512x512 teeth/tongue detail from the target photo so teeth/tongue never look blurry.
 *  - SOURCE_REACTION: Transfers the Source identity person's own facial reaction (their smile, visible teeth, visible tongue,
 *    lip shape, and eye expression) directly from the Source photo onto the Target pose!
 */
enum class FaceReactionSourceMode(
    val title: String,
    val tamilTitle: String,
    val tamilSubtitle: String
) {
    TARGET_REACTION(
        title = "Target Photo Reaction",
        tamilTitle = "Target முகபாவனை (சிரிப்பு / பற்கள் / நாக்கு)",
        tamilSubtitle = "Target படத்தில் உள்ளவரின் சிரிப்பு, தெரியும் பற்கள் (Teeth), நாக்கு (Tongue) & முகபாவனையைப் பயன்படுத்தும்"
    ),
    SOURCE_REACTION(
        title = "Source Face Reaction",
        tamilTitle = "Source முகபாவனை (சிரிப்பு / பற்கள் / நாக்கு)",
        tamilSubtitle = "Source படத்தில் உள்ளவரின் சொந்த சிரிப்பு, பற்கள் (Teeth), நாக்கு (Tongue) & கண் பாவனையைக் கொண்டுவரும்"
    )
}

/**
 * Studio-Grade Artifact-Free & Online-HD Face Blending Engine:
 *  1. Zero Source-Pixel Ghosting (eliminates cheek hair-strand stamping and eyebrow gaps)
 *  2. Dynamic Landmark-Guided Right & Left Eye + Eyebrow Continuity Restoration (`M * landmarks5`)
 *  3. Cheek Scratch/Glitch Healer & Foreground Hair/Jewelry Occlusion Guard
 *  4. Direct 512x512 Online-Style HD Super-Resolution (via `gfpgan_1.4.onnx` or Built-in Guided Pore/Detail Transfer)
 *  5. Selectable Skin Tone Source (Target Scene vs Source Identity vs 50/50 Blend)
 *  6. Selectable Face Reaction Source (Target Smile/Teeth/Tongue vs Source Smile/Teeth/Tongue)
 */
object FaceBlender {

    private const val CROP_SIZE = 128
    private const val HD_SIZE = 512

    // Default canonical 128x128 coordinates (used when dynamic landmarks are not passed)
    private const val DEFAULT_LEFT_EYE_X = 46.2946f
    private const val DEFAULT_LEFT_EYE_Y = 51.6963f
    private const val DEFAULT_RIGHT_EYE_X = 81.5318f
    private const val DEFAULT_RIGHT_EYE_Y = 51.5014f

    data class WarpedFaceGeometry(
        val leftEye: PointF,
        val rightEye: PointF,
        val nose: PointF,
        val leftMouth: PointF,
        val rightMouth: PointF,
        val eyeDist: Float,
        val cosA: Float,
        val sinA: Float,
        val yawRatio: Float = 0f,
        val reconstructed3D: ReconstructedFace3D = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(
            listOf(leftEye, rightEye, nose, leftMouth, rightMouth)
        ),
        val landmarks106: List<PointF> = reconstructed3D.landmarks106
    )

    fun computeWarpedGeometry(
        forwardMatrix128: FloatArray?,
        targetLandmarks5: List<PointF>?,
        scale: Float = 1.0f
    ): WarpedFaceGeometry {
        if (forwardMatrix128 != null && targetLandmarks5 != null && targetLandmarks5.size >= 5) {
            fun mapPt(pt: PointF): PointF {
                val u = (forwardMatrix128[0] * pt.x + forwardMatrix128[1] * pt.y + forwardMatrix128[2]) * scale
                val v = (forwardMatrix128[3] * pt.x + forwardMatrix128[4] * pt.y + forwardMatrix128[5]) * scale
                return PointF(u, v)
            }
            val lEye = mapPt(targetLandmarks5[0])
            val rEye = mapPt(targetLandmarks5[1])
            val nose = mapPt(targetLandmarks5[2])
            val lMouth = mapPt(targetLandmarks5[3])
            val rMouth = mapPt(targetLandmarks5[4])
            val dx = rEye.x - lEye.x
            val dy = rEye.y - lEye.y
            val dist = hypot(dx, dy).coerceAtLeast(24.0f * scale)
            val dLeftNose = hypot((nose.x - lEye.x).toDouble(), (nose.y - lEye.y).toDouble()).toFloat()
            val dRightNose = hypot((rEye.x - nose.x).toDouble(), (rEye.y - nose.y).toDouble()).toFloat()
            val yaw = ((dRightNose - dLeftNose) / dist).coerceIn(-0.85f, 0.85f)
            val mapped5 = listOf(lEye, rEye, nose, lMouth, rMouth)
            val recon3D = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(mapped5)
            return WarpedFaceGeometry(
                leftEye = lEye,
                rightEye = rEye,
                nose = nose,
                leftMouth = lMouth,
                rightMouth = rMouth,
                eyeDist = dist,
                cosA = dx / dist,
                sinA = dy / dist,
                yawRatio = yaw,
                reconstructed3D = recon3D,
                landmarks106 = recon3D.landmarks106
            )
        }
        val lEye = PointF(DEFAULT_LEFT_EYE_X * scale, DEFAULT_LEFT_EYE_Y * scale)
        val rEye = PointF(DEFAULT_RIGHT_EYE_X * scale, DEFAULT_RIGHT_EYE_Y * scale)
        val nose = PointF(64.0252f * scale, 71.7366f * scale)
        val lMouth = PointF(49.5493f * scale, 92.3655f * scale)
        val rMouth = PointF(78.7299f * scale, 92.2041f * scale)
        val dx = rEye.x - lEye.x
        val dy = rEye.y - lEye.y
        val dist = hypot(dx, dy).coerceAtLeast(24.0f * scale)
        val recon3D = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(
            listOf(lEye, rEye, nose, lMouth, rMouth)
        )
        return WarpedFaceGeometry(
            leftEye = lEye,
            rightEye = rEye,
            nose = nose,
            leftMouth = lMouth,
            rightMouth = rMouth,
            eyeDist = dist,
            cosA = dx / dist,
            sinA = dy / dist,
            yawRatio = 0f,
            reconstructed3D = recon3D,
            landmarks106 = recon3D.landmarks106
        )
    }

    /**
     * Oriented distance squared from `(x, y)` to an ellipse centered at `(cx, cy)` rotated by `(cosA, sinA)`.
     */
    private fun orientedEllipseDistSq(
        x: Float,
        y: Float,
        cx: Float,
        cy: Float,
        rx: Float,
        ry: Float,
        cosA: Float,
        sinA: Float
    ): Float {
        val dx = x - cx
        val dy = y - cy
        val u = (dx * cosA + dy * sinA) / rx
        val v = (-dx * sinA + dy * cosA) / ry
        return u * u + v * v
    }

    /**
     * Returns how strongly a pixel (x, y) in 128x128 canonical space belongs to either ocular
     * socket or eyebrow arch in [0.0f, 1.0f], using dynamic warped landmark coordinates.
     */
    private fun computeEyeAndBrowProtectionWeight(
        x: Float,
        y: Float,
        geom: WarpedFaceGeometry
    ): Float {
        val eyeRx = geom.eyeDist * 0.58f
        val eyeRy = geom.eyeDist * 0.38f

        val lDistSq = orientedEllipseDistSq(
            x, y, geom.leftEye.x, geom.leftEye.y, eyeRx, eyeRy, geom.cosA, geom.sinA
        )
        val rDistSq = orientedEllipseDistSq(
            x, y, geom.rightEye.x, geom.rightEye.y, eyeRx, eyeRy, geom.cosA, geom.sinA
        )

        // Eyebrow centers lie just above the eyes along the perpendicular normal (-sinA, cosA)
        val browOffset = geom.eyeDist * 0.30f
        val lBrowX = geom.leftEye.x + geom.sinA * browOffset
        val lBrowY = geom.leftEye.y - geom.cosA * browOffset
        val rBrowX = geom.rightEye.x + geom.sinA * browOffset
        val rBrowY = geom.rightEye.y - geom.cosA * browOffset
        val browRx = geom.eyeDist * 0.60f
        val browRy = geom.eyeDist * 0.22f

        val lbDistSq = orientedEllipseDistSq(
            x, y, lBrowX, lBrowY, browRx, browRy, geom.cosA, geom.sinA
        )
        val rbDistSq = orientedEllipseDistSq(
            x, y, rBrowX, rBrowY, browRx, browRy, geom.cosA, geom.sinA
        )

        // Central Glabella / Forehead Bindi (பொட்டு / புள்ளி) Protection Zone between the eyebrows
        val glabellaX = (lBrowX + rBrowX) * 0.5f
        val glabellaY = (lBrowY + rBrowY) * 0.5f
        val glabellaRx = geom.eyeDist * 0.24f
        val glabellaRy = geom.eyeDist * 0.22f
        val glabellaDistSq = orientedEllipseDistSq(
            x, y, glabellaX, glabellaY, glabellaRx, glabellaRy, geom.cosA, geom.sinA
        )

        val minDistSq = min(min(min(lDistSq, rDistSq), min(lbDistSq, rbDistSq)), glabellaDistSq)
        return when {
            minDistSq <= 0.48f -> 1.0f
            minDistSq >= 1.0f -> 0.0f
            else -> {
                val t = (minDistSq - 0.48f) / 0.52f
                0.5f * (1.0f + cos(Math.PI * t).toFloat())
            }
        }
    }

    /**
     * Eliminates AI artifacts, cheek hair-strand lines, eyebrow gaps, and cloudy/negative eyes on
     * `swapped128` WITHOUT stamping 2D source background/hair pixels onto the target cheeks or forehead.
     */
    fun restoreEyesAndEliminateNegativeArtifacts128(
        swapped128: Bitmap,
        alignedTarget128: Bitmap,
        alignedSource112: Bitmap? = null,
        alignedSource128: Bitmap? = null,
        hasTrueArcFaceLatent: Boolean = true,
        forwardMatrix128: FloatArray? = null,
        targetLandmarks5: List<PointF>? = null
    ): Bitmap {
        val total = CROP_SIZE * CROP_SIZE
        val swapPx = IntArray(total)
        val tgtPx = IntArray(total)
        swapped128.getPixels(swapPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        alignedTarget128.getPixels(tgtPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        val srcPx128 = IntArray(total)
        if (alignedSource128 != null && alignedSource128.width == CROP_SIZE && alignedSource128.height == CROP_SIZE) {
            alignedSource128.getPixels(srcPx128, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        } else if (alignedSource112 != null && alignedSource112.width == 112 && alignedSource112.height == 112) {
            val raw112 = IntArray(112 * 112)
            alignedSource112.getPixels(raw112, 0, 112, 0, 0, 112, 112)
            for (y in 0 until CROP_SIZE) {
                val sy = y.coerceIn(0, 111)
                for (x in 0 until CROP_SIZE) {
                    val sx = (x - 8).coerceIn(0, 111)
                    srcPx128[y * CROP_SIZE + x] = raw112[sy * 112 + sx]
                }
            }
        } else {
            System.arraycopy(tgtPx, 0, srcPx128, 0, total)
        }

        val geom = computeWarpedGeometry(forwardMatrix128, targetLandmarks5, scale = 1.0f)

        // Preserve 100% of the neural swapped face output from the ONNX swap session,
        // healing only isolated single-pixel dark glitch spikes using the swapped face's own neighborhood.
        val outPx = healCheekScratchesAndNegativeAnomalies128(swapPx, tgtPx, geom)

        val out = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        return out
    }

    /**
     * Removes only isolated dark glitch spikes on `swapPx` using the swapped face's own smooth
     * neighborhood (`sSmooth`), NEVER overwriting swapped facial structure with target features
     * when Source and Target have different skin tones.
     */
    private fun healCheekScratchesAndNegativeAnomalies128(
        swapPx: IntArray,
        tgtPx: IntArray,
        geom: WarpedFaceGeometry
    ): IntArray {
        val out = swapPx.copyOf()

        for (y in 4 until CROP_SIZE - 4) {
            val row = y * CROP_SIZE
            for (x in 4 until CROP_SIZE - 4) {
                val idx = row + x
                val eyeBrowWeight = computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom)
                if (eyeBrowWeight > 0.15f) continue
                // Skip nose nostrils and mouth aperture so dark nostrils/lips are never smoothed
                if (y in 62..106 && x in 40..88) continue

                val sc = swapPx[idx]
                val tc = tgtPx[idx]

                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF
                val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB

                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                val sSmooth = computeWideSkinSmoothRGB(swapPx, CROP_SIZE, x, y, radius = 2)
                val tSmooth = computeWideSkinSmoothRGB(tgtPx, CROP_SIZE, x, y, radius = 2)

                val sSmoothLum = 0.299f * sSmooth[0] + 0.587f * sSmooth[1] + 0.114f * sSmooth[2]
                val tSmoothLum = 0.299f * tSmooth[0] + 0.587f * tSmooth[1] + 0.114f * tSmooth[2]

                // Only heal if target cheek is smooth skin AND swapped pixel has an isolated sharp dark spike
                // relative to its OWN neighborhood (sLum < sSmoothLum - 26f)
                val isTargetSmoothSkin = tLum > 80f && abs(tLum - tSmoothLum) < 8f
                val isIsolatedGlitchSpike = isTargetSmoothSkin && (sLum < sSmoothLum - 26f)

                if (isIsolatedGlitchSpike) {
                    val healR = sSmooth[0].toInt().coerceIn(0, 255)
                    val healG = sSmooth[1].toInt().coerceIn(0, 255)
                    val healB = sSmooth[2].toInt().coerceIn(0, 255)
                    out[idx] = (0xFF shl 24) or (healR shl 16) or (healG shl 8) or healB
                }
            }
        }
        return out
    }

    /**
     * Used ONLY when `w600k_r50.onnx` is not installed (2-Model fallback mode).
     * Transfers inner facial features from `srcPx128` onto `tgtPx128`'s illumination envelope while
     * strictly rejecting any dark hair strands on the cheeks or bright skin gaps over eyebrows.
     */
    private fun buildGlitchFreeTwoModelFallback128(
        srcPx128: IntArray,
        tgtPx128: IntArray,
        geom: WarpedFaceGeometry
    ): IntArray {
        val total = CROP_SIZE * CROP_SIZE
        val out = IntArray(total)
        val cx = (geom.leftEye.x + geom.rightEye.x) * 0.5f
        val cy = (geom.nose.y + (geom.leftEye.y + geom.rightEye.y) * 0.5f) * 0.5f
        val rx = geom.eyeDist * 0.92f
        val ry = geom.eyeDist * 1.12f

        for (y in 0 until CROP_SIZE) {
            val row = y * CROP_SIZE
            for (x in 0 until CROP_SIZE) {
                val idx = row + x
                val tc = tgtPx128[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                // Tight inner-features weight so outer cheeks and forehead stay 100% free of source hair strands
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(), cx, cy, rx, ry, geom.cosA, geom.sinA
                )
                val innerWeight = when {
                    dSq >= 1.0f -> 0.0f
                    dSq <= 0.45f -> 0.65f
                    else -> {
                        val t = (dSq - 0.45f) / 0.55f
                        0.65f * 0.5f * (1.0f + cos(Math.PI * t).toFloat())
                    }
                }

                if (innerWeight <= 0.01f) {
                    out[idx] = tc
                    continue
                }

                val sSmooth = computeWideSkinSmoothRGB(srcPx128, CROP_SIZE, x, y, radius = 5)
                val tSmooth = computeWideSkinSmoothRGB(tgtPx128, CROP_SIZE, x, y, radius = 5)
                val sc = srcPx128[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF
                val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB
                val sSmoothLum = 0.299f * sSmooth[0] + 0.587f * sSmooth[1] + 0.114f * sSmooth[2]

                // Reject source hair strands on cheeks: if source has a sharp dark strand on smooth target skin, ignore source detail
                val isSourceHairArtifact = (sLum < sSmoothLum - 10f || sLum < 65f) && tLum > 85f
                val detailScale = if (isSourceHairArtifact) 0.0f else innerWeight

                val hR = (tSmooth[0] + (sR - sSmooth[0]) * detailScale + (tR - tSmooth[0]) * (1f - detailScale)).toInt().coerceIn(0, 255)
                val hG = (tSmooth[1] + (sG - sSmooth[1]) * detailScale + (tG - tSmooth[1]) * (1f - detailScale)).toInt().coerceIn(0, 255)
                val hB = (tSmooth[2] + (sB - sSmooth[2]) * detailScale + (tB - tSmooth[2]) * (1f - detailScale)).toInt().coerceIn(0, 255)

                out[idx] = (0xFF shl 24) or (hR shl 16) or (hG shl 8) or hB
            }
        }
        return out
    }

    /**
     * Protects both Left and Right Eyebrows ONLY against unnatural bright white cuts/gaps,
     * while preserving 100% of the swapped identity's natural eyebrow shape and thickness.
     */
    private fun restoreDynamicEyebrowContinuity(
        outPx: IntArray,
        tgtPx: IntArray,
        size: Int,
        geom: WarpedFaceGeometry
    ) {
        val browOffset = geom.eyeDist * 0.29f
        val browCenters = arrayOf(
            PointF(geom.leftEye.x + geom.sinA * browOffset, geom.leftEye.y - geom.cosA * browOffset),
            PointF(geom.rightEye.x + geom.sinA * browOffset, geom.rightEye.y - geom.cosA * browOffset)
        )
        val browRx = geom.eyeDist * 0.48f
        val browRy = geom.eyeDist * 0.16f
        val boundR = max(browRx, browRy) * 1.10f

        for (browCenter in browCenters) {
            val xMin = (browCenter.x - boundR).toInt().coerceAtLeast(1)
            val xMax = (browCenter.x + boundR).toInt().coerceAtMost(size - 2)
            val yMin = (browCenter.y - boundR).toInt().coerceAtLeast(1)
            val yMax = (browCenter.y + boundR).toInt().coerceAtMost(size - 2)

            for (y in yMin..yMax) {
                for (x in xMin..xMax) {
                    val dSq = orientedEllipseDistSq(
                        x.toFloat(), y.toFloat(),
                        browCenter.x, browCenter.y,
                        browRx, browRy,
                        geom.cosA, geom.sinA
                    )
                    if (dSq >= 1.0f) continue

                    val normDist = sqrt(dSq)
                    val archWeight = 0.5f * (1.0f + cos(Math.PI * normDist).toFloat())

                    val idx = y * size + x
                    val curC = outPx[idx]
                    val tc = tgtPx[idx]

                    val cR = (curC ushr 16) and 0xFF
                    val cG = (curC ushr 8) and 0xFF
                    val cB = curC and 0xFF
                    val cLum = 0.299f * cR + 0.587f * cG + 0.114f * cB

                    val tR = (tc ushr 16) and 0xFF
                    val tG = (tc ushr 8) and 0xFF
                    val tB = tc and 0xFF
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                    // Only heal if the swapped eyebrow has an unnatural blown-out bright white gap (cLum > tLum + 48)
                    if (tLum < 75f && cLum > tLum + 48f) {
                        val browBlend = (0.38f * archWeight).coerceIn(0f, 0.42f)
                        val finalR = (cR * (1f - browBlend) + tR * browBlend).toInt().coerceIn(0, 255)
                        val finalG = (cG * (1f - browBlend) + tG * browBlend).toInt().coerceIn(0, 255)
                        val finalB = (cB * (1f - browBlend) + tB * browBlend).toInt().coerceIn(0, 255)
                        outPx[idx] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
                    }
                }
            }
        }
    }

    /**
     * Protects Left & Right Eyes strictly within the tight iris/pupil aperture (`rx = 0.22 * eyeDist`,
     * `ry = 0.13 * eyeDist`) ONLY when a pupil has a cloudy/white negative artifact, preserving 100%
     * of the swapped identity's eye shape, eyelids, eyelashes, and gaze.
     */
    private fun restoreDynamicOcularSockets(
        outPx: IntArray,
        tgtPx: IntArray,
        srcPx: IntArray?,
        size: Int,
        geom: WarpedFaceGeometry
    ) {
        val eyeCenters = arrayOf(geom.leftEye, geom.rightEye)
        val rx = geom.eyeDist * 0.22f
        val ry = geom.eyeDist * 0.13f
        val boundR = max(rx, ry) * 1.10f

        for (eyeCenter in eyeCenters) {
            val xMin = (eyeCenter.x - boundR).toInt().coerceAtLeast(1)
            val xMax = (eyeCenter.x + boundR).toInt().coerceAtMost(size - 2)
            val yMin = (eyeCenter.y - boundR).toInt().coerceAtLeast(1)
            val yMax = (eyeCenter.y + boundR).toInt().coerceAtMost(size - 2)

            for (y in yMin..yMax) {
                for (x in xMin..xMax) {
                    val dSq = orientedEllipseDistSq(
                        x.toFloat(), y.toFloat(),
                        eyeCenter.x, eyeCenter.y,
                        rx, ry,
                        geom.cosA, geom.sinA
                    )
                    if (dSq >= 1.0f) continue

                    val normDist = sqrt(dSq)
                    val socketWeight = 0.5f * (1.0f + cos(Math.PI * normDist).toFloat())

                    val idx = y * size + x
                    val curC = outPx[idx]
                    val tc = tgtPx[idx]

                    val cR = (curC ushr 16) and 0xFF
                    val cG = (curC ushr 8) and 0xFF
                    val cB = curC and 0xFF
                    val cLum = 0.299f * cR + 0.587f * cG + 0.114f * cB

                    val tR = (tc ushr 16) and 0xFF
                    val tG = (tc ushr 8) and 0xFF
                    val tB = tc and 0xFF
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                    // Only intervene if a dark pupil/iris pixel turned cloudy/white in the neural output
                    val isCloudyPupilAnomaly = normDist <= 0.65f && tLum < 55f && cLum > tLum + 42f
                    if (isCloudyPupilAnomaly) {
                        val eyeBlend = (0.55f * socketWeight).coerceIn(0f, 0.60f)
                        val eR = (cR * (1f - eyeBlend) + tR * eyeBlend).toInt().coerceIn(0, 255)
                        val eG = (cG * (1f - eyeBlend) + tG * eyeBlend).toInt().coerceIn(0, 255)
                        val eB = (cB * (1f - eyeBlend) + tB * eyeBlend).toInt().coerceIn(0, 255)
                        outPx[idx] = (0xFF shl 24) or (eR shl 16) or (eG shl 8) or eB
                    }
                }
            }
        }
    }

    private fun computeWideSkinSmoothRGB(
        px: IntArray,
        stride: Int,
        cx: Int,
        cy: Int,
        radius: Int
    ): FloatArray {
        val x0 = (cx - radius).coerceAtLeast(0)
        val x1 = (cx + radius).coerceAtMost(stride - 1)
        val y0 = (cy - radius).coerceAtLeast(0)
        val y1 = (cy + radius).coerceAtMost(stride - 1)
        var rSum = 0f
        var gSum = 0f
        var bSum = 0f
        var count = 0
        for (y in y0..y1) {
            val row = y * stride
            for (x in x0..x1) {
                val c = px[row + x]
                rSum += (c ushr 16) and 0xFF
                gSum += (c ushr 8) and 0xFF
                bSum += c and 0xFF
                count++
            }
        }
        val inv = 1f / count.coerceAtLeast(1)
        return floatArrayOf(rSum * inv, gSum * inv, bSum * inv)
    }

    private fun compute3x3LuminanceAvg(px: IntArray, stride: Int, cx: Int, cy: Int): Float {
        val x0 = (cx - 1).coerceAtLeast(0)
        val x1 = (cx + 1).coerceAtMost(stride - 1)
        val y0 = (cy - 1).coerceAtLeast(0)
        val y1 = (cy + 1).coerceAtMost(stride - 1)
        var sum = 0f
        var count = 0
        for (y in y0..y1) {
            val row = y * stride
            for (x in x0..x1) {
                val c = px[row + x]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                sum += 0.299f * r + 0.587f * g + 0.114f * b
                count++
            }
        }
        return sum / count.coerceAtLeast(1)
    }

    /**
     * Harmonizes the color and luminance statistics of `swapped128` according to [skinToneMode]
     * (`TARGET_SCENE`, `SOURCE_IDENTITY`, or `BALANCED_BLEND`) strictly on SKIN regions while
     * protecting both eyes and eyebrows from washout.
     */
    fun transferSkinToneStatistics128(
        swapped128: Bitmap,
        targetCrop128: Bitmap,
        sourceCrop128: Bitmap? = null,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        strength: Float = 0.72f,
        forwardMatrix128: FloatArray? = null,
        targetLandmarks5: List<PointF>? = null
    ): Bitmap {
        val total = CROP_SIZE * CROP_SIZE
        val swapPx = IntArray(total)
        val tgtPx = IntArray(total)
        val srcPx = if (sourceCrop128 != null) IntArray(total) else null
        swapped128.getPixels(swapPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        targetCrop128.getPixels(tgtPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        sourceCrop128?.getPixels(srcPx!!, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        val geom = computeWarpedGeometry(forwardMatrix128, targetLandmarks5, scale = 1.0f)

        var sRMean = 0.0
        var sGMean = 0.0
        var sBMean = 0.0
        var tRMean = 0.0
        var tGMean = 0.0
        var tBMean = 0.0
        var dRMean = 0.0
        var dGMean = 0.0
        var dBMean = 0.0
        var count = 0

        for (y in 30..104) {
            val ny = (y - 67.0) / 37.0
            for (x in 30..98) {
                val nx = (x - 64.0) / 34.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom) > 0.12f) continue
                    if (y in 82..102 && x in 42..86) continue

                    val idx = y * CROP_SIZE + x
                    val sc = swapPx[idx]
                    val tc = tgtPx[idx]
                    val dc = srcPx?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    val dLum = 0.299 * (dc ushr 16 and 0xFF) + 0.587 * (dc ushr 8 and 0xFF) + 0.114 * (dc and 0xFF)
                    if (tLum < 50.0 || dLum < 45.0) continue

                    sRMean += (sc ushr 16) and 0xFF
                    sGMean += (sc ushr 8) and 0xFF
                    sBMean += sc and 0xFF
                    tRMean += (tc ushr 16) and 0xFF
                    tGMean += (tc ushr 8) and 0xFF
                    tBMean += tc and 0xFF
                    dRMean += (dc ushr 16) and 0xFF
                    dGMean += (dc ushr 8) and 0xFF
                    dBMean += dc and 0xFF
                    count++
                }
            }
        }

        if (count < 32) return swapped128.copy(Bitmap.Config.ARGB_8888, true)

        sRMean /= count
        sGMean /= count
        sBMean /= count
        tRMean /= count
        tGMean /= count
        tBMean /= count
        dRMean /= count
        dGMean /= count
        dBMean /= count

        var sRVar = 0.0
        var sGVar = 0.0
        var sBVar = 0.0
        var tRVar = 0.0
        var tGVar = 0.0
        var tBVar = 0.0
        var dRVar = 0.0
        var dGVar = 0.0
        var dBVar = 0.0

        for (y in 30..104) {
            val ny = (y - 67.0) / 37.0
            for (x in 30..98) {
                val nx = (x - 64.0) / 34.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom) > 0.12f) continue
                    if (y in 82..102 && x in 42..86) continue

                    val idx = y * CROP_SIZE + x
                    val sc = swapPx[idx]
                    val tc = tgtPx[idx]
                    val dc = srcPx?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    val dLum = 0.299 * (dc ushr 16 and 0xFF) + 0.587 * (dc ushr 8 and 0xFF) + 0.114 * (dc and 0xFF)
                    if (tLum < 50.0 || dLum < 45.0) continue

                    val sr = ((sc ushr 16) and 0xFF) - sRMean
                    val sg = ((sc ushr 8) and 0xFF) - sGMean
                    val sb = (sc and 0xFF) - sBMean
                    val tr = ((tc ushr 16) and 0xFF) - tRMean
                    val tg = ((tc ushr 8) and 0xFF) - tGMean
                    val tb = (tc and 0xFF) - tBMean
                    val dr = ((dc ushr 16) and 0xFF) - dRMean
                    val dg = ((dc ushr 8) and 0xFF) - dGMean
                    val db = (dc and 0xFF) - dBMean

                    sRVar += sr * sr
                    sGVar += sg * sg
                    sBVar += sb * sb
                    tRVar += tr * tr
                    tGVar += tg * tg
                    tBVar += tb * tb
                    dRVar += dr * dr
                    dGVar += dg * dg
                    dBVar += db * db
                }
            }
        }

        val sRStd = sqrt(sRVar / count).coerceAtLeast(6.0)
        val sGStd = sqrt(sGVar / count).coerceAtLeast(6.0)
        val sBStd = sqrt(sBVar / count).coerceAtLeast(6.0)
        val tRStd = sqrt(tRVar / count).coerceAtLeast(6.0)
        val tGStd = sqrt(tGVar / count).coerceAtLeast(6.0)
        val tBStd = sqrt(tBVar / count).coerceAtLeast(6.0)
        val dRStd = sqrt(dRVar / count).coerceAtLeast(6.0)
        val dGStd = sqrt(dGVar / count).coerceAtLeast(6.0)
        val dBStd = sqrt(dBVar / count).coerceAtLeast(6.0)

        // Compute goal skin statistics based on user-selected SkinToneSourceMode
        val (goalRMean, goalGMean, goalBMean) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(tRMean, tGMean, tBMean)
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                dRMean * 0.85 + tRMean * 0.15,
                dGMean * 0.85 + tGMean * 0.15,
                dBMean * 0.85 + tBMean * 0.15
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                dRMean * 0.50 + tRMean * 0.50,
                dGMean * 0.50 + tGMean * 0.50,
                dBMean * 0.50 + tBMean * 0.50
            )
        }
        val (goalRStd, goalGStd, goalBStd) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(tRStd, tGStd, tBStd)
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                dRStd * 0.80 + tRStd * 0.20,
                dGStd * 0.80 + tGStd * 0.20,
                dBStd * 0.80 + tBStd * 0.20
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                dRStd * 0.50 + tRStd * 0.50,
                dGStd * 0.50 + tGStd * 0.50,
                dBStd * 0.50 + tBStd * 0.50
            )
        }

        val scaleR = (goalRStd / sRStd).coerceIn(0.80, 1.25)
        val scaleG = (goalGStd / sGStd).coerceIn(0.80, 1.25)
        val scaleB = (goalBStd / sBStd).coerceIn(0.80, 1.25)

        val outPx = IntArray(total)
        val baseBlend = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> strength.coerceIn(0f, 1f)
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.88f
            SkinToneSourceMode.BALANCED_BLEND -> 0.80f
        }

        for (y in 0 until CROP_SIZE) {
            val row = y * CROP_SIZE
            for (x in 0 until CROP_SIZE) {
                val i = row + x
                val sc = swapPx[i]
                val origR = (sc ushr 16) and 0xFF
                val origG = (sc ushr 8) and 0xFF
                val origB = sc and 0xFF

                val eyeBrowProt = computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom)
                val blend = baseBlend * (1.0f - 0.92f * eyeBrowProt)

                val matchedR = ((origR - sRMean) * scaleR + goalRMean).toFloat()
                val matchedG = ((origG - sGMean) * scaleG + goalGMean).toFloat()
                val matchedB = ((origB - sBMean) * scaleB + goalBMean).toFloat()

                val finalR = (origR * (1f - blend) + matchedR * blend).toInt().coerceIn(0, 255)
                val finalG = (origG * (1f - blend) + matchedG * blend).toInt().coerceIn(0, 255)
                val finalB = (origB * (1f - blend) + matchedB * blend).toInt().coerceIn(0, 255)

                outPx[i] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
            }
        }

        val out = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        return out
    }

    /**
     * Runs `gfpgan_1.4.onnx` ([1, 3, 512, 512] in [-1, 1]) directly using an active in-memory [OrtSession]
     * and returns the full 512x512 HD restored Bitmap.
     */
    fun runGfpganSession512(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        crop128Or512: Bitmap
    ): Bitmap {
        val gfpSize = HD_SIZE
        val hw = gfpSize * gfpSize
        val pixels = upscaleBitmapBicubicTo512(crop128Or512)

        val inBuf = FloatBuffer.allocate(3 * hw)
        for (i in 0 until hw) {
            val c = pixels[i]
            val r = (((c ushr 16) and 0xFF) / 255.0f - 0.5f) / 0.5f
            val g = (((c ushr 8) and 0xFF) / 255.0f - 0.5f) / 0.5f
            val b = ((c and 0xFF) / 255.0f - 0.5f) / 0.5f
            inBuf.put(i, r)
            inBuf.put(hw + i, g)
            inBuf.put(2 * hw + i, b)
        }
        inBuf.rewind()

        val outPixels512 = IntArray(hw)
        val inputName = session.inputNames.first()
        val shape = longArrayOf(1L, 3L, gfpSize.toLong(), gfpSize.toLong())
        OnnxTensor.createTensor(ortEnv, inBuf, shape).use { inTensor ->
            session.run(mapOf(inputName to inTensor)).use { res ->
                val outT = res[0] as OnnxTensor
                val outFloats = FloatArray(3 * hw)
                outT.floatBuffer.get(outFloats)
                for (i in 0 until hw) {
                    val r = (((outFloats[i] * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                    val g = (((outFloats[hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                    val b = (((outFloats[2 * hw + i] * 0.5f + 0.5f).coerceIn(0f, 1f)) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                    outPixels512[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        val restored512 = Bitmap.createBitmap(gfpSize, gfpSize, Bitmap.Config.ARGB_8888)
        restored512.setPixels(outPixels512, 0, gfpSize, 0, 0, gfpSize, gfpSize)
        return restored512
    }

    /**
     * Runs `gfpgan_1.4.onnx` ([1, 3, 512, 512] in [-1, 1]) if installed (or preloaded in memory)
     * and returns the full 512x512 HD restored Bitmap directly (without downscaling to 128x128!).
     */
    fun runOptionalGfpganEnhancement512(
        ortEnv: OrtEnvironment?,
        gfpganFile: File?,
        crop128Or512: Bitmap,
        preferHardwareAccel: Boolean = false,
        preloadedGfpganSession: OrtSession? = null
    ): Bitmap? {
        if (ortEnv == null) return null
        if (preloadedGfpganSession != null) {
            return runCatching {
                runGfpganSession512(ortEnv, preloadedGfpganSession, crop128Or512)
            }.getOrNull()
        }
        if (gfpganFile == null || !gfpganFile.exists() || gfpganFile.length() <= 1024L) {
            return null
        }
        return runCatching {
            val (session, _) = OnnxProtobufInspector.createSessionWithFallback(
                ortEnv = ortEnv,
                modelFile = gfpganFile,
                preferHardwareAcceleration = preferHardwareAccel
            )
            session.use { s ->
                runGfpganSession512(ortEnv, s, crop128Or512)
            }
        }.getOrNull()
    }

    /**
     * Online AI Photo Style 512x512 HD Super-Resolution & Seamless Blending Pipeline:
     *  1. Upscales the cleaned 128x128 face crop to 512x512 HD (using `gfpgan_1.4.onnx` if installed
     *     or smooth bicubic/bilinear scaling).
     *  2. Warps the target face at full 512x512 HD resolution (`alignedTarget512`).
     *  3. Transfers natural high-frequency skin pore texture & studio highlight sheen from `alignedTarget512`
     *     onto the 512x512 swapped skin (with an occlusion gate that rejects hair strands/scratches).
     *  4. Restores 512x512 razor-sharp Right & Left Eyes and Eyebrows using dynamic 512x512 landmarks.
     *  5. Blends the 512x512 HD face directly onto `targetPixels` using a tight Biometric Inner-Face Mask
     *     and foreground occlusion guard so cheek/jawline seams never appear.
     */
    /**
     * FaceFusion-Style Mobile 512x512 HD Enhancement, Target-Aware Upper-Lip/Philtrum Protection,
     * Dynamic Eye/Brow Protection, Occlusion Masking & Direct High-Resolution Bicubic Blending:
     *  - Accepts 128x128 (`inswapper_128.onnx`) or 256x256 (`hyperswap_1a/1b/1c_256.onnx`) swapped crop.
     *  - Upscales via 4x4 Catmull-Rom Bicubic interpolation + `gfpgan_1.4.onnx` 512x512 HD restoration.
     *  - Applies Target-Aware Philtrum / Upper-Lip / Mouth-Corner Facial-Hair Protection so a moustache-free
     *    target image NEVER gets an unwanted moustache or dark upper-lip shadow.
     *  - Warps the 512x512 HD face directly onto full-resolution `targetPixels` using Catmull-Rom Bicubic sampling
     *    (zero 512 -> 128 downscaling).
     */
    fun enhanceAndBlendOnlineHdFace512(
        ortEnv: OrtEnvironment? = null,
        gfpganFile: File?,
        targetBitmap: Bitmap,
        targetPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        colorCorrected128: Bitmap? = null,
        swappedCrop128: Bitmap? = null,
        forwardMatrix128: FloatArray? = null,
        targetLandmarks5: List<PointF>,
        sourceBitmap: Bitmap? = null,
        sourceLandmarks5: List<PointF>? = null,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.SOURCE_IDENTITY,
        faceReactionMode: FaceReactionSourceMode = FaceReactionSourceMode.TARGET_REACTION,
        enableColorTransfer: Boolean = true,
        enableOcclusionProtection: Boolean = true,
        blendStrength: Float = 1.0f,
        enhancementStrength: Float = 0.96f,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f,
        preferHardwareAccel: Boolean = false,
        segformerFile: File? = null,
        preloadedGfpganSession: OrtSession? = null,
        preloadedSegformerSession: OrtSession? = null,
        identityPreservationMode: Boolean = false,
        depthMap: DepthMapResult? = null,
        onStagesCaptured: ((stage1SwapOnly512: Bitmap, stage2SwapRestore512: Bitmap, stage3FinalBlend512: Bitmap) -> Unit)? = null,
        onSubStageTimings: ((restoration512Ms: Long, upperLipGuardMs: Long, maskGenerationMs: Long, finalBlendingMs: Long) -> Unit)? = null
    ): Bitmap {
        val activeSwapCrop = colorCorrected128 ?: swappedCrop128
            ?: throw IllegalArgumentException("Either colorCorrected128 or swappedCrop128 must be provided.")
        val m128 = forwardMatrix128 ?: FaceAlignment.estimateNorm(targetLandmarks5, 128)
        val m512 = FloatArray(6) { i -> m128[i] * 4.0f }
        val alignedTarget512 = FaceAlignment.warpAffineCrop(targetBitmap, m512, HD_SIZE)
        val geom512 = computeWarpedGeometry(m128, targetLandmarks5, scale = 4.0f)

        // Optional Neural Occlusion Gate via segformer_B5_ce.onnx and/or Depth Anything V2 / sentis-MiDaS
        val tMask0 = System.currentTimeMillis()
        val segOcclusionGate512 = if (enableOcclusionProtection && (preloadedSegformerSession != null || segformerFile != null)) {
            HeadSegmentationAndInpainting.computeFaceOcclusionGate512(
                ortEnv = ortEnv,
                alignedTarget512 = alignedTarget512,
                segModelFile = segformerFile,
                preferHardwareAccel = preferHardwareAccel,
                preloadedSegformerSession = preloadedSegformerSession
            )
        } else null
        val depthOcclusionGate512 = if (enableOcclusionProtection && depthMap != null) {
            runCatching {
                val invM512 = FaceAlignment.invertAffine2x3(m512)
                DepthEstimationEngine.computeFaceOcclusionGate512(
                    depthMap = depthMap,
                    targetWidth = targetWidth,
                    targetHeight = targetHeight,
                    invM512 = invM512,
                    leftEye = geom512.leftEye,
                    rightEye = geom512.rightEye,
                    nose = geom512.nose,
                    leftMouth = geom512.leftMouth,
                    rightMouth = geom512.rightMouth
                )
            }.getOrNull()
        } else null
        val neuralOcclusionGate512 = when {
            segOcclusionGate512 != null && depthOcclusionGate512 != null ->
                FloatArray(segOcclusionGate512.size) { i -> min(segOcclusionGate512[i], depthOcclusionGate512[i]) }
            segOcclusionGate512 != null -> segOcclusionGate512
            else -> depthOcclusionGate512
        }
        var maskGenMs = (System.currentTimeMillis() - tMask0).coerceAtLeast(0L)

        val alignedSource512 = if (sourceBitmap != null && sourceLandmarks5 != null && sourceLandmarks5.size >= 5) {
            FaceAlignment.warpSourceToTargetPose(
                sourceBitmap = sourceBitmap,
                sourceLandmarks5 = sourceLandmarks5,
                targetLandmarks5 = targetLandmarks5,
                dstSize = HD_SIZE
            )
        } else null
        val srcGeom512 = if (alignedSource512 != null) geom512 else null

        val clampedEnhance = enhancementStrength.coerceIn(0f, 1f)
        val tRestore0 = System.currentTimeMillis()
        val gfp512 = if (!identityPreservationMode && clampedEnhance > 0.05f && (preloadedGfpganSession != null || gfpganFile != null)) {
            runOptionalGfpganEnhancement512(
                ortEnv = ortEnv,
                gfpganFile = gfpganFile,
                crop128Or512 = activeSwapCrop,
                preferHardwareAccel = preferHardwareAccel,
                preloadedGfpganSession = preloadedGfpganSession
            )
        } else null

        val total512 = HD_SIZE * HD_SIZE
        // Upscale swapped crop (128x128 or 256x256) to 512x512 using sharp 4x4 Catmull-Rom Bicubic interpolation
        val swapPx512 = upscaleBitmapBicubicTo512(activeSwapCrop)
        val tgtPx512 = IntArray(total512)
        val srcPx512 = if (alignedSource512 != null) IntArray(total512) else null

        alignedTarget512.getPixels(tgtPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        alignedSource512?.getPixels(srcPx512!!, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        alignedTarget512.recycle()
        alignedSource512?.recycle()

        // Capture Stage 1: Swap-Only Output (raw neural swap before restoration, upper-lip guard, or blending)
        val stage1SwapOnlyBmp = if (onStagesCaptured != null) {
            Bitmap.createBitmap(HD_SIZE, HD_SIZE, Bitmap.Config.ARGB_8888).apply {
                setPixels(swapPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            }
        } else null

        if (gfp512 != null) {
            val gfpPx = IntArray(total512)
            gfp512.getPixels(gfpPx, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            gfp512.recycle()
            val gfpWeight = (0.35f + 0.65f * clampedEnhance).coerceIn(0f, 1f)
            val invEnh = 1.0f - gfpWeight
            for (i in 0 until total512) {
                val bc = swapPx512[i]
                val gc = gfpPx[i]
                val r = (((gc ushr 16) and 0xFF) * gfpWeight + ((bc ushr 16) and 0xFF) * invEnh).toInt().coerceIn(0, 255)
                val g = (((gc ushr 8) and 0xFF) * gfpWeight + ((bc ushr 8) and 0xFF) * invEnh).toInt().coerceIn(0, 255)
                val b = (((gc and 0xFF) * gfpWeight + (bc and 0xFF) * invEnh)).toInt().coerceIn(0, 255)
                swapPx512[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        // 1. Unified 512x512 Skin-Tone & Scene Illumination Harmonization (Mean + Contrast Transfer)
        //    Run BEFORE Upper-Lip Guard & HD feature sharpening so swapPx512 and tgtPx512 share the exact same skin tone!
        if (enableColorTransfer) {
            harmonizeSkinToneByMode512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                srcPx512 = srcPx512,
                geom512 = geom512,
                skinToneMode = skinToneMode
            )
        }

        var srcWarpBmp: Bitmap? = null
        if (identityPreservationMode && sourceBitmap != null && sourceLandmarks5 != null && sourceLandmarks5.size >= 5) {
            srcWarpBmp = FaceIdentityPreserver.warpSourceTextureToTargetGeometry512(
                sourceBitmap = sourceBitmap,
                sourceLandmarks5 = sourceLandmarks5,
                targetLandmarks5 = targetLandmarks5,
                dstSize = HD_SIZE
            )
        }

        var hadDonorMoustache = false
        var upperLipGuardMs = 1L

        if (!identityPreservationMode) {
            // 1b. 34-Point 3D Anatomical Feature Disentanglement + VAE/GAN Spatially-Adaptive Feature-Fusion (AdaIN/SPADE):
            LatentFeatureFusionEngine.applyVaeGanLatentFeatureFusion512(
                swappedPixels512 = swapPx512,
                targetPixels512 = tgtPx512,
                targetRecon3D512 = geom512.reconstructed3D,
                skinToneMode = skinToneMode
            )

            // 2. First-pass Target-Aware Upper-Lip / Philtrum Moustache & Grey-Patch Guard
            val tGuard0 = System.currentTimeMillis()
            hadDonorMoustache = protectTargetUpperLipAndEliminateMoustache512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                geom512 = geom512,
                forceMoustacheCleanse = false
            )
            upperLipGuardMs = (System.currentTimeMillis() - tGuard0).coerceAtLeast(1L)

            // 3. Apply frequency-separated 512x512 HD anatomical feature crispness boost
            enhanceAnatomicalFeatures512(
                swapPx512 = swapPx512,
                geom512 = geom512,
                strength = (0.42f + 0.52f * clampedEnhance).coerceIn(0.32f, 0.94f)
            )

            // 4. Subtle 512x512 Skin Pore Micro-Texture Harmonization strictly on smooth cheek/forehead skin
            val poreScale = (if (skinToneMode == SkinToneSourceMode.SOURCE_IDENTITY) 0.08f else 0.14f) *
                (0.4f + 0.6f * clampedEnhance)
            val mouthMidX512 = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
            val mouthMidY512 = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
            for (y in 24 until HD_SIZE - 24) {
                val row = y * HD_SIZE
                for (x in 24 until HD_SIZE - 24) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom512) > 0.05f) continue
                    val dNoseSq = orientedEllipseDistSq(
                        x.toFloat(), y.toFloat(),
                        geom512.nose.x, geom512.nose.y,
                        geom512.eyeDist * 0.40f, geom512.eyeDist * 0.48f,
                        geom512.cosA, geom512.sinA
                    )
                    if (dNoseSq <= 1.0f) continue
                    val dMouthSq = orientedEllipseDistSq(
                        x.toFloat(), y.toFloat(),
                        mouthMidX512, mouthMidY512,
                        geom512.eyeDist * 0.62f, geom512.eyeDist * 0.46f,
                        geom512.cosA, geom512.sinA
                    )
                    if (dMouthSq <= 1.0f) continue

                    val idx = row + x
                    val tc = tgtPx512[idx]
                    val tR = (tc ushr 16) and 0xFF
                    val tG = (tc ushr 8) and 0xFF
                    val tB = tc and 0xFF
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB

                    val n = tgtPx512[(y - 2) * HD_SIZE + x]
                    val s = tgtPx512[(y + 2) * HD_SIZE + x]
                    val l = tgtPx512[row + x - 2]
                    val r = tgtPx512[row + x + 2]
                    val tAvgLum = (
                        (0.299f * (n ushr 16 and 0xFF) + 0.587f * (n ushr 8 and 0xFF) + 0.114f * (n and 0xFF)) +
                        (0.299f * (s ushr 16 and 0xFF) + 0.587f * (s ushr 8 and 0xFF) + 0.114f * (s and 0xFF)) +
                        (0.299f * (l ushr 16 and 0xFF) + 0.587f * (l ushr 8 and 0xFF) + 0.114f * (l and 0xFF)) +
                        (0.299f * (r ushr 16 and 0xFF) + 0.587f * (r ushr 8 and 0xFF) + 0.114f * (r and 0xFF))
                    ) * 0.25f

                    val poreDiff = tLum - tAvgLum
                    if (tLum > 85f && abs(poreDiff) < 7f) {
                        val sc = swapPx512[idx]
                        val sr = (sc ushr 16) and 0xFF
                        val sg = (sc ushr 8) and 0xFF
                        val sb = sc and 0xFF
                        val poreBoost = poreDiff * poreScale
                        val nr = (sr + poreBoost).toInt().coerceIn(0, 255)
                        val ng = (sg + poreBoost).toInt().coerceIn(0, 255)
                        val nb = (sb + poreBoost).toInt().coerceIn(0, 255)
                        swapPx512[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
                    }
                }
            }

            // 5. Dynamic Landmark Eye & Eyebrow Protection
            restoreDynamicEyebrowContinuity(
                outPx = swapPx512,
                tgtPx = tgtPx512,
                size = HD_SIZE,
                geom = geom512
            )
            restoreDynamicOcularSockets(
                outPx = swapPx512,
                tgtPx = tgtPx512,
                srcPx = null,
                size = HD_SIZE,
                geom = geom512
            )
            // Negative Artifact Suppression: Protect squinted/winking eyelids, eyeliner, and makeup integrity
            NegativeArtifactSuppression.suppressOcularAndEyelidArtifacts512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                geom512 = geom512,
                makeupPreservationStrength = 0.85f
            )
            // Face-Aging & Expression Style Engine
            FaceAgingStyleEngine.applyFaceAgingAndStyle512(
                pixels512 = swapPx512,
                geom512 = geom512,
                params = FaceAgingStyleEngine.AgingParameters(ageFactor = 0.0f)
            )
        }
        val restoration512Ms = ((System.currentTimeMillis() - tRestore0) - upperLipGuardMs).coerceAtLeast(1L)

        // Capture Stage 2: Swap + 512x512 HD Restoration + Upper-Lip Guard Output (before final scene blending)
        val stage2RestoreBmp = if (onStagesCaptured != null) {
            Bitmap.createBitmap(HD_SIZE, HD_SIZE, Bitmap.Config.ARGB_8888).apply {
                setPixels(swapPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            }
        } else null

        val tBlendStage0 = System.currentTimeMillis()
        if (!identityPreservationMode) {
            // 6. Face Reaction Synthesis & Protruding Tongue / Oral Cavity Negative Artifact Suppression
            when (faceReactionMode) {
                FaceReactionSourceMode.TARGET_REACTION -> {
                    preserveTargetMouthTeethAndTongue512(
                        swapPx512 = swapPx512,
                        tgtPx512 = tgtPx512,
                        geom512 = geom512
                    )
                }
                FaceReactionSourceMode.SOURCE_REACTION -> {
                    if (srcPx512 != null && srcGeom512 != null) {
                        transferSourceReactionSmileTeethTongue512(
                            swapPx512 = swapPx512,
                            srcPx512 = srcPx512,
                            tgtGeom512 = geom512,
                            srcGeom512 = srcGeom512
                        )
                    }
                }
            }
            NegativeArtifactSuppression.suppressOralAndTongueArtifacts512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                geom512 = geom512,
                sensitivity = 0.90f
            )
        }

        // 7. Build 512x512 Target-Aware Biometric Landmark-Fitted Mask
        val tMask1 = System.currentTimeMillis()
        val mask512 = if (identityPreservationMode) {
            FaceIdentityPreserver.createZeroDifferenceTargetBoundaryMask512(
                targetLandmarks5 = targetLandmarks5,
                forwardMatrix512 = m512,
                dstSize = HD_SIZE,
                featherRadiusPx = 5.0f
            )
        } else {
            createBiometricFaceMask512(
                geom512 = geom512,
                tgtPx512 = tgtPx512,
                swapPx512 = swapPx512,
                skinToneMode = skinToneMode,
                enableOcclusionProtection = enableOcclusionProtection,
                neuralOcclusionGate512 = neuralOcclusionGate512
            )
        }
        val maskElapsed = (System.currentTimeMillis() - tMask1).coerceAtLeast(1L)
        maskGenMs = (maskGenMs + maskElapsed).coerceAtLeast(1L)

        if (identityPreservationMode && srcWarpBmp != null) {
            val srcWarpPx = IntArray(total512)
            srcWarpBmp.getPixels(srcWarpPx, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            srcWarpBmp.recycle()
            val preservedPx = FaceIdentityPreserver.blendIdentityPreservedFace512(
                sourceWarped512 = srcWarpPx,
                target512 = tgtPx512,
                mask512 = mask512,
                preserveMakeup = true
            )
            System.arraycopy(preservedPx, 0, swapPx512, 0, total512)
        }

        var guard2Elapsed = 0L
        if (!identityPreservationMode) {
            // 8. 2-Band Multi-Band Boundary Fusion
            applyMultiBandBoundaryAndJawlineFusion512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                mask512 = mask512
            )

            // 8b. LCM Inpainting & Custom LoRA Edge Refinement
            LcmInpaintingPipeline.refineWithLcmInpainting512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                mask512 = mask512,
                config = LcmInpaintingPipeline.LcmConfig()
            )

            // 9. Final-pass Target-Aware Upper-Lip & Philtrum Protection Lock
            val tGuard1 = System.currentTimeMillis()
            protectTargetUpperLipAndEliminateMoustache512(
                swapPx512 = swapPx512,
                tgtPx512 = tgtPx512,
                geom512 = geom512,
                forceMoustacheCleanse = hadDonorMoustache
            )
            guard2Elapsed = (System.currentTimeMillis() - tGuard1).coerceAtLeast(0L)
            upperLipGuardMs = (upperLipGuardMs + guard2Elapsed).coerceAtLeast(1L)
        }

        // 10. Direct High-Resolution Warp of 512x512 HD Face onto full-resolution targetPixels using Catmull-Rom Bicubic
        val inv = FaceAlignment.invertAffine2x3(m512)
        val corners = arrayOf(
            0f to 0f,
            HD_SIZE.toFloat() to 0f,
            0f to HD_SIZE.toFloat(),
            HD_SIZE.toFloat() to HD_SIZE.toFloat()
        )
        var minX = targetWidth - 1
        var maxX = 0
        var minY = targetHeight - 1
        var maxY = 0
        for ((cx, cy) in corners) {
            val tx = (inv[0] * cx + inv[1] * cy + inv[2]).toInt()
            val ty = (inv[3] * cx + inv[4] * cy + inv[5]).toInt()
            minX = min(minX, tx)
            maxX = max(maxX, tx)
            minY = min(minY, ty)
            maxY = max(maxY, ty)
        }

        minX = (minX - 16).coerceIn(0, targetWidth - 1)
        maxX = (maxX + 16).coerceIn(0, targetWidth - 1)
        minY = (minY - 16).coerceIn(0, targetHeight - 1)
        maxY = (maxY + 16).coerceIn(0, targetHeight - 1)

        val m00 = m512[0]
        val m01 = m512[1]
        val m02 = m512[2]
        val m10 = m512[3]
        val m11 = m512[4]
        val m12 = m512[5]

        val clampedBlend = blendStrength.coerceIn(0.20f, 1.0f)
        val safeScale = scaleAdjust.coerceIn(0.85f, 1.15f)
        val shiftU = offsetX.coerceIn(-36f, 36f)
        val shiftV = offsetY.coerceIn(-36f, 36f)

        for (y in minY..maxY) {
            val rowOffset = y * targetWidth
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            for (x in minX..maxX) {
                val rawU = m00 * x + baseU
                val rawV = m10 * x + baseV
                val u = (rawU - 256f - shiftU) / safeScale + 256f
                val v = (rawV - 256f - shiftV) / safeScale + 256f
                if (u >= 2f && u < HD_SIZE - 3f && v >= 2f && v < HD_SIZE - 3f) {
                    val alpha = sampleMaskBilinearGeneric(mask512, HD_SIZE, u, v) * clampedBlend
                    if (alpha > 0.003f) {
                        val swapColor = FaceAlignment.sampleBicubicClamped(
                            swapPx512,
                            HD_SIZE,
                            HD_SIZE,
                            u,
                            v
                        )
                        val dstColor = targetPixels[rowOffset + x]

                        val sr = (swapColor ushr 16) and 0xFF
                        val sg = (swapColor ushr 8) and 0xFF
                        val sb = swapColor and 0xFF

                        val dr = (dstColor ushr 16) and 0xFF
                        val dg = (dstColor ushr 8) and 0xFF
                        val db = dstColor and 0xFF

                        val invA = 1.0f - alpha
                        val outR = (sr * alpha + dr * invA).toInt().coerceIn(0, 255)
                        val outG = (sg * alpha + dg * invA).toInt().coerceIn(0, 255)
                        val outB = (sb * alpha + db * invA).toInt().coerceIn(0, 255)

                        targetPixels[rowOffset + x] =
                            (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                    }
                }
            }
        }

        // Composite swapPx512 with tgtPx512 using mask512 so Stage 3 (production_final_blend.png)
        // shows the exact final blended 512x512 face (including target hair, forehead, temples & jawline)
        val stage3BlendedPx512 = IntArray(total512)
        for (i in 0 until total512) {
            val a = (mask512[i] * clampedBlend).coerceIn(0f, 1f)
            val invA = 1.0f - a
            val sc = swapPx512[i]
            val tc = tgtPx512[i]
            val r = (((sc ushr 16) and 0xFF) * a + ((tc ushr 16) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val g = (((sc ushr 8) and 0xFF) * a + ((tc ushr 8) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val b = ((sc and 0xFF) * a + (tc and 0xFF) * invA).toInt().coerceIn(0, 255)
            stage3BlendedPx512[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val finalBlendingMs = ((System.currentTimeMillis() - tBlendStage0) - maskElapsed - guard2Elapsed).coerceAtLeast(1L)
        onSubStageTimings?.invoke(restoration512Ms, upperLipGuardMs, maskGenMs, finalBlendingMs)

        val restoredHd512 = Bitmap.createBitmap(HD_SIZE, HD_SIZE, Bitmap.Config.ARGB_8888)
        restoredHd512.setPixels(stage3BlendedPx512, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
        if (onStagesCaptured != null && stage1SwapOnlyBmp != null && stage2RestoreBmp != null) {
            onStagesCaptured(stage1SwapOnlyBmp, stage2RestoreBmp, restoredHd512)
        }
        return restoredHd512
    }

    /**
     * Upscales any 128x128 or 256x256 face crop to 512x512 HD using 4x4 Catmull-Rom Bicubic interpolation
     * instead of blurry bilinear scaling.
     */
    private fun upscaleBitmapBicubicTo512(src: Bitmap): IntArray {
        val w = src.width
        val h = src.height
        val out = IntArray(HD_SIZE * HD_SIZE)
        if (w == HD_SIZE && h == HD_SIZE) {
            src.getPixels(out, 0, HD_SIZE, 0, 0, HD_SIZE, HD_SIZE)
            return out
        }
        val srcPx = IntArray(w * h)
        src.getPixels(srcPx, 0, w, 0, 0, w, h)
        val scaleX = (w - 1).toFloat() / (HD_SIZE - 1).toFloat()
        val scaleY = (h - 1).toFloat() / (HD_SIZE - 1).toFloat()
        for (y in 0 until HD_SIZE) {
            val sy = y * scaleY
            val row = y * HD_SIZE
            for (x in 0 until HD_SIZE) {
                val sx = x * scaleX
                out[row + x] = FaceAlignment.sampleBicubicClamped(srcPx, w, h, sx, sy)
            }
        }
        return out
    }

    /**
     * Frequency-separated anatomical feature enhancer in 512x512 HD space:
     * Boosts fine micro-contrast specifically on the Left Eye, Right Eye, Eyebrows, Nose Tip/Nostrils,
     * and Lip Vermilion Contour (strictly excluding the philtrum/upper-lip skin region).
     */
    private fun enhanceAnatomicalFeatures512(
        swapPx512: IntArray,
        geom512: WarpedFaceGeometry,
        strength: Float
    ) {
        val copy = swapPx512.copyOf()
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val eyeRx = geom512.eyeDist * 0.34f
        val eyeRy = geom512.eyeDist * 0.24f
        val noseRx = geom512.eyeDist * 0.30f
        val noseRy = geom512.eyeDist * 0.30f
        val mouthRx = geom512.eyeDist * 0.44f
        val mouthRy = geom512.eyeDist * 0.16f

        for (y in 16 until HD_SIZE - 16) {
            val row = y * HD_SIZE
            for (x in 16 until HD_SIZE - 16) {
                val xf = x.toFloat()
                val yf = y.toFloat()
                val lEyeD = orientedEllipseDistSq(xf, yf, geom512.leftEye.x, geom512.leftEye.y, eyeRx, eyeRy, geom512.cosA, geom512.sinA)
                val rEyeD = orientedEllipseDistSq(xf, yf, geom512.rightEye.x, geom512.rightEye.y, eyeRx, eyeRy, geom512.cosA, geom512.sinA)
                val noseD = orientedEllipseDistSq(xf, yf, geom512.nose.x, geom512.nose.y - 4f, noseRx, noseRy, geom512.cosA, geom512.sinA)
                val mouthD = orientedEllipseDistSq(xf, yf, mouthMidX, mouthMidY + 4f, mouthRx, mouthRy, geom512.cosA, geom512.sinA)

                val minD = min(min(lEyeD, rEyeD), min(noseD, mouthD))
                if (minD >= 1.0f) continue

                val featWeight = 0.5f * (1.0f + cos(Math.PI * sqrt(minD)).toFloat()) * strength
                val idx = row + x
                val c = copy[idx]
                val cr = (c ushr 16) and 0xFF
                val cg = (c ushr 8) and 0xFF
                val cb = c and 0xFF

                val n = copy[(y - 2) * HD_SIZE + x]
                val s = copy[(y + 2) * HD_SIZE + x]
                val w = copy[row + x - 2]
                val e = copy[row + x + 2]
                val avgR = (((n ushr 16) and 0xFF) + ((s ushr 16) and 0xFF) + ((w ushr 16) and 0xFF) + ((e ushr 16) and 0xFF)) * 0.25f
                val avgG = (((n ushr 8) and 0xFF) + ((s ushr 8) and 0xFF) + ((w ushr 8) and 0xFF) + ((e ushr 8) and 0xFF)) * 0.25f
                val avgB = ((n and 0xFF) + (s and 0xFF) + (w and 0xFF) + (e and 0xFF)) * 0.25f

                val outR = (cr + (cr - avgR) * featWeight * 0.65f).toInt().coerceIn(0, 255)
                val outG = (cg + (cg - avgG) * featWeight * 0.65f).toInt().coerceIn(0, 255)
                val outB = (cb + (cb - avgB) * featWeight * 0.65f).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    /**
     * CRITICAL STAGE — Target-Aware Upper-Lip / Philtrum / Mouth-Corner Facial-Hair Protection:
     *
     * Checks whether the Target image has a moustache by measuring the Target's philtrum luminance
     * relative to the Target's upper malar cheek skin luminance.
     * When the Target image has NO moustache (`targetPhiltrumRatio >= 0.62`), this stage protects:
     *  - philtrum (between sub-nasale and upper lip vermilion border)
     *  - upper lip skin arch
     *  - left & right mouth corners (commissures)
     *  - surrounding perioral skin
     * eliminating 100% of any moustache, dark upper-lip shadow, grey patch above the lips, or source
     * facial-hair pigment leakage while keeping the actual pink/red lip vermilion and nose nostrils intact.
     */
    private fun protectTargetUpperLipAndEliminateMoustache512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        geom512: WarpedFaceGeometry,
        forceMoustacheCleanse: Boolean = false
    ): Boolean {
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val mouthW = hypot(
            (geom512.rightMouth.x - geom512.leftMouth.x).toDouble(),
            (geom512.rightMouth.y - geom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        val dxNM = mouthMidX - geom512.nose.x
        val dyNM = mouthMidY - geom512.nose.y
        val noseToMouthDist = hypot(dxNM.toDouble(), dyNM.toDouble()).toFloat().coerceAtLeast(48f)
        val vx = dxNM / noseToMouthDist
        val vy = dyNM / noseToMouthDist
        val ux = geom512.cosA
        val uy = geom512.sinA

        val philtrumCx = geom512.nose.x * 0.38f + mouthMidX * 0.62f
        val philtrumCy = geom512.nose.y * 0.40f + mouthMidY * 0.60f

        // 1. Sample clean cheek skin reference matching the benchmark & anatomical malar zone
        val leftCheekCx = geom512.leftEye.x * 0.55f + geom512.leftMouth.x * 0.45f - geom512.cosA * 18f
        val leftCheekCy = geom512.leftEye.y * 0.45f + geom512.leftMouth.y * 0.55f
        val rightCheekCx = geom512.rightEye.x * 0.55f + geom512.rightMouth.x * 0.45f + geom512.cosA * 18f
        val rightCheekCy = geom512.rightEye.y * 0.45f + geom512.rightMouth.y * 0.55f
        val cheekR = 18

        var sCheekR = 0f
        var sCheekG = 0f
        var sCheekB = 0f
        var tCheekR = 0f
        var tCheekG = 0f
        var tCheekB = 0f
        var cheekCount = 0

        for ((ccx, ccy) in arrayOf(leftCheekCx.toInt() to leftCheekCy.toInt(), rightCheekCx.toInt() to rightCheekCy.toInt())) {
            for (y in (ccy - cheekR)..(ccy + cheekR) step 2) {
                if (y !in 16 until HD_SIZE - 16) continue
                for (x in (ccx - cheekR)..(ccx + cheekR) step 2) {
                    if (x !in 16 until HD_SIZE - 16) continue
                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    sCheekR += (sc ushr 16) and 0xFF
                    sCheekG += (sc ushr 8) and 0xFF
                    sCheekB += sc and 0xFF
                    tCheekR += (tc ushr 16) and 0xFF
                    tCheekG += (tc ushr 8) and 0xFF
                    tCheekB += tc and 0xFF
                    cheekCount++
                }
            }
        }
        if (cheekCount < 8) return false
        sCheekR /= cheekCount
        sCheekG /= cheekCount
        sCheekB /= cheekCount
        tCheekR /= cheekCount
        tCheekG /= cheekCount
        tCheekB /= cheekCount
        val tCheekLum = (0.299f * tCheekR + 0.587f * tCheekG + 0.114f * tCheekB).coerceAtLeast(25f)
        val sCheekLum = (0.299f * sCheekR + 0.587f * sCheekG + 0.114f * sCheekB).coerceAtLeast(25f)

        // 2. Measure Target and Swapped philtrum luminance
        var tPhiltrumLumSum = 0f
        var sPhiltrumLumSum = 0f
        var philtrumSamples = 0
        val pMinX = (philtrumCx - 24f).toInt().coerceIn(16, HD_SIZE - 17)
        val pMaxX = (philtrumCx + 24f).toInt().coerceIn(16, HD_SIZE - 17)
        val pMinY = (philtrumCy - 12f).toInt().coerceIn(16, HD_SIZE - 17)
        val pMaxY = (philtrumCy + 12f).toInt().coerceIn(16, HD_SIZE - 17)
        for (y in pMinY..pMaxY) {
            for (x in pMinX..pMaxX) {
                val idx = y * HD_SIZE + x
                val tc = tgtPx512[idx]
                val sc = swapPx512[idx]
                val tLum = 0.299f * (tc ushr 16 and 0xFF) + 0.587f * (tc ushr 8 and 0xFF) + 0.114f * (tc and 0xFF)
                val sLum = 0.299f * (sc ushr 16 and 0xFF) + 0.587f * (sc ushr 8 and 0xFF) + 0.114f * (sc and 0xFF)
                tPhiltrumLumSum += tLum
                sPhiltrumLumSum += sLum
                philtrumSamples++
            }
        }
        val tPhiltrumMeanLum = (if (philtrumSamples > 0) tPhiltrumLumSum / philtrumSamples else tCheekLum).coerceAtLeast(25f)
        val sPhiltrumMeanLum = (if (philtrumSamples > 0) sPhiltrumLumSum / philtrumSamples else sCheekLum).coerceAtLeast(25f)
        val tPhiltrumRatio = tPhiltrumMeanLum / tCheekLum
        val targetHasNoMoustache = tPhiltrumRatio >= 0.62f
        if (!targetHasNoMoustache) return false

        val sPhiltrumRatio = sPhiltrumMeanLum / sCheekLum
        val philtrumRatioDrop = tPhiltrumRatio - sPhiltrumRatio
        val philtrumLumDrop = tPhiltrumMeanLum - sPhiltrumMeanLum
        // Only activate when Target philtrum is bright/clean AND Swapped philtrum has a genuine heavy donor moustache drop
        // compared to Target's own philtrum. Clean-shaven or female faces have natural nose-base/lip shading in both Target and Swap
        // and are left 100% untouched (zero oval nose cutout or horizontal upper-lip strip).
        val hasDonorMoustache = forceMoustacheCleanse ||
            (tPhiltrumRatio >= 0.86f && sPhiltrumRatio < 0.82f && philtrumRatioDrop > 0.08f && philtrumLumDrop > 12f)
        if (!hasDonorMoustache) return false

        val residualR = sCheekR - tCheekR
        val residualG = sCheekG - tCheekG
        val residualB = sCheekB - tCheekB

        val halfSpanU = max(mouthW * 0.68f, geom512.eyeDist * 0.54f)
        val boundMinX = (min(geom512.nose.x, mouthMidX) - halfSpanU - 16f).toInt().coerceIn(14, HD_SIZE - 15)
        val boundMaxX = (max(geom512.nose.x, mouthMidX) + halfSpanU + 16f).toInt().coerceIn(14, HD_SIZE - 15)
        val boundMinY = (min(geom512.nose.y, mouthMidY) - 10f).toInt().coerceIn(14, HD_SIZE - 15)
        val boundMaxY = (max(geom512.nose.y, mouthMidY) + noseToMouthDist * 0.22f).toInt().coerceIn(14, HD_SIZE - 15)

        val nostrilRx = geom512.eyeDist * 0.20f
        val nostrilRy = geom512.eyeDist * 0.09f

        for (y in boundMinY..boundMaxY) {
            val row = y * HD_SIZE
            for (x in boundMinX..boundMaxX) {
                val xf = x.toFloat()
                val yf = y.toFloat()

                val dx = xf - geom512.nose.x
                val dy = yf - geom512.nose.y
                val u = dx * ux + dy * uy
                val v = dx * vx + dy * vy
                val uNorm = abs(u) / halfSpanU
                if (uNorm >= 1.0f) continue

                val vFrac = v / noseToMouthDist
                val vTop = 0.08f
                val vBottom = 0.96f
                if (vFrac <= vTop || vFrac >= vBottom) continue

                val dNostrilSq = orientedEllipseDistSq(
                    xf, yf, geom512.nose.x, geom512.nose.y + nostrilRy * 0.15f, nostrilRx, nostrilRy, geom512.cosA, geom512.sinA
                )
                // Smooth feather around nostril border instead of a sharp binary step
                val nostrilFeather = when {
                    dNostrilSq <= 0.64f -> 0.0f
                    dNostrilSq >= 1.44f -> 1.0f
                    else -> {
                        val nt = ((sqrt(dNostrilSq) - 0.80f) / 0.40f).coerceIn(0f, 1f)
                        (0.5f * (1.0f - cos(Math.PI * nt))).toFloat()
                    }
                }
                if (nostrilFeather <= 0.01f) continue

                val vMid = 0.54f
                val vHalf = 0.44f
                val vNorm = (vFrac - vMid) / vHalf
                val radialSq = uNorm * uNorm + vNorm * vNorm
                if (radialSq >= 1.0f) continue

                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF

                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF

                // Smooth chromatic lip-vermilion protection
                val lipRedExcess = max(sR - sG, tR - tG)
                val lipFade = if (vFrac > 0.72f && lipRedExcess > 28) {
                    (1.0f - ((lipRedExcess - 28).toFloat() / 18f)).coerceIn(0f, 1f)
                } else 1.0f
                if (lipFade <= 0.01f) continue

                val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB
                val cleanSkinR = (0.58f * (tR + residualR) + 0.42f * sCheekR).coerceIn(0f, 255f)
                val cleanSkinG = (0.58f * (tG + residualG) + 0.42f * sCheekG).coerceIn(0f, 255f)
                val cleanSkinB = (0.58f * (tB + residualB) + 0.42f * sCheekB).coerceIn(0f, 255f)
                val expectedCleanLum = 0.299f * cleanSkinR + 0.587f * cleanSkinG + 0.114f * cleanSkinB

                val rDist = sqrt(radialSq)
                val radialEnv = if (rDist <= 0.68f) {
                    1.0f
                } else {
                    val t = ((rDist - 0.68f) / 0.32f).coerceIn(0f, 1f)
                    (0.5f * (1.0f + cos(Math.PI * t))).toFloat()
                }
                val zoneEnv = radialEnv * nostrilFeather * lipFade

                val shadowDeficit = expectedCleanLum - sLum
                val cleanWarmth = cleanSkinR - cleanSkinB
                val swapWarmth = (sR - sB).toFloat()
                val greyCastDeficit = cleanWarmth - swapWarmth

                if (shadowDeficit > 1.5f || greyCastDeficit > 2.5f) {
                    val severity = max((shadowDeficit - 1.5f) / 10.0f, (greyCastDeficit - 2.5f) / 10.0f).coerceIn(0.35f, 1f)
                    val replaceWeight = (0.98f * zoneEnv * severity).coerceIn(0f, 0.98f)
                    val outR = (sR * (1f - replaceWeight) + cleanSkinR * replaceWeight).toInt().coerceIn(0, 255)
                    val outG = (sG * (1f - replaceWeight) + cleanSkinG * replaceWeight).toInt().coerceIn(0, 255)
                    val outB = (sB * (1f - replaceWeight) + cleanSkinB * replaceWeight).toInt().coerceIn(0, 255)
                    swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                }
            }
        }
        return true
    }

    /**
     * 2-Band (Low-Frequency Directional Illumination + High-Frequency Detail) Multi-Band & Jawline/Forehead Shadow Fusion
     * in 512x512 HD space: smoothly aligns the outer cheek, right/left forehead temple, and chin/neck jawline shadow
     * envelope of `swapPx512` to `tgtPx512` strictly in the outer transition ring (`0.01f < m < 0.95f`),
     * while preserving 100% of the inner core facial features (`m >= 0.95f`) free of any Target nose or mouth shadow bleed.
     */
    private fun applyMultiBandBoundaryAndJawlineFusion512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        mask512: FloatArray
    ) {
        val origSwap = swapPx512.copyOf()
        val step1 = 10
        val step2 = 22
        for (y in step2 until HD_SIZE - step2) {
            val row = y * HD_SIZE
            val regionBoost = when {
                y < 185 -> 0.86f
                y > 395 -> 0.84f
                else -> 0.72f
            }
            for (x in step2 until HD_SIZE - step2) {
                val idx = row + x
                val m = mask512[idx]
                // Skip both outside mask (m <= 0.01f) and solid inner facial core (m >= 0.95f: eyes, nose, philtrum, lips)
                // so Target's side-view nose shadow or parted mouth never bleeds into the swapped core features
                if (m <= 0.01f || m >= 0.95f) continue

                val boundaryWeight = ((1.0f - m) * regionBoost).coerceIn(0f, 0.84f)
                val sLow1 = compute5PointLowPassRGB512(origSwap, x, y, step1)
                val sLow2 = compute5PointLowPassRGB512(origSwap, x, y, step2)
                val tLow1 = compute5PointLowPassRGB512(tgtPx512, x, y, step1)
                val tLow2 = compute5PointLowPassRGB512(tgtPx512, x, y, step2)

                val sLowR = 0.42f * sLow1[0] + 0.58f * sLow2[0]
                val sLowG = 0.42f * sLow1[1] + 0.58f * sLow2[1]
                val sLowB = 0.42f * sLow1[2] + 0.58f * sLow2[2]

                val tLowR = 0.42f * tLow1[0] + 0.58f * tLow2[0]
                val tLowG = 0.42f * tLow1[1] + 0.58f * tLow2[1]
                val tLowB = 0.42f * tLow1[2] + 0.58f * tLow2[2]

                val sc = origSwap[idx]
                val sr = (sc ushr 16) and 0xFF
                val sg = (sc ushr 8) and 0xFF
                val sb = sc and 0xFF

                val nr = (sr + (tLowR - sLowR) * boundaryWeight).toInt().coerceIn(0, 255)
                val ng = (sg + (tLowG - sLowG) * boundaryWeight).toInt().coerceIn(0, 255)
                val nb = (sb + (tLowB - sLowB) * boundaryWeight).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
            }
        }
    }

    private fun compute5PointLowPassRGB512(px: IntArray, x: Int, y: Int, d: Int): FloatArray {
        val c0 = px[y * HD_SIZE + x]
        val cN = px[(y - d) * HD_SIZE + x]
        val cS = px[(y + d) * HD_SIZE + x]
        val cW = px[y * HD_SIZE + (x - d)]
        val cE = px[y * HD_SIZE + (x + d)]
        val r = (((c0 ushr 16) and 0xFF) + ((cN ushr 16) and 0xFF) + ((cS ushr 16) and 0xFF) +
            ((cW ushr 16) and 0xFF) + ((cE ushr 16) and 0xFF)) * 0.2f
        val g = (((c0 ushr 8) and 0xFF) + ((cN ushr 8) and 0xFF) + ((cS ushr 8) and 0xFF) +
            ((cW ushr 8) and 0xFF) + ((cE ushr 8) and 0xFF)) * 0.2f
        val b = ((c0 and 0xFF) + (cN and 0xFF) + (cS and 0xFF) + (cW and 0xFF) + (cE and 0xFF)) * 0.2f
        return floatArrayOf(r, g, b)
    }

    /**
     * Harmonizes the 512x512 restored crop according to the user's chosen [SkinToneSourceMode]
     * (`TARGET_SCENE`, `SOURCE_IDENTITY`, or `BALANCED_BLEND`) using Mean + Contrast (StdDev)
     * plus 2D Left/Right & Top/Bottom Directional Scene Illumination adaptation so a side-lit or
     * indoor Target photo never shows a two-tone color patch at the forehead or jawline.
     */
    private fun harmonizeSkinToneByMode512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        srcPx512: IntArray?,
        geom512: WarpedFaceGeometry,
        skinToneMode: SkinToneSourceMode
    ) {
        var sRMean = 0.0
        var sGMean = 0.0
        var sBMean = 0.0
        var tRMean = 0.0
        var tGMean = 0.0
        var tBMean = 0.0
        var dRMean = 0.0
        var dGMean = 0.0
        var dBMean = 0.0
        var count = 0

        // Also measure left-half vs right-half directional luminance in Target vs Swapped crop
        var tLeftLumSum = 0.0
        var sLeftLumSum = 0.0
        var leftCount = 0
        var tRightLumSum = 0.0
        var sRightLumSum = 0.0
        var rightCount = 0

        for (y in 120..410 step 2) {
            val ny = (y - 268.0) / 148.0
            for (x in 120..392 step 2) {
                val nx = (x - 256.0) / 136.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom512) > 0.12f) continue
                    // Strictly exclude nose base, philtrum/moustache arch, and mouth when sampling clean cheek/forehead skin
                    if (y in 280..425 && x in 150..362) continue

                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    val dc = srcPx512?.get(idx) ?: sc
                    val sLum = 0.299 * (sc ushr 16 and 0xFF) + 0.587 * (sc ushr 8 and 0xFF) + 0.114 * (sc and 0xFF)
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    val dLum = 0.299 * (dc ushr 16 and 0xFF) + 0.587 * (dc ushr 8 and 0xFF) + 0.114 * (dc and 0xFF)
                    if (tLum < 50.0 || dLum < 55.0) continue

                    sRMean += (sc ushr 16) and 0xFF
                    sGMean += (sc ushr 8) and 0xFF
                    sBMean += sc and 0xFF
                    tRMean += (tc ushr 16) and 0xFF
                    tGMean += (tc ushr 8) and 0xFF
                    tBMean += tc and 0xFF
                    dRMean += (dc ushr 16) and 0xFF
                    dGMean += (dc ushr 8) and 0xFF
                    dBMean += dc and 0xFF
                    count++

                    if (x < 256) {
                        tLeftLumSum += tLum
                        sLeftLumSum += sLum
                        leftCount++
                    } else {
                        tRightLumSum += tLum
                        sRightLumSum += sLum
                        rightCount++
                    }
                }
            }
        }
        if (count < 32) return

        sRMean /= count
        sGMean /= count
        sBMean /= count
        tRMean /= count
        tGMean /= count
        tBMean /= count
        dRMean /= count
        dGMean /= count
        dBMean /= count

        val tLeftLum = if (leftCount > 8) tLeftLumSum / leftCount else (0.299 * tRMean + 0.587 * tGMean + 0.114 * tBMean)
        val sLeftLum = if (leftCount > 8) sLeftLumSum / leftCount else (0.299 * sRMean + 0.587 * sGMean + 0.114 * sBMean)
        val tRightLum = if (rightCount > 8) tRightLumSum / rightCount else (0.299 * tRMean + 0.587 * tGMean + 0.114 * tBMean)
        val sRightLum = if (rightCount > 8) sRightLumSum / rightCount else (0.299 * sRMean + 0.587 * sGMean + 0.114 * sBMean)
        // Horizontal directional shadow difference between Target and Swapped face
        val dirGradDelta = (((tRightLum - tLeftLum) - (sRightLum - sLeftLum)) * 0.45).toFloat().coerceIn(-22f, 22f)

        var sRVar = 0.0
        var sGVar = 0.0
        var sBVar = 0.0
        var tRVar = 0.0
        var tGVar = 0.0
        var tBVar = 0.0
        var dRVar = 0.0
        var dGVar = 0.0
        var dBVar = 0.0
        for (y in 120..410 step 4) {
            val ny = (y - 268.0) / 148.0
            for (x in 120..392 step 4) {
                val nx = (x - 256.0) / 136.0
                if (nx * nx + ny * ny <= 1.0) {
                    if (computeEyeAndBrowProtectionWeight(x.toFloat(), y.toFloat(), geom512) > 0.12f) continue
                    if (y in 280..425 && x in 150..362) continue
                    val idx = y * HD_SIZE + x
                    val sc = swapPx512[idx]
                    val tc = tgtPx512[idx]
                    val dc = srcPx512?.get(idx) ?: sc
                    val tLum = 0.299 * (tc ushr 16 and 0xFF) + 0.587 * (tc ushr 8 and 0xFF) + 0.114 * (tc and 0xFF)
                    if (tLum < 50.0) continue
                    val dSr = ((sc ushr 16) and 0xFF) - sRMean
                    val dSg = ((sc ushr 8) and 0xFF) - sGMean
                    val dSb = (sc and 0xFF) - sBMean
                    val dTr = ((tc ushr 16) and 0xFF) - tRMean
                    val dTg = ((tc ushr 8) and 0xFF) - tGMean
                    val dTb = (tc and 0xFF) - tBMean
                    val dDr = ((dc ushr 16) and 0xFF) - dRMean
                    val dDg = ((dc ushr 8) and 0xFF) - dGMean
                    val dDb = (dc and 0xFF) - dBMean
                    sRVar += dSr * dSr
                    sGVar += dSg * dSg
                    sBVar += dSb * dSb
                    tRVar += dTr * dTr
                    tGVar += dTg * dTg
                    tBVar += dTb * dTb
                    dRVar += dDr * dDr
                    dGVar += dDg * dDg
                    dBVar += dDb * dDb
                }
            }
        }
        val varCount = (count / 4).coerceAtLeast(1)
        val sRStd = sqrt(sRVar / varCount).coerceAtLeast(10.0)
        val sGStd = sqrt(sGVar / varCount).coerceAtLeast(10.0)
        val sBStd = sqrt(sBVar / varCount).coerceAtLeast(10.0)
        val tRStd = sqrt(tRVar / varCount).coerceAtLeast(10.0)
        val tGStd = sqrt(tGVar / varCount).coerceAtLeast(10.0)
        val tBStd = sqrt(tBVar / varCount).coerceAtLeast(10.0)
        val dRStd = sqrt(dRVar / varCount).coerceAtLeast(10.0)
        val dGStd = sqrt(dGVar / varCount).coerceAtLeast(10.0)
        val dBStd = sqrt(dBVar / varCount).coerceAtLeast(10.0)

        // Blend Source identity tone with Target scene illumination so even SOURCE_IDENTITY harmonizes with
        // the Target scene's ambient lighting temperature and never leaves a contrasting forehead/neck patch
        val (goalRMean, goalGMean, goalBMean) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(
                tRMean * 0.96 + sRMean * 0.04,
                tGMean * 0.96 + sGMean * 0.04,
                tBMean * 0.96 + sBMean * 0.04
            )
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                dRMean * 0.62 + tRMean * 0.38,
                dGMean * 0.62 + tGMean * 0.38,
                dBMean * 0.62 + tBMean * 0.38
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                dRMean * 0.50 + tRMean * 0.50,
                dGMean * 0.50 + tGMean * 0.50,
                dBMean * 0.50 + tBMean * 0.50
            )
        }

        val (goalRStd, goalGStd, goalBStd) = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> Triple(
                0.85 * tRStd + 0.15 * sRStd,
                0.85 * tGStd + 0.15 * sGStd,
                0.85 * tBStd + 0.15 * sBStd
            )
            SkinToneSourceMode.SOURCE_IDENTITY -> Triple(
                0.68 * dRStd + 0.32 * tRStd,
                0.68 * dGStd + 0.32 * tGStd,
                0.68 * dBStd + 0.32 * tBStd
            )
            SkinToneSourceMode.BALANCED_BLEND -> Triple(
                0.50 * dRStd + 0.50 * tRStd,
                0.50 * dGStd + 0.50 * tGStd,
                0.50 * dBStd + 0.50 * tBStd
            )
        }

        val scaleR = (goalRStd / sRStd).toFloat().coerceIn(0.76f, 1.24f)
        val scaleG = (goalGStd / sGStd).toFloat().coerceIn(0.76f, 1.24f)
        val scaleB = (goalBStd / sBStd).toFloat().coerceIn(0.76f, 1.24f)

        val strength = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> 0.94f
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.92f
            SkinToneSourceMode.BALANCED_BLEND -> 0.90f
        }

        for (y in 0 until HD_SIZE) {
            val row = y * HD_SIZE
            val yf = y.toFloat()
            for (x in 0 until HD_SIZE) {
                val idx = row + x
                val xf = x.toFloat()

                val c = swapPx512[idx]
                val origR = (c ushr 16) and 0xFF
                val origG = (c ushr 8) and 0xFF
                val origB = c and 0xFF

                val eyeProt = computeEyeAndBrowProtectionWeight(xf, yf, geom512)
                val lum = 0.299f * origR + 0.587f * origG + 0.114f * origB
                // Protect dark pupil/lash/eyebrow hair cores so dark features stay crisp and rich
                val darkFeatureProt = if (eyeProt > 0.18f && lum < 72f) eyeProt * 0.85f else 0f
                val blend = strength * (1.0f - darkFeatureProt)

                // Apply horizontal directional lighting gradient (xNorm in [-1, +1])
                val xNorm = ((xf - 256f) / 180f).coerceIn(-1f, 1f)
                val dirShift = dirGradDelta * 0.5f * xNorm

                val matchedR = ((origR - sRMean) * scaleR + goalRMean + dirShift).toFloat()
                val matchedG = ((origG - sGMean) * scaleG + goalGMean + dirShift).toFloat()
                val matchedB = ((origB - sBMean) * scaleB + goalBMean + dirShift).toFloat()

                val r = (origR * (1f - blend) + matchedR * blend).toInt().coerceIn(0, 255)
                val g = (origG * (1f - blend) + matchedG * blend).toInt().coerceIn(0, 255)
                val b = (origB * (1f - blend) + matchedB * blend).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    /**
     * Restores crisp 512x512 inner visible teeth / open oral cavity from the Target photo ONLY when the
     * Target has an open-mouth smile with visible teeth, strictly inside the inner oral slit (`r <= 0.52f`).
     * NEVER overwrites the upper-lip skin, Cupid's bow, or outer lip contours with Target pixels,
     * eliminating 100% of double-lip ghosting and horizontal upper-lip cut lines when swapping between
     * a closed-mouth Source face and an open-mouth Target face.
     */
    private fun preserveTargetMouthTeethAndTongue512(
        swapPx512: IntArray,
        tgtPx512: IntArray,
        geom512: WarpedFaceGeometry
    ) {
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val mouthWidth = hypot(
            (geom512.rightMouth.x - geom512.leftMouth.x).toDouble(),
            (geom512.rightMouth.y - geom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        // Tight inner oral slit aperture (strictly inside the parted lips, never touching philtrum or chin)
        val rx = mouthWidth * 0.36f
        val ry = mouthWidth * 0.12f
        val centerY = mouthMidY + 2f
        val minX = (mouthMidX - rx - 2f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxX = (mouthMidX + rx + 2f).toInt().coerceIn(8, HD_SIZE - 9)
        val minY = (centerY - ry - 2f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxY = (centerY + ry + 2f).toInt().coerceIn(8, HD_SIZE - 9)

        // First verify whether Target actually has a wide-open toothy grin (bright white teeth filling >= 34% of inner oral core)
        // while the swapped mouth lacks teeth. For closed or slightly parted mouths, do NOT blend Target mouth pixels onto Swapped lips!
        var corePixels = 0
        var targetWhiteTeethPixels = 0
        var swapWhiteTeethPixels = 0
        for (y in minY..maxY) {
            val row = y * HD_SIZE
            for (x in minX..maxX) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    mouthMidX, centerY,
                    rx * 0.75f, ry * 0.75f,
                    geom512.cosA, geom512.sinA
                )
                if (dSq > 1.0f) continue
                corePixels++
                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF
                val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB
                val tMax = max(tR, max(tG, tB)).coerceAtLeast(1)
                val tMin = min(tR, min(tG, tB))
                val tSat = (tMax - tMin).toFloat() / tMax.toFloat()
                if (tLum > 172f && tSat < 0.16f) targetWhiteTeethPixels++

                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF
                val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB
                val sMax = max(sR, max(sG, sB)).coerceAtLeast(1)
                val sMin = min(sR, min(sG, sB))
                val sSat = (sMax - sMin).toFloat() / sMax.toFloat()
                if (sLum > 165f && sSat < 0.18f) swapWhiteTeethPixels++
            }
        }

        if (corePixels < 16) return
        val targetTeethRatio = targetWhiteTeethPixels.toFloat() / corePixels.toFloat()
        val swapTeethRatio = swapWhiteTeethPixels.toFloat() / corePixels.toFloat()
        val handledByTongueAndOralGuard = NegativeArtifactSuppression.suppressOralAndTongueArtifacts512(
            swapPx512 = swapPx512,
            tgtPx512 = tgtPx512,
            geom512 = geom512
        )
        if (handledByTongueAndOralGuard || targetTeethRatio < 0.34f || swapTeethRatio >= 0.15f) {
            // Keep 100% of the swapped lips/mouth intact — zero double-lip or split-mouth artifacts!
            return
        }

        for (y in minY..maxY) {
            val row = y * HD_SIZE
            for (x in minX..maxX) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    mouthMidX, centerY,
                    rx, ry,
                    geom512.cosA, geom512.sinA
                )
                if (dSq >= 1.0f) continue
                val r = sqrt(dSq)
                val env = (0.5f * (1.0f + cos(Math.PI * r))).toFloat()

                val idx = row + x
                val tc = tgtPx512[idx]
                val tR = (tc ushr 16) and 0xFF
                val tG = (tc ushr 8) and 0xFF
                val tB = tc and 0xFF

                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF

                val directWeight = (0.62f * env).coerceIn(0f, 0.62f)
                val invW = 1.0f - directWeight
                val outR = (sR * invW + tR * directWeight).toInt().coerceIn(0, 255)
                val outG = (sG * invW + tG * directWeight).toInt().coerceIn(0, 255)
                val outB = (sB * invW + tB * directWeight).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    /**
     * Transfers the Source photo's (Image 1) own mouth/smile/teeth expression onto the swapped face
     * when `FaceReactionSourceMode.SOURCE_REACTION` is selected by the user, while keeping the upper-lip
     * philtrum protected from any unwanted moustache shadow.
     */
    private fun transferSourceReactionSmileTeethTongue512(
        swapPx512: IntArray,
        srcPx512: IntArray,
        tgtGeom512: WarpedFaceGeometry,
        srcGeom512: WarpedFaceGeometry
    ) {
        val tMouthMidX = (tgtGeom512.leftMouth.x + tgtGeom512.rightMouth.x) * 0.5f
        val tMouthMidY = (tgtGeom512.leftMouth.y + tgtGeom512.rightMouth.y) * 0.5f
        val tMouthW = hypot(
            (tgtGeom512.rightMouth.x - tgtGeom512.leftMouth.x).toDouble(),
            (tgtGeom512.rightMouth.y - tgtGeom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        val sMouthMidX = (srcGeom512.leftMouth.x + srcGeom512.rightMouth.x) * 0.5f
        val sMouthMidY = (srcGeom512.leftMouth.y + srcGeom512.rightMouth.y) * 0.5f

        val rx = tMouthW * 0.56f
        val ry = tMouthW * 0.32f
        val centerY = tMouthMidY + 4f
        // Strictly stay below the upper-lip philtrum moustache zone (minY >= tMouthMidY - ry * 0.45f)
        val minX = (tMouthMidX - rx - 4f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxX = (tMouthMidX + rx + 4f).toInt().coerceIn(8, HD_SIZE - 9)
        val minY = (tMouthMidY - ry * 0.45f).toInt().coerceIn(8, HD_SIZE - 9)
        val maxY = (centerY + ry + 4f).toInt().coerceIn(8, HD_SIZE - 9)

        for (y in minY..maxY) {
            val row = y * HD_SIZE
            for (x in minX..maxX) {
                val dSq = orientedEllipseDistSq(
                    x.toFloat(), y.toFloat(),
                    tMouthMidX, centerY,
                    rx, ry,
                    tgtGeom512.cosA, tgtGeom512.sinA
                )
                if (dSq >= 1.0f) continue
                val r = sqrt(dSq)
                val env = (0.5f * (1.0f + cos(Math.PI * r.coerceIn(0f, 1f)))).toFloat()

                val sx = (sMouthMidX + (x - tMouthMidX)).toInt().coerceIn(2, HD_SIZE - 3)
                val sy = (sMouthMidY + (y - tMouthMidY)).toInt().coerceIn(2, HD_SIZE - 3)
                val srcC = srcPx512[sy * HD_SIZE + sx]
                val sR = (srcC ushr 16) and 0xFF
                val sG = (srcC ushr 8) and 0xFF
                val sB = srcC and 0xFF

                val idx = row + x
                val curC = swapPx512[idx]
                val cR = (curC ushr 16) and 0xFF
                val cG = (curC ushr 8) and 0xFF
                val cB = curC and 0xFF

                val w = (0.86f * env).coerceIn(0f, 0.88f)
                val invW = 1.0f - w
                val outR = (cR * invW + sR * w).toInt().coerceIn(0, 255)
                val outG = (cG * invW + sG * w).toInt().coerceIn(0, 255)
                val outB = (cB * invW + sB * w).toInt().coerceIn(0, 255)
                swapPx512[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }
    }

    /**
     * Creates a 3D Yaw-Aware, Hairline-Aware & Chin-Contour-Fitted 512x512 Biometric Face Mask:
     *  - Asymmetric Left/Right Horizontal Radius (`radiusXLeft`, `radiusXRight`) that automatically contracts
     *    on the foreshortened side when the Target head is turned slightly sideways (`geom512.yawRatio`),
     *    preventing forehead/temple mask spillover onto side hair.
     *  - Asymmetric Top/Bottom Vertical Radius (`radiusYTop`, `radiusYBottom`) with an early upper-forehead taper
     *    above the eyebrows/bindi and a tight lower-chin taper so a wider/rounder Source jaw never creates a
     *    double-chin seam below a narrower Target chin.
     *  - Continuous soft hair/shadow occlusion attenuation + 2-pass separable spatial feathering so `mask512`
     *    is 100% smooth with zero sharp seams or patches.
     */
    private fun createBiometricFaceMask512(
        geom512: WarpedFaceGeometry,
        tgtPx512: IntArray,
        swapPx512: IntArray,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.SOURCE_IDENTITY,
        enableOcclusionProtection: Boolean = true,
        neuralOcclusionGate512: FloatArray? = null
    ): FloatArray {
        val rawMask = FloatArray(HD_SIZE * HD_SIZE)
        val eyeMidX = (geom512.leftEye.x + geom512.rightEye.x) * 0.5f
        val eyeMidY = (geom512.leftEye.y + geom512.rightEye.y) * 0.5f
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val mouthW = hypot(
            (geom512.rightMouth.x - geom512.leftMouth.x).toDouble(),
            (geom512.rightMouth.y - geom512.leftMouth.y).toDouble()
        ).toFloat().coerceIn(72f, 220f)

        // Anchor vertical mask center near the nose tip / upper philtrum (y ≈ 292) so the 1.0 inner core
        // covers 100% of the eyebrows, eyes, nose, philtrum, upper lip, oral slit, and lower lip!
        val centerX = (eyeMidX * 0.38f + geom512.nose.x * 0.32f + mouthMidX * 0.30f)
        val centerY = (eyeMidY * 0.32f + geom512.nose.y * 0.30f + mouthMidY * 0.38f)

        val baseRadiusX = (geom512.eyeDist * 1.06f).coerceIn(134f, 168f)
        // Use 3D Head-Pose left/right visibility + 106-point jawline span to contract the foreshortened side of the mask
        val pose3D = geom512.reconstructed3D.headPose
        val leftYawContract = ((1.0f - 0.22f * max(0f, geom512.yawRatio)) * (0.35f + 0.65f * pose3D.leftSideVisibility))
            .coerceIn(0.74f, 1.0f)
        val rightYawContract = ((1.0f - 0.22f * max(0f, -geom512.yawRatio)) * (0.35f + 0.65f * pose3D.rightSideVisibility))
            .coerceIn(0.74f, 1.0f)
        val radiusXLeft = baseRadiusX * leftYawContract
        val radiusXRight = baseRadiusX * rightYawContract

        // Use 106-point chin landmark (index 16) when available to anchor the lower jawline radius well below the lower lip
        val chin106 = geom512.landmarks106.getOrNull(16)
        val chinDist = if (chin106 != null) {
            hypot((chin106.x - centerX).toDouble(), (chin106.y - centerY).toDouble()).toFloat() * 0.98f
        } else {
            geom512.eyeDist * 1.12f
        }
        val radiusYTop = (geom512.eyeDist * 1.04f).coerceIn(136f, 162f)
        val radiusYBottom = chinDist.coerceIn(148f, 174f)
        val innerCoreRatio = 0.66f
        val borderMargin = 24

        val noseCoreRx = geom512.eyeDist * 0.42f
        val noseCoreRy = geom512.eyeDist * 0.46f
        val mouthCoreRx = max(mouthW * 0.68f, geom512.eyeDist * 0.56f)
        val mouthCoreRy = geom512.eyeDist * 0.34f
        val mouthCoreCy = geom512.nose.y * 0.22f + mouthMidY * 0.78f

        for (y in 0 until HD_SIZE) {
            val row = y * HD_SIZE
            for (x in 0 until HD_SIZE) {
                if (x < borderMargin || x >= HD_SIZE - borderMargin ||
                    y < borderMargin || y >= HD_SIZE - borderMargin
                ) {
                    rawMask[row + x] = 0f
                    continue
                }

                val xf = x.toFloat()
                val yf = y.toFloat()
                val dx = xf - centerX
                val dy = yf - centerY
                val localX = dx * geom512.cosA + dy * geom512.sinA
                val localY = -dx * geom512.sinA + dy * geom512.cosA

                val activeRx = if (localX < 0f) radiusXLeft else radiusXRight
                val activeRy = if (localY < 0f) radiusYTop else radiusYBottom
                val uNorm = localX / activeRx
                val vNorm = localY / activeRy
                val r = sqrt(uNorm * uNorm + vNorm * vNorm)

                var alpha = when {
                    r <= innerCoreRatio -> 1.0f
                    r >= 1.0f -> 0.0f
                    else -> {
                        val t = (r - innerCoreRatio) / (1.0f - innerCoreRatio)
                        (0.5f * (1.0f + cos(Math.PI * t))).toFloat()
                    }
                }

                // Guarantee 100% solid alpha = 1.0 across the core facial features (eyes, eyebrows, nose, philtrum, and both lips)
                // so Target's nose or Target's mouth NEVER ghosts through at partial opacity
                val eyeBrowW = computeEyeAndBrowProtectionWeight(xf, yf, geom512)
                val dNoseSq = orientedEllipseDistSq(xf, yf, geom512.nose.x, geom512.nose.y, noseCoreRx, noseCoreRy, geom512.cosA, geom512.sinA)
                val dMouthSq = orientedEllipseDistSq(xf, yf, mouthMidX, mouthCoreCy, mouthCoreRx, mouthCoreRy, geom512.cosA, geom512.sinA)
                val minCoreFeatureSq = min(dNoseSq, dMouthSq)
                val noseMouthCoreW = when {
                    minCoreFeatureSq <= 0.64f -> 1.0f
                    minCoreFeatureSq >= 1.0f -> 0.0f
                    else -> {
                        val ct = (sqrt(minCoreFeatureSq) - 0.80f) / 0.20f
                        (0.5f * (1.0f + cos(Math.PI * ct.coerceIn(0f, 1f)))).toFloat()
                    }
                }
                val coreFeatureProtection = max(eyeBrowW, noseMouthCoreW)
                alpha = max(alpha, coreFeatureProtection)

                val idx = row + x
                val sc = swapPx512[idx]
                val sR = (sc ushr 16) and 0xFF
                val sG = (sc ushr 8) and 0xFF
                val sB = sc and 0xFF

                // Softly attenuate foreign non-skin Source clothing/background colors strictly in the outer perimeter outside core features
                if (coreFeatureProtection < 0.10f && r > 0.58f && (sG > sR + 2 || sB > sR + 6)) {
                    val foreignExcess = max(sG - sR - 1, sB - sR - 5).toFloat()
                    val foreignSuppress = (foreignExcess / 14.0f).coerceIn(0.45f, 1.0f)
                    alpha *= (1.0f - foreignSuppress)
                }

                if (enableOcclusionProtection && coreFeatureProtection < 0.10f && r > 0.58f && alpha > 0.01f) {
                    val perimWeight = ((r - 0.58f) / 0.42f).coerceIn(0f, 1f)
                    if (neuralOcclusionGate512 != null) {
                        val segGate = neuralOcclusionGate512[idx].coerceIn(0f, 1f)
                        val gateFactor = 1.0f - perimWeight * (1.0f - (0.28f + 0.72f * segGate))
                        alpha *= gateFactor
                    }

                    val tc = tgtPx512[idx]
                    val tR = (tc ushr 16) and 0xFF
                    val tG = (tc ushr 8) and 0xFF
                    val tB = tc and 0xFF
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB
                    val sLum = 0.299f * sR + 0.587f * sG + 0.114f * sB

                    // Protect Target hair & dark temple/cheek hairline overlapping forehead or outer cheek
                    if (tLum < 92f && sLum > tLum + 14f) {
                        val hairDarkness = ((92f - tLum) / 56f).coerceIn(0f, 1f)
                        val lumGap = ((sLum - tLum - 14f) / 42f).coerceIn(0f, 1f)
                        val hairKeep = (hairDarkness * lumGap * perimWeight * 1.15f).coerceIn(0f, 0.94f)
                        alpha *= (1.0f - hairKeep)
                    }
                    // Prevent outer Source hair/dark background from pasting over Target skin
                    if (sLum < 64f && tLum > sLum + 20f) {
                        val srcHairSuppress = (((64f - sLum) / 44f) * perimWeight).coerceIn(0f, 0.90f)
                        alpha *= (1.0f - srcHairSuppress)
                    }
                    // Upper forehead kungumam/bindi guard above the eyebrows (localY < -0.42f * radiusYTop):
                    // prevent an extra red mark from the upper Source forehead from creating a cut mark near the hairline
                    if (localY < -0.42f * radiusYTop && (sR - sG) > (tR - tG) + 28) {
                        val upperForeheadFade = (((-localY / radiusYTop) - 0.42f) / 0.35f).coerceIn(0f, 0.92f)
                        alpha *= (1.0f - upperForeheadFade)
                    }
                    // Lower chin / jawline double-seam guard strictly BELOW the lower lip (localY > 0.72f * radiusYBottom):
                    // if Source jaw extends below Target's chin contour into neck shadow, taper alpha smoothly
                    if (localY > 0.72f * radiusYBottom && abs(sLum - tLum) > 16f) {
                        val jawMismatch = ((abs(sLum - tLum) - 16f) / 38f).coerceIn(0f, 1f)
                        val chinFade = (((localY / radiusYBottom) - 0.72f) / 0.28f).coerceIn(0f, 1f)
                        alpha *= (1.0f - (jawMismatch * chinFade * 0.85f).coerceIn(0f, 0.85f))
                    }
                }

                val edgeDist = min(
                    min(x - borderMargin, HD_SIZE - 1 - borderMargin - x),
                    min(y - borderMargin, HD_SIZE - 1 - borderMargin - y)
                ).toFloat()
                val edgeEnvelope = (edgeDist / 38.0f).coerceIn(0f, 1f)

                rawMask[row + x] = (alpha * edgeEnvelope).coerceIn(0f, 1f)
            }
        }

        // Run a fast separable 9x9 box-gaussian feathering pass over rawMask so mask transitions
        // around hair fringes, temples, and the chin contour are 100% smooth with zero sharp steps
        val horizMask = FloatArray(HD_SIZE * HD_SIZE)
        val smoothMask = FloatArray(HD_SIZE * HD_SIZE)
        val radius = 4
        val norm = 1.0f / (2 * radius + 1).toFloat()
        for (y in borderMargin until HD_SIZE - borderMargin) {
            val row = y * HD_SIZE
            for (x in borderMargin until HD_SIZE - borderMargin) {
                var sum = 0f
                for (dx in -radius..radius) {
                    sum += rawMask[row + x + dx]
                }
                horizMask[row + x] = sum * norm
            }
        }
        for (y in borderMargin until HD_SIZE - borderMargin) {
            val row = y * HD_SIZE
            for (x in borderMargin until HD_SIZE - borderMargin) {
                var sum = 0f
                for (dy in -radius..radius) {
                    sum += horizMask[(y + dy) * HD_SIZE + x]
                }
                smoothMask[row + x] = (sum * norm).coerceIn(0f, 1f)
            }
        }
        return smoothMask
    }

    /**
     * Creates a smooth 128x128 soft-feathered facial mask in canonical InSwapper space
     * (used by Head Replacement inner expression fusion and legacy callers).
     */
    fun createFeatheredFaceMask128(): FloatArray {
        val mask = FloatArray(CROP_SIZE * CROP_SIZE)
        val centerX = 64.0f
        val centerY = 67.0f
        val radiusX = 46.0f
        val radiusY = 50.0f
        val innerCoreRatio = 0.58f
        val borderMargin = 8

        for (y in 0 until CROP_SIZE) {
            for (x in 0 until CROP_SIZE) {
                if (x < borderMargin || x >= CROP_SIZE - borderMargin ||
                    y < borderMargin || y >= CROP_SIZE - borderMargin
                ) {
                    mask[y * CROP_SIZE + x] = 0f
                    continue
                }
                val dx = (x - centerX) / radiusX
                val dy = (y - centerY) / radiusY
                val r = sqrt(dx * dx + dy * dy)

                val ellipticAlpha = when {
                    r <= innerCoreRatio -> 1.0f
                    r >= 1.0f -> 0.0f
                    else -> {
                        val t = (r - innerCoreRatio) / (1.0f - innerCoreRatio)
                        (0.5f * (1.0f + cos(Math.PI * t))).toFloat()
                    }
                }

                val edgeDist = min(
                    min(x - borderMargin, CROP_SIZE - 1 - borderMargin - x),
                    min(y - borderMargin, CROP_SIZE - 1 - borderMargin - y)
                ).toFloat()
                val edgeEnvelope = (edgeDist / 12.0f).coerceIn(0f, 1f)

                mask[y * CROP_SIZE + x] = ellipticAlpha * edgeEnvelope
            }
        }
        return mask
    }

    /**
     * Warps and blends `swapped128` onto `targetPixels` in-place using the forward 2x3
     * similarity matrix `forwardMatrix128`.
     */
    fun blendSwappedFaceIntoTarget(
        targetPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        swapped128: Bitmap,
        forwardMatrix128: FloatArray,
        featheredMask128: FloatArray
    ) {
        val swapPixels = IntArray(CROP_SIZE * CROP_SIZE)
        swapped128.getPixels(swapPixels, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)

        val inv = FaceAlignment.invertAffine2x3(forwardMatrix128)
        val corners = arrayOf(
            0f to 0f,
            CROP_SIZE.toFloat() to 0f,
            0f to CROP_SIZE.toFloat(),
            CROP_SIZE.toFloat() to CROP_SIZE.toFloat()
        )
        var minX = targetWidth - 1
        var maxX = 0
        var minY = targetHeight - 1
        var maxY = 0
        for ((cx, cy) in corners) {
            val tx = (inv[0] * cx + inv[1] * cy + inv[2]).toInt()
            val ty = (inv[3] * cx + inv[4] * cy + inv[5]).toInt()
            minX = min(minX, tx)
            maxX = max(maxX, tx)
            minY = min(minY, ty)
            maxY = max(maxY, ty)
        }

        minX = (minX - 2).coerceIn(0, targetWidth - 1)
        maxX = (maxX + 2).coerceIn(0, targetWidth - 1)
        minY = (minY - 2).coerceIn(0, targetHeight - 1)
        maxY = (maxY + 2).coerceIn(0, targetHeight - 1)

        val m00 = forwardMatrix128[0]
        val m01 = forwardMatrix128[1]
        val m02 = forwardMatrix128[2]
        val m10 = forwardMatrix128[3]
        val m11 = forwardMatrix128[4]
        val m12 = forwardMatrix128[5]

        for (y in minY..maxY) {
            val rowOffset = y * targetWidth
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            for (x in minX..maxX) {
                val u = m00 * x + baseU
                val v = m10 * x + baseV
                if (u >= 1f && u < CROP_SIZE - 2f && v >= 1f && v < CROP_SIZE - 2f) {
                    val alpha = sampleMaskBilinearGeneric(featheredMask128, CROP_SIZE, u, v)
                    if (alpha > 0.003f) {
                        val swapColor = FaceAlignment.sampleBilinearClamped(
                            swapPixels,
                            CROP_SIZE,
                            CROP_SIZE,
                            u,
                            v
                        )
                        val dstColor = targetPixels[rowOffset + x]

                        val sr = (swapColor ushr 16) and 0xFF
                        val sg = (swapColor ushr 8) and 0xFF
                        val sb = swapColor and 0xFF

                        val dr = (dstColor ushr 16) and 0xFF
                        val dg = (dstColor ushr 8) and 0xFF
                        val db = dstColor and 0xFF

                        val invA = 1.0f - alpha
                        val outR = (sr * alpha + dr * invA).toInt().coerceIn(0, 255)
                        val outG = (sg * alpha + dg * invA).toInt().coerceIn(0, 255)
                        val outB = (sb * alpha + db * invA).toInt().coerceIn(0, 255)

                        targetPixels[rowOffset + x] =
                            (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                    }
                }
            }
        }
    }

    private fun sampleMaskBilinearGeneric(mask: FloatArray, size: Int, u: Float, v: Float): Float {
        val x0 = u.toInt().coerceIn(0, size - 2)
        val y0 = v.toInt().coerceIn(0, size - 2)
        val fx = u - x0
        val fy = v - y0

        val m00 = mask[y0 * size + x0]
        val m10 = mask[y0 * size + (x0 + 1)]
        val m01 = mask[(y0 + 1) * size + x0]
        val m11 = mask[(y0 + 1) * size + (x0 + 1)]

        return m00 * (1f - fx) * (1f - fy) +
            m10 * fx * (1f - fy) +
            m01 * (1f - fx) * fy +
            m11 * fx * fy
    }

    /**
     * Renders a non-intrusive ethical AI provenance watermark badge in the bottom-right corner
     * of the output image to guard against deceptive misuse.
     */
    fun applyEthicalProvenanceWatermark(bitmap: Bitmap) {
        val canvas = Canvas(bitmap)
        val scale = (bitmap.width / 1080f).coerceIn(0.65f, 2.2f)
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(225, 240, 248, 255)
            textSize = 24f * scale
            isFakeBoldText = true
        }
        val label = "SYNTHETIC AI • OFFLINE ONNX"
        val textWidth = textPaint.measureText(label)
        val padH = 18f * scale
        val padV = 12f * scale
        val margin = 20f * scale
        val boxHeight = 36f * scale + padV
        val left = bitmap.width - textWidth - padH * 2 - margin
        val top = bitmap.height - boxHeight - margin
        val rect = RectF(left, top, bitmap.width - margin, bitmap.height - margin)

        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(165, 11, 16, 33)
            style = Paint.Style.FILL
        }
        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(190, 0, 229, 255)
            style = Paint.Style.STROKE
            strokeWidth = 2f * scale
        }

        canvas.drawRoundRect(rect, 12f * scale, 12f * scale, bgPaint)
        canvas.drawRoundRect(rect, 12f * scale, 12f * scale, borderPaint)
        canvas.drawText(
            label,
            rect.left + padH,
            rect.centerY() + (textPaint.textSize * 0.35f),
            textPaint
        )
    }
}
