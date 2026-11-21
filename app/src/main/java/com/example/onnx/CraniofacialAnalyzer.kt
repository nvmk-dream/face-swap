package com.example.onnx

import android.graphics.Bitmap
import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Craniofacial anthropometric measurements combining all three clinical/3D diagrams:
 *  - Diagram 1 (Proportion Grid): Cheekbone = 1.0, Square Jaw (~0.7), Lower Jaw (~0.4), Vertical Thirds (1 : 1 : 0.8~1)
 *  - Diagram 2 (NIOSH/BSI Anthropometry A–P): O (Interpupillary), B (Bizygomatic), C (Bigonial), D (Face length),
 *    E (Lower face length), K (Bitragion-menton arc), L (Bitragion-subnasale arc), M (Nose protrusion), P (Face depth)
 *  - Diagram 3 (3D Landmark Topology 0..11): Forehead width (8-9), Face width (1-2), Jaw width (6-7),
 *    Chin-to-mouth distance (4-5), Face height (3-5)
 */
data class CraniofacialProfile(
    val cheekWidthNormalized: Float,     // Reference = 1.0f (Diagram 1 Cheekbone / Diagram 2 B / Diagram 3 1-2)
    val squareJawRatio: Float,           // Square jaw (6-7 / C) / Cheekbone (1-2 / B), canonical ~0.70f
    val lowerJawRatio: Float,            // Lower chin width / Cheekbone, canonical ~0.40f
    val foreheadWidthRatio: Float,       // Forehead (8-9) / Cheekbone (1-2), canonical ~0.82f
    val upperThirdRatio: Float,          // Diagram 1 Top 1/3 normalized (= 1.0f reference)
    val midThirdRatio: Float,            // Diagram 1 Middle 1/3 relative to upper third (~1.0f)
    val lowerThirdRatio: Float,          // Diagram 1 Lower 1/3 relative to upper third (~0.8f..1.0f)
    val interpupillaryDistO: Float,      // Diagram 2 O (px)
    val bizygomaticWidthB: Float,        // Diagram 2 B / Diagram 3 (1-2) in px
    val bigonialJawWidthC: Float,        // Diagram 2 C / Diagram 3 (6-7) in px
    val faceLengthD: Float,              // Diagram 2 D / Diagram 3 (3-5) in px
    val lowerFaceLengthE: Float,         // Diagram 2 E (Subnasale-Menton) in px
    val bitragionChinArcK: Float,        // Diagram 2 K (3D Ear-to-Chin arc length) / O
    val bitragionSubnasaleArcL: Float,   // Diagram 2 L (3D Ear-to-Subnasale arc length) / O
    val noseProtrusionM: Float,          // Diagram 2 M (3D Nose Z protrusion) / O
    val faceDepthP: Float,               // Diagram 2 P (3D Cranial Depth Span) / O
    val chinToMouthRatio: Float,         // Diagram 3 (4-5) / Face height (3-5)
    val yawDeg: Float,
    val pitchDeg: Float,
    val rollDeg: Float,
    val mouthOpenRatio: Float,
    val smileCurvatureCoeff: Float
)

data class OptionDCraniofacialFitRatios(
    val sourceCheekNormalized: Float = 1.0f,
    val sourceSquareJawRatio: Float = 0.70f,
    val sourceLowerJawRatio: Float = 0.40f,
    val sourceForeheadRatio: Float = 0.82f,
    val sourceJawToTargetJawScale: Float = 1.0f,
    val sourceChinToTargetChinScale: Float = 1.0f,
    val klArcNeckBlendStrength: Float = 0.28f
)

data class CraniofacialPairReport(
    val sourceProfile: CraniofacialProfile,
    val targetProfile: CraniofacialProfile,
    val overallCompatibilityPct: Int,
    val jawMatchSummary: String,
    val foreheadThirdsSummary: String,
    val klArcSummary: String,
    val poseDeltaSummary: String,
    val reactionSyncSummary: String,
    val diagnosticWarnings: List<String>,
    val optionDFitRatios: OptionDCraniofacialFitRatios
)

object CraniofacialAnalyzer {

    fun extractProfile(face: DetectedFace, bitmap: Bitmap? = null): CraniofacialProfile {
        val recon = face.reconstructed3D
        val lm5 = face.landmarks5
        val pts106 = recon.landmarks106
        val verts3D = recon.vertices3D

        val lEye = lm5.getOrNull(0) ?: PointF(92f, 102f)
        val rEye = lm5.getOrNull(1) ?: PointF(162f, 102f)
        val nose = lm5.getOrNull(2) ?: PointF(127f, 144f)
        val lMouth = lm5.getOrNull(3) ?: PointF(99f, 184f)
        val rMouth = lm5.getOrNull(4) ?: PointF(157f, 184f)

        val interpupillaryO = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble())
            .toFloat()
            .coerceAtLeast(16f)

        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val mouthMidX = (lMouth.x + rMouth.x) * 0.5f
        val mouthMidY = (lMouth.y + rMouth.y) * 0.5f

        fun dist2D(i: Int, j: Int, fallback: Float): Float {
            val a = pts106.getOrNull(i) ?: return fallback
            val b = pts106.getOrNull(j) ?: return fallback
            return hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble()).toFloat().coerceAtLeast(8f)
        }

        fun dist3D(i: Int, j: Int, fallback: Float): Float {
            val a = verts3D.getOrNull(i) ?: return fallback
            val b = verts3D.getOrNull(j) ?: return fallback
            val dx = (b.x - a.x).toDouble()
            val dy = (b.y - a.y).toDouble()
            val dz = (b.z - a.z).toDouble()
            return kotlin.math.sqrt(dx * dx + dy * dy + dz * dz).toFloat().coerceAtLeast(8f)
        }

        // Bizygomatic width (1-2 / B): upper cheek contour landmarks [2]..[30], modulated by bounding box if non-trivial
        val landmarkCheekW = dist2D(2, 30, interpupillaryO * 2.04f)
        val boxCheekW = face.boundingBox.width().takeIf { it > interpupillaryO * 1.4f } ?: landmarkCheekW
        val bizygomaticB = (landmarkCheekW * 0.68f + boxCheekW * 0.32f).coerceIn(interpupillaryO * 1.55f, interpupillaryO * 2.85f)

        // Square Jaw width (6-7 / C): bigonial landmarks [7]..[25] scaled to realistic anatomical ~0.70 envelope
        val rawGonialSpan = dist2D(7, 25, bizygomaticB * 0.72f)
        val bigonialJawC = (rawGonialSpan * 0.88f).coerceIn(bizygomaticB * 0.54f, bizygomaticB * 0.92f)
        val squareJawRatio = (bigonialJawC / bizygomaticB).coerceIn(0.54f, 0.92f)

        // Lower Chin width (Diagram 1 0.40): landmarks [12]..[20] near menton curve
        val lowerChinSpan = dist2D(12, 20, bizygomaticB * 0.41f)
        val lowerJawRatio = (lowerChinSpan / bizygomaticB).coerceIn(0.28f, 0.62f)

        // Forehead width (Diagram 3 8-9): temple boundary [0]..[32] slightly narrower than cheekbones
        val foreheadPx = (dist2D(0, 32, bizygomaticB * 0.86f) * 0.85f)
            .coerceIn(bizygomaticB * 0.66f, bizygomaticB * 0.96f)
        val foreheadWidthRatio = (foreheadPx / bizygomaticB).coerceIn(0.66f, 0.96f)

        // Vertical thirds (Diagram 1: Top 1 : Mid 1 : Bottom 0.8~1) & Diagram 2 D, E
        val browMidY = (
            (pts106.getOrNull(35)?.y ?: (eyeMidY - interpupillaryO * 0.27f)) +
                (pts106.getOrNull(45)?.y ?: (eyeMidY - interpupillaryO * 0.27f))
            ) * 0.5f
        val subnasalePt = pts106.getOrNull(81) ?: PointF(nose.x, nose.y + interpupillaryO * 0.08f)
        val chinTipPt = pts106.getOrNull(16) ?: PointF(mouthMidX, mouthMidY + interpupillaryO * 0.62f)

        val midThirdPx = abs(subnasalePt.y - browMidY).coerceAtLeast(interpupillaryO * 0.45f)
        val lowerThirdPx = hypot((chinTipPt.x - subnasalePt.x).toDouble(), (chinTipPt.y - subnasalePt.y).toDouble())
            .toFloat()
            .coerceAtLeast(interpupillaryO * 0.42f)
        val upperThirdPx = if (face.boundingBox.height() > (midThirdPx + lowerThirdPx) * 1.15f) {
            (face.boundingBox.height() - midThirdPx - lowerThirdPx).coerceIn(midThirdPx * 0.72f, midThirdPx * 1.32f)
        } else {
            midThirdPx
        }
        val midThirdRatio = (midThirdPx / upperThirdPx).coerceIn(0.72f, 1.32f)
        val lowerThirdRatio = (lowerThirdPx / upperThirdPx).coerceIn(0.65f, 1.28f)

        // Diagram 2: D (Nasion-Menton), E (Subnasale-Menton), K (Bitragion-Menton 3D arc), L (Bitragion-Subnasale 3D arc)
        val faceLengthD = (upperThirdPx + midThirdPx + lowerThirdPx).coerceAtLeast(interpupillaryO * 1.6f)
        val kLeftArc = dist3D(0, 16, bizygomaticB * 0.92f)
        val kRightArc = dist3D(32, 16, bizygomaticB * 0.92f)
        val bitragionChinArcK = ((kLeftArc + kRightArc) / interpupillaryO).coerceIn(1.6f, 4.4f)

        val lLeftArc = dist3D(0, 81, bizygomaticB * 0.75f)
        val lRightArc = dist3D(32, 81, bizygomaticB * 0.75f)
        val bitragionSubnasaleArcL = ((lLeftArc + lRightArc) / interpupillaryO).coerceIn(1.3f, 3.8f)

        val noseZ = verts3D.getOrNull(76)?.z ?: (interpupillaryO * 0.44f)
        val alarZ = verts3D.getOrNull(81)?.z ?: (interpupillaryO * 0.24f)
        val templeZ = ((verts3D.getOrNull(0)?.z ?: -interpupillaryO * 0.55f) +
            (verts3D.getOrNull(32)?.z ?: -interpupillaryO * 0.55f)) * 0.5f

        val noseProtrusionM = ((noseZ - alarZ) / interpupillaryO).coerceIn(0.08f, 0.55f)
        val faceDepthP = ((noseZ - templeZ) / interpupillaryO).coerceIn(0.45f, 1.65f)
        val chinMouthPx = hypot((chinTipPt.x - mouthMidX).toDouble(), (chinTipPt.y - mouthMidY).toDouble()).toFloat()
        val chinToMouthRatio = (chinMouthPx / faceLengthD).coerceIn(0.10f, 0.34f)

        // Touch bitmap bounds safely only as optional sanity clamp
        val _hasBitmap = bitmap != null && !bitmap.isRecycled

        return CraniofacialProfile(
            cheekWidthNormalized = 1.0f,
            squareJawRatio = squareJawRatio,
            lowerJawRatio = lowerJawRatio,
            foreheadWidthRatio = foreheadWidthRatio,
            upperThirdRatio = 1.0f,
            midThirdRatio = midThirdRatio,
            lowerThirdRatio = lowerThirdRatio,
            interpupillaryDistO = interpupillaryO,
            bizygomaticWidthB = bizygomaticB,
            bigonialJawWidthC = bigonialJawC,
            faceLengthD = faceLengthD,
            lowerFaceLengthE = lowerThirdPx,
            bitragionChinArcK = bitragionChinArcK,
            bitragionSubnasaleArcL = bitragionSubnasaleArcL,
            noseProtrusionM = noseProtrusionM,
            faceDepthP = faceDepthP,
            chinToMouthRatio = chinToMouthRatio,
            yawDeg = recon.headPose.yawDeg,
            pitchDeg = recon.headPose.pitchDeg,
            rollDeg = recon.headPose.rollDeg,
            mouthOpenRatio = recon.mouthOpenRatio,
            smileCurvatureCoeff = recon.smileCurvatureCoeff
        )
    }

    fun analyzePair(
        sourceFace: DetectedFace,
        sourceBitmap: Bitmap?,
        targetFace: DetectedFace,
        targetBitmap: Bitmap?
    ): CraniofacialPairReport {
        val src = extractProfile(sourceFace, sourceBitmap)
        val tgt = extractProfile(targetFace, targetBitmap)

        val yawDiff = abs(src.yawDeg - tgt.yawDeg)
        val pitchDiff = abs(src.pitchDeg - tgt.pitchDeg)
        val rollDiff = abs(src.rollDeg - tgt.rollDeg)

        val jawDeltaPct = abs(src.squareJawRatio - tgt.squareJawRatio) * 100f
        val chinDeltaPct = abs(src.lowerThirdRatio - tgt.lowerThirdRatio) * 100f
        val foreheadDeltaPct = abs(src.foreheadWidthRatio - tgt.foreheadWidthRatio) * 100f
        val klArcDelta = abs(src.bitragionChinArcK - tgt.bitragionChinArcK) /
            max(src.bitragionChinArcK, tgt.bitragionChinArcK).coerceAtLeast(0.1f)

        val penalty = (yawDiff * 0.72f) +
            (pitchDiff * 0.65f) +
            (rollDiff * 0.25f) +
            (jawDeltaPct * 0.55f) +
            (chinDeltaPct * 0.40f) +
            (foreheadDeltaPct * 0.35f) +
            (klArcDelta * 35f)
        val compatibilityPct = (100f - penalty).roundToInt().coerceIn(35, 99)

        val warnings = mutableListOf<String>()
        if (yawDiff > 16f) {
            warnings += "தலைத் திருப்பம் (Yaw Δ ${yawDiff.roundToInt()}°) அதிகம் — பக்கவாட்டு கன்ன ஓரத்தில் சற்று மென்மை குறையலாம்."
        }
        if (pitchDiff > 16f) {
            warnings += "மேல்/கீழ் கோணம் (Pitch Δ ${pitchDiff.roundToInt()}°) வேறுபாடு — நாடி (Chin) உயரத்தை மென்மையாகச் சரிசெய்கிறது."
        }
        if (jawDeltaPct > 10f) {
            warnings += "தாடை அகல வேறுபாடு (${"%.0f".format(jawDeltaPct)}%) — Option D Source வடிவத்தை வைத்து K/L கீழ் ஓரம் மட்டும் கழுத்துடன் இணைக்கும்."
        }
        if (klArcDelta > 0.16f) {
            warnings += "காது-நாடி வளைவு (K/L Arc Δ ${(klArcDelta * 100).roundToInt()}%) — கழுத்துப் பொருத்தத்திற்கு soft transition பயன்படுத்தப்படுகிறது."
        }

        val srcToTgtJawScale = (src.squareJawRatio / tgt.squareJawRatio.coerceAtLeast(0.4f)).coerceIn(0.82f, 1.22f)
        val srcToTgtChinScale = (src.lowerThirdRatio / tgt.lowerThirdRatio.coerceAtLeast(0.4f)).coerceIn(0.84f, 1.18f)
        val klNeckBlend = (0.24f + klArcDelta * 1.15f).coerceIn(0.24f, 0.58f)

        val optionDRatios = OptionDCraniofacialFitRatios(
            sourceCheekNormalized = src.cheekWidthNormalized,
            sourceSquareJawRatio = src.squareJawRatio,
            sourceLowerJawRatio = src.lowerJawRatio,
            sourceForeheadRatio = src.foreheadWidthRatio,
            sourceJawToTargetJawScale = srcToTgtJawScale,
            sourceChinToTargetChinScale = srcToTgtChinScale,
            klArcNeckBlendStrength = klNeckBlend
        )

        val jawSummary = "Source 1 : ${"%.2f".format(src.squareJawRatio)} : ${"%.2f".format(src.lowerJawRatio)} " +
            "(Tgt ${"%.2f".format(tgt.squareJawRatio)})"
        val foreheadSummary = "Forehead 8-9: ${(src.foreheadWidthRatio * 100).roundToInt()}% • " +
            "Thirds 1 : ${"%.2f".format(src.midThirdRatio)} : ${"%.2f".format(src.lowerThirdRatio)}"
        val klSummary = "K-Arc ${"%.2f".format(src.bitragionChinArcK)} vs ${"%.2f".format(tgt.bitragionChinArcK)} • " +
            "L-Arc ${"%.2f".format(src.bitragionSubnasaleArcL)} • P ${"%.2f".format(src.faceDepthP)}"
        val poseSummary = "Yaw Δ ${"%.0f".format(yawDiff)}° • Pitch Δ ${"%.0f".format(pitchDiff)}° • Roll Δ ${"%.0f".format(rollDiff)}°"
        val reactionSummary = "Target Reaction Sync • Mouth Open ${"%.2f".format(tgt.mouthOpenRatio)} • " +
            "Smile ${"%.2f".format(tgt.smileCurvatureCoeff)}"

        return CraniofacialPairReport(
            sourceProfile = src,
            targetProfile = tgt,
            overallCompatibilityPct = compatibilityPct,
            jawMatchSummary = jawSummary,
            foreheadThirdsSummary = foreheadSummary,
            klArcSummary = klSummary,
            poseDeltaSummary = poseSummary,
            reactionSyncSummary = reactionSummary,
            diagnosticWarnings = warnings,
            optionDFitRatios = optionDRatios
        )
    }
}
