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
    val klArcNeckBlendStrength: Float = 0.28f,
    val upperJawMaxillaScale: Float = 1.0f,    // Region D: Upper jaw / maxilla width ratio
    val jawBoneBigonialScale: Float = 1.0f,    // Region B: Bigonial jaw bone width ratio
    val lowerJawBodyScale: Float = 1.0f,       // Region C: Lower jaw mandibular body width ratio
    val mandibularLengthScale: Float = 1.0f    // Region A: Ear -> Gonion -> Chin length ratio
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
    val optionDFitRatios: OptionDCraniofacialFitRatios,
    val mandibularSummary: String = "D UpperJaw ×1.00 • B JawBone ×1.00 • C LowerJaw ×1.00 • A Mandible ×1.00"
)

object CraniofacialAnalyzer {

    fun extractProfile(face: DetectedFace, bitmap: Bitmap? = null): CraniofacialProfile {
        val recon = face.reconstructed3D
        val lm5 = face.landmarks5
        val pts106 = recon.landmarks106
        val verts3D = recon.vertices3D
        val contour = face.faceContourPoints

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

        val isSyntheticFullImageBox = bitmap != null &&
            face.boundingBox.left <= 2f &&
            face.boundingBox.top <= 2f &&
            face.boundingBox.width() >= bitmap.width * 0.90f &&
            face.boundingBox.height() >= bitmap.height * 0.90f

        fun dist2D(i: Int, j: Int, fallback: Float): Float {
            val a = pts106.getOrNull(i) ?: return fallback
            val b = pts106.getOrNull(j) ?: return fallback
            return hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble()).toFloat().coerceAtLeast(8f)
        }

        fun distContourFrac(fracFromTop: Float, fallback: Float): Float {
            if (contour.size < 24) return fallback
            // ML Kit / face contour typically runs clockwise or symmetric along the perimeter;
            // measure widest horizontal chord near target vertical slice [0.0..1.0].
            var minY = Float.MAX_VALUE
            var maxY = -Float.MAX_VALUE
            for (pt in contour) {
                if (pt.y < minY) minY = pt.y
                if (pt.y > maxY) maxY = pt.y
            }
            val height = (maxY - minY).coerceAtLeast(interpupillaryO)
            val targetY = minY + height * fracFromTop.coerceIn(0.05f, 0.95f)
            val tol = height * 0.12f
            var minX = Float.MAX_VALUE
            var maxX = -Float.MAX_VALUE
            var hits = 0
            for (pt in contour) {
                if (abs(pt.y - targetY) <= tol) {
                    if (pt.x < minX) minX = pt.x
                    if (pt.x > maxX) maxX = pt.x
                    hits++
                }
            }
            return if (hits >= 2 && maxX > minX + 8f) (maxX - minX) else fallback
        }

        fun dist3D(i: Int, j: Int, fallback: Float): Float {
            val a = verts3D.getOrNull(i) ?: return fallback
            val b = verts3D.getOrNull(j) ?: return fallback
            val dx = (b.x - a.x).toDouble()
            val dy = (b.y - a.y).toDouble()
            val dz = (b.z - a.z).toDouble()
            return kotlin.math.sqrt(dx * dx + dy * dy + dz * dz).toFloat().coerceAtLeast(8f)
        }

        // Bizygomatic width (1-2 / B): combine upper cheek landmarks [2]..[30], real oval contour, and non-synthetic box
        val landmarkCheekW = dist2D(2, 30, interpupillaryO * 2.04f)
        val contourCheekW = distContourFrac(0.52f, landmarkCheekW)
        val rawBoxCheekW = if (!isSyntheticFullImageBox &&
            face.boundingBox.width() in (interpupillaryO * 1.45f)..(interpupillaryO * 3.10f)
        ) {
            face.boundingBox.width()
        } else {
            contourCheekW
        }
        val bizygomaticB = (landmarkCheekW * 0.38f + contourCheekW * 0.38f + rawBoxCheekW * 0.24f)
            .coerceIn(interpupillaryO * 1.55f, interpupillaryO * 2.95f)

        // Square Jaw / Jaw Bone width (6-7 / B): bigonial landmarks [7]..[25] blended with contour lower-mid chord
        val landmarkGonialSpan = dist2D(7, 25, bizygomaticB * 0.72f)
        val contourGonialSpan = distContourFrac(0.73f, landmarkGonialSpan)
        val bigonialJawC = ((landmarkGonialSpan * 0.42f + contourGonialSpan * 0.58f) * 0.90f)
            .coerceIn(bizygomaticB * 0.52f, bizygomaticB * 0.95f)
        val squareJawRatio = (bigonialJawC / bizygomaticB).coerceIn(0.52f, 0.95f)

        // Lower Jaw / Chin width (Diagram 1 0.40 & Diagram 3 C): landmarks [12]..[20] blended with lower contour
        val landmarkChinSpan = dist2D(12, 20, bizygomaticB * 0.41f)
        val contourChinSpan = distContourFrac(0.87f, landmarkChinSpan)
        val lowerChinSpan = (landmarkChinSpan * 0.40f + contourChinSpan * 0.60f)
        val lowerJawRatio = (lowerChinSpan / bizygomaticB).coerceIn(0.28f, 0.68f)

        // Forehead width (Diagram 3 8-9): temple boundary [0]..[32] slightly narrower than cheekbones
        val contourForeheadW = distContourFrac(0.24f, bizygomaticB * 0.84f)
        val foreheadPx = (dist2D(0, 32, bizygomaticB * 0.86f) * 0.45f + contourForeheadW * 0.55f)
            .coerceIn(bizygomaticB * 0.64f, bizygomaticB * 0.98f)
        val foreheadWidthRatio = (foreheadPx / bizygomaticB).coerceIn(0.64f, 0.98f)

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
        val upperThirdPx = if (!isSyntheticFullImageBox &&
            face.boundingBox.height() > (midThirdPx + lowerThirdPx) * 1.15f &&
            face.boundingBox.height() < interpupillaryO * 4.5f
        ) {
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

    /**
     * Measures real pixel-level mandibular contours across aligned 512×512 Source & Target crops
     * for all 4 jaw/mandible parameters requested:
     *  - D: Upper jaw / maxilla width (`nose.y + eyeDist * 0.16`)
     *  - B: Bigonial jaw bone width (`mouthMidY + eyeDist * 0.04`)
     *  - C: Lower jaw mandibular body width (`mouthMidY + eyeDist * 0.28`)
     *  - A: Mandibular length (`Ear/Gonion -> Menton chin depth` + vertical chin drop)
     */
    fun computeMandibularFitRatios512(
        srcPx512: IntArray,
        tgtPx512: IntArray,
        geom512: FaceBlender.WarpedFaceGeometry,
        baseRatios: OptionDCraniofacialFitRatios
    ): OptionDCraniofacialFitRatios {
        val size = 512
        if (srcPx512.size < size * size || tgtPx512.size < size * size) return baseRatios

        val eyeMidX = (geom512.leftEye.x + geom512.rightEye.x) * 0.5f
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val centerX = (eyeMidX * 0.38f + geom512.nose.x * 0.32f + mouthMidX * 0.30f)
        val eyeDist = geom512.eyeDist.coerceIn(72f, 240f)
        val cx = centerX.toInt().coerceIn(28, size - 29)

        data class SkinRef(val r: Float, val g: Float, val b: Float, val lum: Float)

        fun sampleInnerCheekRef(px: IntArray): SkinRef {
            val rx1 = (cx - (eyeDist * 0.36f).toInt()).coerceIn(16, size - 17)
            val rx2 = (cx + (eyeDist * 0.36f).toInt()).coerceIn(16, size - 17)
            val ry = geom512.nose.y.toInt().coerceIn(16, size - 17)
            val c1 = px[ry * size + rx1]
            val c2 = px[ry * size + rx2]
            val r = (((c1 ushr 16) and 0xFF) + ((c2 ushr 16) and 0xFF)) * 0.5f
            val g = (((c1 ushr 8) and 0xFF) + ((c2 ushr 8) and 0xFF)) * 0.5f
            val b = ((c1 and 0xFF) + (c2 and 0xFF)) * 0.5f
            val lum = 0.299f * r + 0.587f * g + 0.114f * b
            return SkinRef(r, g, b, lum)
        }

        fun scanHorizontalSpan(px: IntArray, rowY: Int, minOffsetRatio: Float, maxOffsetRatio: Float): Pair<Float, Boolean> {
            val ref = sampleInnerCheekRef(px)
            if (ref.lum < 22f) return eyeDist * (minOffsetRatio + 0.35f) to false
            val refWarmth = ref.r - ref.b
            val refChroma = max(ref.r, max(ref.g, ref.b)) - min(ref.r, min(ref.g, ref.b))
            val y = rowY.coerceIn(16, size - 17)
            val minOff = (eyeDist * minOffsetRatio).toInt().coerceAtLeast(16)
            val maxOff = (eyeDist * maxOffsetRatio).toInt().coerceAtLeast(minOff + 8)
            var hitBorder = false

            fun scanDir(step: Int): Int {
                var lastSkin = minOff
                var missRun = 0
                var d = minOff
                while (d <= maxOff) {
                    val xx = cx + step * d
                    if (xx < 8 || xx >= size - 8) {
                        hitBorder = true
                        break
                    }
                    val pc = px[y * size + xx]
                    val r = ((pc ushr 16) and 0xFF).toFloat()
                    val g = ((pc ushr 8) and 0xFF).toFloat()
                    val b = (pc and 0xFF).toFloat()
                    val lum = 0.299f * r + 0.587f * g + 0.114f * b
                    val diff = abs(r - ref.r) + abs(g - ref.g) + abs(b - ref.b)
                    val warmth = r - b
                    val chroma = max(r, max(g, b)) - min(r, min(g, b))
                    val isDarkHairOrBg = lum < 64f && ref.lum > 84f
                    val isDeepJawCrease = lum < ref.lum * 0.58f && diff > 52f
                    // Reject grey/white/cool studio backgrounds (e.g. passport backdrop R≈G≈B)
                    // where luminance resembles skin but warmth & skin chroma vanish.
                    val isNeutralOrCoolBg = (refWarmth >= 18f && warmth < max(8f, refWarmth * 0.28f)) ||
                        (refChroma >= 22f && chroma < refChroma * 0.32f && diff > 36f) ||
                        (g > r + 6f)
                    val isSkinLike = !isDarkHairOrBg && !isDeepJawCrease && !isNeutralOrCoolBg &&
                        diff < 118f && r >= b - 8f
                    if (isSkinLike) {
                        lastSkin = d
                        missRun = 0
                    } else {
                        missRun++
                        if (missRun >= 3) break
                    }
                    d++
                }
                return lastSkin
            }
            return (scanDir(-1) + scanDir(1)).toFloat() to hitBorder
        }

        val chinStartY = (mouthMidY + eyeDist * 0.14f).toInt().coerceIn(24, size - 48)
        fun estimateChinDrop(px: IntArray): Float {
            val ref = sampleInnerCheekRef(px)
            if (ref.lum < 22f) return eyeDist * 0.52f
            val refWarmth = ref.r - ref.b
            val refChroma = max(ref.r, max(ref.g, ref.b)) - min(ref.r, min(ref.g, ref.b))
            val xOffsets = intArrayOf(-((eyeDist * 0.10f).toInt()), 0, (eyeDist * 0.10f).toInt())
            var accDepth = 0f
            for (xo in xOffsets) {
                val sampleX = (cx + xo).coerceIn(16, size - 17)
                var depth = (eyeDist * 0.28f).toInt()
                var missRun = 0
                val maxDown = (eyeDist * 0.96f).toInt()
                for (dy in depth..maxDown) {
                    val yy = chinStartY + dy
                    if (yy >= size - 10) break
                    val pc = px[yy * size + sampleX]
                    val r = ((pc ushr 16) and 0xFF).toFloat()
                    val g = ((pc ushr 8) and 0xFF).toFloat()
                    val b = (pc and 0xFF).toFloat()
                    val lum = 0.299f * r + 0.587f * g + 0.114f * b
                    val diff = abs(r - ref.r) + abs(g - ref.g) + abs(b - ref.b)
                    val warmth = r - b
                    val chroma = max(r, max(g, b)) - min(r, min(g, b))
                    val isNeutralOrCoolBg = (refWarmth >= 18f && warmth < max(8f, refWarmth * 0.28f)) ||
                        (refChroma >= 22f && chroma < refChroma * 0.32f && diff > 36f)
                    val isNeckShadowOrEdge = lum < ref.lum * 0.65f || diff > 114f || isNeutralOrCoolBg
                    if (!isNeckShadowOrEdge) {
                        depth = dy
                        missRun = 0
                    } else {
                        missRun++
                        if (missRun >= 4) break
                    }
                }
                accDepth += depth.toFloat()
            }
            return (accDepth / xOffsets.size.toFloat()).coerceAtLeast(eyeDist * 0.26f)
        }

        // Exact anatomical rows in 512x512 aligned crop for Diagram 3 (D, B, C, A)
        val rowUpperJawD = (geom512.nose.y + eyeDist * 0.16f).toInt()
        val rowJawBoneB = (mouthMidY + eyeDist * 0.04f).toInt()
        val rowLowerJawC = (mouthMidY + eyeDist * 0.28f).toInt()

        val (srcSpanD, srcBorderD) = scanHorizontalSpan(srcPx512, rowUpperJawD, 0.52f, 1.48f)
        val (tgtSpanD, _) = scanHorizontalSpan(tgtPx512, rowUpperJawD, 0.52f, 1.48f)

        val (srcSpanB, srcBorderB) = scanHorizontalSpan(srcPx512, rowJawBoneB, 0.44f, 1.44f)
        val (tgtSpanB, _) = scanHorizontalSpan(tgtPx512, rowJawBoneB, 0.44f, 1.44f)

        val (srcSpanC, srcBorderC) = scanHorizontalSpan(srcPx512, rowLowerJawC, 0.30f, 1.34f)
        val (tgtSpanC, _) = scanHorizontalSpan(tgtPx512, rowLowerJawC, 0.30f, 1.34f)

        val srcChinA = estimateChinDrop(srcPx512)
        val tgtChinA = estimateChinDrop(tgtPx512).coerceAtLeast(16f)

        val safeTgtD = tgtSpanD.coerceAtLeast(16f)
        val safeTgtB = tgtSpanB.coerceAtLeast(16f)
        val safeTgtC = tgtSpanC.coerceAtLeast(16f)

        val srcDiagA = hypot((srcSpanB * 0.5f).toDouble(), srcChinA.toDouble()).toFloat()
        val tgtDiagA = hypot((safeTgtB * 0.5f).toDouble(), tgtChinA.toDouble()).toFloat().coerceAtLeast(16f)

        val rawPixelD = (srcSpanD / safeTgtD).coerceIn(0.80f, 1.28f)
        val rawPixelB = (srcSpanB / safeTgtB).coerceIn(0.78f, 1.32f)
        val rawPixelC = (srcSpanC / safeTgtC).coerceIn(0.76f, 1.34f)
        val pixelScaleD = if (srcBorderD) max(rawPixelD, baseRatios.upperJawMaxillaScale) else rawPixelD
        val pixelScaleB = if (srcBorderB) max(rawPixelB, baseRatios.sourceJawToTargetJawScale) else rawPixelB
        val pixelScaleC = if (srcBorderC) max(rawPixelC, baseRatios.lowerJawBodyScale) else rawPixelC
        val pixelScaleA = ((srcChinA / tgtChinA) * 0.62f + (srcDiagA / tgtDiagA) * 0.38f).coerceIn(0.84f, 1.26f)

        val finalD = (pixelScaleD * 0.78f + baseRatios.upperJawMaxillaScale * 0.22f).coerceIn(0.80f, 1.28f)
        val finalB = (pixelScaleB * 0.78f + baseRatios.sourceJawToTargetJawScale * 0.22f).coerceIn(0.78f, 1.32f)
        val finalC = (pixelScaleC * 0.78f + baseRatios.lowerJawBodyScale * 0.22f).coerceIn(0.76f, 1.34f)
        val finalA = (pixelScaleA * 0.74f + baseRatios.sourceChinToTargetChinScale * 0.26f).coerceIn(0.84f, 1.26f)

        return baseRatios.copy(
            sourceJawToTargetJawScale = (finalB * 0.58f + finalC * 0.42f).coerceIn(0.78f, 1.32f),
            sourceChinToTargetChinScale = finalA,
            upperJawMaxillaScale = finalD,
            jawBoneBigonialScale = finalB,
            lowerJawBodyScale = finalC,
            mandibularLengthScale = finalA
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
            warnings += "தாடை அகல வேறுபாடு (${"%.0f".format(jawDeltaPct)}%) — Option D Source (D/B/C/A) வடிவத்தை வைத்து K/L கீழ் ஓரம் மட்டும் கழுத்துடன் இணைக்கும்."
        }
        if (klArcDelta > 0.16f) {
            warnings += "காது-நாடி வளைவு (K/L Arc Δ ${(klArcDelta * 100).roundToInt()}%) — கழுத்துப் பொருத்தத்திற்கு soft transition பயன்படுத்தப்படுகிறது."
        }

        val baseUpperJawD = (src.bizygomaticWidthB / src.interpupillaryDistO) /
            (tgt.bizygomaticWidthB / tgt.interpupillaryDistO).coerceAtLeast(0.5f)
        val srcToTgtJawScale = (src.squareJawRatio / tgt.squareJawRatio.coerceAtLeast(0.4f)).coerceIn(0.80f, 1.26f)
        val srcToTgtLowerJawC = (src.lowerJawRatio / tgt.lowerJawRatio.coerceAtLeast(0.25f)).coerceIn(0.78f, 1.28f)
        val srcToTgtChinScale = (src.lowerThirdRatio / tgt.lowerThirdRatio.coerceAtLeast(0.4f)).coerceIn(0.84f, 1.22f)
        val clampedD = baseUpperJawD.coerceIn(0.82f, 1.24f)
        val klNeckBlend = (0.24f + klArcDelta * 1.15f).coerceIn(0.24f, 0.58f)

        val optionDRatios = OptionDCraniofacialFitRatios(
            sourceCheekNormalized = src.cheekWidthNormalized,
            sourceSquareJawRatio = src.squareJawRatio,
            sourceLowerJawRatio = src.lowerJawRatio,
            sourceForeheadRatio = src.foreheadWidthRatio,
            sourceJawToTargetJawScale = srcToTgtJawScale,
            sourceChinToTargetChinScale = srcToTgtChinScale,
            klArcNeckBlendStrength = klNeckBlend,
            upperJawMaxillaScale = clampedD,
            jawBoneBigonialScale = srcToTgtJawScale,
            lowerJawBodyScale = srcToTgtLowerJawC,
            mandibularLengthScale = srcToTgtChinScale
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
        val mandibularSummary = "D UpperJaw ×${"%.2f".format(clampedD)} • B JawBone ×${"%.2f".format(srcToTgtJawScale)} • " +
            "C LowerJaw ×${"%.2f".format(srcToTgtLowerJawC)} • A Mandible ×${"%.2f".format(srcToTgtChinScale)}"

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
            optionDFitRatios = optionDRatios,
            mandibularSummary = mandibularSummary
        )
    }
}
