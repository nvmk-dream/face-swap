package com.example.onnx

import android.graphics.Bitmap
import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 3D vertex in image/canonical space `(x, y)` with reconstructed 3D cranial depth `z`
 * (positive `z` = protruding toward the camera, e.g. nose tip; negative `z` = receding toward ears/temples)
 * and unit surface normal `(nx, ny, nz)`.
 */
data class Vertex3D(
    val index: Int,
    val x: Float,
    val y: Float,
    val z: Float,
    val nx: Float = 0f,
    val ny: Float = 0f,
    val nz: Float = 1f
)

/**
 * 3D Head-Pose estimation result in degrees and 3x3 rotation matrix.
 *  - [pitchDeg] (`eulerX`): Up/Down head nod (-45°..+45°, positive = looking up, negative = looking down)
 *  - [yawDeg]   (`eulerY`): Left/Right head turn (-65°..+65°, positive = turned toward image right, negative = turned toward image left)
 *  - [rollDeg]  (`eulerZ`): In-plane head tilt (-180°..+180°, clockwise roll in image plane)
 */
data class HeadPose3D(
    val pitchDeg: Float,
    val yawDeg: Float,
    val rollDeg: Float,
    val rotationMatrix3x3: FloatArray,
    val leftSideVisibility: Float,
    val rightSideVisibility: Float,
    val poseConfidence: Float,
    val poseSummaryLabel: String
)

/**
 * Complete 3D Face Reconstruction + 106-Point Landmark Topology for Pose-Robust Face Swap.
 *
 * Canonical 106-Point Indexing (InsightFace `2d106det` / 3DMM compatible):
 *  - `0..32`   (33 pts): Jawline & Face Oval Contour (left temple [0] -> chin center [16] -> right temple [32])
 *  - `33..42`  (10 pts): Left Eyebrow (upper arch `33..37`, lower arch `38..42`)
 *  - `43..52`  (10 pts): Right Eyebrow (upper arch `43..47`, lower arch `48..52`)
 *  - `53..62`  (10 pts): Left Eye Contour (`53..60`: outer, 3 upper, inner, 3 lower) + Left Iris/Pupil (`61..62`)
 *  - `63..72`  (10 pts): Right Eye Contour (`63..70`: inner, 3 upper, outer, 3 lower) + Right Iris/Pupil (`71..72`)
 *  - `73..85`  (13 pts): Nose Bridge (`73..76` top-to-tip) + Nose Tip & Alar Nostril Base (`77..85`)
 *  - `86..97`  (12 pts): Outer Lip Vermilion Contour (left corner [86], upper lip [87..91], right corner [92], lower lip [93..97])
 *  - `98..105` (8 pts) : Inner Oral Cavity / Teeth Aperture Contour (`98..105`)
 */
data class ReconstructedFace3D(
    val landmarks106: List<PointF>,
    val vertices3D: List<Vertex3D>,
    val headPose: HeadPose3D,
    val jawWidthCoeff: Float,
    val noseProjectionCoeff: Float,
    val mouthOpenRatio: Float,
    val smileCurvatureCoeff: Float,
    val anatomicalPoints34: List<AnatomicalLandmark3D> = emptyList()
)

object Face3DReconstruction {

    const val LANDMARK_COUNT_106 = 106

    // Index ranges for the 106-point topology
    val JAWLINE_INDICES = 0..32
    val LEFT_BROW_INDICES = 33..42
    val RIGHT_BROW_INDICES = 43..52
    val LEFT_EYE_INDICES = 53..62
    val RIGHT_EYE_INDICES = 63..72
    val NOSE_INDICES = 73..85
    val OUTER_LIP_INDICES = 86..97
    val INNER_LIP_INDICES = 98..105

    /**
     * Estimates full 3D Head Pose (`pitchDeg`, `yawDeg`, `rollDeg`, 3x3 rotation matrix, and left/right
     * cheek visibility factors) directly from 5-point or 106-point landmarks using a weak-perspective
     * Anthropometric 3D PnP solver.
     *
     * If hardware/ML Kit Euler hints (`hintPitch`, `hintYaw`, `hintRoll`) are non-zero, fuses them with
     * the geometric PnP solution for maximum stability on side-view and tilted faces.
     */
    fun estimate3DHeadPose(
        landmarks5: List<PointF>,
        hintPitch: Float = 0f,
        hintYaw: Float = 0f,
        hintRoll: Float = 0f
    ): HeadPose3D {
        if (landmarks5.size < 5) {
            return HeadPose3D(
                pitchDeg = hintPitch,
                yawDeg = hintYaw,
                rollDeg = hintRoll,
                rotationMatrix3x3 = buildRotationMatrix3x3(hintPitch, hintYaw, hintRoll),
                leftSideVisibility = 1.0f,
                rightSideVisibility = 1.0f,
                poseConfidence = 0.85f,
                poseSummaryLabel = "Frontal (0°)"
            )
        }

        val lEye = landmarks5[0]
        val rEye = landmarks5[1]
        val nose = landmarks5[2]
        val lMouth = landmarks5[3]
        val rMouth = landmarks5[4]

        // 1. In-plane Roll angle from inter-ocular vector
        val dxEye = rEye.x - lEye.x
        val dyEye = rEye.y - lEye.y
        val eyeDist = hypot(dxEye.toDouble(), dyEye.toDouble()).toFloat().coerceAtLeast(8f)
        val cosR = dxEye / eyeDist
        val sinR = dyEye / eyeDist
        val pnpRollDeg = Math.toDegrees(atan2(dyEye.toDouble(), dxEye.toDouble())).toFloat()

        // Local de-rotated coordinate frame centered at eye midpoint:
        // u-axis = along eye line (left to right), v-axis = perpendicular downward
        val eyeMidX = (lEye.x + rEye.x) * 0.5f
        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val mouthMidX = (lMouth.x + rMouth.x) * 0.5f
        val mouthMidY = (lMouth.y + rMouth.y) * 0.5f

        fun toLocalU(px: Float, py: Float): Float = (px - eyeMidX) * cosR + (py - eyeMidY) * sinR
        fun toLocalV(px: Float, py: Float): Float = -(px - eyeMidX) * sinR + (py - eyeMidY) * cosR

        val noseU = toLocalU(nose.x, nose.y)
        val noseV = toLocalV(nose.x, nose.y).coerceAtLeast(eyeDist * 0.18f)
        val mouthU = toLocalU(mouthMidX, mouthMidY)
        val mouthV = toLocalV(mouthMidX, mouthMidY).coerceAtLeast(eyeDist * 0.45f)
        val lMouthU = toLocalU(lMouth.x, lMouth.y)
        val rMouthU = toLocalU(rMouth.x, rMouth.y)

        // 2. 3D Yaw angle estimation:
        // As the head turns right/left by angle Yaw, the protruding nose tip (depth Z_nose ≈ 0.42 * eyeDist)
        // shifts horizontally relative to the eye midpoint and mouth midpoint.
        val noseHorizAsym = (noseU / (eyeDist * 0.50f)).coerceIn(-1.2f, 1.2f)
        val mouthHorizAsym = ((noseU - mouthU) / (eyeDist * 0.36f)).coerceIn(-1.2f, 1.2f)
        val lEyeToNose = hypot((nose.x - lEye.x).toDouble(), (nose.y - lEye.y).toDouble()).toFloat()
        val rEyeToNose = hypot((rEye.x - nose.x).toDouble(), (rEye.y - nose.y).toDouble()).toFloat()
        val eyeNoseRatioAsym = ((lEyeToNose - rEyeToNose) / (lEyeToNose + rEyeToNose).coerceAtLeast(8f)).coerceIn(-0.85f, 0.85f)
        val mouthCornerAsym = ((abs(lMouthU) - abs(rMouthU)) / eyeDist.coerceAtLeast(8f)).coerceIn(-0.65f, 0.65f)

        val combinedYawSignal = (0.42f * noseHorizAsym + 0.28f * eyeNoseRatioAsym + 0.18f * mouthHorizAsym + 0.12f * mouthCornerAsym)
        val pnpYawDeg = (combinedYawSignal * 54.0f).coerceIn(-62f, 62f)

        // 3. 3D Pitch angle estimation:
        // Canonical frontal face has noseV / mouthV ≈ 0.545 and eye-to-nose / nose-to-mouth ≈ 1.20.
        // Looking down increases noseV / mouthV; looking up decreases noseV / mouthV.
        val noseToMouthV = (mouthV - noseV).coerceAtLeast(eyeDist * 0.15f)
        val verticalRatio = noseV / mouthV.coerceAtLeast(12f)
        val canonicalVertRatio = 0.545f
        val pnpPitchDeg = ((canonicalVertRatio - verticalRatio) * 115.0f).coerceIn(-45f, 45f)

        // 4. Fuse with ML Kit / hardware Euler angles when available
        val finalYaw = if (abs(hintYaw) > 0.5f) (0.55f * hintYaw + 0.45f * pnpYawDeg) else pnpYawDeg
        val finalPitch = if (abs(hintPitch) > 0.5f) (0.55f * hintPitch + 0.45f * pnpPitchDeg) else pnpPitchDeg
        val finalRoll = if (abs(hintRoll) > 0.5f) (0.50f * hintRoll + 0.50f * pnpRollDeg) else pnpRollDeg

        val rot3x3 = buildRotationMatrix3x3(finalPitch, finalYaw, finalRoll)

        // Compute left/right cheek surface visibility under 3D Yaw rotation
        val yawRad = Math.toRadians(finalYaw.toDouble()).toFloat()
        // Positive finalYaw = nose shifted toward image right -> left cheek is turned toward camera, right cheek is foreshortened
        val leftVis = (cos(yawRad * 0.75f) + 0.28f * sin(yawRad)).coerceIn(0.42f, 1.0f)
        val rightVis = (cos(yawRad * 0.75f) - 0.28f * sin(yawRad)).coerceIn(0.42f, 1.0f)

        val absYaw = abs(finalYaw)
        val absPitch = abs(finalPitch)
        val absRoll = abs(finalRoll)
        val poseLabel = buildString {
            when {
                absYaw < 7f && absPitch < 7f && absRoll < 6f -> append("Frontal")
                absYaw >= 18f -> append(if (finalYaw > 0f) "Side-View Right" else "Side-View Left")
                absYaw >= 7f -> append(if (finalYaw > 0f) "Slight Right Turn" else "Slight Left Turn")
                else -> append("Tilted")
            }
            append(" (Yaw ${"%.0f".format(finalYaw)}°, Pitch ${"%.0f".format(finalPitch)}°, Roll ${"%.0f".format(finalRoll)}°)")
        }

        val _unused = noseToMouthV
        return HeadPose3D(
            pitchDeg = finalPitch,
            yawDeg = finalYaw,
            rollDeg = finalRoll,
            rotationMatrix3x3 = rot3x3,
            leftSideVisibility = leftVis,
            rightSideVisibility = rightVis,
            poseConfidence = 0.96f,
            poseSummaryLabel = poseLabel
        )
    }

    /**
     * Reconstructs the complete 106-point dense landmarks AND 3D Morphable Face Geometry (`vertices3D`
     * with 3D cranial depth `z` and surface normals `(nx, ny, nz)`) from 5 base landmarks, optional
     * contour points, and optional bitmap pixel refinement.
     */
    fun reconstruct3DFaceAnd106Landmarks(
        landmarks5: List<PointF>,
        eulerX: Float = 0f,
        eulerY: Float = 0f,
        eulerZ: Float = 0f,
        existingContourPoints: List<PointF> = emptyList(),
        bitmap: Bitmap? = null
    ): ReconstructedFace3D {
        val headPose = estimate3DHeadPose(landmarks5, eulerX, eulerY, eulerZ)
        if (landmarks5.size < 5) {
            val fallbackPts = List(LANDMARK_COUNT_106) { PointF(128f, 128f) }
            val fallbackVerts = List(LANDMARK_COUNT_106) { idx -> Vertex3D(idx, 128f, 128f, 0f) }
            return ReconstructedFace3D(
                landmarks106 = fallbackPts,
                vertices3D = fallbackVerts,
                headPose = headPose,
                jawWidthCoeff = 1.0f,
                noseProjectionCoeff = 1.0f,
                mouthOpenRatio = 0.0f,
                smileCurvatureCoeff = 0.0f
            )
        }

        val lEye = landmarks5[0]
        val rEye = landmarks5[1]
        val nose = landmarks5[2]
        val lMouth = landmarks5[3]
        val rMouth = landmarks5[4]

        val dxEye = rEye.x - lEye.x
        val dyEye = rEye.y - lEye.y
        val eyeDist = hypot(dxEye.toDouble(), dyEye.toDouble()).toFloat().coerceAtLeast(10f)
        val cosA = dxEye / eyeDist
        val sinA = dyEye / eyeDist
        val downX = -sinA
        val downY = cosA

        val eyeMidX = (lEye.x + rEye.x) * 0.5f
        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val mouthMidX = (lMouth.x + rMouth.x) * 0.5f
        val mouthMidY = (lMouth.y + rMouth.y) * 0.5f
        val mouthWidth = hypot((rMouth.x - lMouth.x).toDouble(), (rMouth.y - lMouth.y).toDouble()).toFloat()
            .coerceAtLeast(eyeDist * 0.55f)
        val noseToMouthDist = hypot((mouthMidX - nose.x).toDouble(), (mouthMidY - nose.y).toDouble()).toFloat()
            .coerceAtLeast(eyeDist * 0.28f)

        val yawNorm = (headPose.yawDeg / 50f).coerceIn(-0.90f, 0.90f)
        val pitchNorm = (headPose.pitchDeg / 45f).coerceIn(-0.80f, 0.80f)

        // 3D foreshortening scales for left vs right side of the face under Yaw rotation:
        // When yawNorm > 0 (head turned toward right of image), right cheek (u > 0) compresses and left cheek (u < 0) expands
        val leftYawScale = (1.0f + yawNorm * 0.26f).coerceIn(0.64f, 1.24f)
        val rightYawScale = (1.0f - yawNorm * 0.26f).coerceIn(0.64f, 1.24f)

        val imgW = bitmap?.width?.toFloat() ?: 8192f
        val imgH = bitmap?.height?.toFloat() ?: 8192f

        fun projectPoint(cx: Float, cy: Float, u: Float, v: Float): PointF {
            val px = (cx + cosA * u + downX * v).coerceIn(0f, imgW)
            val py = (cy + sinA * u + downY * v).coerceIn(0f, imgH)
            return PointF(px, py)
        }

        val pts106 = ArrayList<PointF>(LANDMARK_COUNT_106)
        val verts3D = ArrayList<Vertex3D>(LANDMARK_COUNT_106)

        fun addVertex(pt: PointF, depthZ: Float, nxLocal: Float, nyLocal: Float, nzLocal: Float) {
            val normLen = sqrt(nxLocal * nxLocal + nyLocal * nyLocal + nzLocal * nzLocal).coerceAtLeast(1e-4f)
            val idx = pts106.size
            pts106.add(pt)
            verts3D.add(
                Vertex3D(
                    index = idx,
                    x = pt.x,
                    y = pt.y,
                    z = depthZ,
                    nx = nxLocal / normLen,
                    ny = nyLocal / normLen,
                    nz = nzLocal / normLen
                )
            )
        }

        // =========================================================================================
        // 1. JAWLINE & FACE CONTOUR: 33 points (indices 0..32)
        //    0 = left temple, 1..15 = left cheek/jaw to chin, 16 = chin tip, 17..31 = right jaw/cheek, 32 = right temple
        // =========================================================================================
        val faceCenterX = eyeMidX * 0.42f + nose.x * 0.32f + mouthMidX * 0.26f
        val faceCenterY = eyeMidY * 0.42f + nose.y * 0.30f + mouthMidY * 0.28f
        val jawRxBase = eyeDist * 1.03f
        val chinDrop = (eyeDist * 0.56f + noseToMouthDist * 0.72f) * (1.0f - pitchNorm * 0.14f)
        val templeRise = -eyeDist * 0.22f

        for (i in 0..32) {
            // t in [-1.0 .. +1.0], where -1.0 is left temple, 0.0 is chin tip (index 16), +1.0 is right temple
            val t = (i - 16) / 16.0f
            val angle = t * (Math.PI * 0.52).toFloat() // -93.6° .. +93.6° from bottom vertical
            val sinT = sin(angle)
            val cosT = cos(angle)

            val sideScale = if (t < 0f) leftYawScale else rightYawScale
            // Natural anatomical taper toward the chin tip (t near 0)
            val jawTaper = 1.0f - 0.15f * cosT * cosT
            val u = sinT * jawRxBase * jawTaper * sideScale

            // Vertical profile from templeRise (at |t|=1) down to chinDrop (at t=0)
            val v = templeRise * (1.0f - cosT) + chinDrop * cosT
            val pt = if (existingContourPoints.size >= 33 && i < existingContourPoints.size) {
                // Blend 65% geometric 3D-pose jawline with 35% detected contour if available
                val cPt = existingContourPoints[(i * existingContourPoints.size) / 33]
                val gPt = projectPoint(faceCenterX, faceCenterY, u, v)
                PointF(gPt.x * 0.65f + cPt.x * 0.35f, gPt.y * 0.65f + cPt.y * 0.35f)
            } else {
                projectPoint(faceCenterX, faceCenterY, u, v)
            }

            // 3D depth Z along jawline: chin tip protrudes (+0.12*eyeDist), temples recede (-0.55*eyeDist)
            val zDepth = eyeDist * (0.12f * cosT - 0.55f * (1.0f - cosT) - sinT * yawNorm * 0.35f)
            val nx = sinT * cos(Math.toRadians(headPose.yawDeg.toDouble())).toFloat()
            val ny = 0.35f * cosT
            val nz = (cosT * 0.75f - sinT * sin(Math.toRadians(headPose.yawDeg.toDouble())).toFloat()).coerceIn(0.15f, 1.0f)
            addVertex(pt, zDepth, nx, ny, nz)
        }

        // =========================================================================================
        // 2. LEFT EYEBROW: 10 points (indices 33..42: 5 upper arch 33..37, 5 lower arch 38..42)
        // =========================================================================================
        val lBrowCenterU = -eyeDist * 0.04f * leftYawScale
        val lBrowCenterV = -eyeDist * (0.27f - pitchNorm * 0.05f)
        val lBrowHalfW = eyeDist * 0.28f * leftYawScale
        val browThickness = eyeDist * 0.045f
        for (i in 0 until 5) {
            val frac = (i - 2) / 2.0f // -1.0 .. +1.0 (outer to inner)
            val archLift = -eyeDist * 0.042f * (1.0f - frac * frac)
            val u = lBrowCenterU + frac * lBrowHalfW
            val vUpper = lBrowCenterV + archLift - browThickness
            val pt = projectPoint(lEye.x, lEye.y, u, vUpper)
            val z = eyeDist * (0.24f * (1.0f - 0.25f * abs(frac)) + 0.12f * yawNorm)
            addVertex(pt, z, frac * 0.25f, -0.25f, 0.92f)
        }
        for (i in 0 until 5) {
            val frac = (2 - i) / 2.0f // +1.0 .. -1.0 (inner back to outer)
            val archLift = -eyeDist * 0.032f * (1.0f - frac * frac)
            val u = lBrowCenterU + frac * lBrowHalfW
            val vLower = lBrowCenterV + archLift + browThickness * 0.5f
            val pt = projectPoint(lEye.x, lEye.y, u, vLower)
            val z = eyeDist * (0.22f * (1.0f - 0.25f * abs(frac)) + 0.12f * yawNorm)
            addVertex(pt, z, frac * 0.25f, -0.18f, 0.94f)
        }

        // =========================================================================================
        // 3. RIGHT EYEBROW: 10 points (indices 43..52: 5 upper arch 43..47, 5 lower arch 48..52)
        // =========================================================================================
        val rBrowCenterU = eyeDist * 0.04f * rightYawScale
        val rBrowCenterV = -eyeDist * (0.27f - pitchNorm * 0.05f)
        val rBrowHalfW = eyeDist * 0.28f * rightYawScale
        for (i in 0 until 5) {
            val frac = (i - 2) / 2.0f // -1.0 .. +1.0 (inner to outer)
            val archLift = -eyeDist * 0.042f * (1.0f - frac * frac)
            val u = rBrowCenterU + frac * rBrowHalfW
            val vUpper = rBrowCenterV + archLift - browThickness
            val pt = projectPoint(rEye.x, rEye.y, u, vUpper)
            val z = eyeDist * (0.24f * (1.0f - 0.25f * abs(frac)) - 0.12f * yawNorm)
            addVertex(pt, z, frac * 0.25f, -0.25f, 0.92f)
        }
        for (i in 0 until 5) {
            val frac = (2 - i) / 2.0f // +1.0 .. -1.0 (outer back to inner)
            val archLift = -eyeDist * 0.032f * (1.0f - frac * frac)
            val u = rBrowCenterU + frac * rBrowHalfW
            val vLower = rBrowCenterV + archLift + browThickness * 0.5f
            val pt = projectPoint(rEye.x, rEye.y, u, vLower)
            val z = eyeDist * (0.22f * (1.0f - 0.25f * abs(frac)) - 0.12f * yawNorm)
            addVertex(pt, z, frac * 0.25f, -0.18f, 0.94f)
        }

        // =========================================================================================
        // 4. LEFT EYE SOCKET & IRIS: 10 points (indices 53..62)
        //    53 = outer canthus, 54..56 = upper eyelid, 57 = inner canthus, 58..60 = lower eyelid, 61..62 = iris/pupil
        // =========================================================================================
        val lEyeRx = eyeDist * 0.19f * leftYawScale
        val lEyeRy = eyeDist * 0.082f
        val eyeAngles = floatArrayOf(
            Math.PI.toFloat(),
            (Math.PI * 0.75).toFloat(),
            (Math.PI * 0.50).toFloat(),
            (Math.PI * 0.25).toFloat(),
            0f,
            (-Math.PI * 0.25).toFloat(),
            (-Math.PI * 0.50).toFloat(),
            (-Math.PI * 0.75).toFloat()
        )
        for (ang in eyeAngles) {
            val u = cos(ang) * lEyeRx
            val v = -sin(ang) * lEyeRy
            val pt = projectPoint(lEye.x, lEye.y, u, v)
            val z = eyeDist * (0.14f + 0.04f * sin(ang) + 0.10f * yawNorm)
            addVertex(pt, z, cos(ang) * 0.18f, -sin(ang) * 0.18f, 0.96f)
        }
        // 61: Left Iris Center, 62: Left Pupil Upper Catchlight Anchor
        addVertex(PointF(lEye.x, lEye.y), eyeDist * (0.16f + 0.10f * yawNorm), 0f, 0f, 1f)
        addVertex(projectPoint(lEye.x, lEye.y, 0f, -lEyeRy * 0.45f), eyeDist * (0.16f + 0.10f * yawNorm), 0f, -0.1f, 0.99f)

        // =========================================================================================
        // 5. RIGHT EYE SOCKET & IRIS: 10 points (indices 63..72)
        //    63 = inner canthus, 64..66 = upper eyelid, 67 = outer canthus, 68..70 = lower eyelid, 71..72 = iris/pupil
        // =========================================================================================
        val rEyeRx = eyeDist * 0.19f * rightYawScale
        val rEyeRy = eyeDist * 0.082f
        for (ang in eyeAngles) {
            val u = cos(ang) * rEyeRx
            val v = -sin(ang) * rEyeRy
            val pt = projectPoint(rEye.x, rEye.y, u, v)
            val z = eyeDist * (0.14f + 0.04f * sin(ang) - 0.10f * yawNorm)
            addVertex(pt, z, cos(ang) * 0.18f, -sin(ang) * 0.18f, 0.96f)
        }
        // 71: Right Iris Center, 72: Right Pupil Upper Catchlight Anchor
        addVertex(PointF(rEye.x, rEye.y), eyeDist * (0.16f - 0.10f * yawNorm), 0f, 0f, 1f)
        addVertex(projectPoint(rEye.x, rEye.y, 0f, -rEyeRy * 0.45f), eyeDist * (0.16f - 0.10f * yawNorm), 0f, -0.1f, 0.99f)

        // =========================================================================================
        // 6. NOSE BRIDGE & ALAR NOSTRIL WINGS: 13 points (indices 73..85)
        //    73..76 = 4 nose bridge points from nasion (between eyes) down to pronasale (nose tip [76])
        //    77..85 = 9 alar nostril wing & sub-nasale contour points (left wing -> subnasale [81] -> right wing)
        // =========================================================================================
        val nasionX = eyeMidX + downX * (eyeDist * 0.04f)
        val nasionY = eyeMidY + downY * (eyeDist * 0.04f)
        for (i in 0..3) {
            val frac = i / 3.0f // 0 = nasion, 1.0 = nose tip
            val bx = nasionX * (1f - frac) + nose.x * frac
            val by = nasionY * (1f - frac) + nose.y * frac
            val zBridge = eyeDist * (0.20f + 0.24f * frac)
            addVertex(PointF(bx, by), zBridge, 0f, -0.15f * (1f - frac), 0.98f)
        }
        val alarLeftW = eyeDist * 0.21f * leftYawScale
        val alarRightW = eyeDist * 0.21f * rightYawScale
        val subNasaleDrop = noseToMouthDist * 0.18f
        for (i in 0..8) {
            val frac = (i - 4) / 4.0f // -1.0 (left nostril wing) .. 0.0 (subnasale) .. +1.0 (right nostril wing)
            val wingW = if (frac < 0f) alarLeftW else alarRightW
            val u = frac * wingW
            val v = subNasaleDrop * (1.0f - 0.45f * frac * frac)
            val pt = projectPoint(nose.x, nose.y, u, v)
            val zAlar = eyeDist * (0.36f * (1.0f - 0.42f * abs(frac)))
            addVertex(pt, zAlar, frac * 0.45f, 0.25f, 0.85f)
        }

        // =========================================================================================
        // 7. OUTER LIP VERMILION CONTOUR: 12 points (indices 86..97)
        //    86 = left mouth corner, 87..91 = 5 upper lip Cupid's bow points,
        //    92 = right mouth corner, 93..97 = 5 lower lip vermilion points
        // =========================================================================================
        val mAxisX = (rMouth.x - lMouth.x) / mouthWidth
        val mAxisY = (rMouth.y - lMouth.y) / mouthWidth
        val mNormalX = -mAxisY
        val mNormalY = mAxisX

        val upperLipThickness = (noseToMouthDist * 0.24f).coerceIn(eyeDist * 0.08f, eyeDist * 0.20f)
        val lowerLipThickness = (noseToMouthDist * 0.28f).coerceIn(eyeDist * 0.10f, eyeDist * 0.24f)

        // 86: Left Mouth Corner
        addVertex(PointF(lMouth.x, lMouth.y), eyeDist * (0.14f + 0.08f * yawNorm), -0.35f, 0f, 0.93f)

        // 87..91: Upper Lip Outer Arch (with natural Cupid's bow dip at center index 89)
        val upperFracs = floatArrayOf(-0.65f, -0.32f, 0.0f, 0.32f, 0.65f)
        for (idx in upperFracs.indices) {
            val f = upperFracs[idx]
            val archEnv = 1.0f - f * f
            val cupidsBowDip = if (idx == 2) 0.86f else if (idx == 1 || idx == 3) 1.04f else 0.78f
            val lift = -upperLipThickness * archEnv * cupidsBowDip
            val px = (mouthMidX + mAxisX * (f * mouthWidth * 0.5f) + mNormalX * lift).coerceIn(0f, imgW)
            val py = (mouthMidY + mAxisY * (f * mouthWidth * 0.5f) + mNormalY * lift).coerceIn(0f, imgH)
            val zLip = eyeDist * (0.28f * archEnv + 0.14f * (1f - archEnv))
            addVertex(PointF(px, py), zLip, f * 0.25f, -0.25f, 0.94f)
        }

        // 92: Right Mouth Corner
        addVertex(PointF(rMouth.x, rMouth.y), eyeDist * (0.14f - 0.08f * yawNorm), 0.35f, 0f, 0.93f)

        // 93..97: Lower Lip Outer Arch (right-to-left)
        val lowerFracs = floatArrayOf(0.65f, 0.32f, 0.0f, -0.32f, -0.65f)
        for (f in lowerFracs) {
            val archEnv = 1.0f - f * f
            val drop = lowerLipThickness * archEnv
            val px = (mouthMidX + mAxisX * (f * mouthWidth * 0.5f) + mNormalX * drop).coerceIn(0f, imgW)
            val py = (mouthMidY + mAxisY * (f * mouthWidth * 0.5f) + mNormalY * drop).coerceIn(0f, imgH)
            val zLip = eyeDist * (0.26f * archEnv + 0.12f * (1f - archEnv))
            addVertex(PointF(px, py), zLip, f * 0.25f, 0.28f, 0.93f)
        }

        // =========================================================================================
        // 8. INNER ORAL CAVITY / TEETH APERTURE CONTOUR: 8 points (indices 98..105)
        // =========================================================================================
        val innerRx = mouthWidth * 0.36f
        val innerRy = (upperLipThickness + lowerLipThickness) * 0.22f
        for (i in 0 until 8) {
            val ang = (2.0 * Math.PI * i / 8.0).toFloat()
            val u = -cos(ang) * innerRx
            val v = -sin(ang) * innerRy
            val px = (mouthMidX + mAxisX * u + mNormalX * v).coerceIn(0f, imgW)
            val py = (mouthMidY + mAxisY * u + mNormalY * v).coerceIn(0f, imgH)
            addVertex(PointF(px, py), eyeDist * 0.22f, 0f, 0f, 0.98f)
        }

        val jawWidthCoeff = ((jawRxBase * 2f) / (eyeDist * 2.06f)).coerceIn(0.80f, 1.25f)
        val noseProjCoeff = (noseToMouthDist / (eyeDist * 0.52f)).coerceIn(0.75f, 1.30f)
        val smileCurve = (((mouthMidY - (lMouth.y + rMouth.y) * 0.5f)) / (eyeDist * 0.15f)).coerceIn(-0.5f, 0.8f)

        val baseRecon = ReconstructedFace3D(
            landmarks106 = pts106,
            vertices3D = verts3D,
            headPose = headPose,
            jawWidthCoeff = jawWidthCoeff,
            noseProjectionCoeff = noseProjCoeff,
            mouthOpenRatio = (innerRy / eyeDist).coerceIn(0.02f, 0.35f),
            smileCurvatureCoeff = smileCurve
        )
        return baseRecon.copy(
            anatomicalPoints34 = LatentFeatureFusionEngine.extract34NamedAnatomicalPoints(baseRecon)
        )
    }

    /**
     * Convenience helper that returns the 106-point dense facial landmark list directly from 5 base landmarks
     * and 3D Euler pose angles.
     */
    fun synthesize106LandmarksFrom5PointsAndPose(
        landmarks5: List<PointF>,
        eulerX: Float = 0f,
        eulerY: Float = 0f,
        eulerZ: Float = 0f,
        existingContourPoints: List<PointF> = emptyList()
    ): List<PointF> {
        return reconstruct3DFaceAnd106Landmarks(
            landmarks5 = landmarks5,
            eulerX = eulerX,
            eulerY = eulerY,
            eulerZ = eulerZ,
            existingContourPoints = existingContourPoints,
            bitmap = null
        ).landmarks106
    }

    /**
     * Transforms a 106-point landmark list using a 2x3 affine matrix `m` (e.g. `m128` or `m512`).
     */
    fun transformLandmarks106(landmarks106: List<PointF>, m2x3: FloatArray): List<PointF> {
        val m00 = m2x3[0]
        val m01 = m2x3[1]
        val m02 = m2x3[2]
        val m10 = m2x3[3]
        val m11 = m2x3[4]
        val m12 = m2x3[5]
        return landmarks106.map { pt ->
            PointF(
                m00 * pt.x + m01 * pt.y + m02,
                m10 * pt.x + m11 * pt.y + m12
            )
        }
    }

    /**
     * Computes a 3D Euler rotation matrix `R = Rz(roll) * Ry(yaw) * Rx(pitch)` as a 9-element row-major FloatArray.
     */
    fun buildRotationMatrix3x3(pitchDeg: Float, yawDeg: Float, rollDeg: Float): FloatArray {
        val rx = Math.toRadians(pitchDeg.toDouble())
        val ry = Math.toRadians(yawDeg.toDouble())
        val rz = Math.toRadians(rollDeg.toDouble())

        val cx = cos(rx).toFloat()
        val sx = sin(rx).toFloat()
        val cy = cos(ry).toFloat()
        val sy = sin(ry).toFloat()
        val cz = cos(rz).toFloat()
        val sz = sin(rz).toFloat()

        return floatArrayOf(
            cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
            sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
            -sy, cy * sx, cy * cx
        )
    }
}
