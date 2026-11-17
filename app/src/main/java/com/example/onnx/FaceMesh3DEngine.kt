package com.example.onnx

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 3D Facial Mesh Extraction + Piecewise 3D Triangle-Mesh Head Warping Engine.
 *
 * Supports `facemesh-v2-v1.onnx` (468 / 478 MediaPipe 3D facial vertices with X, Y, Z depth)
 * with auto-detected input/output tensor shape & FP16/FP32 precision, and falls back smoothly
 * to [Face3DReconstruction.reconstruct3DFaceAnd106Landmarks] when `facemesh-v2-v1.onnx` is not installed.
 */
data class FaceMesh3DResult(
    val landmarks: List<PointF>,
    val depthsZ: FloatArray,
    val headPose: HeadPose3D,
    val controlPoints31: List<PointF>,
    val usedFaceMeshOnnx: Boolean,
    val sourceLabel: String
)

object FaceMesh3DEngine {

    // MediaPipe 468-point canonical anchor indices
    private val MP_LEFT_EYE_RING = intArrayOf(33, 133, 159, 145)
    private val MP_RIGHT_EYE_RING = intArrayOf(362, 263, 386, 374)
    private const val MP_NOSE_TIP = 1
    private const val MP_NOSE_BRIDGE = 6
    private const val MP_LEFT_MOUTH = 61
    private const val MP_RIGHT_MOUTH = 291
    private const val MP_UPPER_LIP = 13
    private const val MP_LOWER_LIP = 14
    private const val MP_LEFT_BROW = 105
    private const val MP_RIGHT_BROW = 334
    private const val MP_FOREHEAD_TOP = 10
    private const val MP_CHIN = 152
    private const val MP_LEFT_TEMPLE = 127
    private const val MP_RIGHT_TEMPLE = 356
    private const val MP_LEFT_CHEEK = 234
    private const val MP_RIGHT_CHEEK = 454
    private const val MP_LEFT_JAW = 172
    private const val MP_RIGHT_JAW = 397

    /**
     * Unified 31-Point Anatomical Control Lattice for Head/Hair/Neck 3D Mesh Warping:
     *  0..17 : Face oval, inner features, brow, jaw, chin (18 pts)
     *  18..24: Cranial crown & left/right hair wings (7 pts)
     *  25..30: Left/Center/Right upper neck & collar anchors (6 pts)
     */
    const val CONTROL_POINT_COUNT = 31

    // Fixed, topologically valid triangle mesh over the 31 anatomical control points.
    // Every triangle covers face, jaw, skull crown, side hair wings, or neck-to-collar seamlessly.
    private val HEAD_MESH_TRIANGLES: Array<IntArray> = arrayOf(
        // Inner ocular & nasal core
        intArrayOf(0, 8, 2), intArrayOf(8, 9, 2), intArrayOf(9, 1, 2),
        intArrayOf(0, 2, 3), intArrayOf(1, 2, 3),
        intArrayOf(8, 10, 9), intArrayOf(8, 0, 10), intArrayOf(9, 1, 10),
        // Cheeks & mouth
        intArrayOf(0, 3, 11), intArrayOf(1, 3, 12),
        intArrayOf(3, 6, 11), intArrayOf(3, 6, 12),
        intArrayOf(3, 4, 6), intArrayOf(3, 5, 6),
        intArrayOf(4, 6, 7), intArrayOf(5, 6, 7),
        intArrayOf(4, 11, 7), intArrayOf(5, 12, 7),
        // Jawline & chin perimeter
        intArrayOf(0, 11, 13), intArrayOf(11, 13, 15), intArrayOf(11, 4, 15),
        intArrayOf(4, 15, 17), intArrayOf(4, 7, 17),
        intArrayOf(1, 12, 14), intArrayOf(12, 14, 16), intArrayOf(12, 5, 16),
        intArrayOf(5, 16, 17), intArrayOf(5, 7, 17),
        // Forehead & temples
        intArrayOf(0, 8, 13), intArrayOf(1, 9, 14),
        intArrayOf(8, 10, 18), intArrayOf(9, 10, 18),
        intArrayOf(8, 13, 19), intArrayOf(8, 18, 19),
        intArrayOf(9, 14, 20), intArrayOf(9, 18, 20),
        // Cranial crown & hair dome
        intArrayOf(18, 19, 21), intArrayOf(18, 21, 22), intArrayOf(18, 20, 22),
        intArrayOf(13, 19, 23), intArrayOf(19, 21, 23),
        intArrayOf(14, 20, 24), intArrayOf(20, 22, 24),
        // Side hair wings to jaw
        intArrayOf(13, 15, 23), intArrayOf(14, 16, 24),
        // Upper neck & chin transition
        intArrayOf(15, 17, 25), intArrayOf(17, 25, 26),
        intArrayOf(16, 17, 27), intArrayOf(17, 27, 26),
        intArrayOf(15, 23, 25), intArrayOf(16, 24, 27),
        // Lower neck to collar anchors
        intArrayOf(25, 26, 28), intArrayOf(26, 28, 29),
        intArrayOf(26, 27, 29), intArrayOf(27, 29, 30)
    )

    fun extractFaceMesh3D(
        ortEnv: OrtEnvironment?,
        bitmap: Bitmap,
        face: DetectedFace,
        faceMeshFile: File?,
        preferHardwareAccel: Boolean = false
    ): FaceMesh3DResult {
        if (ortEnv != null && faceMeshFile != null && faceMeshFile.exists() && faceMeshFile.length() > 1024L) {
            val onnxRes = runCatching {
                runFaceMeshOnnx(
                    ortEnv = ortEnv,
                    faceMeshFile = faceMeshFile,
                    bitmap = bitmap,
                    face = face,
                    preferHardwareAccel = preferHardwareAccel
                )
            }.getOrNull()
            if (onnxRes != null) return onnxRes
        }

        val recon = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(
            landmarks5 = face.landmarks5,
            eulerX = face.eulerX,
            eulerY = face.eulerY,
            eulerZ = face.eulerZ,
            existingContourPoints = face.faceContourPoints,
            bitmap = bitmap
        )
        val pts = recon.landmarks106
        val zArr = FloatArray(recon.vertices3D.size) { i -> recon.vertices3D[i].z }
        val ctrl = buildControlPointsFrom106(pts, face.landmarks5, recon.headPose)
        return FaceMesh3DResult(
            landmarks = pts,
            depthsZ = zArr,
            headPose = recon.headPose,
            controlPoints31 = ctrl,
            usedFaceMeshOnnx = false,
            sourceLabel = "3D-PnP-106 Fallback"
        )
    }

    private fun runFaceMeshOnnx(
        ortEnv: OrtEnvironment,
        faceMeshFile: File,
        bitmap: Bitmap,
        face: DetectedFace,
        preferHardwareAccel: Boolean
    ): FaceMesh3DResult? {
        val session = OnnxProtobufInspector.getOrCreateCachedSession(
            ortEnv = ortEnv,
            file = faceMeshFile,
            preferHardwareAcceleration = preferHardwareAccel
        ) ?: return null

        val inputEntry = session.inputInfo.entries.firstOrNull() ?: return null
        val inputName = inputEntry.key
        val tInfo = inputEntry.value.info as? TensorInfo ?: return null
        val shape = tInfo.shape
        val isNhwc = shape.size == 4 && (shape[3] == 3L || shape[3] == 4L)
        val hDim = if (shape.size == 4) {
            val raw = if (isNhwc) shape[1] else shape[2]
            if (raw in 64L..512L) raw.toInt() else 192
        } else 192
        val wDim = if (shape.size == 4) {
            val raw = if (isNhwc) shape[2] else shape[3]
            if (raw in 64L..512L) raw.toInt() else hDim
        } else hDim
        val cropSize = max(hDim, wDim)

        // Build a generous square crop (1.55x face bounding box) centered on the face
        val box = face.boundingBox
        val cx = box.centerX()
        val cy = box.centerY()
        val half = max(box.width(), box.height()) * 0.78f
        val left = (cx - half).coerceIn(0f, (bitmap.width - 2).toFloat())
        val top = (cy - half).coerceIn(0f, (bitmap.height - 2).toFloat())
        val side = (half * 2f).coerceAtLeast(16f)
        val cropRect = RectF(left, top, left + side, top + side)

        val cropBmp = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        val cropPixels = IntArray(cropSize * cropSize)
        val srcW = bitmap.width
        val srcH = bitmap.height
        val srcPixels = IntArray(srcW * srcH)
        bitmap.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)

        for (y in 0 until cropSize) {
            val sy = cropRect.top + (y + 0.5f) / cropSize * side
            for (x in 0 until cropSize) {
                val sx = cropRect.left + (x + 0.5f) / cropSize * side
                cropPixels[y * cropSize + x] = FaceAlignment.sampleBilinearClamped(srcPixels, srcW, srcH, sx, sy)
            }
        }
        cropBmp.setPixels(cropPixels, 0, cropSize, 0, 0, cropSize, cropSize)
        cropBmp.recycle()

        val hw = cropSize * cropSize
        val isFp16Input = tInfo.type == OnnxJavaType.FLOAT16
        val tensorShape = if (isNhwc) {
            longArrayOf(1L, cropSize.toLong(), cropSize.toLong(), 3L)
        } else {
            longArrayOf(1L, 3L, cropSize.toLong(), cropSize.toLong())
        }

        val rawCoords: FloatArray
        if (isFp16Input) {
            val shortBuf = ShortBuffer.allocate(3 * hw)
            if (isNhwc) {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    val r = ((c ushr 16) and 0xFF) / 255.0f
                    val g = ((c ushr 8) and 0xFF) / 255.0f
                    val b = (c and 0xFF) / 255.0f
                    shortBuf.put(i * 3, DepthEstimationEngine.floatToFloat16(r))
                    shortBuf.put(i * 3 + 1, DepthEstimationEngine.floatToFloat16(g))
                    shortBuf.put(i * 3 + 2, DepthEstimationEngine.floatToFloat16(b))
                }
            } else {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    shortBuf.put(i, DepthEstimationEngine.floatToFloat16(((c ushr 16) and 0xFF) / 255.0f))
                    shortBuf.put(hw + i, DepthEstimationEngine.floatToFloat16(((c ushr 8) and 0xFF) / 255.0f))
                    shortBuf.put(2 * hw + i, DepthEstimationEngine.floatToFloat16((c and 0xFF) / 255.0f))
                }
            }
            shortBuf.rewind()
            rawCoords = OnnxTensor.createTensor(ortEnv, shortBuf, tensorShape, OnnxJavaType.FLOAT16).use { inT ->
                session.run(mapOf(inputName to inT)).use { res -> extractLandmarkTensor(res) }
            } ?: return null
        } else {
            val floatBuf = FloatBuffer.allocate(3 * hw)
            if (isNhwc) {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    floatBuf.put(i * 3, ((c ushr 16) and 0xFF) / 255.0f)
                    floatBuf.put(i * 3 + 1, ((c ushr 8) and 0xFF) / 255.0f)
                    floatBuf.put(i * 3 + 2, (c and 0xFF) / 255.0f)
                }
            } else {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    floatBuf.put(i, ((c ushr 16) and 0xFF) / 255.0f)
                    floatBuf.put(hw + i, ((c ushr 8) and 0xFF) / 255.0f)
                    floatBuf.put(2 * hw + i, (c and 0xFF) / 255.0f)
                }
            }
            floatBuf.rewind()
            rawCoords = OnnxTensor.createTensor(ortEnv, floatBuf, tensorShape).use { inT ->
                session.run(mapOf(inputName to inT)).use { res -> extractLandmarkTensor(res) }
            } ?: return null
        }

        val numPts = (rawCoords.size / 3).coerceAtMost(478)
        if (numPts < 468) return null

        // Auto-detect whether output coordinates are normalized [0..1] or crop-pixel [0..cropSize]
        var maxCoord = 0f
        for (i in 0 until numPts) {
            val x = rawCoords[i * 3]
            val y = rawCoords[i * 3 + 1]
            if (x.isFinite() && y.isFinite()) {
                maxCoord = max(maxCoord, max(abs(x), abs(y)))
            }
        }
        val isNormalized = maxCoord <= 2.2f

        val pts = ArrayList<PointF>(numPts)
        val zArr = FloatArray(numPts)
        for (i in 0 until numPts) {
            val rx = rawCoords[i * 3]
            val ry = rawCoords[i * 3 + 1]
            val rz = rawCoords[i * 3 + 2]
            val nx = if (isNormalized) rx else rx / cropSize.toFloat()
            val ny = if (isNormalized) ry else ry / cropSize.toFloat()
            val nz = if (isNormalized) rz else rz / cropSize.toFloat()
            val imgX = cropRect.left + nx * side
            val imgY = cropRect.top + ny * side
            pts.add(PointF(imgX, imgY))
            zArr[i] = nz * side
        }

        // Verify landmark sanity (must span reasonable eye/face distance)
        val lEye = avgPoint(pts, MP_LEFT_EYE_RING)
        val rEye = avgPoint(pts, MP_RIGHT_EYE_RING)
        val meshEyeDist = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble()).toFloat()
        val detEyeDist = if (face.landmarks5.size >= 2) {
            hypot(
                (face.landmarks5[1].x - face.landmarks5[0].x).toDouble(),
                (face.landmarks5[1].y - face.landmarks5[0].y).toDouble()
            ).toFloat()
        } else {
            box.width() * 0.38f
        }
        if (!meshEyeDist.isFinite() || meshEyeDist < detEyeDist * 0.45f || meshEyeDist > detEyeDist * 2.2f) {
            return null
        }

        // Fuse 3D Z-depth yaw & pitch cues from FaceMesh V2 with 5-point PnP solver
        val nose = pts[MP_NOSE_TIP]
        val lMouth = pts[MP_LEFT_MOUTH]
        val rMouth = pts[MP_RIGHT_MOUTH]
        val basePose = Face3DReconstruction.estimate3DHeadPose(
            landmarks5 = listOf(lEye, rEye, nose, lMouth, rMouth),
            hintPitch = face.eulerX,
            hintYaw = face.eulerY,
            hintRoll = face.eulerZ
        )

        val zLeftCheek = zArr[MP_LEFT_CHEEK]
        val zRightCheek = zArr[MP_RIGHT_CHEEK]
        val zYawSignal = ((zLeftCheek - zRightCheek) / meshEyeDist.coerceAtLeast(8f) * 42f).coerceIn(-45f, 45f)
        val zForehead = zArr[MP_FOREHEAD_TOP]
        val zChin = zArr[MP_CHIN]
        val zPitchSignal = ((zChin - zForehead) / meshEyeDist.coerceAtLeast(8f) * 38f).coerceIn(-35f, 35f)

        val fusedYaw = (0.65f * basePose.yawDeg + 0.35f * zYawSignal).coerceIn(-62f, 62f)
        val fusedPitch = (0.65f * basePose.pitchDeg + 0.35f * zPitchSignal).coerceIn(-45f, 45f)
        val fusedPose = basePose.copy(
            yawDeg = fusedYaw,
            pitchDeg = fusedPitch,
            poseSummaryLabel = "3D-FaceMesh (${"%.0f".format(fusedYaw)}°Y, ${"%.0f".format(fusedPitch)}°P, ${"%.0f".format(basePose.rollDeg)}°R)"
        )

        val ctrl = buildControlPointsFrom468(pts, fusedPose)
        return FaceMesh3DResult(
            landmarks = pts,
            depthsZ = zArr,
            headPose = fusedPose,
            controlPoints31 = ctrl,
            usedFaceMeshOnnx = true,
            sourceLabel = "facemesh-v2-v1.onnx (468-Pt 3D)"
        )
    }

    private fun extractLandmarkTensor(res: OrtSession.Result): FloatArray? {
        for (i in 0 until res.size()) {
            val t = res[i] as? OnnxTensor ?: continue
            val shape = t.info.shape
            val total = shape.fold(1L) { acc, v -> acc * max(1L, v) }.toInt()
            if (total >= 1404) {
                return if (t.info.type == OnnxJavaType.FLOAT16) {
                    val sArr = ShortArray(total)
                    t.shortBuffer.get(sArr)
                    FloatArray(total) { idx -> DepthEstimationEngine.float16ToFloat(sArr[idx]) }
                } else {
                    val fArr = FloatArray(total)
                    t.floatBuffer.get(fArr)
                    fArr
                }
            }
        }
        return null
    }

    private fun avgPoint(pts: List<PointF>, indices: IntArray): PointF {
        var sx = 0f
        var sy = 0f
        var cnt = 0
        for (idx in indices) {
            if (idx in pts.indices) {
                sx += pts[idx].x
                sy += pts[idx].y
                cnt++
            }
        }
        return if (cnt > 0) PointF(sx / cnt, sy / cnt) else PointF(0f, 0f)
    }

    private fun buildControlPointsFrom468(
        pts: List<PointF>,
        pose: HeadPose3D
    ): List<PointF> {
        val lEye = avgPoint(pts, MP_LEFT_EYE_RING)
        val rEye = avgPoint(pts, MP_RIGHT_EYE_RING)
        val nose = pts[MP_NOSE_TIP]
        val lMouth = pts[MP_LEFT_MOUTH]
        val rMouth = pts[MP_RIGHT_MOUTH]
        val uLip = pts[MP_UPPER_LIP]
        val lLip = pts[MP_LOWER_LIP]
        val noseBridge = pts[MP_NOSE_BRIDGE]
        val lBrow = pts[MP_LEFT_BROW]
        val rBrow = pts[MP_RIGHT_BROW]
        val forehead = pts[MP_FOREHEAD_TOP]
        val lCheek = pts[MP_LEFT_CHEEK]
        val rCheek = pts[MP_RIGHT_CHEEK]
        val lTemple = pts[MP_LEFT_TEMPLE]
        val rTemple = pts[MP_RIGHT_TEMPLE]
        val lJaw = pts[MP_LEFT_JAW]
        val rJaw = pts[MP_RIGHT_JAW]
        val chin = pts[MP_CHIN]

        return appendCranialAndNeckCage(
            base18 = listOf(
                lEye, rEye, nose, lMouth, rMouth, uLip, lLip, noseBridge,
                lBrow, rBrow, forehead, lCheek, rCheek, lTemple, rTemple,
                lJaw, rJaw, chin
            ),
            pose = pose
        )
    }

    private fun buildControlPointsFrom106(
        pts106: List<PointF>,
        landmarks5: List<PointF>,
        pose: HeadPose3D
    ): List<PointF> {
        fun safePt(idx: Int, fallback: PointF): PointF =
            if (idx in pts106.indices) pts106[idx] else fallback

        val lEye = landmarks5.getOrElse(0) { PointF(100f, 110f) }
        val rEye = landmarks5.getOrElse(1) { PointF(156f, 110f) }
        val nose = landmarks5.getOrElse(2) { PointF(128f, 145f) }
        val lMouth = landmarks5.getOrElse(3) { PointF(105f, 178f) }
        val rMouth = landmarks5.getOrElse(4) { PointF(151f, 178f) }

        val uLip = safePt(89, PointF((lMouth.x + rMouth.x) * 0.5f, (lMouth.y + rMouth.y) * 0.5f - 4f))
        val lLip = safePt(95, PointF((lMouth.x + rMouth.x) * 0.5f, (lMouth.y + rMouth.y) * 0.5f + 5f))
        val noseBridge = safePt(73, PointF((lEye.x + rEye.x) * 0.5f, (lEye.y + rEye.y) * 0.5f))
        val lBrow = safePt(35, PointF(lEye.x, lEye.y - 16f))
        val rBrow = safePt(45, PointF(rEye.x, rEye.y - 16f))
        val eyeMidX = (lEye.x + rEye.x) * 0.5f
        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val eyeDist = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble()).toFloat().coerceAtLeast(12f)
        val dx = (rEye.x - lEye.x) / eyeDist
        val dy = (rEye.y - lEye.y) / eyeDist
        val downX = -dy
        val downY = dx
        val forehead = PointF(eyeMidX - downX * eyeDist * 0.92f, eyeMidY - downY * eyeDist * 0.92f)
        val lCheek = safePt(4, PointF(lEye.x - dx * eyeDist * 0.55f + downX * eyeDist * 0.55f, lEye.y - dy * eyeDist * 0.55f + downY * eyeDist * 0.55f))
        val rCheek = safePt(28, PointF(rEye.x + dx * eyeDist * 0.55f + downX * eyeDist * 0.55f, rEye.y + dy * eyeDist * 0.55f + downY * eyeDist * 0.55f))
        val lTemple = safePt(0, PointF(lEye.x - dx * eyeDist * 0.65f, lEye.y - dy * eyeDist * 0.65f))
        val rTemple = safePt(32, PointF(rEye.x + dx * eyeDist * 0.65f, rEye.y + dy * eyeDist * 0.65f))
        val lJaw = safePt(9, PointF(lMouth.x - dx * eyeDist * 0.45f + downX * eyeDist * 0.35f, lMouth.y - dy * eyeDist * 0.45f + downY * eyeDist * 0.35f))
        val rJaw = safePt(23, PointF(rMouth.x + dx * eyeDist * 0.45f + downX * eyeDist * 0.35f, rMouth.y + dy * eyeDist * 0.45f + downY * eyeDist * 0.35f))
        val chin = safePt(16, PointF((lMouth.x + rMouth.x) * 0.5f + downX * eyeDist * 0.72f, (lMouth.y + rMouth.y) * 0.5f + downY * eyeDist * 0.72f))

        return appendCranialAndNeckCage(
            base18 = listOf(
                lEye, rEye, nose, lMouth, rMouth, uLip, lLip, noseBridge,
                lBrow, rBrow, forehead, lCheek, rCheek, lTemple, rTemple,
                lJaw, rJaw, chin
            ),
            pose = pose
        )
    }

    /**
     * Expands the 18 facial + jawline control points into a full 31-point Cranial Crown + Hair Wings +
     * Neck-to-Collar structural cage aligned to the head's 3D roll/yaw vector.
     */
    private fun appendCranialAndNeckCage(
        base18: List<PointF>,
        pose: HeadPose3D
    ): List<PointF> {
        val lEye = base18[0]
        val rEye = base18[1]
        val lTemple = base18[13]
        val rTemple = base18[14]
        val lJaw = base18[15]
        val rJaw = base18[16]
        val chin = base18[17]

        val dx = rEye.x - lEye.x
        val dy = rEye.y - lEye.y
        val eyeDist = hypot(dx.toDouble(), dy.toDouble()).toFloat().coerceAtLeast(10f)
        val rightX = dx / eyeDist
        val rightY = dy / eyeDist
        val downX = -rightY
        val downY = rightX
        val eyeMidX = (lEye.x + rEye.x) * 0.5f
        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val yawShift = (pose.yawDeg / 55f).coerceIn(-0.45f, 0.45f) * eyeDist * 0.28f

        fun pt(uRight: Float, vDown: Float): PointF = PointF(
            eyeMidX + uRight * rightX + vDown * downX,
            eyeMidY + uRight * rightY + vDown * downY
        )

        // 18: crownTop, 19: crownLeft, 20: crownRight, 21: hairDomeLeft, 22: hairDomeRight,
        // 23: hairWingLeft, 24: hairWingRight
        val crownTop = pt(-yawShift, -eyeDist * 1.95f)
        val crownLeft = pt(-eyeDist * 1.05f - yawShift * 0.6f, -eyeDist * 1.35f)
        val crownRight = pt(eyeDist * 1.05f - yawShift * 0.6f, -eyeDist * 1.35f)
        val hairDomeLeft = pt(-eyeDist * 1.48f - yawShift * 0.8f, -eyeDist * 1.75f)
        val hairDomeRight = pt(eyeDist * 1.48f - yawShift * 0.8f, -eyeDist * 1.75f)
        val hairWingLeft = PointF(
            lTemple.x - rightX * eyeDist * 0.92f + downX * eyeDist * 0.55f,
            lTemple.y - rightY * eyeDist * 0.92f + downY * eyeDist * 0.55f
        )
        val hairWingRight = PointF(
            rTemple.x + rightX * eyeDist * 0.92f + downX * eyeDist * 0.55f,
            rTemple.y + rightY * eyeDist * 0.92f + downY * eyeDist * 0.55f
        )

        // 25..27: Upper Neck (left, center, right) just beneath jaw/chin
        val neckCenter = PointF(chin.x + downX * eyeDist * 0.52f, chin.y + downY * eyeDist * 0.52f)
        val neckLeft = PointF(lJaw.x - rightX * eyeDist * 0.08f + downX * eyeDist * 0.62f, lJaw.y - rightY * eyeDist * 0.08f + downY * eyeDist * 0.62f)
        val neckRight = PointF(rJaw.x + rightX * eyeDist * 0.08f + downX * eyeDist * 0.62f, rJaw.y + rightY * eyeDist * 0.08f + downY * eyeDist * 0.62f)

        // 28..30: Lower Neck / Collar Junction
        val collarCenter = PointF(neckCenter.x + downX * eyeDist * 0.68f, neckCenter.y + downY * eyeDist * 0.68f)
        val collarLeft = PointF(neckLeft.x - rightX * eyeDist * 0.22f + downX * eyeDist * 0.68f, neckLeft.y - rightY * eyeDist * 0.22f + downY * eyeDist * 0.68f)
        val collarRight = PointF(neckRight.x + rightX * eyeDist * 0.22f + downX * eyeDist * 0.68f, neckRight.y + rightY * eyeDist * 0.22f + downY * eyeDist * 0.68f)

        return base18 + listOf(
            crownTop, crownLeft, crownRight, hairDomeLeft, hairDomeRight,
            hairWingLeft, hairWingRight,
            neckLeft, neckCenter, neckRight,
            collarLeft, collarCenter, collarRight
        )
    }

    data class WarpedHeadCanvasResult(
        val bitmap: Bitmap,
        val inBoundsMask: FloatArray
    )

    /**
     * Maps the 31 control points into target canvas coordinates, applies user fine-tune adjustments
     * (`offsetX`, `offsetY`, `scaleAdjust`), and blends target's 3D jawline/neck pose into the mesh.
     */
    fun warpSourceHeadToTarget3DPose(
        sourceBitmap: Bitmap,
        sourceMesh: FaceMesh3DResult,
        targetMesh: FaceMesh3DResult,
        targetCanvasMatrix2x3: FloatArray,
        canvasSize: Int,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f
    ): Bitmap {
        return warpSourceHeadToTarget3DPoseWithBounds(
            sourceBitmap = sourceBitmap,
            sourceMesh = sourceMesh,
            targetMesh = targetMesh,
            targetCanvasMatrix2x3 = targetCanvasMatrix2x3,
            canvasSize = canvasSize,
            offsetX = offsetX,
            offsetY = offsetY,
            scaleAdjust = scaleAdjust
        ).bitmap
    }

    /**
     * Same 3D pose warp as [warpSourceHeadToTarget3DPose], also returning a per-pixel
     * [WarpedHeadCanvasResult.inBoundsMask] in `[0f, 1f]` that smoothly tapers to `0` near or
     * outside the source photo edges so clamped background/ceiling strips never leak into the portrait.
     */
    fun warpSourceHeadToTarget3DPoseWithBounds(
        sourceBitmap: Bitmap,
        sourceMesh: FaceMesh3DResult,
        targetMesh: FaceMesh3DResult,
        targetCanvasMatrix2x3: FloatArray,
        canvasSize: Int,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f
    ): WarpedHeadCanvasResult {
        val srcCtrl = sourceMesh.controlPoints31
        val tgtCtrlRaw = targetMesh.controlPoints31
        if (srcCtrl.size < CONTROL_POINT_COUNT || tgtCtrlRaw.size < CONTROL_POINT_COUNT) {
            return WarpedHeadCanvasResult(
                bitmap = Bitmap.createBitmap(canvasSize, canvasSize, Bitmap.Config.ARGB_8888),
                inBoundsMask = FloatArray(canvasSize * canvasSize)
            )
        }

        // Transform target control points into canvasSize x canvasSize space
        val m00 = targetCanvasMatrix2x3[0]
        val m01 = targetCanvasMatrix2x3[1]
        val m02 = targetCanvasMatrix2x3[2]
        val m10 = targetCanvasMatrix2x3[3]
        val m11 = targetCanvasMatrix2x3[4]
        val m12 = targetCanvasMatrix2x3[5]

        val rawCanvasPts = Array(CONTROL_POINT_COUNT) { i ->
            val p = tgtCtrlRaw[i]
            PointF(m00 * p.x + m01 * p.y + m02, m10 * p.x + m11 * p.y + m12)
        }

        // Preserve Source cranial/hair proportions on outer dome (indices 18..24) while locking eyes/jaw/neck (0..17, 25..30) to Target 3D pose!
        val srcEyeDist = hypot(
            (srcCtrl[1].x - srcCtrl[0].x).toDouble(),
            (srcCtrl[1].y - srcCtrl[0].y).toDouble()
        ).toFloat().coerceAtLeast(8f)
        val tgtEyeDist = hypot(
            (rawCanvasPts[1].x - rawCanvasPts[0].x).toDouble(),
            (rawCanvasPts[1].y - rawCanvasPts[0].y).toDouble()
        ).toFloat().coerceAtLeast(8f)
        val srcEyeMidX = (srcCtrl[0].x + srcCtrl[1].x) * 0.5f
        val srcEyeMidY = (srcCtrl[0].y + srcCtrl[1].y) * 0.5f
        val tgtEyeMidX = (rawCanvasPts[0].x + rawCanvasPts[1].x) * 0.5f
        val tgtEyeMidY = (rawCanvasPts[0].y + rawCanvasPts[1].y) * 0.5f

        val srcDirX = (srcCtrl[1].x - srcCtrl[0].x) / srcEyeDist
        val srcDirY = (srcCtrl[1].y - srcCtrl[0].y) / srcEyeDist
        val tgtDirX = (rawCanvasPts[1].x - rawCanvasPts[0].x) / tgtEyeDist
        val tgtDirY = (rawCanvasPts[1].y - rawCanvasPts[0].y) / tgtEyeDist
        val poseScale = tgtEyeDist / srcEyeDist

        val tgtCanvasPts = Array(CONTROL_POINT_COUNT) { i ->
            val basePt = if (i in 18..24) {
                // Project Source crown/hair wing vector rotated & scaled into Target head pose so Source hairstyle volume keeps its authentic silhouette
                val vx = srcCtrl[i].x - srcEyeMidX
                val vy = srcCtrl[i].y - srcEyeMidY
                val localU = (vx * srcDirX + vy * srcDirY) * poseScale
                val localV = (-vx * srcDirY + vy * srcDirX) * poseScale
                PointF(
                    tgtEyeMidX + localU * tgtDirX - localV * tgtDirY,
                    tgtEyeMidY + localU * tgtDirY + localV * tgtDirX
                )
            } else {
                rawCanvasPts[i]
            }
            // Apply user scaleAdjust & offset around eye midpoint
            val safeScale = scaleAdjust.coerceIn(0.80f, 1.25f)
            val adjX = tgtEyeMidX + (basePt.x - tgtEyeMidX) * safeScale + offsetX * (canvasSize / 320f)
            val adjY = tgtEyeMidY + (basePt.y - tgtEyeMidY) * safeScale + offsetY * (canvasSize / 320f)
            PointF(adjX, adjY)
        }

        val srcW = sourceBitmap.width
        val srcH = sourceBitmap.height
        val srcPixels = IntArray(srcW * srcH)
        sourceBitmap.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)

        // Fallback global similarity transform for pixels slightly outside the 31-pt cage
        val globalInv = estimateSimilarity2x3(tgtCanvasPts.toList(), srcCtrl)
        val outPixels = IntArray(canvasSize * canvasSize)
        val inBoundsMask = FloatArray(canvasSize * canvasSize)
        val filled = BooleanArray(canvasSize * canvasSize)
        val edgeMarginPx = (min(srcW, srcH) * 0.03f).coerceIn(6f, 14f)

        fun computeBoundsAlpha(sx: Float, sy: Float): Float {
            val edgeDist = min(min(sx, (srcW - 1f) - sx), min(sy, (srcH - 1f) - sy))
            return when {
                edgeDist <= 0.5f -> 0f
                edgeDist >= edgeMarginPx -> 1f
                else -> ((edgeDist - 0.5f) / (edgeMarginPx - 0.5f)).coerceIn(0f, 1f)
            }
        }

        // Rasterize each triangle in target canvas space using inverse barycentric coordinates into sourceBitmap
        for (tri in HEAD_MESH_TRIANGLES) {
            val i0 = tri[0]
            val i1 = tri[1]
            val i2 = tri[2]
            val p0 = tgtCanvasPts[i0]
            val p1 = tgtCanvasPts[i1]
            val p2 = tgtCanvasPts[i2]

            val s0 = srcCtrl[i0]
            val s1 = srcCtrl[i1]
            val s2 = srcCtrl[i2]

            val denom = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y)
            if (abs(denom) < 1e-3f) continue
            val invDenom = 1.0f / denom

            val minX = floorInt(min(p0.x, min(p1.x, p2.x))).coerceIn(0, canvasSize - 1)
            val maxX = ceilInt(max(p0.x, max(p1.x, p2.x))).coerceIn(0, canvasSize - 1)
            val minY = floorInt(min(p0.y, min(p1.y, p2.y))).coerceIn(0, canvasSize - 1)
            val maxY = ceilInt(max(p0.y, max(p1.y, p2.y))).coerceIn(0, canvasSize - 1)

            for (y in minY..maxY) {
                val py = y + 0.5f
                val rowOffset = y * canvasSize
                for (x in minX..maxX) {
                    val px = x + 0.5f
                    val w0 = ((p1.y - p2.y) * (px - p2.x) + (p2.x - p1.x) * (py - p2.y)) * invDenom
                    val w1 = ((p2.y - p0.y) * (px - p2.x) + (p0.x - p2.x) * (py - p2.y)) * invDenom
                    val w2 = 1.0f - w0 - w1
                    if (w0 >= -0.015f && w1 >= -0.015f && w2 >= -0.015f) {
                        val sx = w0 * s0.x + w1 * s1.x + w2 * s2.x
                        val sy = w0 * s0.y + w1 * s1.y + w2 * s2.y
                        val idx = rowOffset + x
                        outPixels[idx] = FaceAlignment.sampleBilinearClamped(srcPixels, srcW, srcH, sx, sy)
                        inBoundsMask[idx] = computeBoundsAlpha(sx, sy)
                        filled[idx] = true
                    }
                }
            }
        }

        // Fill exterior hair/background perimeter smoothly using global similarity transform so long hair tips aren't clipped
        val g00 = globalInv[0]
        val g01 = globalInv[1]
        val g02 = globalInv[2]
        val g10 = globalInv[3]
        val g11 = globalInv[4]
        val g12 = globalInv[5]
        for (y in 0 until canvasSize) {
            val rowOffset = y * canvasSize
            for (x in 0 until canvasSize) {
                val idx = rowOffset + x
                if (!filled[idx]) {
                    val sx = g00 * x + g01 * y + g02
                    val sy = g10 * x + g11 * y + g12
                    outPixels[idx] = FaceAlignment.sampleBilinearClamped(srcPixels, srcW, srcH, sx, sy)
                    inBoundsMask[idx] = computeBoundsAlpha(sx, sy)
                }
            }
        }

        val outBmp = Bitmap.createBitmap(canvasSize, canvasSize, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(outPixels, 0, canvasSize, 0, 0, canvasSize, canvasSize)
        return WarpedHeadCanvasResult(
            bitmap = outBmp,
            inBoundsMask = inBoundsMask
        )
    }

    /**
     * Builds a smooth `[0f..1f]` Inner-Face Identity Protection Mask in `canvasSize x canvasSize` space.
     * `1.0f` = strictly protect Source's eyes, brows, nose, lips & central cheeks during Stage 5 Diffusion Bridge.
     */
    fun buildInnerFaceIdentityMask(
        targetMesh: FaceMesh3DResult,
        targetCanvasMatrix2x3: FloatArray,
        canvasSize: Int,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f
    ): FloatArray {
        val mask = FloatArray(canvasSize * canvasSize)
        val ctrl = targetMesh.controlPoints31
        if (ctrl.size < CONTROL_POINT_COUNT) return mask

        val m00 = targetCanvasMatrix2x3[0]
        val m01 = targetCanvasMatrix2x3[1]
        val m02 = targetCanvasMatrix2x3[2]
        val m10 = targetCanvasMatrix2x3[3]
        val m11 = targetCanvasMatrix2x3[4]
        val m12 = targetCanvasMatrix2x3[5]

        fun toCanvas(p: PointF): PointF = PointF(m00 * p.x + m01 * p.y + m02, m10 * p.x + m11 * p.y + m12)
        val lEye = toCanvas(ctrl[0])
        val rEye = toCanvas(ctrl[1])
        val nose = toCanvas(ctrl[2])
        val chin = toCanvas(ctrl[17])

        val eyeMidX = (lEye.x + rEye.x) * 0.5f + offsetX * (canvasSize / 320f)
        val eyeMidY = (lEye.y + rEye.y) * 0.5f + offsetY * (canvasSize / 320f)
        val eyeDist = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble()).toFloat()
            .coerceAtLeast(10f) * scaleAdjust.coerceIn(0.80f, 1.25f)
        val cx = nose.x + offsetX * (canvasSize / 320f)
        val cy = (eyeMidY * 0.45f + chin.y * 0.55f) + offsetY * (canvasSize / 320f)

        val dx = (rEye.x - lEye.x)
        val dy = (rEye.y - lEye.y)
        val len = hypot(dx.toDouble(), dy.toDouble()).toFloat().coerceAtLeast(1f)
        val ux = dx / len
        val uy = dy / len
        val vx = -uy
        val vy = ux

        val rx = eyeDist * 0.92f
        val ry = eyeDist * 1.15f

        for (y in 0 until canvasSize) {
            val rowOffset = y * canvasSize
            for (x in 0 until canvasSize) {
                val px = x - cx
                val py = y - cy
                val lx = (px * ux + py * uy) / rx
                val ly = (px * vx + py * vy) / ry
                val d = sqrt(lx * lx + ly * ly)
                mask[rowOffset + x] = when {
                    d <= 0.72f -> 1.0f
                    d >= 1.08f -> 0.0f
                    else -> ((1.08f - d) / 0.36f).coerceIn(0f, 1f)
                }
            }
        }
        return mask
    }

    private fun estimateSimilarity2x3(
        srcPts: List<PointF>,
        dstPts: List<PointF>
    ): FloatArray {
        val n = min(min(srcPts.size, dstPts.size), 18)
        if (n < 3) return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        var sMx = 0f
        var sMy = 0f
        var dMx = 0f
        var dMy = 0f
        for (i in 0 until n) {
            sMx += srcPts[i].x
            sMy += srcPts[i].y
            dMx += dstPts[i].x
            dMy += dstPts[i].y
        }
        sMx /= n
        sMy /= n
        dMx /= n
        dMy /= n

        var numA = 0f
        var numB = 0f
        var denom = 0f
        for (i in 0 until n) {
            val u = srcPts[i].x - sMx
            val v = srcPts[i].y - sMy
            val xp = dstPts[i].x - dMx
            val yp = dstPts[i].y - dMy
            numA += u * xp + v * yp
            numB += u * yp - v * xp
            denom += u * u + v * v
        }
        if (denom < 1e-6f) return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        val a = numA / denom
        val b = numB / denom
        val tx = dMx - (a * sMx - b * sMy)
        val ty = dMy - (b * sMx + a * sMy)
        return floatArrayOf(a, -b, tx, b, a, ty)
    }

    private fun floorInt(v: Float): Int = kotlin.math.floor(v.toDouble()).toInt()
    private fun ceilInt(v: Float): Int = kotlin.math.ceil(v.toDouble()).toInt()
}
