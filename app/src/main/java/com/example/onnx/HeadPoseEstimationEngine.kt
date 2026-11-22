package com.example.onnx

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import java.io.File
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.util.concurrent.CancellationException
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class SixDRepNetPoseResult(
    val pitchDeg: Float,
    val yawDeg: Float,
    val rollDeg: Float,
    val rotationMatrix3x3: FloatArray,
    val sourceLabel: String
)

/**
 * On-device ONNX Runtime inference engine for `SixDRepNet.onnx` (6D Rotation Head Pose Network).
 *
 * Automatically adapts to common ONNX export conventions:
 *  - Input:  `NCHW` or `NHWC`, dynamic or static spatial size (default `224x224`), `FLOAT` or `FLOAT16`,
 *            RGB normalized with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
 *  - Output: `[1, 3]` direct Euler degrees, `[1, 6]` 6D continuous ortho vectors -> Gram-Schmidt 3x3 -> Euler,
 *            `[1, 3, 3]` / `[1, 9]` 3x3 rotation matrix -> Euler, or 3 separate `yaw/pitch/roll` tensors.
 */
object HeadPoseEstimationEngine {

    private const val DEFAULT_INPUT_SIZE = 224
    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun resolveInstalledModelFile(context: Context): File? {
        val file = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.HEAD_POSE)
        return if (file.exists() && file.length() > 1024L) file else null
    }

    /**
     * Converts raw 6D ortho representation `(a1[0..2], a2[0..2])` into a column-orthonormal 3x3 rotation matrix
     * (row-major 9 floats) via Gram-Schmidt orthogonalization matching `6DRepNet` (`compute_rotation_matrix_from_ortho6d`).
     */
    fun computeRotationMatrixFrom6D(raw6D: FloatArray, offset: Int = 0): FloatArray {
        if (raw6D.size < offset + 6) {
            return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        }
        val a1x = raw6D[offset]
        val a1y = raw6D[offset + 1]
        val a1z = raw6D[offset + 2]
        val a2x = raw6D[offset + 3]
        val a2y = raw6D[offset + 4]
        val a2z = raw6D[offset + 5]

        val n1 = sqrt((a1x * a1x + a1y * a1y + a1z * a1z).toDouble()).toFloat()
        val bx: Float
        val by: Float
        val bz: Float
        if (n1 > 1e-6f && n1.isFinite()) {
            bx = a1x / n1
            by = a1y / n1
            bz = a1z / n1
        } else {
            bx = 1f; by = 0f; bz = 0f
        }

        // b3 = normalize(b1 x a2)
        val cx = by * a2z - bz * a2y
        val cy = bz * a2x - bx * a2z
        val cz = bx * a2y - by * a2x
        val n3 = sqrt((cx * cx + cy * cy + cz * cz).toDouble()).toFloat()
        val zx: Float
        val zy: Float
        val zz: Float
        if (n3 > 1e-6f && n3.isFinite()) {
            zx = cx / n3
            zy = cy / n3
            zz = cz / n3
        } else {
            zx = 0f; zy = 0f; zz = 1f
        }

        // b2 = b3 x b1
        val yx = zy * bz - zz * by
        val yy = zz * bx - zx * bz
        val yz = zx * by - zy * bx

        // Column vectors [b1 | b2 | b3] in row-major 3x3
        return floatArrayOf(
            bx, yx, zx,
            by, yy, zy,
            bz, yz, zz
        )
    }

    /**
     * Extracts `(pitchDeg, yawDeg, rollDeg)` from a row-major 3x3 rotation matrix matching
     * `6DRepNet`'s `compute_euler_angles_from_rotation_matrices`.
     */
    fun rotationMatrixToEulerDeg(r: FloatArray): Triple<Float, Float, Float> {
        if (r.size < 9) return Triple(0f, 0f, 0f)
        val r00 = r[0].toDouble()
        val r10 = r[3].toDouble()
        val r11 = r[4].toDouble()
        val r12 = r[5].toDouble()
        val r20 = r[6].toDouble()
        val r21 = r[7].toDouble()
        val r22 = r[8].toDouble()

        val sy = sqrt(r00 * r00 + r10 * r10)
        val xRad: Double
        val yRad: Double
        val zRad: Double
        if (sy >= 1e-6) {
            xRad = atan2(r21, r22)
            yRad = atan2(-r20, sy)
            zRad = atan2(r10, r00)
        } else {
            xRad = atan2(-r12, r11)
            yRad = atan2(-r20, sy)
            zRad = 0.0
        }

        val pitchDeg = Math.toDegrees(xRad).toFloat().coerceIn(-75f, 75f)
        val yawDeg = (-Math.toDegrees(yRad).toFloat()).coerceIn(-85f, 85f)
        val rollDeg = Math.toDegrees(zRad).toFloat().coerceIn(-180f, 180f)
        return Triple(pitchDeg, yawDeg, rollDeg)
    }

    /**
     * Parses raw ONNX output tensors from any common SixDRepNet / head-pose model export into [SixDRepNetPoseResult].
     * Exposed internally for unit testing without native ONNX binaries.
     */
    internal fun parsePoseOutputs(
        outputs: List<Triple<String, LongArray, FloatArray>>,
        geometricRollHintDeg: Float? = null,
        sourceLabel: String = "SixDRepNet.onnx"
    ): SixDRepNetPoseResult? {
        if (outputs.isEmpty()) return null

        // Case 1: Named separate yaw / pitch / roll tensors (either scalar [1,1] or binned [1,66])
        if (outputs.size >= 3) {
            var yaw: Float? = null
            var pitch: Float? = null
            var roll: Float? = null
            for ((name, _, data) in outputs) {
                val lower = name.lowercase()
                val angle = extractSingleOrBinnedAngle(data) ?: continue
                when {
                    lower.contains("yaw") -> yaw = angle
                    lower.contains("pitch") -> pitch = angle
                    lower.contains("roll") -> roll = angle
                }
            }
            if (yaw != null && pitch != null && roll != null) {
                val p = pitch.coerceIn(-75f, 75f)
                val y = yaw.coerceIn(-85f, 85f)
                val r = roll.coerceIn(-180f, 180f)
                return SixDRepNetPoseResult(
                    pitchDeg = p,
                    yawDeg = y,
                    rollDeg = r,
                    rotationMatrix3x3 = Face3DReconstruction.buildRotationMatrix3x3(p, y, r),
                    sourceLabel = sourceLabel
                )
            }
        }

        // Case 2: Primary tensor -> check [1, 3, 3]/[1, 9], [1, 6], or [1, 3]
        val (primaryName, primaryShape, primaryData) = outputs.first()
        if (primaryData.any { !it.isFinite() }) return null

        val isMatrix3x3 = (primaryShape.size == 3 && primaryShape[1] == 3L && primaryShape[2] == 3L) ||
            (primaryData.size == 9)
        if (isMatrix3x3 && primaryData.size >= 9) {
            val mat = primaryData.copyOfRange(0, 9)
            val (p, y, r) = rotationMatrixToEulerDeg(mat)
            return SixDRepNetPoseResult(
                pitchDeg = p,
                yawDeg = y,
                rollDeg = r,
                rotationMatrix3x3 = mat,
                sourceLabel = sourceLabel
            )
        }

        if (primaryData.size == 6) {
            val mat = computeRotationMatrixFrom6D(primaryData, 0)
            val (p, y, r) = rotationMatrixToEulerDeg(mat)
            return SixDRepNetPoseResult(
                pitchDeg = p,
                yawDeg = y,
                rollDeg = r,
                rotationMatrix3x3 = mat,
                sourceLabel = sourceLabel
            )
        }

        if (primaryData.size == 3) {
            val a0 = primaryData[0]
            val a1 = primaryData[1]
            val a2 = primaryData[2]
            val lowerName = primaryName.lowercase()

            // Most SixDRepNet ONNX wrappers export [yaw, pitch, roll] (ypr), while some export [pitch, yaw, roll].
            val isPitchYawRoll = when {
                lowerName.contains("pyr") || lowerName.contains("pitch_yaw_roll") -> true
                lowerName.contains("ypr") || lowerName.contains("yaw_pitch_roll") -> false
                geometricRollHintDeg != null && abs(geometricRollHintDeg) > 10f -> {
                    // Roll is always index 2 in both conventions; check if index 0 or index 2 matches roll just in case
                    abs(a2 - geometricRollHintDeg) <= abs(a0 - geometricRollHintDeg)
                }
                else -> false
            }

            val pitch: Float
            val yaw: Float
            val roll: Float
            if (isPitchYawRoll) {
                pitch = a0.coerceIn(-75f, 75f)
                yaw = a1.coerceIn(-85f, 85f)
                roll = a2.coerceIn(-180f, 180f)
            } else {
                // Default SixDRepNet demo/ONNX export order: [yaw, pitch, roll]
                yaw = a0.coerceIn(-85f, 85f)
                pitch = a1.coerceIn(-75f, 75f)
                roll = a2.coerceIn(-180f, 180f)
            }
            return SixDRepNetPoseResult(
                pitchDeg = pitch,
                yawDeg = yaw,
                rollDeg = roll,
                rotationMatrix3x3 = Face3DReconstruction.buildRotationMatrix3x3(pitch, yaw, roll),
                sourceLabel = sourceLabel
            )
        }

        return null
    }

    private fun extractSingleOrBinnedAngle(data: FloatArray): Float? {
        if (data.isEmpty() || data.any { !it.isFinite() }) return null
        if (data.size == 1) return data[0]
        // HopeNet/6DRepNet style 66-bin softmax over [-99..+99] step 3 deg
        if (data.size in 60..72) {
            var maxVal = -Float.MAX_VALUE
            for (v in data) if (v > maxVal) maxVal = v
            var sumExp = 0.0
            var expectedIdx = 0.0
            for (i in data.indices) {
                val e = kotlin.math.exp((data[i] - maxVal).toDouble())
                sumExp += e
                expectedIdx += e * i
            }
            if (sumExp <= 0.0) return null
            val idxMean = (expectedIdx / sumExp).toFloat()
            val halfSpan = (data.size - 1) * 1.5f
            return idxMean * 3.0f - halfSpan
        }
        return null
    }

    fun estimateHeadPoseOnnx(
        ortEnv: OrtEnvironment?,
        headPoseFile: File?,
        bitmap: Bitmap,
        face: DetectedFace,
        preferHardwareAccel: Boolean = false,
        isCancelled: () -> Boolean = { false }
    ): SixDRepNetPoseResult? {
        if (isCancelled()) throw CancellationException("Head pose estimation cancelled by user.")
        if (ortEnv == null || headPoseFile == null || !headPoseFile.exists() || headPoseFile.length() <= 1024L) {
            return null
        }

        val session = OnnxProtobufInspector.getOrCreateCachedSession(
            ortEnv = ortEnv,
            file = headPoseFile,
            preferHardwareAcceleration = preferHardwareAccel
        ) ?: return null

        if (isCancelled()) throw CancellationException("Head pose estimation cancelled by user.")

        val inputEntry = session.inputInfo.entries.firstOrNull() ?: return null
        val inputName = inputEntry.key
        val tInfo = inputEntry.value.info as? TensorInfo ?: return null
        val shape = tInfo.shape
        val isNhwc = shape.size == 4 && (shape[3] == 3L || shape[3] == 4L)
        val hDim = if (shape.size == 4) {
            val raw = if (isNhwc) shape[1] else shape[2]
            if (raw in 64L..512L) raw.toInt() else DEFAULT_INPUT_SIZE
        } else {
            DEFAULT_INPUT_SIZE
        }
        val wDim = if (shape.size == 4) {
            val raw = if (isNhwc) shape[2] else shape[3]
            if (raw in 64L..512L) raw.toInt() else hDim
        } else {
            hDim
        }

        // Generous square face crop (1.40x expanded bbox) matching SixDRepNet training framing
        val box = face.boundingBox
        val cx = box.centerX()
        val cy = box.centerY()
        val half = (max(box.width(), box.height()) * 0.70f).coerceAtLeast(16f)
        val cropRect = RectF(cx - half, cy - half, cx + half, cy + half)
        val side = cropRect.width().coerceAtLeast(16f)

        val srcW = bitmap.width
        val srcH = bitmap.height
        val srcPixels = IntArray(srcW * srcH)
        bitmap.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)

        val hw = wDim * hDim
        val cropPixels = IntArray(hw)
        for (y in 0 until hDim) {
            val sy = cropRect.top + (y + 0.5f) / hDim * side
            for (x in 0 until wDim) {
                val sx = cropRect.left + (x + 0.5f) / wDim * side
                cropPixels[y * wDim + x] = FaceAlignment.sampleBilinearClamped(srcPixels, srcW, srcH, sx, sy)
            }
        }

        if (isCancelled()) throw CancellationException("Head pose estimation cancelled by user.")

        val tensorShape = if (isNhwc) {
            longArrayOf(1L, hDim.toLong(), wDim.toLong(), 3L)
        } else {
            longArrayOf(1L, 3L, hDim.toLong(), wDim.toLong())
        }

        val rawOutputs: List<Triple<String, LongArray, FloatArray>> = if (tInfo.type == OnnxJavaType.FLOAT16) {
            val shortBuf = ShortBuffer.allocate(3 * hw)
            if (isNhwc) {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    val r = (((c ushr 16) and 0xFF) / 255.0f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                    val g = (((c ushr 8) and 0xFF) / 255.0f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                    val b = ((c and 0xFF) / 255.0f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                    shortBuf.put(i * 3, DepthEstimationEngine.floatToFloat16(r))
                    shortBuf.put(i * 3 + 1, DepthEstimationEngine.floatToFloat16(g))
                    shortBuf.put(i * 3 + 2, DepthEstimationEngine.floatToFloat16(b))
                }
            } else {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    val r = (((c ushr 16) and 0xFF) / 255.0f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                    val g = (((c ushr 8) and 0xFF) / 255.0f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                    val b = ((c and 0xFF) / 255.0f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                    shortBuf.put(i, DepthEstimationEngine.floatToFloat16(r))
                    shortBuf.put(hw + i, DepthEstimationEngine.floatToFloat16(g))
                    shortBuf.put(2 * hw + i, DepthEstimationEngine.floatToFloat16(b))
                }
            }
            shortBuf.rewind()
            OnnxTensor.createTensor(ortEnv, shortBuf, tensorShape, OnnxJavaType.FLOAT16).use { inT ->
                session.run(mapOf(inputName to inT)).use { res -> extractAllOutputTensors(session, res) }
            }
        } else {
            val floatBuf = FloatBuffer.allocate(3 * hw)
            if (isNhwc) {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    val r = (((c ushr 16) and 0xFF) / 255.0f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                    val g = (((c ushr 8) and 0xFF) / 255.0f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                    val b = ((c and 0xFF) / 255.0f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                    floatBuf.put(i * 3, r)
                    floatBuf.put(i * 3 + 1, g)
                    floatBuf.put(i * 3 + 2, b)
                }
            } else {
                for (i in 0 until hw) {
                    val c = cropPixels[i]
                    val r = (((c ushr 16) and 0xFF) / 255.0f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                    val g = (((c ushr 8) and 0xFF) / 255.0f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                    val b = ((c and 0xFF) / 255.0f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                    floatBuf.put(i, r)
                    floatBuf.put(hw + i, g)
                    floatBuf.put(2 * hw + i, b)
                }
            }
            floatBuf.rewind()
            OnnxTensor.createTensor(ortEnv, floatBuf, tensorShape).use { inT ->
                session.run(mapOf(inputName to inT)).use { res -> extractAllOutputTensors(session, res) }
            }
        }

        val geometricRoll = if (face.landmarks5.size >= 2) {
            val lEye = face.landmarks5[0]
            val rEye = face.landmarks5[1]
            Math.toDegrees(atan2((rEye.y - lEye.y).toDouble(), (rEye.x - lEye.x).toDouble())).toFloat()
        } else {
            null
        }

        return parsePoseOutputs(
            outputs = rawOutputs,
            geometricRollHintDeg = geometricRoll,
            sourceLabel = headPoseFile.name
        )
    }

    private fun extractAllOutputTensors(
        session: OrtSession,
        res: OrtSession.Result
    ): List<Triple<String, LongArray, FloatArray>> {
        val outputNames = session.outputNames.toList()
        val list = ArrayList<Triple<String, LongArray, FloatArray>>(res.size())
        for (i in 0 until res.size()) {
            val t = res[i] as? OnnxTensor ?: continue
            val shape = t.info.shape
            val total = shape.fold(1L) { acc, v -> acc * max(1L, v) }.toInt().coerceAtLeast(1)
            val floats = if (t.info.type == OnnxJavaType.FLOAT16) {
                val sArr = ShortArray(total)
                t.shortBuffer.get(sArr)
                FloatArray(total) { idx -> DepthEstimationEngine.float16ToFloat(sArr[idx]) }
            } else {
                val fArr = FloatArray(total)
                t.floatBuffer.get(fArr)
                fArr
            }
            val outName = outputNames.getOrElse(i) { "output_$i" }
            list.add(Triple(outName, shape, floats))
        }
        return list
    }

    /**
     * Enriches a single [DetectedFace] with SixDRepNet 3D Pitch/Yaw/Roll and updates its
     * 36-point contour & 106-point 3DMM mesh when `SixDRepNet.onnx` is present and runs cleanly.
     */
    fun enrichFaceWithPose(
        ortEnv: OrtEnvironment?,
        headPoseFile: File?,
        bitmap: Bitmap,
        face: DetectedFace,
        preferHardwareAccel: Boolean = false,
        isCancelled: () -> Boolean = { false }
    ): Pair<DetectedFace, Boolean> {
        val poseRes = runCatching {
            estimateHeadPoseOnnx(
                ortEnv = ortEnv,
                headPoseFile = headPoseFile,
                bitmap = bitmap,
                face = face,
                preferHardwareAccel = preferHardwareAccel,
                isCancelled = isCancelled
            )
        }.onFailure { err ->
            if (err is CancellationException) throw err
        }.getOrNull() ?: return face to false

        val fusedPose = Face3DReconstruction.estimate3DHeadPose(
            landmarks5 = face.landmarks5,
            hintPitch = poseRes.pitchDeg,
            hintYaw = poseRes.yawDeg,
            hintRoll = poseRes.rollDeg,
            neuralPoseWeight = 0.85f
        )
        val updatedContour = if (face.landmarks5.size >= 5) {
            ScrfdFaceDetector.build3DBiometricFaceOvalContour(
                landmarks5 = face.landmarks5,
                eulerX = fusedPose.pitchDeg,
                eulerY = fusedPose.yawDeg,
                eulerZ = fusedPose.rollDeg,
                imgW = bitmap.width,
                imgH = bitmap.height
            )
        } else {
            face.faceContourPoints
        }
        val recon3D = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(
            landmarks5 = face.landmarks5,
            eulerX = fusedPose.pitchDeg,
            eulerY = fusedPose.yawDeg,
            eulerZ = fusedPose.rollDeg,
            existingContourPoints = updatedContour,
            bitmap = bitmap,
            neuralPoseWeight = 0.85f
        )
        val updatedFace = face.copy(
            detectorSource = if (face.detectorSource.contains("6DRepNet")) {
                face.detectorSource
            } else {
                "${face.detectorSource} + SixDRepNet"
            },
            faceContourPoints = updatedContour,
            eulerX = recon3D.headPose.pitchDeg,
            eulerY = recon3D.headPose.yawDeg,
            eulerZ = recon3D.headPose.rollDeg,
            reconstructed3D = recon3D,
            landmarks106 = recon3D.landmarks106
        )
        return updatedFace to true
    }

    fun enrichFacesIfModelPresent(
        context: Context,
        ortEnv: OrtEnvironment?,
        bitmap: Bitmap,
        faces: List<DetectedFace>,
        preferHardwareAccel: Boolean = false,
        isCancelled: () -> Boolean = { false }
    ): Pair<List<DetectedFace>, Boolean> {
        if (faces.isEmpty() || ortEnv == null) return faces to false
        val file = resolveInstalledModelFile(context) ?: return faces to false
        var anyUsed = false
        val enriched = faces.map { face ->
            if (isCancelled()) throw CancellationException("Head pose estimation cancelled by user.")
            val (updated, used) = enrichFaceWithPose(
                ortEnv = ortEnv,
                headPoseFile = file,
                bitmap = bitmap,
                face = face,
                preferHardwareAccel = preferHardwareAccel,
                isCancelled = isCancelled
            )
            if (used) anyUsed = true
            updated
        }
        return enriched to anyUsed
    }
}
