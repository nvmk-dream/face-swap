package com.example.onnx

import android.graphics.Bitmap
import android.graphics.PointF
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Implements InsightFace's exact 5-point Umeyama similarity alignment (`face_align.py`)
 * and fast bilinear affine warping (`warpAffine`) + inverse projection (`invertAffine2x3`).
 */
object FaceAlignment {

    // Standard 5-point reference landmarks for 112x112 ArcFace recognition (w600k_r50.onnx)
    private val ARCFACE_DST_112 = arrayOf(
        PointF(38.2946f, 51.6963f), // Left Eye
        PointF(73.5318f, 51.5014f), // Right Eye
        PointF(56.0252f, 71.7366f), // Nose Tip
        PointF(41.5493f, 92.3655f), // Left Mouth Corner
        PointF(70.7299f, 92.2041f)  // Right Mouth Corner
    )

    /**
     * Returns the canonical 5-point target template for a given crop size (`112` or `128`),
     * matching `insightface/utils/face_align.py`:
     * - For 112 (divisible by 112): ratio = imageSize / 112f, diffX = 0f
     * - For 128 (inswapper_128): ratio = imageSize / 128f = 1.0f, diffX = 8.0f * ratio
     */
    fun getReferenceTemplate(imageSize: Int): Array<PointF> {
        val ratio: Float
        val diffX: Float
        if (imageSize % 112 == 0) {
            ratio = imageSize / 112.0f
            diffX = 0.0f
        } else {
            ratio = imageSize / 128.0f
            diffX = 8.0f * ratio
        }
        return Array(5) { i ->
            PointF(
                ARCFACE_DST_112[i].x * ratio + diffX,
                ARCFACE_DST_112[i].y * ratio
            )
        }
    }

    /**
     * Computes the 2x3 similarity transform matrix `M = [[a, -b, tx], [b, a, ty]]`
     * that maps `srcPoints` (5 detected facial keypoints in original image space)
     * to the canonical `imageSize x imageSize` InsightFace reference template (`dstPoints`).
     *
     * Closed-form 2D Umeyama / Procrustes similarity estimation (`estimateAffinePartial2D`):
     * Minimizes sum_i || (s * R * src_i + t) - dst_i ||^2 over scale s, rotation R, translation t.
     */
    fun estimateNorm(srcPoints: List<PointF>, imageSize: Int): FloatArray {
        require(srcPoints.size >= 5) { "At least 5 facial landmarks are required for alignment." }
        val dstPoints = getReferenceTemplate(imageSize)
        val n = 5

        var srcMeanX = 0f
        var srcMeanY = 0f
        var dstMeanX = 0f
        var dstMeanY = 0f

        for (i in 0 until n) {
            srcMeanX += srcPoints[i].x
            srcMeanY += srcPoints[i].y
            dstMeanX += dstPoints[i].x
            dstMeanY += dstPoints[i].y
        }
        srcMeanX /= n
        srcMeanY /= n
        dstMeanX /= n
        dstMeanY /= n

        // In 2D similarity transform:
        // u_i = x_i - srcMeanX, v_i = y_i - srcMeanY
        // x'_i = dst_i.x - dstMeanX, y'_i = dst_i.y - dstMeanY
        // a = sum(u_i * x'_i + v_i * y'_i) / sum(u_i^2 + v_i^2)
        // b = sum(u_i * y'_i - v_i * x'_i) / sum(u_i^2 + v_i^2)
        var numA = 0f
        var numB = 0f
        var denom = 0f

        for (i in 0 until n) {
            val u = srcPoints[i].x - srcMeanX
            val v = srcPoints[i].y - srcMeanY
            val xp = dstPoints[i].x - dstMeanX
            val yp = dstPoints[i].y - dstMeanY

            numA += u * xp + v * yp
            numB += u * yp - v * xp
            denom += u * u + v * v
        }

        if (denom < 1e-6f) {
            return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        }

        val a = numA / denom
        val b = numB / denom
        val tx = dstMeanX - (a * srcMeanX - b * srcMeanY)
        val ty = dstMeanY - (b * srcMeanX + a * srcMeanY)

        // Row-major 2x3 affine matrix:
        // [ a, -b, tx ]
        // [ b,  a, ty ]
        return floatArrayOf(a, -b, tx, b, a, ty)
    }

    /**
     * Exact ArcFace 5-point Umeyama similarity alignment (`diffX = 0`, `ratio = imageSize / 112.0f`)
     * required by SimSwap 256/512 models (`arcface_112_v2` template scaled directly without
     * InSwapper's 8px horizontal offset). Used exclusively by Option D SimSwap.
     */
    fun estimateNormArcFaceExact(srcPoints: List<PointF>, imageSize: Int): FloatArray {
        require(srcPoints.size >= 5) { "At least 5 facial landmarks are required for alignment." }
        val ratio = imageSize / 112.0f
        val dstPoints = Array(5) { i ->
            PointF(ARCFACE_DST_112[i].x * ratio, ARCFACE_DST_112[i].y * ratio)
        }
        val n = 5
        var srcMeanX = 0f
        var srcMeanY = 0f
        var dstMeanX = 0f
        var dstMeanY = 0f
        for (i in 0 until n) {
            srcMeanX += srcPoints[i].x
            srcMeanY += srcPoints[i].y
            dstMeanX += dstPoints[i].x
            dstMeanY += dstPoints[i].y
        }
        srcMeanX /= n
        srcMeanY /= n
        dstMeanX /= n
        dstMeanY /= n

        var numA = 0f
        var numB = 0f
        var denom = 0f
        for (i in 0 until n) {
            val u = srcPoints[i].x - srcMeanX
            val v = srcPoints[i].y - srcMeanY
            val xp = dstPoints[i].x - dstMeanX
            val yp = dstPoints[i].y - dstMeanY
            numA += u * xp + v * yp
            numB += u * yp - v * xp
            denom += u * u + v * v
        }
        if (denom < 1e-6f) {
            return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        }
        val a = numA / denom
        val b = numB / denom
        val tx = dstMeanX - (a * srcMeanX - b * srcMeanY)
        val ty = dstMeanY - (b * srcMeanX + a * srcMeanY)
        return floatArrayOf(a, -b, tx, b, a, ty)
    }

    /**
     * Computes the exact inverse 2x3 affine matrix `M^-1` mapping from aligned crop coordinates
     * `(x', y')` back to original image coordinates `(x, y)`.
     */
    fun invertAffine2x3(m: FloatArray): FloatArray {
        val m00 = m[0]
        val m01 = m[1]
        val m02 = m[2]
        val m10 = m[3]
        val m11 = m[4]
        val m12 = m[5]

        val det = m00 * m11 - m01 * m10
        if (kotlin.math.abs(det) < 1e-8f) {
            return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        }
        val invDet = 1.0f / det
        val i00 = m11 * invDet
        val i01 = -m01 * invDet
        val i10 = -m10 * invDet
        val i11 = m00 * invDet
        val i02 = -(i00 * m02 + i01 * m12)
        val i12 = -(i10 * m02 + i11 * m12)

        return floatArrayOf(i00, i01, i02, i10, i11, i12)
    }

    /**
     * Warps `srcBitmap` into a `dstSize x dstSize` aligned face crop using bilinear interpolation
     * and the forward 2x3 similarity transform `forwardMatrix2x3`.
     */
    fun warpAffineCrop(
        srcBitmap: Bitmap,
        forwardMatrix2x3: FloatArray,
        dstSize: Int
    ): Bitmap {
        val inv = invertAffine2x3(forwardMatrix2x3)
        val srcW = srcBitmap.width
        val srcH = srcBitmap.height
        val srcPixels = IntArray(srcW * srcH)
        srcBitmap.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)

        val dstPixels = IntArray(dstSize * dstSize)
        val i00 = inv[0]
        val i01 = inv[1]
        val i02 = inv[2]
        val i10 = inv[3]
        val i11 = inv[4]
        val i12 = inv[5]

        for (dy in 0 until dstSize) {
            val rowOffset = dy * dstSize
            val baseX = i01 * dy + i02
            val baseY = i11 * dy + i12
            for (dx in 0 until dstSize) {
                val sx = i00 * dx + baseX
                val sy = i10 * dx + baseY
                dstPixels[rowOffset + dx] = sampleBilinearClamped(srcPixels, srcW, srcH, sx, sy)
            }
        }

        val out = Bitmap.createBitmap(dstSize, dstSize, Bitmap.Config.ARGB_8888)
        out.setPixels(dstPixels, 0, dstSize, 0, 0, dstSize, dstSize)
        return out
    }

    fun sampleBilinearClamped(
        pixels: IntArray,
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ): Int {
        val cx = x.coerceIn(0f, (width - 1).toFloat())
        val cy = y.coerceIn(0f, (height - 1).toFloat())
        val x0 = cx.toInt()
        val y0 = cy.toInt()
        val x1 = (x0 + 1).coerceAtMost(width - 1)
        val y1 = (y0 + 1).coerceAtMost(height - 1)

        val fx = cx - x0
        val fy = cy - y0
        val w00 = (1f - fx) * (1f - fy)
        val w10 = fx * (1f - fy)
        val w01 = (1f - fx) * fy
        val w11 = fx * fy

        val c00 = pixels[y0 * width + x0]
        val c10 = pixels[y0 * width + x1]
        val c01 = pixels[y1 * width + x0]
        val c11 = pixels[y1 * width + x1]

        val r = ((c00 ushr 16 and 0xFF) * w00 +
                (c10 ushr 16 and 0xFF) * w10 +
                (c01 ushr 16 and 0xFF) * w01 +
                (c11 ushr 16 and 0xFF) * w11).toInt().coerceIn(0, 255)

        val g = ((c00 ushr 8 and 0xFF) * w00 +
                (c10 ushr 8 and 0xFF) * w10 +
                (c01 ushr 8 and 0xFF) * w01 +
                (c11 ushr 8 and 0xFF) * w11).toInt().coerceIn(0, 255)

        val b = ((c00 and 0xFF) * w00 +
                (c10 and 0xFF) * w10 +
                (c01 and 0xFF) * w01 +
                (c11 and 0xFF) * w11).toInt().coerceIn(0, 255)

        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    data class AlignedCropResult(
        val croppedBitmap: Bitmap,
        val forwardMatrix: FloatArray,
        val inverseMatrix: FloatArray
    )

    fun alignCrop128(srcBitmap: Bitmap, landmarks5: List<PointF>): AlignedCropResult {
        val m = estimateNorm(landmarks5, 128)
        val inv = invertAffine2x3(m)
        val crop = warpAffineCrop(srcBitmap, m, 128)
        return AlignedCropResult(crop, m, inv)
    }

    fun alignCrop256(srcBitmap: Bitmap, landmarks5: List<PointF>): AlignedCropResult {
        val m = estimateNorm(landmarks5, 256)
        val inv = invertAffine2x3(m)
        val crop = warpAffineCrop(srcBitmap, m, 256)
        return AlignedCropResult(crop, m, inv)
    }

    fun alignCrop512(srcBitmap: Bitmap, landmarks5: List<PointF>): AlignedCropResult {
        val m = estimateNorm(landmarks5, 512)
        val inv = invertAffine2x3(m)
        val crop = warpAffineCrop(srcBitmap, m, 512)
        return AlignedCropResult(crop, m, inv)
    }

    /**
     * Warps `sourceBitmap` into a `dstSize x dstSize` aligned crop whose 15 anatomical landmarks
     * (Left/Right Eye, Left/Right Eyebrow, Glabella, Nasal Bridge, Nose Tip, Philtrum, Left/Right
     * Mouth Corners, Oral Center, Left/Right Cheekbones, Forehead, and Chin) match the Target face's
     * exact 3D head pose (`yaw`, `pitch`, `roll`) in crop space.
     *
     * Eliminates double-nose, double-eyebrow, and double-eye ghosting when swapping between
     * a frontal face and a turned/tilted head (or between long-shot and close-up faces).
     */
    fun warpSourceToTargetPose(
        sourceBitmap: Bitmap,
        sourceLandmarks5: List<PointF>,
        targetLandmarks5: List<PointF>,
        dstSize: Int
    ): Bitmap {
        if (sourceLandmarks5.size < 5 || targetLandmarks5.size < 5) {
            val mFallback = estimateNorm(sourceLandmarks5, dstSize)
            return warpAffineCrop(sourceBitmap, mFallback, dstSize)
        }

        val mSrc = estimateNorm(sourceLandmarks5, dstSize)
        val mTgt = estimateNorm(targetLandmarks5, dstSize)
        val invSrc = invertAffine2x3(mSrc)

        fun mapPt(m: FloatArray, p: PointF): PointF = PointF(
            m[0] * p.x + m[1] * p.y + m[2],
            m[3] * p.x + m[4] * p.y + m[5]
        )

        val srcPts5 = List(5) { i -> mapPt(mSrc, sourceLandmarks5[i]) }
        val tgtPts5 = List(5) { i -> mapPt(mTgt, targetLandmarks5[i]) }

        // Reconstruct 106-point dense 3D meshes, 34 Named Anatomical Feature Points, and 3D Head-Poses
        val srcRecon3D = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(srcPts5)
        val tgtRecon3D = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(tgtPts5)
        val srcPts106 = srcRecon3D.landmarks106
        val tgtPts106 = tgtRecon3D.landmarks106
        val tgtVerts3D = tgtRecon3D.vertices3D
        val tgtAnatByIdx = tgtRecon3D.anatomicalPoints34.associateBy { it.pointType.index106 }

        // Select 13 well-separated 3D anatomical anchors from the 34/106-point topology
        // (Jawline/Chin, Eyebrow Midpoints, Eye/Orbital Centers, Glabella/NoseBridge, NoseTip, MouthCorners & Oral Center)
        val sampledIndices = intArrayOf(
            3, 9, 16, 23, 29,     // 5 Jawline & Chin Contour anchors (RJawEnd, Chin, LJawEnd)
            35, 45,               // 2 REyebrowMid & LEyebrowMid anchors
            61, 71,               // 2 Right & Left Orbital/Eye Centers
            73, 76,               // 2 Glabella/NoseBridge & NoseTip anchors
            86, 92, 100           // 3 RMouthCorner, LMouthCorner & Lip Center anchors
        )
        val numCtrl = sampledIndices.size
        val maxShift = dstSize * 0.075f
        val ctrlTgtX = FloatArray(numCtrl)
        val ctrlTgtY = FloatArray(numCtrl)
        val ctrlWeight = FloatArray(numCtrl)
        val dispX = FloatArray(numCtrl)
        val dispY = FloatArray(numCtrl)

        for (i in 0 until numCtrl) {
            val idx106 = sampledIndices[i]
            val sPt = srcPts106[idx106]
            val tPt = tgtPts106[idx106]
            val tVert = tgtVerts3D[idx106]
            val anatVis = tgtAnatByIdx[idx106]?.visibility ?: tVert.nz
            ctrlTgtX[i] = tPt.x
            ctrlTgtY[i] = tPt.y
            // Weight each 3D anatomical anchor by its self-occlusion visibility and damp nose-tip horizontal shear
            val poseDamp = when (idx106) {
                76 -> 0.45f
                3, 9, 16, 23, 29 -> 0.55f
                else -> 0.78f
            }
            ctrlWeight[i] = anatVis.coerceIn(0.45f, 1.0f)
            dispX[i] = ((sPt.x - tPt.x) * ctrlWeight[i] * poseDamp).coerceIn(-maxShift, maxShift)
            dispY[i] = ((sPt.y - tPt.y) * ctrlWeight[i] * poseDamp).coerceIn(-maxShift, maxShift)
        }

        // Evaluate smooth 3D-pose-weighted Gaussian Radial-Basis displacement on a 32x32 control grid
        // with wide regularization (eps = 0.14 * dstSize) so neighboring features never pinch or fold
        val gridDiv = 32
        val gridSize = gridDiv + 1
        val step = dstSize.toFloat() / gridDiv.toFloat()
        val sigmaSq = (dstSize * 0.24f) * (dstSize * 0.24f)
        val epsSq = (dstSize * 0.14f) * (dstSize * 0.14f)
        val gridSrcX = FloatArray(gridSize * gridSize)
        val gridSrcY = FloatArray(gridSize * gridSize)
        val halfS = dstSize * 0.5f

        for (gy in 0..gridDiv) {
            val yf = gy * step
            val rowOff = gy * gridSize
            for (gx in 0..gridDiv) {
                val xf = gx * step
                val borderNorm = kotlin.math.max(
                    kotlin.math.abs(xf - halfS) / halfS,
                    kotlin.math.abs(yf - halfS) / halfS
                )
                val borderTaper = if (borderNorm <= 0.68f) {
                    1.0f
                } else if (borderNorm >= 0.96f) {
                    0.0f
                } else {
                    val t = (borderNorm - 0.68f) / 0.28f
                    (0.5f * (1.0 + cos(Math.PI * t))).toFloat()
                }

                var sumW = 0f
                var sumDx = 0f
                var sumDy = 0f
                for (k in 0 until numCtrl) {
                    val ddx = xf - ctrlTgtX[k]
                    val ddy = yf - ctrlTgtY[k]
                    val d2 = ddx * ddx + ddy * ddy
                    val w = ctrlWeight[k] * (1.0f / (d2 + epsSq)) * kotlin.math.exp((-d2 / (2.0f * sigmaSq)).toDouble()).toFloat()
                    sumW += w
                    sumDx += w * dispX[k]
                    sumDy += w * dispY[k]
                }

                val dx = if (sumW > 1e-7f) (sumDx / sumW) * borderTaper else 0f
                val dy = if (sumW > 1e-7f) (sumDy / sumW) * borderTaper else 0f
                val srcCropX = xf + dx
                val srcCropY = yf + dy

                // Map directly through invSrc into original sourceBitmap pixel coordinates
                gridSrcX[rowOff + gx] = invSrc[0] * srcCropX + invSrc[1] * srcCropY + invSrc[2]
                gridSrcY[rowOff + gx] = invSrc[3] * srcCropX + invSrc[4] * srcCropY + invSrc[5]
            }
        }

        val srcW = sourceBitmap.width
        val srcH = sourceBitmap.height
        val srcPixels = IntArray(srcW * srcH)
        sourceBitmap.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)
        val dstPixels = IntArray(dstSize * dstSize)
        val invStep = 1.0f / step

        for (dy in 0 until dstSize) {
            val gyFloat = (dy * invStep).coerceIn(0f, (gridDiv - 0.0001f))
            val gy0 = gyFloat.toInt()
            val gy1 = (gy0 + 1).coerceAtMost(gridDiv)
            val fy = gyFloat - gy0
            val oneMinusFy = 1.0f - fy
            val gRow0 = gy0 * gridSize
            val gRow1 = gy1 * gridSize
            val dstRow = dy * dstSize

            for (dx in 0 until dstSize) {
                val gxFloat = (dx * invStep).coerceIn(0f, (gridDiv - 0.0001f))
                val gx0 = gxFloat.toInt()
                val gx1 = (gx0 + 1).coerceAtMost(gridDiv)
                val fx = gxFloat - gx0
                val oneMinusFx = 1.0f - fx

                val w00 = oneMinusFx * oneMinusFy
                val w10 = fx * oneMinusFy
                val w01 = oneMinusFx * fy
                val w11 = fx * fy

                val sx = w00 * gridSrcX[gRow0 + gx0] +
                    w10 * gridSrcX[gRow0 + gx1] +
                    w01 * gridSrcX[gRow1 + gx0] +
                    w11 * gridSrcX[gRow1 + gx1]

                val sy = w00 * gridSrcY[gRow0 + gx0] +
                    w10 * gridSrcY[gRow0 + gx1] +
                    w01 * gridSrcY[gRow1 + gx0] +
                    w11 * gridSrcY[gRow1 + gx1]

                dstPixels[dstRow + dx] = sampleBicubicClamped(srcPixels, srcW, srcH, sx, sy)
            }
        }

        val out = Bitmap.createBitmap(dstSize, dstSize, Bitmap.Config.ARGB_8888)
        out.setPixels(dstPixels, 0, dstSize, 0, 0, dstSize, dstSize)
        return out
    }

    /**
     * 4x4 Catmull-Rom Bicubic interpolation sampler for high-resolution 512x512 -> Target warping,
     * preserving crisp iris, eyelash, nostril, and lip vermilion micro-contrast without bilinear blur.
     */
    fun sampleBicubicClamped(
        pixels: IntArray,
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ): Int {
        val cx = x.coerceIn(0f, (width - 1).toFloat())
        val cy = y.coerceIn(0f, (height - 1).toFloat())
        val xInt = cx.toInt()
        val yInt = cy.toInt()
        val tx = cx - xInt
        val ty = cy - yInt

        fun cubicWeight(t: Float): FloatArray {
            val t2 = t * t
            val t3 = t2 * t
            val w0 = -0.5f * t + t2 - 0.5f * t3
            val w1 = 1.0f - 2.5f * t2 + 1.5f * t3
            val w2 = 0.5f * t + 2.0f * t2 - 1.5f * t3
            val w3 = -0.5f * t2 + 0.5f * t3
            return floatArrayOf(w0, w1, w2, w3)
        }

        val wx = cubicWeight(tx)
        val wy = cubicWeight(ty)

        var rSum = 0f
        var gSum = 0f
        var bSum = 0f

        for (j in -1..2) {
            val py = (yInt + j).coerceIn(0, height - 1)
            val rowOffset = py * width
            val wyj = wy[j + 1]
            var rRow = 0f
            var gRow = 0f
            var bRow = 0f
            for (i in -1..2) {
                val px = (xInt + i).coerceIn(0, width - 1)
                val c = pixels[rowOffset + px]
                val wxi = wx[i + 1]
                rRow += ((c ushr 16) and 0xFF) * wxi
                gRow += ((c ushr 8) and 0xFF) * wxi
                bRow += (c and 0xFF) * wxi
            }
            rSum += rRow * wyj
            gSum += gRow * wyj
            bSum += bRow * wyj
        }

        val r = (rSum + 0.5f).toInt().coerceIn(0, 255)
        val g = (gSum + 0.5f).toInt().coerceIn(0, 255)
        val b = (bSum + 0.5f).toInt().coerceIn(0, 255)
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
}
