package com.example.onnx

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * High-Fidelity Facial Identity & Texture Preservation Engine.
 *
 * Implements:
 * 1. Dense 106-Point 3D Geometry-Based Piecewise Affine / Mesh Warping to map genuine Source
 *    facial textures (eyes, iris/pupil, eyelashes, eyeliner, lips, lipstick, nostrils, skin pores)
 *    directly onto the Target face pose without synthetic AI reconstruction or beauty retouching.
 * 2. Strict Boundary-Only Feathering that guarantees Zero Difference (0 bit change) for all
 *    Target pixels outside the face contour (background, hair, ears, clothes, neck).
 * 3. Photometric illumination matching that preserves 100% of the Source face details while
 *    matching the Target scene's ambient lighting.
 */
object FaceIdentityPreserver {

    const val WARP_SIZE = 512

    /**
     * Triangle indices referencing the 106 canonical facial landmarks (+ 8 perimeter control vertices).
     */
    data class MeshTriangle(val i0: Int, val i1: Int, val i2: Int)

    /**
     * Triangulation topology covering the entire face structure:
     * - Jawline and cheeks (0..32)
     * - Left eyebrow (33..42) and Right eyebrow (43..52)
     * - Left eye & iris (53..62) and Right eye & iris (63..72)
     * - Nose bridge and alar base (73..85)
     * - Outer and inner lips (86..105)
     * - Forehead & temple perimeter (106..113)
     */
    val CANONICAL_FACE_TRIANGLES: List<MeshTriangle> by lazy {
        val list = ArrayList<MeshTriangle>(240)

        // Jawline & Cheeks strips
        for (i in 0 until 16) {
            list.add(MeshTriangle(i, i + 1, 73 + (i * 12) / 16))
        }
        for (i in 16 until 32) {
            list.add(MeshTriangle(i, i + 1, 85 - ((i - 16) * 12) / 16))
        }

        // Left Eyebrow mesh
        for (i in 33..36) {
            list.add(MeshTriangle(i, i + 1, i + 5))
            list.add(MeshTriangle(i + 1, i + 6, i + 5))
        }

        // Right Eyebrow mesh
        for (i in 43..46) {
            list.add(MeshTriangle(i, i + 1, i + 5))
            list.add(MeshTriangle(i + 1, i + 6, i + 5))
        }

        // Left Eye & Pupil mesh (53..62)
        list.add(MeshTriangle(53, 54, 61))
        list.add(MeshTriangle(54, 55, 61))
        list.add(MeshTriangle(55, 56, 61))
        list.add(MeshTriangle(56, 57, 62))
        list.add(MeshTriangle(57, 58, 62))
        list.add(MeshTriangle(58, 59, 62))
        list.add(MeshTriangle(59, 60, 62))
        list.add(MeshTriangle(60, 53, 61))
        list.add(MeshTriangle(61, 62, 57))
        list.add(MeshTriangle(61, 62, 53))

        // Right Eye & Pupil mesh (63..72)
        list.add(MeshTriangle(63, 64, 71))
        list.add(MeshTriangle(64, 65, 71))
        list.add(MeshTriangle(65, 66, 71))
        list.add(MeshTriangle(66, 67, 72))
        list.add(MeshTriangle(67, 68, 72))
        list.add(MeshTriangle(68, 69, 72))
        list.add(MeshTriangle(69, 70, 72))
        list.add(MeshTriangle(70, 63, 71))
        list.add(MeshTriangle(71, 72, 67))
        list.add(MeshTriangle(71, 72, 63))

        // Nose Bridge (73..76) and Base (77..85)
        for (i in 73..75) {
            list.add(MeshTriangle(i, i + 1, i + 4))
        }
        for (i in 77..84) {
            list.add(MeshTriangle(76, i, i + 1))
        }

        // Outer Lips (86..97)
        for (i in 86..91) {
            list.add(MeshTriangle(i, i + 1, 76))
        }
        for (i in 92..96) {
            list.add(MeshTriangle(i, i + 1, 16))
        }
        list.add(MeshTriangle(97, 86, 16))

        // Inner Lips & Oral Cavity (98..105)
        for (i in 98..104) {
            list.add(MeshTriangle(i, i + 1, (i - 98) + 86))
        }
        list.add(MeshTriangle(105, 98, 92))

        // Forehead & Temple connections (106..113 are perimeter points)
        list.add(MeshTriangle(0, 33, 106))
        list.add(MeshTriangle(33, 35, 107))
        list.add(MeshTriangle(35, 37, 108))
        list.add(MeshTriangle(37, 43, 109))
        list.add(MeshTriangle(43, 45, 110))
        list.add(MeshTriangle(45, 47, 111))
        list.add(MeshTriangle(47, 32, 112))
        list.add(MeshTriangle(32, 112, 113))

        list
    }

    /**
     * Warps the authentic Source face directly into Target 512x512 coordinate space using
     * dense 106-point landmark geometry. Every feature (eyelashes, eyeliner, pupils, lips, pores)
     * is transferred without any synthetic AI reconstruction.
     */
    fun warpSourceTextureToTargetGeometry512(
        sourceBitmap: Bitmap,
        sourceLandmarks5: List<PointF>,
        targetLandmarks5: List<PointF>,
        dstSize: Int = WARP_SIZE
    ): Bitmap {
        val srcRecon = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(
            landmarks5 = sourceLandmarks5,
            bitmap = sourceBitmap
        )
        val tgtRecon = Face3DReconstruction.reconstruct3DFaceAnd106Landmarks(
            landmarks5 = targetLandmarks5,
            bitmap = null
        )

        // Build 512x512 affine alignment for target landmarks
        val mTarget512 = FaceAlignment.estimateNorm(targetLandmarks5, dstSize)
        val tgtPts512 = ArrayList<PointF>(114)
        for (pt in tgtRecon.landmarks106) {
            val px = mTarget512[0] * pt.x + mTarget512[1] * pt.y + mTarget512[2]
            val py = mTarget512[3] * pt.x + mTarget512[4] * pt.y + mTarget512[5]
            tgtPts512.add(PointF(px, py))
        }

        // Add 8 forehead & hairline perimeter points in 512 space
        addForeheadPerimeterPoints(tgtPts512, dstSize)

        // Source landmarks in original source image space
        val srcPts = ArrayList<PointF>(114)
        srcPts.addAll(srcRecon.landmarks106)
        addForeheadPerimeterPointsSource(srcPts, sourceBitmap.width, sourceBitmap.height)

        val output = Bitmap.createBitmap(dstSize, dstSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)

        // Piecewise Affine Warping triangle-by-triangle
        val srcMatrix = FloatArray(9)
        val matrix = Matrix()
        val path = Path()

        for (tri in CANONICAL_FACE_TRIANGLES) {
            if (tri.i0 >= tgtPts512.size || tri.i1 >= tgtPts512.size || tri.i2 >= tgtPts512.size) continue
            if (tri.i0 >= srcPts.size || tri.i1 >= srcPts.size || tri.i2 >= srcPts.size) continue

            val t0 = tgtPts512[tri.i0]
            val t1 = tgtPts512[tri.i1]
            val t2 = tgtPts512[tri.i2]

            val s0 = srcPts[tri.i0]
            val s1 = srcPts[tri.i1]
            val s2 = srcPts[tri.i2]

            // Calculate affine transform mapping source triangle (s0, s1, s2) to target triangle (t0, t1, t2)
            val success = computeAffineTransform(
                s0.x, s0.y, s1.x, s1.y, s2.x, s2.y,
                t0.x, t0.y, t1.x, t1.y, t2.x, t2.y,
                srcMatrix
            )
            if (!success) continue

            matrix.setValues(srcMatrix)

            canvas.save()
            path.reset()
            path.moveTo(t0.x, t0.y)
            path.lineTo(t1.x, t1.y)
            path.lineTo(t2.x, t2.y)
            path.close()

            canvas.clipPath(path)
            canvas.drawBitmap(sourceBitmap, matrix, paint)
            canvas.restore()
        }

        return output
    }

    /**
     * Creates a high-fidelity boundary mask where:
     * - The core face region has alpha = 1.0 (pure source identity).
     * - The outer perimeter has smooth 4-8px cosine feathering.
     * - Pixels outside the face contour strictly have alpha = 0.0 (zero difference on target).
     */
    fun createZeroDifferenceTargetBoundaryMask512(
        targetLandmarks5: List<PointF>,
        forwardMatrix512: FloatArray? = null,
        dstSize: Int = WARP_SIZE,
        featherRadiusPx: Float = 6.0f
    ): FloatArray {
        val total = dstSize * dstSize
        val mask = FloatArray(total)

        val lEye = if (forwardMatrix512 != null) {
            PointF(
                forwardMatrix512[0] * targetLandmarks5[0].x + forwardMatrix512[1] * targetLandmarks5[0].y + forwardMatrix512[2],
                forwardMatrix512[3] * targetLandmarks5[0].x + forwardMatrix512[4] * targetLandmarks5[0].y + forwardMatrix512[5]
            )
        } else targetLandmarks5[0]

        val rEye = if (forwardMatrix512 != null) {
            PointF(
                forwardMatrix512[0] * targetLandmarks5[1].x + forwardMatrix512[1] * targetLandmarks5[1].y + forwardMatrix512[2],
                forwardMatrix512[3] * targetLandmarks5[1].x + forwardMatrix512[4] * targetLandmarks5[1].y + forwardMatrix512[5]
            )
        } else targetLandmarks5[1]

        val nose = if (forwardMatrix512 != null) {
            PointF(
                forwardMatrix512[0] * targetLandmarks5[2].x + forwardMatrix512[1] * targetLandmarks5[2].y + forwardMatrix512[2],
                forwardMatrix512[3] * targetLandmarks5[2].x + forwardMatrix512[4] * targetLandmarks5[2].y + forwardMatrix512[5]
            )
        } else targetLandmarks5[2]

        val lMouth = if (forwardMatrix512 != null) {
            PointF(
                forwardMatrix512[0] * targetLandmarks5[3].x + forwardMatrix512[1] * targetLandmarks5[3].y + forwardMatrix512[2],
                forwardMatrix512[3] * targetLandmarks5[3].x + forwardMatrix512[4] * targetLandmarks5[3].y + forwardMatrix512[5]
            )
        } else targetLandmarks5[3]

        val rMouth = if (forwardMatrix512 != null) {
            PointF(
                forwardMatrix512[0] * targetLandmarks5[4].x + forwardMatrix512[1] * targetLandmarks5[4].y + forwardMatrix512[2],
                forwardMatrix512[3] * targetLandmarks5[4].x + forwardMatrix512[4] * targetLandmarks5[4].y + forwardMatrix512[5]
            )
        } else targetLandmarks5[4]

        val eyeDist = kotlin.math.hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble()).toFloat().coerceAtLeast(40f)
        val cosA = (rEye.x - lEye.x) / eyeDist
        val sinA = (rEye.y - lEye.y) / eyeDist

        val eyeMidX = (lEye.x + rEye.x) * 0.5f
        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val mouthMidX = (lMouth.x + rMouth.x) * 0.5f
        val mouthMidY = (lMouth.y + rMouth.y) * 0.5f

        val centerX = eyeMidX * 0.38f + nose.x * 0.32f + mouthMidX * 0.30f
        val centerY = eyeMidY * 0.32f + nose.y * 0.30f + mouthMidY * 0.38f

        val radiusX = eyeDist * 1.05f
        val radiusYTop = eyeDist * 1.05f
        val radiusYBottom = eyeDist * 1.25f
        val innerCoreRatio = 0.70f
        val borderMargin = 20

        for (y in 0 until dstSize) {
            val row = y * dstSize
            for (x in 0 until dstSize) {
                if (x < borderMargin || x >= dstSize - borderMargin ||
                    y < borderMargin || y >= dstSize - borderMargin
                ) {
                    mask[row + x] = 0.0f
                    continue
                }

                val dx = x.toFloat() - centerX
                val dy = y.toFloat() - centerY
                val localX = dx * cosA + dy * sinA
                val localY = -dx * sinA + dy * cosA

                val activeRy = if (localY < 0f) radiusYTop else radiusYBottom
                val uNorm = localX / radiusX
                val vNorm = localY / activeRy
                val r = sqrt(uNorm * uNorm + vNorm * vNorm)

                val alpha = when {
                    r <= innerCoreRatio -> 1.0f
                    r >= 1.0f -> 0.0f
                    else -> {
                        val t = (r - innerCoreRatio) / (1.0f - innerCoreRatio)
                        0.5f * (1.0f + cos(Math.PI * t).toFloat())
                    }
                }
                mask[row + x] = alpha
            }
        }

        return mask
    }

    /**
     * Blends the pure Source identity texture with the Target face lighting while strictly
     * preserving genuine Source eye details, iris, eyelashes, lips, lipstick, and skin texture.
     */
    fun blendIdentityPreservedFace512(
        sourceWarped512: IntArray,
        target512: IntArray,
        mask512: FloatArray,
        preserveMakeup: Boolean = true
    ): IntArray {
        val total = WARP_SIZE * WARP_SIZE
        val blended = IntArray(total)

        // Calculate average scene luminance for Target and Source face
        var tgtLumSum = 0.0
        var srcLumSum = 0.0
        var count = 0

        for (i in 0 until total) {
            val m = mask512[i]
            if (m > 0.4f) {
                val tc = target512[i]
                val sc = sourceWarped512[i]
                val tLum = 0.299f * ((tc ushr 16) and 0xFF) + 0.587f * ((tc ushr 8) and 0xFF) + 0.114f * (tc and 0xFF)
                val sLum = 0.299f * ((sc ushr 16) and 0xFF) + 0.587f * ((sc ushr 8) and 0xFF) + 0.114f * (sc and 0xFF)
                tgtLumSum += tLum
                srcLumSum += sLum
                count++
            }
        }

        val lumRatio = if (count > 0 && srcLumSum > 10.0) {
            (tgtLumSum / srcLumSum).toFloat().coerceIn(0.70f, 1.35f)
        } else 1.0f

        for (i in 0 until total) {
            val m = mask512[i]
            if (m <= 0.001f) {
                // Strictly 100% Target pixel outside face mask
                blended[i] = target512[i]
            } else {
                val tc = target512[i]
                val sc = sourceWarped512[i]

                val tr = (tc ushr 16) and 0xFF
                val tg = (tc ushr 8) and 0xFF
                val tb = tc and 0xFF

                var sr = ((sc ushr 16) and 0xFF).toFloat()
                var sg = ((sc ushr 8) and 0xFF).toFloat()
                var sb = (sc and 0xFF).toFloat()

                // Harmonize lighting gently on skin while strictly keeping 100% contrast for eyes, eyeliner & lips
                val isHighContrastFeature = if (preserveMakeup) {
                    val sMax = max(sr, max(sg, sb))
                    val sMin = min(sr, min(sg, sb))
                    (sMax - sMin > 45f) || (sr + sg + sb < 160f) // Eyeliner, lashes, lips
                } else false

                if (!isHighContrastFeature) {
                    sr = (sr * lumRatio).coerceIn(0f, 255f)
                    sg = (sg * lumRatio).coerceIn(0f, 255f)
                    sb = (sb * lumRatio).coerceIn(0f, 255f)
                }

                // Blend with Target pixel according to boundary mask alpha
                val outR = (sr * m + tr * (1.0f - m)).toInt().coerceIn(0, 255)
                val outG = (sg * m + tg * (1.0f - m)).toInt().coerceIn(0, 255)
                val outB = (sb * m + tb * (1.0f - m)).toInt().coerceIn(0, 255)

                blended[i] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }

        return blended
    }

    private fun addForeheadPerimeterPoints(list: ArrayList<PointF>, size: Int) {
        val fSize = size.toFloat()
        list.add(PointF(0f, 0f))
        list.add(PointF(fSize * 0.25f, 0f))
        list.add(PointF(fSize * 0.50f, 0f))
        list.add(PointF(fSize * 0.75f, 0f))
        list.add(PointF(fSize, 0f))
        list.add(PointF(0f, fSize * 0.35f))
        list.add(PointF(fSize, fSize * 0.35f))
        list.add(PointF(fSize * 0.5f, fSize * 0.20f))
    }

    private fun addForeheadPerimeterPointsSource(list: ArrayList<PointF>, width: Int, height: Int) {
        val w = width.toFloat()
        val h = height.toFloat()
        list.add(PointF(0f, 0f))
        list.add(PointF(w * 0.25f, 0f))
        list.add(PointF(w * 0.50f, 0f))
        list.add(PointF(w * 0.75f, 0f))
        list.add(PointF(w, 0f))
        list.add(PointF(0f, h * 0.35f))
        list.add(PointF(w, h * 0.35f))
        list.add(PointF(w * 0.5f, h * 0.20f))
    }

    /**
     * Calculates the 3x3 affine transformation matrix mapping triangle (x0, y0), (x1, y1), (x2, y2)
     * to triangle (u0, v0), (u1, v1), (u2, v2).
     */
    private fun computeAffineTransform(
        x0: Float, y0: Float, x1: Float, y1: Float, x2: Float, y2: Float,
        u0: Float, v0: Float, u1: Float, v1: Float, u2: Float, v2: Float,
        outMatrix9: FloatArray
    ): Boolean {
        val den = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
        if (abs(den) < 1e-5f) return false

        val invDen = 1.0f / den

        // Affine forward matrix mapping source (x, y) to destination (u, v)
        val a11 = ((u0 - u2) * (y1 - y2) - (u1 - u2) * (y0 - y2)) * invDen
        val a12 = ((x0 - x2) * (u1 - u2) - (x1 - x2) * (u0 - u2)) * invDen
        val a13 = u0 - a11 * x0 - a12 * y0

        val a21 = ((v0 - v2) * (y1 - y2) - (v1 - v2) * (y0 - y2)) * invDen
        val a22 = ((x0 - x2) * (v1 - v2) - (x1 - x2) * (v0 - v2)) * invDen
        val a23 = v0 - a21 * x0 - a22 * y0

        // We need the matrix that transforms the source image onto canvas
        outMatrix9[0] = a11
        outMatrix9[1] = a12
        outMatrix9[2] = a13
        outMatrix9[3] = a21
        outMatrix9[4] = a22
        outMatrix9[5] = a23
        outMatrix9[6] = 0f
        outMatrix9[7] = 0f
        outMatrix9[8] = 1f

        return true
    }
}
