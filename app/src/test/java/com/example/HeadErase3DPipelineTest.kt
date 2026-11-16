package com.example

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.CleanEraseHeadReplacementEngine
import com.example.onnx.DetectedFace
import com.example.onnx.FaceMesh3DEngine
import com.example.onnx.GhostHeadReplacementEngine
import com.example.onnx.HeadReplacementPipelineMode
import com.example.onnx.HeadSegmentationAndInpainting
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxProtobufInspector
import com.example.ui.FaceSwapUiState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.math.cos
import kotlin.math.sin

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class HeadErase3DPipelineTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    private fun createPortrait(
        width: Int,
        height: Int,
        bgColor: Int,
        skinColor: Int,
        hairColor: Int,
        hairRadiusScale: Float = 1.0f,
        rollDegrees: Float = 0f,
        eyeDist: Float = width * 0.16f
    ): Pair<Bitmap, DetectedFace> {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height) { bgColor }

        val cx = width * 0.5f
        val cy = height * 0.5f
        val rad = Math.toRadians(rollDegrees.toDouble())
        val cosA = cos(rad).toFloat()
        val sinA = sin(rad).toFloat()

        val hairRx = eyeDist * 1.65f * hairRadiusScale
        val hairRy = eyeDist * 1.75f * hairRadiusScale
        val faceRx = eyeDist * 1.05f
        val faceRy = eyeDist * 1.22f

        for (y in 0 until height) {
            for (x in 0 until width) {
                val dx = x - cx
                val dy = y - cy
                val lx = dx * cosA + dy * sinA
                val ly = -dx * sinA + dy * cosA

                val hNormX = lx / hairRx
                val hNormY = (ly + hairRy * 0.30f) / hairRy
                if (hNormX * hNormX + hNormY * hNormY <= 1.0f) {
                    pixels[y * width + x] = hairColor
                }
                if (kotlin.math.abs(lx) <= eyeDist * 0.55f &&
                    ly >= eyeDist * 0.85f &&
                    ly <= eyeDist * 1.85f
                ) {
                    pixels[y * width + x] = skinColor
                }
                val fNormX = lx / faceRx
                val fNormY = (ly - eyeDist * 0.12f) / faceRy
                if (fNormX * fNormX + fNormY * fNormY <= 1.0f) {
                    pixels[y * width + x] = skinColor
                }
            }
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)

        fun rot(dx: Float, dy: Float): PointF = PointF(
            cx + dx * cosA - dy * sinA,
            cy + dx * sinA + dy * cosA
        )

        val landmarks = listOf(
            rot(-eyeDist * 0.5f, 0f),
            rot(eyeDist * 0.5f, 0f),
            rot(0f, eyeDist * 0.55f),
            rot(-eyeDist * 0.42f, eyeDist * 1.05f),
            rot(eyeDist * 0.42f, eyeDist * 1.05f)
        )
        val face = DetectedFace(
            index = 0,
            boundingBox = RectF(cx - eyeDist * 1.2f, cy - eyeDist * 1.2f, cx + eyeDist * 1.2f, cy + eyeDist * 1.4f),
            score = 0.99f,
            landmarks5 = landmarks,
            detectorSource = "TestPortrait"
        )
        return bmp to face
    }

    @Test
    fun `matchSlotForFileName maps all 3 new omebly ONNX models accurately`() {
        assertEquals(ModelSlot.FACE_MESH_3D, OnnxProtobufInspector.matchSlotForFileName("facemesh-v2-v1.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("gfpgan-v1.4-w16-v1.onnx"))
        assertEquals(ModelSlot.DEPTH, OnnxProtobufInspector.matchSlotForFileName("depth-anything-v2-small-fp16-v1.onnx"))
    }

    @Test
    fun `default FaceSwapUiState selects CLEAN_ERASE_3D_DIFFUSION`() {
        val state = FaceSwapUiState()
        assertEquals(HeadReplacementPipelineMode.CLEAN_ERASE_3D_DIFFUSION, state.headPipelineMode)
    }

    @Test
    fun `FaceMesh3DEngine fallback builds 31 control points, warps head cleanly, and builds inner face lock`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(20, 30, 45),
            skinColor = Color.rgb(230, 185, 155),
            hairColor = Color.rgb(35, 25, 20),
            rollDegrees = -12f
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(210, 225, 240),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(180, 140, 60),
            rollDegrees = 14f
        )

        val srcMesh = FaceMesh3DEngine.extractFaceMesh3D(null, srcBmp, srcFace, null)
        val tgtMesh = FaceMesh3DEngine.extractFaceMesh3D(null, tgtBmp, tgtFace, null)

        assertEquals(FaceMesh3DEngine.CONTROL_POINT_COUNT, srcMesh.controlPoints31.size)
        assertEquals(FaceMesh3DEngine.CONTROL_POINT_COUNT, tgtMesh.controlPoints31.size)

        val mat = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(tgtFace.landmarks5, 256, 0f)
        val warped = FaceMesh3DEngine.warpSourceHeadToTarget3DPose(
            sourceBitmap = srcBmp,
            sourceMesh = srcMesh,
            targetMesh = tgtMesh,
            targetCanvasMatrix2x3 = mat,
            canvasSize = 256
        )
        assertEquals(256, warped.width)
        assertEquals(256, warped.height)

        val lockMask = FaceMesh3DEngine.buildInnerFaceIdentityMask(tgtMesh, mat, 256)
        assertEquals(256 * 256, lockMask.size)
        // Center of inner face should be protected (~1.0), corner should be 0.0
        assertTrue(lockMask[140 * 256 + 128] > 0.7f)
        assertEquals(0f, lockMask[2 * 256 + 2], 0.01f)
    }

    @Test
    fun `CleanEraseHeadReplacementEngine erases wider target hair and preserves exterior background`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(20, 25, 35),
            skinColor = Color.rgb(225, 180, 150),
            hairColor = Color.rgb(30, 20, 20),
            hairRadiusScale = 0.85f
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(220, 235, 250),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(210, 25, 25), // Distinct red wide target hair to verify erasure
            hairRadiusScale = 1.35f
        )

        val stages = mutableListOf<Int>()
        val result = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_DIFFUSION,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = { stages.add(it.stepIndex) }
        )

        assertEquals(256, result.outputBitmap.width)
        assertEquals(256, result.outputBitmap.height)
        assertEquals(listOf(1, 2, 3, 4, 5, 6), stages)
        // Exterior corner must remain untouched
        assertEquals(tgtBmp.getPixel(5, 5), result.outputBitmap.getPixel(5, 5))
        // Upper target hair crown region above narrower source hair should have been cleanly erased toward sky bg
        val erasedCrownPixel = result.outputBitmap.getPixel(128, 48)
        val red = Color.red(erasedCrownPixel)
        val blue = Color.blue(erasedCrownPixel)
        assertTrue("Expected bright sky background after target hair erase (r=$red, b=$blue)", blue > 150)
    }

    @Test
    fun `GhostHeadReplacementEngine still routes LEGACY_COPY_PASTE cleanly when selected`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(25, 30, 40),
            skinColor = Color.rgb(230, 185, 155),
            hairColor = Color.rgb(40, 25, 20)
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(210, 220, 230),
            skinColor = Color.rgb(205, 160, 135),
            hairColor = Color.rgb(160, 120, 60)
        )

        val stages = mutableListOf<Int>()
        val res = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            headPipelineMode = HeadReplacementPipelineMode.LEGACY_COPY_PASTE,
            onProgress = { stages.add(it.stepIndex) }
        )

        assertNotNull(res.outputBitmap)
        assertTrue(stages.containsAll(listOf(1, 2, 3, 4, 5, 6)))
        assertEquals(tgtBmp.getPixel(5, 5), res.outputBitmap.getPixel(5, 5))
    }
}
