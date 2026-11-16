package com.example

import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.DetectedFace
import com.example.onnx.FaceBlender
import com.example.onnx.FaceSwapStrategyOption
import com.example.onnx.GhostHeadReplacementEngine
import com.example.onnx.InSwapperEngine
import com.example.ui.FaceSwapViewModel
import com.example.ui.StudioMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.math.abs

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SwapStrategyOptionsRobolectricTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    private val ortEnv: OrtEnvironment?
        get() = runCatching { OrtEnvironment.getEnvironment() }.getOrNull()

    private fun makePortrait(
        width: Int = 360,
        height: Int = 360,
        faceRadiusX: Float = 84f,
        faceRadiusY: Float = 104f,
        skinColor: Int = Color.rgb(225, 182, 155),
        bgColor: Int = Color.rgb(30, 42, 65)
    ): Pair<Bitmap, DetectedFace> {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val cx = width * 0.5f
        val cy = height * 0.52f
        val px = IntArray(width * height) { bgColor }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val nx = (x - cx) / faceRadiusX
                val ny = (y - cy) / faceRadiusY
                if (nx * nx + ny * ny <= 1f) {
                    val shade = ((x - cx) * 0.08f + (y - cy) * 0.08f).toInt()
                    val r = ((skinColor ushr 16 and 0xFF) + shade).coerceIn(0, 255)
                    val g = ((skinColor ushr 8 and 0xFF) + shade).coerceIn(0, 255)
                    val b = ((skinColor and 0xFF) + shade).coerceIn(0, 255)
                    px[y * width + x] = Color.rgb(r, g, b)
                }
            }
        }
        bmp.setPixels(px, 0, width, 0, 0, width, height)

        val lms = listOf(
            PointF(cx - faceRadiusX * 0.36f, cy - faceRadiusY * 0.16f),
            PointF(cx + faceRadiusX * 0.36f, cy - faceRadiusY * 0.16f),
            PointF(cx, cy + faceRadiusY * 0.10f),
            PointF(cx - faceRadiusX * 0.30f, cy + faceRadiusY * 0.42f),
            PointF(cx + faceRadiusX * 0.30f, cy + faceRadiusY * 0.42f)
        )
        val face = DetectedFace(
            index = 0,
            boundingBox = RectF(cx - faceRadiusX, cy - faceRadiusY, cx + faceRadiusX, cy + faceRadiusY),
            score = 0.99f,
            landmarks5 = lms,
            detectorSource = "TestDetector"
        )
        return bmp to face
    }

    @Test
    fun `passthrough threshold skips retry at cdiff 11_7 and triggers below 7_5`() {
        assertEquals(7.5f, InSwapperEngine.PASSTHROUGH_CENTER_DIFF_THRESHOLD, 0.01f)
        assertFalse(
            InSwapperEngine.shouldRetryNearIdentityPassthrough(
                isHyperSwapModel = true,
                hasCorruption = false,
                lumaCorrelation = 0.99f,
                centerFaceDiff = 11.7f
            )
        )
        assertTrue(
            InSwapperEngine.shouldRetryNearIdentityPassthrough(
                isHyperSwapModel = true,
                hasCorruption = false,
                lumaCorrelation = 0.99f,
                centerFaceDiff = 4.0f
            )
        )
        assertFalse(
            InSwapperEngine.shouldRetryNearIdentityPassthrough(
                isHyperSwapModel = false,
                hasCorruption = false,
                lumaCorrelation = 0.99f,
                centerFaceDiff = 2.0f
            )
        )
    }

    @Test
    fun `option B computes source contour ratios and applies edge-safe warp on 512 crop`() {
        val geom512 = FaceBlender.computeWarpedGeometry(
            forwardMatrix128 = null,
            targetLandmarks5 = null,
            scale = 4.0f
        )
        val total = 512 * 512
        val srcPx512 = IntArray(total) { Color.rgb(30, 40, 60) }
        val tgtPx512 = IntArray(total) { Color.rgb(30, 40, 60) }

        val cx = 256f
        val cy = 264f
        for (y in 0 until 512) {
            for (x in 0 until 512) {
                val dx = x - cx
                val dy = y - cy
                // Wider and longer source oval vs narrower/shorter target oval
                if ((dx / 155f) * (dx / 155f) + (dy / 195f) * (dy / 195f) <= 1f) {
                    srcPx512[y * 512 + x] = Color.rgb(225, 180, 150)
                }
                if ((dx / 118f) * (dx / 118f) + (dy / 155f) * (dy / 155f) <= 1f) {
                    tgtPx512[y * 512 + x] = Color.rgb(220, 175, 145)
                }
            }
        }

        val ratios = FaceBlender.computeSourceToTargetShapeRatios512(
            srcPx512 = srcPx512,
            tgtPx512 = tgtPx512,
            geom512 = geom512
        )
        assertTrue("Expected wider source cheek ratio > 1.04, got ${ratios.cheekWidthScale}", ratios.cheekWidthScale > 1.04f)
        assertTrue("Expected wider source jaw ratio > 1.03, got ${ratios.jawWidthScale}", ratios.jawWidthScale > 1.03f)
        assertTrue("Expected longer source chin ratio > 1.03, got ${ratios.chinLengthScale}", ratios.chinLengthScale > 1.03f)

        // Build a 512x512 crop with green frame border and inner skin square
        val cropPx = IntArray(total)
        for (y in 0 until 512) {
            for (x in 0 until 512) {
                cropPx[y * 512 + x] = when {
                    x < 8 || x >= 504 || y < 8 || y >= 504 -> Color.rgb(10, 200, 30)
                    abs(x - 256) <= 120 && abs(y - 256) <= 120 -> Color.rgb(210, 160 - (abs(x - 256) / 2), 140)
                    else -> Color.rgb(40, 40, 60)
                }
            }
        }
        val centerBefore = cropPx[256 * 512 + 256]
        val cornerBefore = cropPx[4 * 512 + 4]
        FaceBlender.applySourceShapeContourWarp512(
            px512 = cropPx,
            geom512 = geom512,
            ratios = ratios
        )
        // Outer border (hair/background ring) must remain untouched
        assertEquals(cornerBefore, cropPx[4 * 512 + 4])
        assertEquals(Color.rgb(10, 200, 30), cropPx[507 * 512 + 507])
        // Center nose bridge (256, 256) remains anchored
        assertEquals(centerBefore, cropPx[256 * 512 + 256])
    }

    @Test
    fun `viewModel popup confirmation updates swapStrategyOption and cancel leaves it unchanged`() {
        val vm = FaceSwapViewModel(context.applicationContext as Application)
        assertEquals(StudioMode.FACE_SWAP, vm.uiState.value.studioMode)
        assertEquals(FaceSwapStrategyOption.OPTION_B_SHAPE_CONTOUR, vm.uiState.value.swapStrategyOption)
        assertNull(vm.uiState.value.pendingStrategyPopupOption)

        // Tap A -> opens Tamil popup -> Cancel keeps B active
        vm.requestSwapStrategySelection(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY)
        assertEquals(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY, vm.uiState.value.pendingStrategyPopupOption)
        assertEquals(FaceSwapStrategyOption.OPTION_B_SHAPE_CONTOUR, vm.uiState.value.swapStrategyOption)

        vm.dismissSwapStrategyPopup()
        assertNull(vm.uiState.value.pendingStrategyPopupOption)
        assertEquals(FaceSwapStrategyOption.OPTION_B_SHAPE_CONTOUR, vm.uiState.value.swapStrategyOption)

        // Tap B+C -> OK updates active option to B+C
        vm.requestSwapStrategySelection(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD)
        assertEquals(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD, vm.uiState.value.pendingStrategyPopupOption)

        vm.confirmSwapStrategySelection()
        assertNull(vm.uiState.value.pendingStrategyPopupOption)
        assertEquals(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD, vm.uiState.value.swapStrategyOption)
        assertTrue(vm.uiState.value.statusBannerMessage?.contains("B+C") == true)
    }

    @Test
    fun `executeFaceSwap supports Option A and B and fullHead supports B+C`() {
        val (srcBmp, srcFace) = makePortrait(faceRadiusX = 96f, faceRadiusY = 116f, skinColor = Color.rgb(235, 190, 165))
        val (tgtBmp, tgtFace) = makePortrait(faceRadiusX = 80f, faceRadiusY = 98f, skinColor = Color.rgb(205, 160, 138))

        val resA = InSwapperEngine.executeFaceSwap(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            swapStrategyOption = FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY,
            onProgress = {}
        )
        assertTrue("Expected [Opt A] in summary: ${resA.pipelineSummary}", resA.pipelineSummary.contains("[Opt A]"))
        assertEquals(tgtBmp.width, resA.outputBitmap.width)
        assertEquals(tgtBmp.height, resA.outputBitmap.height)

        val resB = InSwapperEngine.executeFaceSwap(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            swapStrategyOption = FaceSwapStrategyOption.OPTION_B_SHAPE_CONTOUR,
            onProgress = {}
        )
        assertTrue("Expected [Opt B] in summary: ${resB.pipelineSummary}", resB.pipelineSummary.contains("[Opt B]"))
        assertEquals(tgtBmp.width, resB.outputBitmap.width)

        val resBC = GhostHeadReplacementEngine.executeFullHeadReplacement(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            onProgress = {}
        )
        assertNotNull(resBC.outputBitmap)
        assertEquals(tgtBmp.width, resBC.outputBitmap.width)
    }
}
