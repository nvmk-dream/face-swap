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
    fun `canonicalize5Landmarks fixes inverted ML Kit eye and mouth landmark ordering`() {
        val inverted = listOf(
            PointF(165f, 110f), // subject LEFT_EYE -> viewer right (inverted!)
            PointF(95f, 110f),  // subject RIGHT_EYE -> viewer left (inverted!)
            PointF(130f, 150f), // nose tip
            PointF(158f, 185f), // subject MOUTH_LEFT -> viewer right (inverted!)
            PointF(102f, 185f)  // subject MOUTH_RIGHT -> viewer left (inverted!)
        )
        val canon = com.example.onnx.ScrfdFaceDetector.canonicalize5Landmarks(inverted)
        assertEquals(95f, canon[0].x, 0.01f)
        assertEquals(165f, canon[1].x, 0.01f)
        assertEquals(130f, canon[2].x, 0.01f)
        assertEquals(102f, canon[3].x, 0.01f)
        assertEquals(158f, canon[4].x, 0.01f)

        // Tilted head (+22 deg roll) with valid viewer ordering must remain unchanged
        val validTilted = listOf(
            PointF(100f, 100f),
            PointF(164f, 126f),
            PointF(118f, 152f),
            PointF(94f, 180f),
            PointF(146f, 200f)
        )
        val kept = com.example.onnx.ScrfdFaceDetector.canonicalize5Landmarks(validTilted)
        assertEquals(validTilted[0].x, kept[0].x, 0.01f)
        assertEquals(validTilted[1].x, kept[1].x, 0.01f)
        assertEquals(validTilted[3].x, kept[3].x, 0.01f)
        assertEquals(validTilted[4].x, kept[4].x, 0.01f)
    }

    @Test
    fun `resolveSwapperSourceVector forces raw ArcFace embedding for HyperSwap 256`() {
        val rawVec = FloatArray(512) { 1.0f }
        val emapProjVec = FloatArray(512) { 9.0f }
        val fakeCrop = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888)
        val embResult = com.example.onnx.SourceEmbeddingResult(
            latentSourceVector512 = emapProjVec,
            rawNormedEmbedding512 = rawVec,
            aligned112Crop = fakeCrop,
            usedArcFaceModel = true,
            usedEmbeddedEmap = true,
            providerSummary = "w600k_r50.onnx"
        )
        val fakeModel = java.io.File(context.cacheDir, "swapper_renamed.onnx")

        val hyperVec = InSwapperEngine.resolveSwapperSourceVector(
            context = context,
            activeSwapFile = fakeModel,
            isHyperSwap256 = true,
            sourceEmbedding = embResult
        )
        assertEquals(1.0f, hyperVec[0], 0.001f)

        val inswapperVec = InSwapperEngine.resolveSwapperSourceVector(
            context = context,
            activeSwapFile = fakeModel,
            isHyperSwap256 = false,
            sourceEmbedding = embResult
        )
        assertEquals(9.0f, inswapperVec[0], 0.001f)
    }

    @Test
    fun `viewModel popup confirmation updates swapStrategyOption and cancel leaves it unchanged`() {
        val vm = FaceSwapViewModel(context.applicationContext as Application)
        assertEquals(StudioMode.FACE_SWAP, vm.uiState.value.studioMode)
        assertEquals(FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY, vm.uiState.value.swapStrategyOption)
        assertNull(vm.uiState.value.pendingStrategyPopupOption)

        // Tap A -> opens Tamil popup -> Cancel keeps B active
        vm.requestSwapStrategySelection(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY)
        assertEquals(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY, vm.uiState.value.pendingStrategyPopupOption)
        assertEquals(FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY, vm.uiState.value.swapStrategyOption)

        vm.dismissSwapStrategyPopup()
        assertNull(vm.uiState.value.pendingStrategyPopupOption)
        assertEquals(FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY, vm.uiState.value.swapStrategyOption)

        // Tap B+C -> OK updates active option to B+C
        vm.requestSwapStrategySelection(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD)
        assertEquals(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD, vm.uiState.value.pendingStrategyPopupOption)

        vm.confirmSwapStrategySelection()
        assertNull(vm.uiState.value.pendingStrategyPopupOption)
        assertEquals(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD, vm.uiState.value.swapStrategyOption)
        assertTrue(vm.uiState.value.statusBannerMessage?.contains("B+C") == true)
    }

    @Test
    fun `executeFaceSwap supports Option A and Clean B and fullHead supports B+C`() {
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
            swapStrategyOption = FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY,
            onProgress = {}
        )
        assertTrue("Expected [Opt B] in summary: ${resB.pipelineSummary}", resB.pipelineSummary.contains("[Opt B]"))
        assertTrue(
            "Expected clean pipeline in summary: ${resB.pipelineSummary}",
            resB.pipelineSummary.contains("Target Lighting Color Match")
        )
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
