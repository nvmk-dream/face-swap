package com.example

import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.CraniofacialAnalyzer
import com.example.onnx.DetectedFace
import com.example.onnx.FaceBlender
import com.example.onnx.FaceReactionSourceMode
import com.example.onnx.FaceSwapStrategyOption
import com.example.onnx.GhostHeadReplacementEngine
import com.example.onnx.InSwapperEngine
import com.example.onnx.SkinToneSourceMode
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
@Config(sdk = [36])
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

        val resD = InSwapperEngine.executeFaceSwap(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            skinToneMode = SkinToneSourceMode.TARGET_SCENE,
            faceReactionMode = FaceReactionSourceMode.TARGET_REACTION,
            swapStrategyOption = FaceSwapStrategyOption.OPTION_D_CRANIOFACIAL_FIT,
            onProgress = {}
        )
        assertTrue("Expected [Opt D] in summary: ${resD.pipelineSummary}", resD.pipelineSummary.contains("[Opt D]"))
        assertEquals(tgtBmp.width, resD.outputBitmap.width)
        assertEquals(tgtBmp.height, resD.outputBitmap.height)
    }

    @Test
    fun `craniofacial analyzer extracts 1-2-3 ratios and option D confirmation applies defaults`() {
        val (srcBmp, srcFace) = makePortrait(faceRadiusX = 96f, faceRadiusY = 116f)
        val (tgtBmp, tgtFace) = makePortrait(faceRadiusX = 80f, faceRadiusY = 98f)
        val sideFace = DetectedFace(
            index = 1,
            boundingBox = tgtFace.boundingBox,
            score = 0.99f,
            landmarks5 = tgtFace.landmarks5,
            detectorSource = "Test",
            eulerX = 0f,
            eulerY = 38f,
            eulerZ = 0f
        )

        val reportAligned = CraniofacialAnalyzer.analyzePair(srcFace, srcBmp, tgtFace, tgtBmp)
        assertEquals(1.0f, reportAligned.sourceProfile.cheekWidthNormalized, 0.001f)
        assertTrue(reportAligned.sourceProfile.squareJawRatio in 0.54f..0.92f)
        assertTrue(reportAligned.sourceProfile.lowerJawRatio in 0.28f..0.62f)
        assertTrue(reportAligned.sourceProfile.foreheadWidthRatio in 0.66f..0.96f)
        assertTrue(reportAligned.overallCompatibilityPct >= 75)

        val reportYawMismatch = CraniofacialAnalyzer.analyzePair(srcFace, srcBmp, sideFace, tgtBmp)
        assertTrue(reportYawMismatch.diagnosticWarnings.isNotEmpty())
        assertTrue(reportYawMismatch.overallCompatibilityPct < reportAligned.overallCompatibilityPct)

        val vm = FaceSwapViewModel(context.applicationContext as Application)
        assertEquals(SkinToneSourceMode.TARGET_SCENE, vm.uiState.value.skinToneMode)
        assertEquals(FaceReactionSourceMode.TARGET_REACTION, vm.uiState.value.faceReactionMode)

        vm.loadSourceAndTargetBitmaps(srcBmp, tgtBmp, srcFace, tgtFace)
        assertNotNull(vm.uiState.value.craniofacialReport)

        vm.setSkinToneMode(SkinToneSourceMode.SOURCE_IDENTITY)
        vm.setFaceReactionMode(FaceReactionSourceMode.SOURCE_REACTION)
        vm.requestSwapStrategySelection(FaceSwapStrategyOption.OPTION_D_CRANIOFACIAL_FIT)
        vm.confirmSwapStrategySelection()
        assertEquals(FaceSwapStrategyOption.OPTION_D_CRANIOFACIAL_FIT, vm.uiState.value.swapStrategyOption)
        assertEquals(SkinToneSourceMode.TARGET_SCENE, vm.uiState.value.skinToneMode)
        assertEquals(FaceReactionSourceMode.TARGET_REACTION, vm.uiState.value.faceReactionMode)
    }

    @Test
    fun `option D measures D B C A mandibular ratios and expands wider source jaw outward without shrinking chin`() {
        val geom512 = FaceBlender.computeWarpedGeometry(
            forwardMatrix128 = null,
            targetLandmarks5 = null,
            scale = 4.0f
        )
        val total = 512 * 512
        val srcPx512 = IntArray(total) { Color.rgb(25, 25, 35) }
        val tgtPx512 = IntArray(total) { Color.rgb(25, 25, 35) }

        val eyeMidX = (geom512.leftEye.x + geom512.rightEye.x) * 0.5f
        val eyeMidY = (geom512.leftEye.y + geom512.rightEye.y) * 0.5f
        val mouthMidX = (geom512.leftMouth.x + geom512.rightMouth.x) * 0.5f
        val mouthMidY = (geom512.leftMouth.y + geom512.rightMouth.y) * 0.5f
        val cx = (eyeMidX * 0.38f + geom512.nose.x * 0.32f + mouthMidX * 0.30f)
        val cy = (eyeMidY * 0.32f + geom512.nose.y * 0.30f + mouthMidY * 0.38f)

        // Source = wide U-shaped jaw (full D/B/C/A); Target = narrow V-shaped chin
        for (y in 0 until 512) {
            for (x in 0 until 512) {
                val dx = abs(x - cx)
                val dy = y - cy
                // Target narrow V taper below center
                val tgtHalfW = if (dy < 0f) 130f else (130f * (1f - (dy / 225f).coerceIn(0f, 0.72f))).coerceAtLeast(36f)
                if (dy in -145f..155f && dx <= tgtHalfW) {
                    tgtPx512[y * 512 + x] = Color.rgb(218, 174, 146)
                }
                // Source wide rounded U-jaw (wider at UpperJaw D, JawBone B, LowerJaw C, longer Mandible A)
                val srcHalfW = if (dy < 0f) 156f else (156f * (1f - 0.34f * (dy / 210f).coerceIn(0f, 1f))).coerceAtLeast(88f)
                if (dy in -155f..190f && dx <= srcHalfW) {
                    srcPx512[y * 512 + x] = Color.rgb(228, 184, 156)
                }
            }
        }

        val (srcBmp, srcFace) = makePortrait(faceRadiusX = 110f, faceRadiusY = 132f)
        val (tgtBmp, tgtFace) = makePortrait(faceRadiusX = 82f, faceRadiusY = 102f)
        val baseReport = CraniofacialAnalyzer.analyzePair(srcFace, srcBmp, tgtFace, tgtBmp)
        assertTrue(baseReport.mandibularSummary.contains("D UpperJaw"))
        assertTrue(baseReport.mandibularSummary.contains("C LowerJaw"))

        val ratios = CraniofacialAnalyzer.computeMandibularFitRatios512(
            srcPx512 = srcPx512,
            tgtPx512 = tgtPx512,
            geom512 = geom512,
            baseRatios = baseReport.optionDFitRatios
        )
        assertTrue("Expected D UpperJaw > 1.05, got ${ratios.upperJawMaxillaScale}", ratios.upperJawMaxillaScale > 1.05f)
        assertTrue("Expected B JawBone > 1.08, got ${ratios.jawBoneBigonialScale}", ratios.jawBoneBigonialScale > 1.08f)
        assertTrue("Expected C LowerJaw > 1.08, got ${ratios.lowerJawBodyScale}", ratios.lowerJawBodyScale > 1.08f)
        assertTrue("Expected A Mandible > 1.04, got ${ratios.mandibularLengthScale}", ratios.mandibularLengthScale > 1.04f)

        // Verify applyOptionDCraniofacialShapeWarp512 pushes jaw skin outward onto narrow perimeter at Row B (Jaw Bone) and Row C (Lower Jaw)
        val warpPx = tgtPx512.copyOf()
        val cornerBefore = warpPx[4 * 512 + 4]
        val lowerJawY = (mouthMidY + geom512.eyeDist * 0.28f).toInt().coerceIn(32, 480)
        // Target narrow V-jaw half-width at lowerJawY is ~63px -> test at cx + 76px (13px outside target contour)
        val testX = (cx + 76f).toInt().coerceIn(32, 480)
        val beforeOuterR = (warpPx[lowerJawY * 512 + testX] ushr 16) and 0xFF
        assertEquals(25, beforeOuterR)

        FaceBlender.applyOptionDCraniofacialShapeWarp512(
            px512 = warpPx,
            geom512 = geom512,
            fitRatios = ratios
        )
        val afterOuterR = (warpPx[lowerJawY * 512 + testX] ushr 16) and 0xFF
        assertTrue("Expected Option D warp to expand lower jaw C outward (red > 150), got $afterOuterR", afterOuterR > 150)
        assertEquals(cornerBefore, warpPx[4 * 512 + 4])
    }
}
