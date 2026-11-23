package com.example

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.CleanEraseHeadReplacementEngine
import com.example.onnx.DetectedFace
import com.example.onnx.FaceMesh3DEngine
import com.example.onnx.FaceReactionSourceMode
import com.example.onnx.GhostHeadReplacementEngine
import com.example.onnx.HeadReplacementPipelineMode
import com.example.onnx.HeadSegmentationAndInpainting
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxProtobufInspector
import com.example.onnx.SdInpaintOnnxEngine
import com.example.onnx.SkinToneSourceMode
import com.example.ui.FaceSwapUiState
import com.example.ui.FaceSwapViewModel
import com.example.ui.ProcessingQualityLevel
import com.example.util.ImageGalleryHelper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.util.concurrent.CancellationException
import kotlin.math.cos
import kotlin.math.sin

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
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
    fun `default FaceSwapUiState selects CLEAN_ERASE_3D_FAST`() {
        val state = FaceSwapUiState()
        assertEquals(HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST, state.headPipelineMode)
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
        assertTrue(
            "Expected 3-Step Portrait-First pipeline summary, got: ${result.pipelineSummary}",
            result.pipelineSummary.contains("3-Step Portrait-First")
        )
        assertNotNull("Expected Step 1 Headless preview", result.stage1SwapOnly512)
        assertNotNull("Expected Step 2 Intermediate Portrait preview", result.stage2SwapRestore512)
        assertNotNull("Expected Step 3 Final 512 preview", result.stage3FinalBlend512)
        assertEquals(512, result.stage1SwapOnly512!!.width)
        assertEquals(512, result.stage2SwapRestore512!!.width)
        assertEquals(512, result.rawSwapped128.width)
        assertEquals(512, result.stage3FinalBlend512!!.width)
        // Exterior corner must remain untouched
        assertEquals(tgtBmp.getPixel(5, 5), result.outputBitmap.getPixel(5, 5))
        // Upper target hair crown region above narrower source hair should have been cleanly erased toward sky bg
        val erasedCrownPixel = result.outputBitmap.getPixel(128, 48)
        val red = Color.red(erasedCrownPixel)
        val blue = Color.blue(erasedCrownPixel)
        assertTrue("Expected bright sky background after target hair erase (r=$red, b=$blue)", blue > 150)
    }

    @Test
    fun `warpSourceHeadToTarget3DPoseWithBounds suppresses out-of-bounds top edge clamping`() {
        // Build source portrait placed high so crownTop goes above y = 0 (bright magenta strip at y = 0)
        val w = 256
        val h = 256
        val px = IntArray(w * h) { Color.rgb(30, 35, 45) }
        for (x in 0 until w) {
            px[x] = Color.rgb(255, 0, 255) // Top border pixel that clamped sampling would smear upward
        }
        val cy = 58f
        val cx = 128f
        val eyeDist = 48f
        for (y in 2 until h) {
            for (x in 0 until w) {
                val dx = (x - cx) / 56f
                val dy = (y - cy) / 68f
                if (dx * dx + dy * dy <= 1f) {
                    px[y * w + x] = Color.rgb(220, 175, 145)
                }
            }
        }
        val srcBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
            setPixels(px, 0, w, 0, 0, w, h)
        }
        val srcFace = DetectedFace(
            index = 0,
            boundingBox = RectF(cx - 56f, 2f, cx + 56f, cy + 72f),
            score = 0.99f,
            landmarks5 = listOf(
                PointF(cx - eyeDist * 0.5f, cy),
                PointF(cx + eyeDist * 0.5f, cy),
                PointF(cx, cy + 26f),
                PointF(cx - 20f, cy + 48f),
                PointF(cx + 20f, cy + 48f)
            ),
            detectorSource = "TestHighCrop"
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(215, 230, 245),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(40, 30, 25)
        )

        val srcMesh = FaceMesh3DEngine.extractFaceMesh3D(null, srcBmp, srcFace, null)
        val tgtMesh = FaceMesh3DEngine.extractFaceMesh3D(null, tgtBmp, tgtFace, null)
        val mat = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(tgtFace.landmarks5, 256, 0f)

        val warped = FaceMesh3DEngine.warpSourceHeadToTarget3DPoseWithBounds(
            sourceBitmap = srcBmp,
            sourceMesh = srcMesh,
            targetMesh = tgtMesh,
            targetCanvasMatrix2x3 = mat,
            canvasSize = 256
        )
        // Top of target canvas maps far above y = 0 in sourceBitmap -> inBoundsMask must be 0f!
        assertEquals(0f, warped.inBoundsMask[4 * 256 + 128], 0.01f)
        // Face center maps well inside sourceBitmap -> inBoundsMask must be 1.0f!
        assertEquals(1f, warped.inBoundsMask[128 * 256 + 128], 0.01f)
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

    @Test
    fun `SD 15 Stage 5 components bypass XNNPACK, keep 4-step seam plan, and respect cancellation`() {
        // 1. Verify XNNPACK bypass for SD 1.5 components (prevents UNet/VAE native stall & RAM doubling)
        assertTrue(OnnxProtobufInspector.shouldBypassXnnpackForFile(File("/data/sd_inpaint/sd15_fp16/unet/model.onnx")))
        assertTrue(OnnxProtobufInspector.shouldBypassXnnpackForFile(File("/data/sd_inpaint/sd15_fp16/vae_encoder/model.onnx")))
        assertTrue(OnnxProtobufInspector.shouldBypassXnnpackForFile(File("/data/sd_inpaint/sd15_fp16/vae_decoder/model.onnx")))
        assertTrue(OnnxProtobufInspector.shouldBypassXnnpackForFile(File("/data/sd_inpaint/sd15_fp16/text_encoder/model.onnx")))
        assertTrue(OnnxProtobufInspector.shouldBypassXnnpackForFile(File("unet.onnx")))
        assertFalse(OnnxProtobufInspector.shouldBypassXnnpackForFile(File("det_10g.onnx")))

        // 2. Verify Stage 5 Diffusion Seam Bridge strength (0.25f fastMode) produces a focused 3-step plan
        val plan = SdInpaintOnnxEngine.computeDenoisePlan(fastMode = true, strength = 0.25f)
        assertEquals(3, plan.denoiseSteps)

        // 3. Verify cooperative cancellation exits cleanly without hanging or swallowing CancellationException
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(20, 25, 35),
            skinColor = Color.rgb(225, 180, 150),
            hairColor = Color.rgb(30, 20, 20)
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(220, 235, 250),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(45, 30, 20)
        )

        var cancelled = false
        try {
            CleanEraseHeadReplacementEngine.execute(
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
                isCancelled = { true },
                onProgress = {}
            )
            fail("Expected CancellationException when isCancelled returns true")
        } catch (_: CancellationException) {
            cancelled = true
        }
        assertTrue(cancelled)
    }

    @Test
    fun `SeamBridge prioritizes INT8 over FP16 and FP32 and caps portrait resolution safely`() {
        val root = com.example.onnx.SdInpaintPackManager.getSdInpaintRootDir(context)
        root.deleteRecursively()
        root.mkdirs()

        listOf(
            com.example.onnx.SdInpaintPackType.SD15_FP32,
            com.example.onnx.SdInpaintPackType.SD15_FP16,
            com.example.onnx.SdInpaintPackType.REALVIS_INT8
        ).forEach { packType ->
            val dir = File(root, packType.id)
            listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").forEach { comp ->
                val compDir = File(dir, comp).apply { mkdirs() }
                File(compDir, "model.onnx").writeBytes(ByteArray(2048) { 7 })
            }
        }

        val chosen = com.example.onnx.SdInpaintPackManager.resolvePackForSeamBridge(context)
        assertEquals(com.example.onnx.SdInpaintPackType.REALVIS_INT8, chosen?.packType)

        // Portrait image capped at 384 or 512 does NOT inflate to 512x768 on memory-constrained fast runs
        val dims384 = SdInpaintOnnxEngine.computeAspectMatchedTargetResolution(720, 1280, true, maxDimensionPx = 384)
        assertEquals(384, dims384.width)
        assertEquals(384, dims384.height)

        val dims512 = SdInpaintOnnxEngine.computeAspectMatchedTargetResolution(720, 1280, true, maxDimensionPx = 512)
        assertEquals(512, dims512.width)
        assertEquals(512, dims512.height)

        val dims768 = SdInpaintOnnxEngine.computeAspectMatchedTargetResolution(720, 1280, true, maxDimensionPx = 768)
        assertEquals(512, dims768.width)
        assertEquals(768, dims768.height)
        root.deleteRecursively()
    }

    @Test
    fun `warpSourceHeadToTarget3DPoseWithBounds preserves open-mouth Source smile teeth without lip collapse`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(30, 35, 45),
            skinColor = Color.rgb(230, 185, 155),
            hairColor = Color.rgb(28, 22, 22)
        )
        // Paint wide open smile teeth band around Source mouth center
        val mouthCy = (srcFace.landmarks5[3].y + srcFace.landmarks5[4].y) * 0.5f
        val mouthCx = 128f
        for (y in (mouthCy - 5).toInt()..(mouthCy + 6).toInt()) {
            for (x in (mouthCx - 16).toInt()..(mouthCx + 16).toInt()) {
                srcBmp.setPixel(x, y, Color.rgb(245, 245, 245))
            }
        }
        val (tgtBmp, rawTgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(120, 75, 45),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(25, 20, 20)
        )
        // Narrower closed-mouth target landmarks
        val closedMouthFace = rawTgtFace.copy(
            landmarks5 = rawTgtFace.landmarks5.mapIndexed { idx, pt ->
                when (idx) {
                    3 -> PointF(pt.x + 6f, pt.y - 8f)
                    4 -> PointF(pt.x - 6f, pt.y - 8f)
                    else -> pt
                }
            }
        )

        val srcMesh = FaceMesh3DEngine.extractFaceMesh3D(null, srcBmp, srcFace, null)
        val tgtMesh = FaceMesh3DEngine.extractFaceMesh3D(null, tgtBmp, closedMouthFace, null)
        val mat = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(closedMouthFace.landmarks5, 256, 0f)
        val warped = FaceMesh3DEngine.warpSourceHeadToTarget3DPoseWithBounds(
            sourceBitmap = srcBmp,
            sourceMesh = srcMesh,
            targetMesh = tgtMesh,
            targetCanvasMatrix2x3 = mat,
            canvasSize = 256
        )

        // Bright white teeth pixels must survive cleanly in the warped mouth zone without black/skin tearing
        var brightTeethCount = 0
        for (y in 160..215) {
            for (x in 100..156) {
                val p = warped.bitmap.getPixel(x, y)
                if (Color.red(p) > 225 && Color.green(p) > 225 && Color.blue(p) > 225) {
                    brightTeethCount++
                }
            }
        }
        assertTrue("Expected open-mouth teeth pixels preserved cleanly, got count=$brightTeethCount", brightTeethCount >= 40)
    }

    @Test
    fun `CleanEraseHeadReplacementEngine strips cool corner glare, keeps cheek smooth, and protects lower dupatta`() {
        val w = 256
        val h = 256
        val (srcBmp, rawSrcFace) = createPortrait(
            width = w,
            height = h,
            bgColor = Color.rgb(85, 185, 195), // Cool cyan window glare behind cropped top hair
            skinColor = Color.rgb(232, 188, 158),
            hairColor = Color.rgb(26, 24, 24)
        )
        // Paint dark black blouse strip near bottom edge of source
        for (y in 215 until h) {
            for (x in 0 until w) {
                srcBmp.setPixel(x, y, Color.rgb(12, 12, 14))
            }
        }
        val (tgtBmp, tgtFace) = createPortrait(
            width = w,
            height = h,
            bgColor = Color.rgb(105, 62, 34), // Warm brown wood background
            skinColor = Color.rgb(212, 168, 140),
            hairColor = Color.rgb(22, 20, 20)
        )
        // Paint olive-gold dupatta band below Target chin
        for (y in 196 until 228) {
            for (x in 64 until 192) {
                tgtBmp.setPixel(x, y, Color.rgb(165, 148, 52))
            }
        }

        val result = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = rawSrcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )

        // 1. Target olive-gold dupatta directly below neck must remain protected (never overwritten by black blouse)
        val dupattaPx = result.outputBitmap.getPixel(128, 212)
        assertTrue(
            "Expected olive-gold dupatta preserved below chin, got RGB=(${Color.red(dupattaPx)},${Color.green(dupattaPx)},${Color.blue(dupattaPx)})",
            Color.red(dupattaPx) > 110 && Color.green(dupattaPx) > 95
        )
        // 2. Step 2 portrait upper crown must not leak strong cyan window glare
        val step2 = result.stage2SwapRestore512!!
        for (y in 135..165 step 10) {
            val crownPx = step2.getPixel(215, y)
            assertTrue(
                "Expected cool cyan cast stripped from crown at y=$y, got RGB=(${Color.red(crownPx)},${Color.green(crownPx)},${Color.blue(crownPx)})",
                Color.blue(crownPx) - Color.red(crownPx) < 18
            )
        }
    }

    @Test
    fun `CleanEraseHeadReplacementEngine preserves peripheral marigold flowers while erasing bright wide hair`() {
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
            hairColor = Color.rgb(210, 25, 25),
            hairRadiusScale = 1.35f
        )
        // Add vivid yellow flower garland cluster in top-left exterior region outside head
        for (y in 24..56) {
            for (x in 20..52) {
                tgtBmp.setPixel(x, y, Color.rgb(245, 200, 20))
            }
        }

        val result = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )

        val flowerPx = result.outputBitmap.getPixel(36, 40)
        assertTrue(
            "Expected peripheral yellow flower preserved, got RGB=(${Color.red(flowerPx)},${Color.green(flowerPx)},${Color.blue(flowerPx)})",
            Color.red(flowerPx) > 190 && Color.green(flowerPx) > 150 && Color.blue(flowerPx) < 80
        )
        val erasedHairPx = result.outputBitmap.getPixel(128, 48)
        assertTrue(
            "Expected wide red target hair erased cleanly toward sky bg, got blue=${Color.blue(erasedHairPx)}",
            Color.blue(erasedHairPx) > 150
        )
    }

    @Test
    fun `CleanEraseHeadReplacementEngine stops immediately between Stage 3 and Stage 4 on cancel`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(20, 25, 35),
            skinColor = Color.rgb(225, 180, 150),
            hairColor = Color.rgb(30, 20, 20)
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(220, 235, 250),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(45, 30, 20)
        )

        val cancelFlag = java.util.concurrent.atomic.AtomicBoolean(false)
        val observedStages = mutableListOf<Int>()
        var cancelled = false
        try {
            CleanEraseHeadReplacementEngine.execute(
                context = context,
                ortEnv = null,
                sourceBitmap = srcBmp,
                sourceFace = srcFace,
                targetBitmap = tgtBmp,
                targetFacesToReplace = listOf(tgtFace),
                pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
                enableColorTransfer = true,
                enableProvenanceWatermark = false,
                allowTwoModelFallbackForTesting = true,
                lowMemoryMode = true,
                isCancelled = { cancelFlag.get() },
                onProgress = { progress ->
                    observedStages.add(progress.stepIndex)
                    if (progress.stepIndex == 3) {
                        cancelFlag.set(true)
                    }
                }
            )
            fail("Expected CancellationException right after Stage 3 when cancelled during Stage 3")
        } catch (_: CancellationException) {
            cancelled = true
        }
        assertTrue(cancelled)
        assertTrue("Expected Stage 3 to start before cancellation, stages=$observedStages", observedStages.contains(3))
        assertFalse("Stage 4 must NOT run after Stage 3 cancellation, stages=$observedStages", observedStages.any { it >= 4 })
    }

    @Test
    fun `CleanEraseHeadReplacementEngine multi-face close proximity preserves Face 1 new hair when replacing Face 2`() {
        val w = 384
        val h = 256
        val bgColor = Color.rgb(215, 230, 245)
        val skinColor = Color.rgb(212, 168, 140)
        val targetRedHair = Color.rgb(215, 30, 30)
        val sourceBlackHair = Color.rgb(24, 22, 20)

        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(25, 30, 40),
            skinColor = Color.rgb(228, 185, 155),
            hairColor = sourceBlackHair,
            hairRadiusScale = 1.05f
        )

        val tgtBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(w * h) { bgColor }
        val eyeDist = 34f
        val cy = 128f
        val centers = listOf(132f, 252f)
        fun paintHead(cx: Float) {
            val hairRx = eyeDist * 1.65f
            val hairRy = eyeDist * 1.75f
            val faceRx = eyeDist * 1.05f
            val faceRy = eyeDist * 1.22f
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val lx = x - cx
                    val ly = y - cy
                    val hnx = lx / hairRx
                    val hny = (ly + hairRy * 0.30f) / hairRy
                    if (hnx * hnx + hny * hny <= 1f) {
                        px[y * w + x] = targetRedHair
                    }
                    if (kotlin.math.abs(lx) <= eyeDist * 0.55f && ly in (eyeDist * 0.85f)..(eyeDist * 1.85f)) {
                        px[y * w + x] = skinColor
                    }
                    val fnx = lx / faceRx
                    val fny = (ly - eyeDist * 0.12f) / faceRy
                    if (fnx * fnx + fny * fny <= 1f) {
                        px[y * w + x] = skinColor
                    }
                }
            }
        }
        centers.forEach(::paintHead)
        tgtBmp.setPixels(px, 0, w, 0, 0, w, h)

        val targetFaces = centers.mapIndexed { idx, cx ->
            DetectedFace(
                index = idx,
                boundingBox = RectF(cx - eyeDist * 1.2f, cy - eyeDist * 1.2f, cx + eyeDist * 1.2f, cy + eyeDist * 1.4f),
                score = 0.99f,
                landmarks5 = listOf(
                    PointF(cx - eyeDist * 0.5f, cy),
                    PointF(cx + eyeDist * 0.5f, cy),
                    PointF(cx, cy + eyeDist * 0.55f),
                    PointF(cx - eyeDist * 0.42f, cy + eyeDist * 1.05f),
                    PointF(cx + eyeDist * 0.42f, cy + eyeDist * 1.05f)
                ),
                detectorSource = "MultiFaceTest"
            )
        }

        val result = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = targetFaces,
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )

        assertEquals(2, result.swappedFacesCount)
        assertEquals(bgColor, result.outputBitmap.getPixel(4, 4))
        // Face 1 inner right hair near Face 2 must retain dark Source hair (not erased by Face 2 or reverted to red)
        val face1RightHairCore = result.outputBitmap.getPixel(154, 78)
        assertTrue(
            "Expected Face 1 interior right hair near Face 2 to retain dark Source hair, got RGB=(${Color.red(face1RightHairCore)},${Color.green(face1RightHairCore)},${Color.blue(face1RightHairCore)})",
            Color.red(face1RightHairCore) < 95 && Color.blue(face1RightHairCore) < 95
        )
        val face1RightHairEdge = result.outputBitmap.getPixel(166, 78)
        assertTrue(
            "Expected Face 1 outer right hair edge near Face 2 not to revert to red or over-erased sky bg, got RGB=(${Color.red(face1RightHairEdge)},${Color.green(face1RightHairEdge)},${Color.blue(face1RightHairEdge)})",
            Color.red(face1RightHairEdge) < 130 && Color.blue(face1RightHairEdge) < 140 && kotlin.math.abs(Color.red(face1RightHairEdge) - Color.blue(face1RightHairEdge)) < 35
        )
        // Face 2 crown (x = 252, y = 72) must also be replaced with dark source hair (not original bright red)
        val face2Crown = result.outputBitmap.getPixel(252, 72)
        assertTrue(
            "Expected Face 2 crown replaced with dark source hair rather than red target hair, got RGB=(${Color.red(face2Crown)},${Color.green(face2Crown)},${Color.blue(face2Crown)})",
            Color.red(face2Crown) < 110 && Color.blue(face2Crown) < 110
        )
    }

    @Test
    fun `CleanEraseHeadReplacementEngine protects rolled gt 25 deg collar and erases wide hat under fallback`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(20, 25, 35),
            skinColor = Color.rgb(225, 180, 150),
            hairColor = Color.rgb(30, 20, 20),
            hairRadiusScale = 0.85f
        )
        val (tgtBmp, rawTgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(220, 235, 250),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(40, 25, 20),
            rollDegrees = 32f
        )
        // Paint bright red wide hat brim across upper forehead/crown
        for (y in 42..66) {
            for (x in 54..202) {
                tgtBmp.setPixel(x, y, Color.rgb(220, 30, 30))
            }
        }
        // Paint emerald collar across lower shoulder/neck band
        for (y in 204..228) {
            for (x in 72..184) {
                tgtBmp.setPixel(x, y, Color.rgb(20, 165, 65))
            }
        }
        // Simulate mild profile yaw compression alongside 32 deg roll
        val rolledProfileFace = rawTgtFace.copy(
            landmarks5 = rawTgtFace.landmarks5.mapIndexed { idx, pt ->
                when (idx) {
                    0 -> PointF(pt.x + 5f, pt.y)
                    1 -> PointF(pt.x - 5f, pt.y)
                    else -> pt
                }
            }
        )

        val result = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(rolledProfileFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )

        // Emerald collar band stays protected even under roll > 25 deg
        val collarLeft = result.outputBitmap.getPixel(95, 214)
        val collarRight = result.outputBitmap.getPixel(162, 214)
        assertTrue(
            "Expected left rolled collar preserved, got RGB=(${Color.red(collarLeft)},${Color.green(collarLeft)},${Color.blue(collarLeft)})",
            Color.green(collarLeft) > Color.red(collarLeft) + 40
        )
        assertTrue(
            "Expected right rolled collar preserved, got RGB=(${Color.red(collarRight)},${Color.green(collarRight)},${Color.blue(collarRight)})",
            Color.green(collarRight) > Color.red(collarRight) + 40
        )
        // Wide red hat brim corner (x = 76, y = 52) above narrower source hair should be erased toward sky blue
        val erasedHatBrim = result.outputBitmap.getPixel(76, 52)
        assertTrue(
            "Expected wide red hat brim erased toward sky background, got RGB=(${Color.red(erasedHatBrim)},${Color.green(erasedHatBrim)},${Color.blue(erasedHatBrim)})",
            Color.blue(erasedHatBrim) > 150
        )
    }

    @Test
    fun `No5 capBitmapMaxDimension and DetectedFace scaled preserve small inputs and cap large 12MP photos cleanly`() {
        val smallBmp = Bitmap.createBitmap(1024, 1536, Bitmap.Config.ARGB_8888)
        val noScale = ImageGalleryHelper.capBitmapMaxDimension(smallBmp, 2048)
        assertFalse(noScale.wasScaled)
        assertSame(smallBmp, noScale.bitmap)
        assertEquals(1.0f, noScale.scaleX, 0.0001f)
        assertEquals(1.0f, noScale.scaleY, 0.0001f)
        assertFalse(smallBmp.isRecycled)

        val large12Mp = Bitmap.createBitmap(3000, 4000, Bitmap.Config.ARGB_8888)
        val capped2048 = ImageGalleryHelper.capBitmapMaxDimension(large12Mp, 2048)
        assertTrue(capped2048.wasScaled)
        assertEquals(1536, capped2048.bitmap.width)
        assertEquals(2048, capped2048.bitmap.height)
        assertEquals(1536f / 3000f, capped2048.scaleX, 0.0005f)
        assertEquals(2048f / 4000f, capped2048.scaleY, 0.0005f)
        assertFalse("Caller bitmap must never be recycled by capBitmapMaxDimension", large12Mp.isRecycled)

        val capped1280 = ImageGalleryHelper.capBitmapMaxDimension(large12Mp, 1280)
        assertTrue(capped1280.wasScaled)
        assertEquals(960, capped1280.bitmap.width)
        assertEquals(1280, capped1280.bitmap.height)
        capped2048.bitmap.recycle()
        capped1280.bitmap.recycle()
        smallBmp.recycle()
        large12Mp.recycle()

        val originalFace = DetectedFace(
            index = 1,
            boundingBox = RectF(1000f, 800f, 2000f, 2200f),
            score = 0.95f,
            landmarks5 = listOf(
                PointF(1300f, 1200f),
                PointF(1700f, 1200f),
                PointF(1500f, 1500f),
                PointF(1340f, 1800f),
                PointF(1660f, 1800f)
            ),
            detectorSource = "TestLarge",
            faceContourPoints = listOf(PointF(1200f, 1000f), PointF(1800f, 2000f)),
            eulerX = 5f,
            eulerY = -10f,
            eulerZ = 3f
        )
        assertSame(originalFace, originalFace.scaled(1.0f, 1.0f))

        val scaledFace = originalFace.scaled(0.512f, 0.512f)
        assertEquals(512f, scaledFace.boundingBox.left, 0.01f)
        assertEquals(409.6f, scaledFace.boundingBox.top, 0.01f)
        assertEquals(665.6f, scaledFace.landmarks5[0].x, 0.05f)
        assertEquals(614.4f, scaledFace.faceContourPoints[0].x, 0.05f)
        assertEquals(5f, scaledFace.eulerX, 0.001f)
        assertEquals(-10f, scaledFace.eulerY, 0.001f)
        assertEquals(3f, scaledFace.eulerZ, 0.001f)
    }

    @Test
    fun `No5 resolveSafeProcessingMaxDimensionPx enforces 1280 low-RAM cap and 2048 working cap`() {
        val vm = FaceSwapViewModel(ApplicationProvider.getApplicationContext<Application>())

        assertEquals(
            2048,
            vm.resolveSafeProcessingMaxDimensionPx(
                FaceSwapUiState(
                    lowMemoryMode = false,
                    runtimeLowRamLatched = false,
                    qualityLevel = ProcessingQualityLevel.HIGH_QUALITY
                )
            )
        )
        assertEquals(
            2048,
            vm.resolveSafeProcessingMaxDimensionPx(
                FaceSwapUiState(
                    lowMemoryMode = false,
                    runtimeLowRamLatched = false,
                    qualityLevel = ProcessingQualityLevel.BALANCED
                )
            )
        )
        assertEquals(
            1280,
            vm.resolveSafeProcessingMaxDimensionPx(
                FaceSwapUiState(
                    lowMemoryMode = false,
                    runtimeLowRamLatched = false,
                    qualityLevel = ProcessingQualityLevel.FAST
                )
            )
        )
        assertEquals(
            1280,
            vm.resolveSafeProcessingMaxDimensionPx(
                FaceSwapUiState(
                    lowMemoryMode = true,
                    runtimeLowRamLatched = false,
                    qualityLevel = ProcessingQualityLevel.HIGH_QUALITY
                )
            )
        )
        assertEquals(
            1280,
            vm.resolveSafeProcessingMaxDimensionPx(
                FaceSwapUiState(
                    lowMemoryMode = false,
                    runtimeLowRamLatched = true,
                    qualityLevel = ProcessingQualityLevel.HIGH_QUALITY
                )
            )
        )
    }

    @Test
    fun `No5 CleanEraseHeadReplacementEngine auto-caps oversized bitmaps and leaves caller inputs unrecycled`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 1920,
            height = 1440,
            bgColor = Color.rgb(25, 30, 40),
            skinColor = Color.rgb(225, 180, 150),
            hairColor = Color.rgb(30, 20, 20),
            eyeDist = 180f
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 2400,
            height = 1800,
            bgColor = Color.rgb(215, 230, 245),
            skinColor = Color.rgb(210, 165, 140),
            hairColor = Color.rgb(45, 30, 20),
            eyeDist = 220f
        )

        val result = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )

        assertEquals(1280, result.outputBitmap.width)
        assertEquals(960, result.outputBitmap.height)
        assertFalse(srcBmp.isRecycled)
        assertFalse(tgtBmp.isRecycled)
        srcBmp.recycle()
        tgtBmp.recycle()
        result.outputBitmap.recycle()
    }

    @Test
    fun `postSwapCrossCheckAndBeautifyPass heals dark neck seam and ear cutout smudges while protecting clothing`() {
        val w = 256
        val h = 256
        val orig = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val origPx = IntArray(w * h) { Color.rgb(180, 145, 110) }
        val outPx = IntArray(w * h) { Color.rgb(180, 145, 110) }

        val eyeDist = 44f
        val cx = 128f
        val eyeY = 96f
        val mouthY = eyeY + eyeDist * 1.05f // ~142
        val chinY = mouthY + (mouthY - eyeY) * 0.60f // ~170

        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                // Healthy lower neck & face skin
                if (kotlin.math.abs(x - cx) < 55f && y in 60..210) {
                    val c = Color.rgb(215, 170, 142)
                    origPx[idx] = c
                    outPx[idx] = c
                }
                // Blue cloth collar below y = 212
                if (y >= 212) {
                    val cloth = Color.rgb(12, 130, 140)
                    origPx[idx] = cloth
                    outPx[idx] = cloth
                }
            }
        }

        // Inject synthetic dark U-shaped neck seam trough right under chin (y = 176..184)
        for (y in 176..184) {
            for (x in 108..148) {
                outPx[y * w + x] = Color.rgb(75, 55, 45)
            }
        }
        // Inject synthetic dark ear-flank cutout hole where orig had warm gold earring / background
        for (y in 115..135) {
            for (x in 178..190) {
                origPx[y * w + x] = Color.rgb(212, 168, 72)
                outPx[y * w + x] = Color.rgb(22, 18, 16)
            }
        }

        orig.setPixels(origPx, 0, w, 0, 0, w, h)
        out.setPixels(outPx, 0, w, 0, 0, w, h)

        val face = DetectedFace(
            index = 0,
            boundingBox = RectF(cx - 50f, eyeY - 40f, cx + 50f, chinY + 10f),
            score = 0.99f,
            landmarks5 = listOf(
                PointF(cx - eyeDist * 0.5f, eyeY),
                PointF(cx + eyeDist * 0.5f, eyeY),
                PointF(cx, eyeY + eyeDist * 0.55f),
                PointF(cx - eyeDist * 0.42f, mouthY),
                PointF(cx + eyeDist * 0.42f, mouthY)
            ),
            detectorSource = "TestCrossCheck"
        )

        val report = CleanEraseHeadReplacementEngine.runPostSwapCrossCheckAndBeautifyPass(
            outputBitmap = out,
            originalTargetBitmap = orig,
            placedFaces = listOf(face),
            enhancementStrength = 0.90f
        )

        val healedNeckPx = out.getPixel(128, 180)
        assertTrue(
            "Expected dark neck seam trough (r=75) to heal significantly, got r=${Color.red(healedNeckPx)}, report=$report",
            Color.red(healedNeckPx) > 135
        )
        val healedEarPx = out.getPixel(184, 125)
        assertTrue(
            "Expected dark ear cutout hole (r=22) to be repaired toward gold/bg, got r=${Color.red(healedEarPx)}, report=$report",
            Color.red(healedEarPx) > 110
        )
        // Collar clothing must stay untouched
        assertEquals(Color.rgb(12, 130, 140), out.getPixel(128, 220))
        assertTrue("Expected report to include neck/ear/polish stats: $report", report.contains("polish="))

        orig.recycle()
        out.recycle()
    }

    @Test
    fun `execute emits PostSwap cross-check summary and preserves tilted 3D alignment cleanly`() {
        val (srcBmp, srcFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(25, 30, 40),
            skinColor = Color.rgb(228, 182, 152),
            hairColor = Color.rgb(30, 22, 20),
            rollDegrees = -15f
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(210, 225, 240),
            skinColor = Color.rgb(212, 168, 140),
            hairColor = Color.rgb(32, 24, 22),
            rollDegrees = 12f
        )

        val result = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )

        assertTrue(
            "Expected PostSwap CrossCheck+Beautify in summary, got: ${result.pipelineSummary}",
            result.pipelineSummary.contains("PostSwap(CrossCheck+Beautify:")
        )
        assertEquals(256, result.outputBitmap.width)
        assertEquals(tgtBmp.getPixel(5, 5), result.outputBitmap.getPixel(5, 5))

        srcBmp.recycle()
        tgtBmp.recycle()
        result.outputBitmap.recycle()
    }

    @Test
    fun `closeSmallNotchesAndFillInteriorHoles seals enclosed brow and ear holes without filling exterior`() {
        val size = 64
        val alpha = FloatArray(size * size)
        val skin = FloatArray(size * size)
        for (y in 12..52) {
            for (x in 14..50) {
                alpha[y * size + x] = 1.0f
                if (y >= 24) skin[y * size + x] = 1.0f
            }
        }
        // Punch enclosed highlight hole inside forehead/brow and cheek
        for (y in 20..24) {
            for (x in 22..26) {
                alpha[y * size + x] = 0.0f
                skin[y * size + x] = 0.0f
            }
        }
        for (y in 34..37) {
            for (x in 34..37) {
                alpha[y * size + x] = 0.0f
                skin[y * size + x] = 0.0f
            }
        }

        HeadSegmentationAndInpainting.closeSmallNotchesAndFillInteriorHoles(alpha, skin, size)

        assertEquals(1.0f, alpha[22 * size + 24], 0.01f)
        assertEquals(1.0f, alpha[35 * size + 35], 0.01f)
        assertEquals(1.0f, skin[35 * size + 35], 0.01f)
        assertEquals("Exterior corners must remain 0f", 0.0f, alpha[4 * size + 4], 0.01f)
    }

    @Test
    fun `detectInnerPhotoContentBounds strips Gallery white UI bars and keeps intact bun uncut`() {
        val w = 240
        val h = 400
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(w * h)
        // Top & bottom Gallery white UI bars (y=0..89 and y=310..399) with dark status bar text glyphs
        for (y in 0 until h) {
            for (x in 0 until w) {
                px[y * w + x] = when {
                    y < 90 || y >= 310 -> {
                        if (x in 20..44 && (y in 18..24 || y in 350..356)) Color.rgb(30, 30, 30)
                        else Color.rgb(252, 252, 252)
                    }
                    else -> Color.rgb(185, 200, 195) // Interior photo wall
                }
            }
        }
        // Tall intact portrait + hair bun inside y=104..308
        val cx = 120f
        val eyeY = 215f
        val eyeDist = 40f
        for (y in 104 until 310) {
            for (x in 0 until w) {
                val hDx = (x - cx) / 62f
                val hDy = (y - 185f) / 76f
                if (hDx * hDx + hDy * hDy <= 1f) {
                    px[y * w + x] = Color.rgb(28, 24, 24)
                }
                val fDx = (x - cx) / 42f
                val fDy = (y - 222f) / 50f
                if (fDx * fDx + fDy * fDy <= 1f) {
                    px[y * w + x] = Color.rgb(228, 182, 152)
                }
            }
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)

        val (minY, maxY) = FaceMesh3DEngine.detectInnerPhotoContentBounds(
            srcPixels = px,
            srcW = w,
            srcH = h,
            srcEyeY = eyeY,
            srcEyeDist = eyeDist,
            srcChinY = 278f
        )
        assertTrue("Expected top white Gallery UI bar excluded (minY=$minY)", minY >= 85f)
        assertTrue("Expected bottom white Gallery UI bar excluded (maxY=$maxY)", maxY <= 312f)

        // Also run full B+C execute with gallery-framed Source to confirm Output 2 cutout strips white UI bar & preserves head core
        val srcFace = DetectedFace(
            index = 0,
            boundingBox = RectF(cx - 46f, eyeY - 40f, cx + 46f, 280f),
            score = 0.99f,
            landmarks5 = listOf(
                PointF(cx - eyeDist * 0.5f, eyeY),
                PointF(cx + eyeDist * 0.5f, eyeY),
                PointF(cx, eyeY + eyeDist * 0.55f),
                PointF(cx - eyeDist * 0.42f, eyeY + eyeDist * 1.05f),
                PointF(cx + eyeDist * 0.42f, eyeY + eyeDist * 1.05f)
            ),
            detectorSource = "GalleryScreenshotSrc"
        )
        val (tgtBmp, tgtFace) = createPortrait(
            width = 256,
            height = 256,
            bgColor = Color.rgb(90, 58, 36),
            skinColor = Color.rgb(212, 168, 140),
            hairColor = Color.rgb(24, 20, 20)
        )

        val res = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = bmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            onProgress = {}
        )
        val cutout = res.sourceHeadCutoutPreview!!
        // Bottom-corner gallery white UI rectangle must have alpha == 0f (completely transparent) in Output 2!
        assertEquals(0, Color.alpha(cutout.getPixel(80, 485)))
        assertEquals(0, Color.alpha(cutout.getPixel(430, 485)))

        bmp.recycle()
        tgtBmp.recycle()
        res.outputBitmap.recycle()
    }

    @Test
    fun `CleanErase TARGET_REACTION preserves Target closed-mouth smile vs SOURCE_REACTION open teeth`() {
        val w = 256
        val h = 256
        val (srcBmp, srcFace) = createPortrait(
            width = w,
            height = h,
            bgColor = Color.rgb(230, 235, 240),
            skinColor = Color.rgb(228, 185, 155),
            hairColor = Color.rgb(26, 22, 22)
        )
        val srcMouthY = ((srcFace.landmarks5[3].y + srcFace.landmarks5[4].y) * 0.5f).toInt()
        for (y in (srcMouthY - 5)..(srcMouthY + 5)) {
            for (x in 114..142) {
                srcBmp.setPixel(x, y, Color.rgb(248, 248, 248)) // Open bright white Source teeth
            }
        }

        val (tgtBmp, tgtFace) = createPortrait(
            width = w,
            height = h,
            bgColor = Color.rgb(120, 80, 50),
            skinColor = Color.rgb(212, 168, 140),
            hairColor = Color.rgb(24, 20, 20)
        )
        val tgtMouthY = ((tgtFace.landmarks5[3].y + tgtFace.landmarks5[4].y) * 0.5f).toInt()
        for (y in (tgtMouthY - 4)..(tgtMouthY + 4)) {
            for (x in 114..142) {
                tgtBmp.setPixel(x, y, Color.rgb(168, 74, 82)) // Closed rose-berry Target smile
            }
        }

        val targetReactRes = CleanEraseHeadReplacementEngine.execute(
            context = context,
            ortEnv = null,
            sourceBitmap = srcBmp,
            sourceFace = srcFace,
            targetBitmap = tgtBmp,
            targetFacesToReplace = listOf(tgtFace),
            pipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
            enableColorTransfer = true,
            enableProvenanceWatermark = false,
            allowTwoModelFallbackForTesting = true,
            lowMemoryMode = true,
            faceReactionMode = FaceReactionSourceMode.TARGET_REACTION,
            onProgress = {}
        )

        val mouthCenterTgt = targetReactRes.outputBitmap.getPixel(128, tgtMouthY)
        assertTrue(
            "Expected TARGET_REACTION mouth center to keep closed Target rose lip (not white Source teeth), got RGB=(${Color.red(mouthCenterTgt)},${Color.green(mouthCenterTgt)},${Color.blue(mouthCenterTgt)})",
            Color.red(mouthCenterTgt) > Color.green(mouthCenterTgt) + 30 && Color.blue(mouthCenterTgt) < 150
        )

        srcBmp.recycle()
        tgtBmp.recycle()
        targetReactRes.outputBitmap.recycle()
    }
}



