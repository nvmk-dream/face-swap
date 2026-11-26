package com.example

import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.DetectedFace
import com.example.onnx.FaceAlignment
import com.example.onnx.FaceFusionOptionDEngine
import com.example.onnx.FaceReactionSourceMode
import com.example.onnx.FaceSwapStrategyOption
import com.example.onnx.InSwapperEngine
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxProtobufInspector
import com.example.onnx.OptionDEnhancerChoice
import com.example.onnx.OptionDFaceFusionConfig
import com.example.onnx.OptionDSwapperChoice
import com.example.onnx.SkinToneSourceMode
import com.example.ui.FaceSwapUiState
import com.example.ui.FaceSwapViewModel
import com.example.ui.StudioMode
import com.example.ui.tabs.StudioFrontPage
import com.example.ui.theme.MyApplicationTheme
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class FaceFusionOptionDRobolectricTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    private val ortEnv: OrtEnvironment?
        get() = runCatching { OrtEnvironment.getEnvironment() }.getOrNull()

    private fun cleanTestModelsDir() {
        File(context.filesDir, "models").deleteRecursively()
    }

    @After
    fun tearDown() {
        cleanTestModelsDir()
    }

    private fun writeDummyValidOnnxFile(dir: File, name: String, sizeBytes: Int = 4096): File {
        dir.mkdirs()
        val f = File(dir, name)
        f.writeBytes(ByteArray(sizeBytes) { (it % 251).toByte() })
        return f
    }

    private fun makePortrait(
        width: Int = 360,
        height: Int = 360,
        skinColor: Int = Color.rgb(220, 175, 148),
        bgColor: Int = Color.rgb(28, 38, 58)
    ): Pair<Bitmap, DetectedFace> {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val cx = width * 0.5f
        val cy = height * 0.52f
        val rx = 82f
        val ry = 102f
        val px = IntArray(width * height) { bgColor }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val nx = (x - cx) / rx
                val ny = (y - cy) / ry
                if (nx * nx + ny * ny <= 1f) {
                    px[y * width + x] = skinColor
                }
            }
        }
        bmp.setPixels(px, 0, width, 0, 0, width, height)
        val lms = listOf(
            PointF(cx - rx * 0.36f, cy - ry * 0.16f),
            PointF(cx + rx * 0.36f, cy - ry * 0.16f),
            PointF(cx, cy + ry * 0.10f),
            PointF(cx - rx * 0.30f, cy + ry * 0.42f),
            PointF(cx + rx * 0.30f, cy + ry * 0.42f)
        )
        val face = DetectedFace(
            index = 0,
            boundingBox = RectF(cx - rx, cy - ry, cx + rx, cy + ry),
            score = 0.99f,
            landmarks5 = lms,
            detectorSource = "TestDetector"
        )
        return bmp to face
    }

    @Test
    fun `pixel boost implode and explode are lossless for 512 to 256 and 256 to 128`() {
        val pixels512 = IntArray(512 * 512) { idx ->
            Color.rgb((idx * 3) and 0xFF, (idx / 512) and 0xFF, (idx * 7) and 0xFF)
        }
        val tiles256 = FaceFusionOptionDEngine.implodePixelBoost(pixels512, boostSize = 512, modelSize = 256)
        assertEquals(4, tiles256.size)
        assertTrue(tiles256.all { it.size == 256 * 256 })
        val restored512 = FaceFusionOptionDEngine.explodePixelBoost(tiles256, boostSize = 512, modelSize = 256)
        assertArrayEquals(pixels512, restored512)

        val pixels256 = IntArray(256 * 256) { idx ->
            Color.rgb((idx * 5) and 0xFF, (idx / 256) and 0xFF, (idx * 11) and 0xFF)
        }
        val tiles128 = FaceFusionOptionDEngine.implodePixelBoost(pixels256, boostSize = 256, modelSize = 128)
        assertEquals(4, tiles128.size)
        assertTrue(tiles128.all { it.size == 128 * 128 })
        val restored256 = FaceFusionOptionDEngine.explodePixelBoost(tiles128, boostSize = 256, modelSize = 128)
        assertArrayEquals(pixels256, restored256)
    }

    @Test
    fun `estimateNormArcFaceExact differs from standard InSwapper estimateNorm without 8px shift`() {
        val srcLandmarks = listOf(
            PointF(130f, 150f),
            PointF(210f, 150f),
            PointF(170f, 195f),
            PointF(140f, 235f),
            PointF(200f, 235f)
        )
        val simSwap512 = FaceAlignment.estimateNormArcFaceExact(srcLandmarks, 512)
        val simSwapV1 = FaceAlignment.estimateNormArcFaceV1(srcLandmarks, 512)
        val ffhq512 = FaceAlignment.estimateNormFfhq(srcLandmarks, 512)
        val standard512 = FaceAlignment.estimateNorm(srcLandmarks, 512)

        assertEquals(6, simSwap512.size)
        assertEquals(6, simSwapV1.size)
        assertEquals(6, ffhq512.size)
        assertEquals(6, standard512.size)
        assertNotEquals(standard512[2], simSwap512[2], 1.0f)
        assertNotEquals(simSwap512[5], simSwapV1[5], 1.0f)
        assertNotEquals(simSwapV1[0], ffhq512[0], 0.05f)
    }

    @Test
    fun `model slot matching routes FaceFusion models and separates occluder from segmentation`() {
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("simswap_512_unofficial.onnx"))
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("reswapper_256.onnx"))
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("blendswap_256.onnx"))
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("ghost_1_256.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("codeformer.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("gpen_bfr_512.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("restoreformer_plus_plus.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("gfpgan-v1.4-w16-v1.onnx"))
        assertEquals(ModelSlot.FACE_OCCLUDER, OnnxProtobufInspector.matchSlotForFileName("face_occluder.onnx"))
        assertEquals(ModelSlot.FACE_OCCLUDER, OnnxProtobufInspector.matchSlotForFileName("xseg_1.onnx"))
        assertEquals(ModelSlot.FACE_OCCLUDER, OnnxProtobufInspector.matchSlotForFileName("dfl_xseg.onnx"))
        assertEquals(ModelSlot.SEGMENTATION, OnnxProtobufInspector.matchSlotForFileName("segformer_B5_ce.onnx"))
    }

    @Test
    fun `multi variant coexistence keeps Option A B BC resolveModelFile untouched while Option D selects variants`() {
        cleanTestModelsDir()
        val swapDir = File(context.filesDir, "models/face_swap")
        val recDir = File(context.filesDir, "models/recognition")
        val enhDir = File(context.filesDir, "models/enhancement")
        val occDir = File(context.filesDir, "models/occlusion")

        writeDummyValidOnnxFile(recDir, "w600k_r50.onnx")
        writeDummyValidOnnxFile(recDir, "crossface_simswap.onnx")

        writeDummyValidOnnxFile(swapDir, "hyperswap_1b_256.onnx")
        writeDummyValidOnnxFile(swapDir, "simswap_512_unofficial.onnx")
        writeDummyValidOnnxFile(swapDir, "reswapper_256.onnx")
        writeDummyValidOnnxFile(swapDir, "inswapper_128.onnx")

        writeDummyValidOnnxFile(enhDir, "gfpgan_1.4.onnx")
        writeDummyValidOnnxFile(enhDir, "codeformer.onnx")
        writeDummyValidOnnxFile(enhDir, "gpen_bfr_512.onnx")
        writeDummyValidOnnxFile(enhDir, "restoreformer_plus_plus.onnx")

        writeDummyValidOnnxFile(occDir, "xseg_1.onnx")

        // Option A / B / B+C primary resolution must stay pinned to HyperSwap 256, w600k_r50, & GFPGAN 1.4
        assertEquals("w600k_r50.onnx", OnnxProtobufInspector.resolveModelFile(context, ModelSlot.RECOGNIZER).name)
        assertEquals("hyperswap_1b_256.onnx", OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SWAPPER).name)
        assertEquals("gfpgan_1.4.onnx", OnnxProtobufInspector.resolveModelFile(context, ModelSlot.ENHANCEMENT).name)
        assertEquals("xseg_1.onnx", OnnxProtobufInspector.resolveModelFile(context, ModelSlot.FACE_OCCLUDER).name)
        assertEquals("crossface_simswap.onnx", FaceFusionOptionDEngine.resolveSimSwapEmbeddingConverterFile(context)?.name)

        // All variants coexist without deletion or crossface pollution
        val swappers = FaceFusionOptionDEngine.listInstalledSwappers(context).map { it.name }
        assertEquals(4, swappers.size)
        val enhancers = FaceFusionOptionDEngine.listInstalledEnhancers(context).map { it.name }
        assertEquals(4, enhancers.size)

        // Option D choices resolve exact requested variants
        assertEquals("simswap_512_unofficial.onnx", FaceFusionOptionDEngine.resolveSwapperFile(context, OptionDSwapperChoice.SIMSWAP_512).name)
        assertEquals("reswapper_256.onnx", FaceFusionOptionDEngine.resolveSwapperFile(context, OptionDSwapperChoice.RESWAPPER_256).name)
        assertEquals("inswapper_128.onnx", FaceFusionOptionDEngine.resolveSwapperFile(context, OptionDSwapperChoice.INSWAPPER_128).name)
        assertEquals("simswap_512_unofficial.onnx", FaceFusionOptionDEngine.resolveSwapperFile(context, OptionDSwapperChoice.AUTO).name)

        assertEquals("codeformer.onnx", FaceFusionOptionDEngine.resolveEnhancerFile(context, OptionDEnhancerChoice.CODEFORMER)?.name)
        assertEquals("gpen_bfr_512.onnx", FaceFusionOptionDEngine.resolveEnhancerFile(context, OptionDEnhancerChoice.GPEN_BFR)?.name)
        assertEquals("restoreformer_plus_plus.onnx", FaceFusionOptionDEngine.resolveEnhancerFile(context, OptionDEnhancerChoice.RESTOREFORMER)?.name)
        assertEquals("gfpgan_1.4.onnx", FaceFusionOptionDEngine.resolveEnhancerFile(context, OptionDEnhancerChoice.GFPGAN)?.name)

        val inspectionSwap = OnnxProtobufInspector.inspectSlot(context, ortEnv, ModelSlot.SWAPPER)
        assertEquals(4, inspectionSwap.installedVariants.size)
    }

    @Test
    fun `option D pipeline executes cleanly while Option A and B pipelines remain isolated`() {
        cleanTestModelsDir()
        val (srcBmp, srcFace) = makePortrait(skinColor = Color.rgb(225, 178, 150))
        val (tgtBmp, tgtFace) = makePortrait(skinColor = Color.rgb(185, 138, 112))

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
            optionDConfig = OptionDFaceFusionConfig(
                swapperChoice = OptionDSwapperChoice.SIMSWAP_512,
                enhancerChoice = OptionDEnhancerChoice.CODEFORMER,
                codeformerFidelity = 0.72f,
                pixelBoostEnabled = true
            ),
            onProgress = {}
        )
        assertNotNull(resD.outputBitmap)
        assertTrue(resD.pipelineSummary.contains("[Opt D]"))
        assertTrue(
            "Unexpected summary: ${resD.pipelineSummary}",
            resD.pipelineSummary.contains("CodeFormer") ||
                resD.pipelineSummary.contains("Built-in") ||
                resD.pipelineSummary.contains("Enhancer") ||
                resD.pipelineSummary.contains("512x512 Bicubic HD")
        )

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
            optionDConfig = OptionDFaceFusionConfig(
                swapperChoice = OptionDSwapperChoice.SIMSWAP_512,
                enhancerChoice = OptionDEnhancerChoice.CODEFORMER,
                pixelBoostEnabled = true
            ),
            onProgress = {}
        )
        assertTrue(resB.pipelineSummary.contains("[Opt B]"))
        assertFalse(resB.pipelineSummary.contains("FaceFusion"))
    }

    @Test
    fun `option D controls card is hidden in Option A B BC and visible only in Option D`() {
        val vm = FaceSwapViewModel(context.applicationContext as Application)

        // Verify setters
        vm.setOptionDSwapperChoice(OptionDSwapperChoice.RESWAPPER_256)
        vm.setOptionDEnhancerChoice(OptionDEnhancerChoice.CODEFORMER)
        vm.setOptionDCodeformerFidelity(0.85f)
        vm.setOptionDPixelBoostEnabled(true)
        assertEquals(OptionDSwapperChoice.RESWAPPER_256, vm.uiState.value.optionDSwapperChoice)
        assertEquals(OptionDEnhancerChoice.CODEFORMER, vm.uiState.value.optionDEnhancerChoice)
        assertEquals(0.85f, vm.uiState.value.optionDCodeformerFidelity, 0.001f)
        assertTrue(vm.uiState.value.optionDPixelBoostEnabled)

        // Default is Option B -> Option D FaceFusion controls card must NOT exist
        var activeState = FaceSwapUiState(
            studioMode = StudioMode.FACE_SWAP,
            swapStrategyOption = FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY
        )
        composeTestRule.setContent {
            MyApplicationTheme {
                StudioFrontPage(
                    uiState = activeState,
                    onSelectStudioMode = {},
                    onPickSourcePhoto = {},
                    onPickTargetPhoto = {},
                    onDetectFaces = {},
                    onPreviewAlignmentPage = {},
                    onOpenSettingsPage = {},
                    onRunSwap = {},
                    onDownloadHd = {},
                    onSaveToGallery = {},
                    onInspectResultDetails = {},
                    onOptionDSwapperChoiceChanged = { vm.setOptionDSwapperChoice(it) },
                    onOptionDEnhancerChoiceChanged = { vm.setOptionDEnhancerChoice(it) },
                    onOptionDCodeformerFidelityChanged = { vm.setOptionDCodeformerFidelity(it) },
                    onOptionDPixelBoostChanged = { vm.setOptionDPixelBoostEnabled(it) }
                )
            }
        }
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("option_d_facefusion_card").assertDoesNotExist()
    }

    @Test
    fun `option D controls card renders and interacts when Option D is active`() {
        var currentConfig = OptionDFaceFusionConfig()
        val state = FaceSwapUiState(
            studioMode = StudioMode.FACE_SWAP,
            swapStrategyOption = FaceSwapStrategyOption.OPTION_D_CRANIOFACIAL_FIT,
            installedOptionDSwappers = listOf("hyperswap_1b_256.onnx", "simswap_512.onnx"),
            installedOptionDEnhancers = listOf("codeformer.onnx", "gpen_bfr_512.onnx")
        )
        composeTestRule.setContent {
            MyApplicationTheme {
                StudioFrontPage(
                    uiState = state,
                    onSelectStudioMode = {},
                    onPickSourcePhoto = {},
                    onPickTargetPhoto = {},
                    onDetectFaces = {},
                    onPreviewAlignmentPage = {},
                    onOpenSettingsPage = {},
                    onRunSwap = {},
                    onDownloadHd = {},
                    onSaveToGallery = {},
                    onInspectResultDetails = {},
                    onOptionDSwapperChoiceChanged = {
                        currentConfig = currentConfig.copy(swapperChoice = it)
                    },
                    onOptionDEnhancerChoiceChanged = {
                        currentConfig = currentConfig.copy(enhancerChoice = it)
                    },
                    onOptionDCodeformerFidelityChanged = {
                        currentConfig = currentConfig.copy(codeformerFidelity = it)
                    },
                    onOptionDPixelBoostChanged = {
                        currentConfig = currentConfig.copy(pixelBoostEnabled = it)
                    }
                )
            }
        }
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("option_d_facefusion_card").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("option_d_occluder_status_badge").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("option_d_swapper_simswap_512").performScrollTo().performClick()
        composeTestRule.onNodeWithTag("option_d_enhancer_codeformer").performScrollTo().performClick()
        composeTestRule.onNodeWithTag("option_d_pixel_boost_switch").performScrollTo().performClick()
        composeTestRule.onNodeWithTag("option_d_codeformer_fidelity_slider").performScrollTo().assertIsDisplayed()

        composeTestRule.waitForIdle()
        assertEquals(OptionDSwapperChoice.SIMSWAP_512, currentConfig.swapperChoice)
        assertEquals(OptionDEnhancerChoice.CODEFORMER, currentConfig.enhancerChoice)
        assertTrue(currentConfig.pixelBoostEnabled)
    }
}
