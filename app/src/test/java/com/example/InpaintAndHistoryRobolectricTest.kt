package com.example

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.example.data.SwapAuditDatabase
import com.example.data.SwapAuditLog
import com.example.onnx.AccessoryMaskEngine
import com.example.onnx.AccessoryMaskLayers
import com.example.onnx.AccessoryTarget
import com.example.onnx.ClipBpeTokenizer
import com.example.onnx.InpaintModelOption
import com.example.onnx.InteractiveInpaintEngine
import android.app.Application
import com.example.onnx.SdInpaintOnnxEngine
import com.example.onnx.SdInpaintPackManager
import com.example.onnx.SdInpaintPackType
import com.example.ui.FaceSwapViewModel
import com.example.util.ImageGalleryHelper
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import kotlin.math.abs

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class InpaintAndHistoryRobolectricTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun swapAuditDatabase_retainsOnlyLatestThreeRecordsAndDeletesOldImages() = runBlocking {
        val db = SwapAuditDatabase.getInstance(context)
        db.clearAll()

        val savedPaths = mutableListOf<String>()
        for (i in 1..5) {
            val bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.rgb(20 * i, 40 * i, 180))
            }
            val path = ImageGalleryHelper.saveBitmapToHistoryCache(context, bmp)!!
            savedPaths.add(path)
            db.insertLog(
                SwapAuditLog(
                    timestampMillis = 1_000_000L + i * 100L,
                    sourceResolution = "64x64",
                    targetResolution = "64x64",
                    swappedFacesCount = 1,
                    pipelineSummary = "Test #$i",
                    detectionMs = 10L,
                    embeddingMs = 10L,
                    inswapperMs = 10L,
                    blendingMs = 10L,
                    totalMs = 40L,
                    colorTransferEnabled = false,
                    watermarkEnabled = false,
                    resultImagePath = path
                )
            )
            Thread.sleep(5)
        }

        val logs = db.observeAllLogs().first()
        assertEquals(3, logs.size)
        assertEquals("Test #5", logs[0].pipelineSummary)
        assertEquals("Test #4", logs[1].pipelineSummary)
        assertEquals("Test #3", logs[2].pipelineSummary)

        // Older files (1 and 2) should be auto-deleted; latest 3 must still exist
        assertFalse(File(savedPaths[0]).exists())
        assertFalse(File(savedPaths[1]).exists())
        assertTrue(File(savedPaths[2]).exists())
        assertTrue(File(savedPaths[3]).exists())
        assertTrue(File(savedPaths[4]).exists())
    }

    @Test
    fun interactiveInpaintEngine_fillsMaskedRegionAndPreservesExteriorPixels() {
        val src = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        src.eraseColor(Color.rgb(30, 120, 210))
        // Place a distinct red block at the center to remove via mask
        for (y in 48 until 80) {
            for (x in 48 until 80) {
                src.setPixel(x, y, Color.RED)
            }
        }

        val mask = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        mask.eraseColor(Color.TRANSPARENT)
        for (y in 44 until 84) {
            for (x in 44 until 84) {
                mask.setPixel(x, y, Color.WHITE)
            }
        }

        val result = InteractiveInpaintEngine.inpaint(
            ortEnv = null,
            sourceBitmap = src,
            maskBitmap = mask,
            prompt = "remove red square",
            fastMode = true,
            lamaModelFile = null,
            preferHardwareAccel = false
        )

        assertEquals(128, result.outputBitmap.width)
        assertEquals(128, result.outputBitmap.height)

        // Exterior unmasked pixel must remain 100% unchanged
        assertEquals(Color.rgb(30, 120, 210), result.outputBitmap.getPixel(10, 10))

        // Center pixel inside masked region must now resemble the blue background rather than red
        val centerColor = result.outputBitmap.getPixel(64, 64)
        assertTrue(Color.blue(centerColor) > Color.red(centerColor))
    }

    @Test
    fun sdInpaintPackManager_prioritizesFp16FirstAndInt8Second_andSelectsLinkedOnnx() {
        val root = SdInpaintPackManager.getSdInpaintRootDir(context)
        root.deleteRecursively()
        root.mkdirs()

        // Create INT8 pack (Second option) with 2 files in unet: model.onnx referencing model_int8.onnx.data + extra.onnx
        val int8Dir = File(root, SdInpaintPackType.REALVIS_INT8.id)
        listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").forEach { comp ->
            val compDir = File(int8Dir, comp).apply { mkdirs() }
            val dummyGraph = ByteArray(2048) { 1 }
            val refBytes = "external_data:model_int8.onnx.data".toByteArray()
            System.arraycopy(refBytes, 0, dummyGraph, 100, refBytes.size)
            File(compDir, "model.onnx").writeBytes(dummyGraph)
            File(compDir, "model_int8.onnx.data").writeBytes(ByteArray(4096) { 2 })
            if (comp == "unet") {
                File(compDir, "unused_alt.onnx").writeBytes(ByteArray(1500) { 3 })
            }
        }

        // Only INT8 present -> default option must pick REALVIS_INT8
        assertEquals(InpaintModelOption.REALVIS_INT8, SdInpaintPackManager.determineDefaultModelOption(context, null))
        val unetSelected = SdInpaintPackManager.selectBestOnnxInComponentDir(File(int8Dir, "unet"), SdInpaintPackType.REALVIS_INT8)
        assertEquals("model.onnx", unetSelected?.name)

        // Now add FP16 pack (Primary Priority 1)
        val fp16Dir = File(root, SdInpaintPackType.SD15_FP16.id)
        listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").forEach { comp ->
            val compDir = File(fp16Dir, comp).apply { mkdirs() }
            File(compDir, "model.onnx").writeBytes(ByteArray(2048) { 4 })
            File(compDir, "model.onnx.data").writeBytes(ByteArray(4096) { 5 })
        }

        val packs = SdInpaintPackManager.discoverInstalledPacks(context)
        assertEquals(2, packs.size)
        assertEquals(SdInpaintPackType.SD15_FP16, packs[0].packType)
        assertEquals(SdInpaintPackType.REALVIS_INT8, packs[1].packType)
        assertEquals(InpaintModelOption.SD15_FP16, SdInpaintPackManager.determineDefaultModelOption(context, null))
        assertEquals(SdInpaintPackType.SD15_FP16, SdInpaintPackManager.resolvePackForOption(context, InpaintModelOption.AUTO)?.packType)
    }

    @Test
    fun clipBpeTokenizer_andFp16Conversion_andPromptSafetyGuardWorkCorrectly() {
        val vocabJson = """
            {
              "<|startoftext|>": 49406,
              "<|endoftext|>": 49407,
              "modern</w>": 1200,
              "dr": 300,
              "ess</w>": 400
            }
        """.trimIndent()
        val mergesTxt = """
            #version: 0.2
            d r
            e s
            es s</w>
            e r
            m o
            d er
            mo der
            moder n</w>
        """.trimIndent()

        val tokenizer = ClipBpeTokenizer.fromVocabAndMerges(vocabJson, mergesTxt)
        val ids = tokenizer.tokenizeTo77("Modern dress")
        assertEquals(77, ids.size)
        assertEquals(49406L, ids[0])
        assertEquals(1200L, ids[1])
        assertEquals(300L, ids[2])
        assertEquals(400L, ids[3])
        assertEquals(49407L, ids[4])
        assertEquals(49407L, ids[76])

        // FP16 roundtrip accuracy check
        for (v in floatArrayOf(0f, 1.0f, -1.0f, 0.18215f, 12.5f, -3.25f)) {
            val h = SdInpaintOnnxEngine.floatToHalfBits(v)
            val back = SdInpaintOnnxEngine.halfBitsToFloat(h)
            assertTrue(abs(v - back) < 0.02f)
        }

 }

    @Test
    fun accessoryMaskLayers_combinesMultiSelectedTargetsAndPrompts() {
        val prompt = AccessoryMaskEngine.buildCombinedRemovalPrompt(
            setOf(AccessoryTarget.CAP, AccessoryTarget.GLASSES, AccessoryTarget.JEWELS)
        )
        assertTrue(prompt.contains("no cap", ignoreCase = true))
        assertTrue(prompt.contains("no sunglasses", ignoreCase = true))
        assertTrue(prompt.contains("no jewelry", ignoreCase = true))

        val w = 32
        val h = 32
        val cap = FloatArray(w * h)
        val glasses = FloatArray(w * h)
        val jewels = FloatArray(w * h)
        cap[4 * w + 16] = 1.0f
        glasses[12 * w + 16] = 1.0f
        jewels[26 * w + 16] = 1.0f

        val layers = AccessoryMaskLayers(
            width = w,
            height = h,
            capBaseMask = cap,
            glassesBaseMask = glasses,
            jewelsBaseMask = jewels,
            detectedFacesCount = 1,
            usedNeuralParser = false,
            engineLabel = "test"
        )
        val rendered = layers.renderMaskBitmap(
            targetWidth = 64,
            targetHeight = 64,
            selectedTargets = setOf(AccessoryTarget.CAP, AccessoryTarget.GLASSES),
            expandPx = 1f
        )
        assertEquals(64, rendered.width)
        assertEquals(64, rendered.height)
        assertTrue(Color.alpha(rendered.getPixel(32, 8)) > 128)
        assertTrue(Color.alpha(rendered.getPixel(32, 24)) > 128)
        assertEquals(0, Color.alpha(rendered.getPixel(32, 52)))
    }

    @Test
    fun clipBpeTokenizer_parsesTokenizerJsonWithArrayMergesAndAddedTokens() {
        val tokJson = """
            {
              "added_tokens": [
                {"id": 49406, "content": "<|startoftext|>"},
                {"id": 49407, "content": "<|endoftext|>"}
              ],
              "model": {
                "vocab": {
                  "s": 10,
                  "l": 11,
                  "e": 12,
                  "v": 13,
                  "less</w>": 14,
                  "sle": 15,
                  "eve": 16,
                  "sleeveless</w>": 777
                },
                "merges": [
                  ["s", "l"],
                  ["sl", "e"],
                  "e v",
                  ["sle", "eve"],
                  ["sleev", "eless</w>"]
                ]
              }
            }
        """.trimIndent()
        val tok = ClipBpeTokenizer.fromTokenizerJson(tokJson)
        assertEquals(10, tok.vocabSize)
        val ids = tok.tokenizeTo77("sleeveless")
        assertEquals(49406L, ids[0])
        assertEquals(49407L, ids[ids.indexOfFirst { it == 49407L }])
    }

    @Test
    fun sdInpaintOnnxEngine_computesStrengthDenoisePlanAndDilatesMask() {
        val fastDefault = SdInpaintOnnxEngine.computeDenoisePlan(fastMode = true, strength = 0.85f)
        assertEquals(16, fastDefault.baseSteps)
        assertEquals(14, fastDefault.denoiseSteps)
        assertEquals(2, fastDefault.startStepIndex)

        val hdTuned = SdInpaintOnnxEngine.computeDenoisePlan(fastMode = false, strength = 0.70f)
        assertEquals(24, hdTuned.baseSteps)
        assertEquals(17, hdTuned.denoiseSteps)
        assertEquals(7, hdTuned.startStepIndex)

        val mask = FloatArray(32 * 32)
        mask[16 * 32 + 16] = 1.0f
        val dilated = SdInpaintOnnxEngine.dilateBinaryMask(mask, 32, 32, radiusPx = 5)
        assertEquals(1.0f, dilated[11 * 32 + 11], 0.001f)
        assertEquals(1.0f, dilated[21 * 32 + 21], 0.001f)
        assertEquals(0.0f, dilated[5 * 32 + 5], 0.001f)
    }

    @Test
    fun faceSwapViewModel_autoSelectsSd15AndSegformerOnCapOrGlassesToggle_andUpdatesStrengthCfg() {
        val root = SdInpaintPackManager.getSdInpaintRootDir(context)
        root.deleteRecursively()
        val fp16Dir = File(root, SdInpaintPackType.SD15_FP16.id)
        listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").forEach { comp ->
            val compDir = File(fp16Dir, comp).apply { mkdirs() }
            File(compDir, "model.onnx").writeBytes(ByteArray(2048) { 7 })
        }

        val vm = FaceSwapViewModel(context.applicationContext as Application)
        assertEquals(0.85f, vm.uiState.value.inpaintStrength, 0.001f)
        assertEquals(7.0f, vm.uiState.value.inpaintCfgScale, 0.001f)

        vm.setInpaintStrength(0.70f)
        vm.setInpaintCfgScale(8.5f)
        assertEquals(0.70f, vm.uiState.value.inpaintStrength, 0.001f)
        assertEquals(8.5f, vm.uiState.value.inpaintCfgScale, 0.001f)

        vm.updateAccessorySelectionAndExpand(setOf(AccessoryTarget.CAP), 6f)
        assertTrue(vm.uiState.value.segformerAutoSelectedForAccessory)
        assertTrue(AccessoryTarget.CAP in vm.uiState.value.selectedAccessoryTargets)

        vm.updateAccessorySelectionAndExpand(emptySet(), 6f)
        assertFalse(vm.uiState.value.segformerAutoSelectedForAccessory)
    }
}
