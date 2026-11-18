package com.example

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.AccessoryTarget
import com.example.onnx.ClipBpeTokenizer
import com.example.onnx.InpaintModelOption
import com.example.onnx.InstalledSdInpaintPack
import com.example.onnx.InteractiveInpaintEngine
import com.example.onnx.OnnxProtobufInspector
import com.example.onnx.SdInpaintOnnxEngine
import com.example.onnx.SdInpaintPackManager
import com.example.onnx.SdInpaintPackType
import com.example.ui.FaceSwapViewModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import android.net.Uri
import java.io.File
import java.util.concurrent.CancellationException

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class InpaintFailurePathRobolectricTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    private fun sampleBitmap(): Bitmap =
        Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.rgb(120, 140, 160))
        }

    private fun sampleMask(): Bitmap =
        Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.TRANSPARENT)
            for (y in 20..40) {
                for (x in 20..40) {
                    setPixel(x, y, Color.WHITE)
                }
            }
        }

    @Test
    fun explicitLamaAndSdModes_throwClearErrorsInsteadOfSilentFallback() {
        val src = sampleBitmap()
        val mask = sampleMask()

        // 1. Explicit LAMA_ONNX with missing file must throw (never silently blend CV fallback)
        try {
            InteractiveInpaintEngine.inpaint(
                ortEnv = null,
                sourceBitmap = src,
                maskBitmap = mask,
                prompt = "remove object",
                fastMode = true,
                lamaModelFile = null,
                selectedOption = InpaintModelOption.LAMA_ONNX
            )
            fail("Expected IllegalStateException when LAMA_ONNX file is missing")
        } catch (e: IllegalStateException) {
            assertTrue(e.message?.contains("LaMa", ignoreCase = true) == true)
        }

        // 2. Explicit SD15_FP16 with null pack must throw
        try {
            InteractiveInpaintEngine.inpaint(
                ortEnv = null,
                sourceBitmap = src,
                maskBitmap = mask,
                prompt = "modern suit",
                fastMode = true,
                lamaModelFile = null,
                selectedOption = InpaintModelOption.SD15_FP16,
                primarySdPack = null
            )
            fail("Expected IllegalStateException when SD15_FP16 pack is missing")
        } catch (e: IllegalStateException) {
            assertTrue(e.message?.contains("SD 1.5", ignoreCase = true) == true)
        }

        // 3. Explicit REALVIS_INT8 with dummy pack & null ortEnv must throw
        val fakeRoot = File(context.filesDir, "fake_sd_pack_test").apply { deleteRecursively(); mkdirs() }
        val compDirs = listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").map {
            File(fakeRoot, it).apply { mkdirs() }
        }
        compDirs.forEach { File(it, "model.onnx").writeBytes(ByteArray(2048) { 1 }) }
        val dummyPack = InstalledSdInpaintPack(
            packType = SdInpaintPackType.REALVIS_INT8,
            packDir = fakeRoot,
            textEncoderOnnx = File(compDirs[0], "model.onnx"),
            unetOnnx = File(compDirs[1], "model.onnx"),
            vaeEncoderOnnx = File(compDirs[2], "model.onnx"),
            vaeDecoderOnnx = File(compDirs[3], "model.onnx"),
            tokenizerDir = null,
            totalSizeBytes = 8192L
        )
        try {
            InteractiveInpaintEngine.inpaint(
                ortEnv = null,
                sourceBitmap = src,
                maskBitmap = mask,
                prompt = "blue jacket",
                fastMode = true,
                lamaModelFile = null,
                selectedOption = InpaintModelOption.REALVIS_INT8,
                primarySdPack = dummyPack
            )
            fail("Expected IllegalStateException when ONNX Runtime is null")
        } catch (e: IllegalStateException) {
            assertTrue(e.message?.contains("ONNX Runtime", ignoreCase = true) == true)
        }
    }

    @Test
    fun inpaintRespectsCancellation_andStopsAutoFallbackOnFatalErrors_andChecksLowRamAndTokenizerGuards() {
        val src = sampleBitmap()
        val mask = sampleMask()

        try {
            InteractiveInpaintEngine.inpaint(
                ortEnv = null,
                sourceBitmap = src,
                maskBitmap = mask,
                prompt = "test cancel",
                fastMode = true,
                lamaModelFile = null,
                isCancelled = { true }
            )
            fail("Expected CancellationException when cancelled before start")
        } catch (_: CancellationException) {
            // Expected
        }

        assertTrue(InteractiveInpaintEngine.shouldStopAutoFallbackOnError(CancellationException("cancelled")))
        assertTrue(
            InteractiveInpaintEngine.shouldStopAutoFallbackOnError(
                SdInpaintOnnxEngine.InsufficientNativeRamException("low ram")
            )
        )
        assertTrue(InteractiveInpaintEngine.shouldStopAutoFallbackOnError(OutOfMemoryError("oom")))
        assertTrue(InteractiveInpaintEngine.shouldStopAutoFallbackOnError(IllegalArgumentException("bad input")))
        assertFalse(InteractiveInpaintEngine.shouldStopAutoFallbackOnError(IllegalStateException("corrupt pack")))

        // Low-RAM cap selects 384x384 cleanly when shapes are dynamic
        val lowRamDims = SdInpaintOnnxEngine.computeAspectMatchedTargetResolution(
            origW = 512,
            origH = 768,
            supportsDynamicShapes = true,
            maxDimensionPx = 384
        )
        assertEquals(384, lowRamDims.width)
        assertEquals(384, lowRamDims.height)
        assertEquals(48, lowRamDims.latentW)
        assertEquals(48, lowRamDims.latentH)

        // When no static restrictions exist, selectCompatibleDims retains low-RAM 384x384 candidate
        val retained = SdInpaintOnnxEngine.selectCompatibleDims(lowRamDims, vaeEncInfo = null, vaeDecInfo = null)
        assertEquals(384, retained.width)
        assertEquals(384, retained.height)

        // ClipBpeTokenizer rejects missing vocab directory cleanly via isLoaded()
        val emptyTokDir = File(context.filesDir, "empty_tok_dir").apply { deleteRecursively(); mkdirs() }
        val unloadedTok = ClipBpeTokenizer.loadFromDirectory(emptyTokDir)
        assertFalse(unloadedTok.isLoaded())
    }

    @Test
    fun viewModel_preservesExplicitLamaSelectionOnAccessoryToggle() {
        val root = SdInpaintPackManager.getSdInpaintRootDir(context)
        root.deleteRecursively()
        val fp32Dir = File(root, SdInpaintPackType.SD15_FP32.id)
        listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").forEach { comp ->
            val compDir = File(fp32Dir, comp).apply { mkdirs() }
            File(compDir, "model.onnx").writeBytes(ByteArray(2048) { 7 })
        }
        val lamaFile = OnnxProtobufInspector.resolveModelFile(context, com.example.onnx.ModelSlot.INPAINTING)
        lamaFile.parentFile?.mkdirs()
        lamaFile.writeBytes(ByteArray(2048) { 3 })

        val vm = FaceSwapViewModel(context.applicationContext as Application)
        val deadline = System.currentTimeMillis() + 3000L
        while (vm.uiState.value.isInspectingModels && System.currentTimeMillis() < deadline) {
            Thread.sleep(20L)
        }

        vm.selectInpaintModelOption(InpaintModelOption.LAMA_ONNX)
        assertEquals(InpaintModelOption.LAMA_ONNX, vm.uiState.value.inpaintModelOption)

        vm.updateAccessorySelectionAndExpand(setOf(AccessoryTarget.CAP), 6f)
        // Must preserve user's explicit LAMA_ONNX choice instead of silently overriding to SD15
        assertEquals(InpaintModelOption.LAMA_ONNX, vm.uiState.value.inpaintModelOption)
        assertTrue(vm.uiState.value.segformerAutoSelectedForAccessory)
    }

    @Test
    fun sharedClipTokenizer_andIndividualFileImport_resolvesAcrossAllSdPacksAndValidates() {
        val root = SdInpaintPackManager.getSdInpaintRootDir(context)
        root.deleteRecursively()

        listOf(SdInpaintPackType.SD15_FP16, SdInpaintPackType.REALVIS_INT8).forEach { pt ->
            val packDir = File(root, pt.id)
            listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").forEach { comp ->
                val cDir = File(packDir, comp).apply { mkdirs() }
                File(cDir, "model.onnx").writeBytes(ByteArray(2048) { 9 })
            }
        }

        val beforeStatus = SdInpaintPackManager.inspectClipTokenizerStatus(context)
        assertFalse(beforeStatus.isReady)
        val packsBefore = SdInpaintPackManager.discoverInstalledPacks(context)
        assertEquals(2, packsBefore.size)
        assertTrue(packsBefore.all { !it.hasTokenizer && !it.isReadyWithTokenizer })

        // Prepare minimal valid CLIP vocab.json (>16 entries) and merges.txt, including browser "(1)" suffix
        val tmpPickDir = File(context.cacheDir, "pick_tok_test").apply { deleteRecursively(); mkdirs() }
        val vocabJson = buildString {
            append("{")
            append("\"<|startoftext|>\":49406,\"<|endoftext|>\":49407")
            val words = listOf("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "red</w>", "silk</w>", "saree</w>")
            words.forEachIndexed { idx, w -> append(",\"$w\":$idx") }
            append("}")
        }
        val vocabFile = File(tmpPickDir, "vocab (1).json").apply { writeText(vocabJson) }
        val mergesFile = File(tmpPickDir, "merges.txt").apply {
            writeText("#version: 0.2\ns i\nsi l\nsil k</w>\nr e\nre d</w>\n")
        }
        val unrelatedFile = File(tmpPickDir, "notes.txt").apply { writeText("hello world") }

        // Reject unrelated file clearly
        val badRes = SdInpaintPackManager.importTokenizerFilesFromUris(
            context,
            listOf(Uri.fromFile(unrelatedFile))
        )
        assertTrue(badRes.isFailure)

        // Import vocab (1).json + merges.txt
        val goodRes = SdInpaintPackManager.importTokenizerFilesFromUris(
            context,
            listOf(Uri.fromFile(vocabFile), Uri.fromFile(mergesFile))
        )
        assertTrue(goodRes.isSuccess)
        val importRes = goodRes.getOrThrow()
        assertTrue(importRes.status.isReady)
        assertTrue(importRes.status.hasVocabJson)
        assertTrue(importRes.status.hasMergesTxt)
        assertTrue(importRes.tokenizerVerifiedLoaded)

        // Both FP16 and INT8 packs must now resolve the tokenizer automatically
        val packsAfter = SdInpaintPackManager.discoverInstalledPacks(context)
        assertEquals(2, packsAfter.size)
        assertTrue(packsAfter.all { it.hasTokenizer && it.isReadyWithTokenizer })
        packsAfter.forEach { p ->
            assertTrue(ClipBpeTokenizer.loadFromDirectory(p.tokenizerDir).isLoaded())
        }
    }
}
