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

import com.example.onnx.DepthEstimationEngine
import android.graphics.Bitmap.CompressFormat
import java.io.FileOutputStream

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

    @Test
    fun vcopJsonWithBom_largeHeaderTokenizerJson_andPackMergesSync_allImportAndVerify() {
        val root = SdInpaintPackManager.getSdInpaintRootDir(context)
        root.deleteRecursively()

        // Pack has merges.txt ONLY inside its tokenizer/ subfolder (matches screenshot scenario)
        val fp16Dir = File(root, SdInpaintPackType.SD15_FP16.id)
        listOf("text_encoder", "unet", "vae_encoder", "vae_decoder").forEach { comp ->
            File(fp16Dir, comp).apply { mkdirs() }.let {
                File(it, "model.onnx").writeBytes(ByteArray(2048) { 4 })
            }
        }
        val packTokDir = File(fp16Dir, "tokenizer").apply { mkdirs() }
        File(packTokDir, "merges.txt").writeText("#version: 0.2\nr e\nre d</w>\ns i\nsi l\nsil k</w>\n")

        val initialStatus = SdInpaintPackManager.inspectClipTokenizerStatus(context)
        assertFalse(initialStatus.isReady)
        assertTrue(initialStatus.hasMergesTxt)
        assertFalse(initialStatus.hasVocabJson)

        // Scenario A: User uploads "Vcop .json" with UTF-8 BOM while merges.txt is in pack dir
        val tmpDir = File(context.cacheDir, "vcop_tok_test").apply { deleteRecursively(); mkdirs() }
        val vocabBody = buildString {
            append("\uFEFF{")
            val words = (0 until 20).map { "w$it</w>" } + listOf("red</w>", "saree</w>")
            words.forEachIndexed { idx, w ->
                if (idx > 0) append(",")
                append("\"$w\":$idx")
            }
            append(",\"<|startoftext|>\":49406,\"<|endoftext|>\":49407}")
        }
        val vcopFile = File(tmpDir, "Vcop .json").apply { writeText(vocabBody) }
        val vcopImport = SdInpaintPackManager.importTokenizerFilesFromUris(
            context,
            listOf(Uri.fromFile(vcopFile)),
            expectedCanonicalHint = "vocab.json"
        )
        assertTrue("Expected Vcop .json import to succeed: ${vcopImport.exceptionOrNull()}", vcopImport.isSuccess)
        val vcopRes = vcopImport.getOrThrow()
        assertTrue(vcopRes.status.isReady)
        assertTrue(vcopRes.status.hasVocabJson)
        assertTrue(vcopRes.status.hasMergesTxt)
        assertTrue("Expected cross-dir sync verification to succeed", vcopRes.tokenizerVerifiedLoaded)

        // Scenario B: Large HuggingFace tokenizer.json (>16 KB before "model" key appears) with generic name
        root.deleteRecursively()
        val bigTokenizerJson = buildString {
            append("{\"version\":\"1.0\",\"added_tokens\":[")
            // Generate > 18 KB of added_tokens before "model"
            for (i in 0 until 280) {
                if (i > 0) append(",")
                append("{\"id\":${50000 + i},\"content\":\"<extra_token_padding_$i>\",\"special\":false}")
            }
            append("],\"model\":{\"type\":\"BPE\",\"vocab\":{")
            val vocabEntries = (0 until 20).map { "\"t$it</w>\":$it" } +
                listOf("\"<|startoftext|>\":49406", "\"<|endoftext|>\":49407")
            append(vocabEntries.joinToString(","))
            append("},\"merges\":[\"r e\",\"re d</w>\",\"s a\",\"sa ree</w>\"]}}")
        }
        assertTrue(bigTokenizerJson.indexOf("\"model\"") > 12_000)
        val genericTokFile = File(tmpDir, "hf_clip_fast_config.json").apply { writeText(bigTokenizerJson) }
        val bigImport = SdInpaintPackManager.importTokenizerFilesFromUris(
            context,
            listOf(Uri.fromFile(genericTokFile))
        )
        assertTrue("Expected >12KB-offset tokenizer.json import to succeed: ${bigImport.exceptionOrNull()}", bigImport.isSuccess)
        assertTrue(bigImport.getOrThrow().status.hasTokenizerJson)
        assertTrue(bigImport.getOrThrow().tokenizerVerifiedLoaded)
    }

    @Test
    fun inpaintEngine_timestepShape_promptNormalize_zeroBleedFeather_andViewModelSelfHeal() {
        // 1. Static [1] timestep must stay [1] even when CFG batch=2
        val staticOne = SdInpaintOnnxEngine.resolveTimestepShape(longArrayOf(1L), batchSize = 2)
        assertEquals(1, staticOne.size)
        assertEquals(1L, staticOne[0])

        val dynamicBatch = SdInpaintOnnxEngine.resolveTimestepShape(longArrayOf(-1L), batchSize = 2)
        assertEquals(1, dynamicBatch.size)
        assertEquals(2L, dynamicBatch[0])

        // 2. Prompt normalization maps sharee -> saree while keeping saree intact
        assertEquals(
            "red saree, realistic photography",
            SdInpaintOnnxEngine.normalizePromptForClip("red sharee, realistic photography")
        )
        assertEquals(
            "red silk saree",
            SdInpaintOnnxEngine.normalizePromptForClip("red silk saree")
        )

        // 3. Zero outward bleed in featherMask across both engines
        val w = 20
        val h = 20
        val mask = FloatArray(w * h)
        for (y in 6..13) {
            for (x in 6..13) {
                mask[y * w + x] = 1.0f
            }
        }
        val sdFeathered = SdInpaintOnnxEngine.featherMask(mask, w, h, radius = 3)
        val interactiveFeathered = InteractiveInpaintEngine.featherMask(mask, w, h, radius = 3)
        // Pixels just outside the mask (e.g. x=5, y=10) MUST be strictly 0.0f
        assertEquals(0f, sdFeathered[10 * w + 5], 1e-6f)
        assertEquals(0f, interactiveFeathered[10 * w + 5], 1e-6f)
        // Deep interior pixel MUST remain 1.0f
        assertEquals(1f, sdFeathered[10 * w + 10], 1e-6f)
        assertEquals(1f, interactiveFeathered[10 * w + 10], 1e-6f)

        // 4. Blank mask rejected + Create pre-flight errors cleanly
        val blankMask = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.TRANSPARENT)
        }
        assertFalse(InteractiveInpaintEngine.hasPaintedMaskPixels(blankMask))

        val vm = FaceSwapViewModel(context.applicationContext as Application)
        val deadline = System.currentTimeMillis() + 3000L
        while (vm.uiState.value.isInspectingModels && System.currentTimeMillis() < deadline) {
            Thread.sleep(20L)
        }
        // Without source image, runInpaint sets error cleanly instead of silent no-op
        vm.runInpaint()
        assertTrue(vm.uiState.value.errorBannerMessage?.contains("Source Image", ignoreCase = true) == true)
        assertFalse(vm.uiState.value.isInpainting)
    }

    @Test
    fun runtimeLmkdBreadcrumb_and3To4GbRamProfile_protectSdAndStage5() {
        OnnxProtobufInspector.clearRuntimeLowRamLatch(context)
        assertFalse(OnnxProtobufInspector.isRuntimeLowRamLatched(context))
        assertFalse(OnnxProtobufInspector.isRuntime384UnsafeLatched(context))

        // 1. Normal completion or CancellationException must NOT latch Low-RAM guard
        val ok = OnnxProtobufInspector.runWithRuntimeCrashBreadcrumb(context, "sd_vae_decoder", 512) { 42 }
        assertEquals(42, ok)
        assertFalse(OnnxProtobufInspector.isRuntimeLowRamLatched(context))

        try {
            OnnxProtobufInspector.runWithRuntimeCrashBreadcrumb(context, "sd_unet", 512) {
                throw CancellationException("user cancel")
            }
            fail("Expected CancellationException")
        } catch (_: CancellationException) {
            // Expected
        }
        assertFalse(OnnxProtobufInspector.isRuntimeLowRamLatched(context))

        // 2. Simulate OS LMKD native kill mid-UNet session.run() (pending breadcrumb never cleared)
        OnnxProtobufInspector.markPendingRuntimeStage(context, "sd_unet", resolutionPx = 512)
        OnnxProtobufInspector.promoteStuckCrashBreadcrumbIfAny(context)
        assertTrue(OnnxProtobufInspector.isRuntimeLowRamLatched(context))
        assertFalse(OnnxProtobufInspector.isRuntime384UnsafeLatched(context))
        assertEquals("sd_unet", OnnxProtobufInspector.getLastRuntimeCrashStage(context))

        // Even with plenty of RAM reported, latched crash forces 384px & 2 threads + disables Stage 5 SD bridge
        val healthyRam = SdInpaintOnnxEngine.DeviceRamStatus(availMb = 2800L, totalMb = 8192L, isLowMemory = false)
        val latchedProfile = SdInpaintOnnxEngine.resolveExecutionProfile(
            ramStatus = healthyRam,
            fastMode = false,
            maxPreferredResolutionPx = 768,
            isRuntimeCrashLatched = true
        )
        assertTrue(latchedProfile.isLowRamConstrained)
        assertEquals(384, latchedProfile.maxDimensionPx)
        assertEquals(2, latchedProfile.intraOpThreads)
        assertEquals(850L, latchedProfile.minRamBeforeVaeDecoderMbFor512)

        val (stage5UseSdLatched, stage5ResLatched) = com.example.onnx.CleanEraseHeadReplacementEngine.resolveStage5BridgePlan(
            lowMemoryMode = false,
            ramStatus = healthyRam,
            isRuntimeCrashLatched = true
        )
        assertFalse(stage5UseSdLatched)
        assertEquals(384, stage5ResLatched)

        // 3. ViewModel boots into Safe Low-RAM mode automatically when runtime latch is on disk, and reset clears it
        val vmLatched = FaceSwapViewModel(context.applicationContext as Application)
        val deadline = System.currentTimeMillis() + 3000L
        while (vmLatched.uiState.value.isInspectingModels && System.currentTimeMillis() < deadline) {
            Thread.sleep(20L)
        }
        assertTrue(vmLatched.uiState.value.runtimeLowRamLatched)
        assertTrue(vmLatched.uiState.value.lowMemoryMode)
        assertEquals(com.example.ui.ProcessingQualityLevel.FAST, vmLatched.uiState.value.qualityLevel)

        vmLatched.resetRuntimeCrashGuard()
        assertFalse(vmLatched.uiState.value.runtimeLowRamLatched)
        assertFalse(OnnxProtobufInspector.isRuntimeLowRamLatched(context))

        // 4. Simulate LMKD kill even at 384px -> flags unsafe_even_384 to prevent crash loops on 384-impossible phones
        OnnxProtobufInspector.markPendingRuntimeStage(context, "sd_vae_decoder", resolutionPx = 384)
        OnnxProtobufInspector.promoteStuckCrashBreadcrumbIfAny(context)
        assertTrue(OnnxProtobufInspector.isRuntimeLowRamLatched(context))
        assertTrue(OnnxProtobufInspector.isRuntime384UnsafeLatched(context))
        OnnxProtobufInspector.clearRuntimeLowRamLatch(context)

        // 5. 3GB-4GB RAM phone with 1300-1600 MB free RAM MUST pick 384px & 2 threads even without prior crash
        val fourGbPhoneRam = SdInpaintOnnxEngine.DeviceRamStatus(availMb = 1480L, totalMb = 3900L, isLowMemory = false)
        val fourGbProfile = SdInpaintOnnxEngine.resolveExecutionProfile(
            ramStatus = fourGbPhoneRam,
            fastMode = false,
            maxPreferredResolutionPx = 768,
            isRuntimeCrashLatched = false
        )
        assertTrue(fourGbProfile.isLowRamConstrained)
        assertEquals(384, fourGbProfile.maxDimensionPx)
        assertEquals(2, fourGbProfile.intraOpThreads)

        // Stage 5 on 1480MB free RAM skips risky SD Diffusion bridge in favor of Fast 3D Seam Bridge
        val (stage5On4GbTight, _) = com.example.onnx.CleanEraseHeadReplacementEngine.resolveStage5BridgePlan(
            lowMemoryMode = false,
            ramStatus = fourGbPhoneRam,
            isRuntimeCrashLatched = false
        )
        assertFalse(stage5On4GbTight)

        // 6. High-RAM phone with 3200MB free RAM keeps full 768px in HQ and 512px Stage 5 SD bridge
        val flagShipRam = SdInpaintOnnxEngine.DeviceRamStatus(availMb = 3200L, totalMb = 7800L, isLowMemory = false)
        val hqProfile = SdInpaintOnnxEngine.resolveExecutionProfile(
            ramStatus = flagShipRam,
            fastMode = false,
            maxPreferredResolutionPx = 768,
            isRuntimeCrashLatched = false
        )
        assertFalse(hqProfile.isLowRamConstrained)
        assertEquals(768, hqProfile.maxDimensionPx)
        val (stage5OnFlagship, stage5FlagshipDim) = com.example.onnx.CleanEraseHeadReplacementEngine.resolveStage5BridgePlan(
            lowMemoryMode = false,
            ramStatus = flagShipRam,
            isRuntimeCrashLatched = false
        )
        assertTrue(stage5OnFlagship)
        assertEquals(512, stage5FlagshipDim)
    }

    @Test
    fun depthEstimation_respectsCancellation_andViewModelCancelAllowsCleanRecreate() {
        val src = sampleBitmap()
        val mask = sampleMask()

        // 1. DepthEstimationEngine must throw CancellationException immediately when cancelled before start
        try {
            DepthEstimationEngine.estimateDepth(
                context = context,
                ortEnv = null,
                sourceBitmap = src,
                isCancelled = { true }
            )
            fail("Expected CancellationException when depth estimation is pre-cancelled")
        } catch (_: CancellationException) {
            // Expected
        }

        // 2. Clean non-cancelled null-env call returns null gracefully
        val nullEnvRes = DepthEstimationEngine.estimateDepth(
            context = context,
            ortEnv = null,
            sourceBitmap = src,
            isCancelled = { false }
        )
        assertEquals(null, nullEnvRes)

        // 3. ViewModel cancel -> immediate recreate flow uses isolated per-job cancel tokens
        SdInpaintPackManager.getSdInpaintRootDir(context).deleteRecursively()
        OnnxProtobufInspector.resolveModelFile(context, com.example.onnx.ModelSlot.INPAINTING).delete()
        val dummyDepthFile = OnnxProtobufInspector.resolveModelFile(context, com.example.onnx.ModelSlot.DEPTH)
        dummyDepthFile.parentFile?.mkdirs()
        dummyDepthFile.writeBytes(ByteArray(2048) { 5 })

        val srcPng = File(context.cacheDir, "vm_cancel_recreate_src.png")
        FileOutputStream(srcPng).use { fos ->
            src.compress(CompressFormat.PNG, 100, fos)
        }

        val mainLooperShadow = org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper())
        val vm = FaceSwapViewModel(context.applicationContext as Application)
        val initDeadline = System.currentTimeMillis() + 3000L
        while (vm.uiState.value.isInspectingModels && System.currentTimeMillis() < initDeadline) {
            mainLooperShadow.idle()
            Thread.sleep(20L)
        }
        mainLooperShadow.idle()

        vm.onInpaintSourceImageSelected(Uri.fromFile(srcPng))
        val loadDeadline = System.currentTimeMillis() + 3000L
        while (vm.uiState.value.inpaintSourceBitmap == null &&
            vm.uiState.value.errorBannerMessage == null &&
            System.currentTimeMillis() < loadDeadline
        ) {
            mainLooperShadow.idle()
            Thread.sleep(20L)
        }
        mainLooperShadow.idle()
        assertTrue(
            "Expected inpaintSourceBitmap to load, err=${vm.uiState.value.errorBannerMessage}",
            vm.uiState.value.inpaintSourceBitmap != null
        )

        vm.setInpaintMaskBitmap(mask)
        vm.selectInpaintModelOption(InpaintModelOption.CV_FALLBACK)
        vm.setInpaintDepthGuardEnabled(true)

        // Start -> immediately cancel -> immediately start a fresh run
        vm.runInpaint()
        vm.cancelActiveInpaint()
        assertFalse(vm.uiState.value.isInpainting)

        vm.runInpaint()
        val finishDeadline = System.currentTimeMillis() + 6000L
        while ((vm.uiState.value.isInpainting ||
                (vm.uiState.value.inpaintResultBitmap == null && vm.uiState.value.errorBannerMessage == null)) &&
            System.currentTimeMillis() < finishDeadline
        ) {
            mainLooperShadow.idle()
            Thread.sleep(25L)
        }
        mainLooperShadow.idle()

        assertFalse(vm.uiState.value.isInpainting)
        assertTrue(
            "Expected recreated inpaint run to finish with result, got err=${vm.uiState.value.errorBannerMessage} status=${vm.uiState.value.statusBannerMessage}",
            vm.uiState.value.inpaintResultBitmap != null
        )

        dummyDepthFile.delete()
        srcPng.delete()
    }
}
