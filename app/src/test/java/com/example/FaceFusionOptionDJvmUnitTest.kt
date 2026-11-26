package com.example

import com.example.onnx.FaceFusionOptionDEngine
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxProtobufInspector
import com.example.onnx.OptionDEnhancerChoice
import com.example.onnx.OptionDSwapperChoice
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class FaceFusionOptionDJvmUnitTest {

    @Test
    fun pixelBoostImplodeExplode_roundTripsLosslessly() {
        val pixels512 = IntArray(512 * 512) { idx ->
            ((idx * 3) and 0xFF shl 16) or (((idx / 512) and 0xFF) shl 8) or ((idx * 7) and 0xFF) or (0xFF shl 24)
        }
        val tiles256 = FaceFusionOptionDEngine.implodePixelBoost(pixels512, boostSize = 512, modelSize = 256)
        assertEquals(4, tiles256.size)
        assertTrue(tiles256.all { it.size == 256 * 256 })
        val restored512 = FaceFusionOptionDEngine.explodePixelBoost(tiles256, boostSize = 512, modelSize = 256)
        assertArrayEquals(pixels512, restored512)

        val pixels256 = IntArray(256 * 256) { idx ->
            ((idx * 5) and 0xFF shl 16) or (((idx / 256) and 0xFF) shl 8) or ((idx * 11) and 0xFF) or (0xFF shl 24)
        }
        val tiles128 = FaceFusionOptionDEngine.implodePixelBoost(pixels256, boostSize = 256, modelSize = 128)
        assertEquals(4, tiles128.size)
        val restored256 = FaceFusionOptionDEngine.explodePixelBoost(tiles128, boostSize = 256, modelSize = 128)
        assertArrayEquals(pixels256, restored256)
    }

    @Test
    fun matchSlotForFileName_routesOptionDModelsAndIsolatesOccluderSlot() {
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("simswap_512_unofficial.onnx"))
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("reswapper_256.onnx"))
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("blendswap_256.onnx"))
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("ghost_1_256.onnx"))
        assertEquals(ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("hyperswap_1b_256.onnx"))

        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("codeformer.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("gpen_bfr_512.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("restoreformer_plus_plus.onnx"))
        assertEquals(ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("gfpgan-v1.4-w16-v1.onnx"))

        assertEquals(ModelSlot.FACE_OCCLUDER, OnnxProtobufInspector.matchSlotForFileName("face_occluder.onnx"))
        assertEquals(ModelSlot.FACE_OCCLUDER, OnnxProtobufInspector.matchSlotForFileName("xseg_1.onnx"))
        assertEquals(ModelSlot.FACE_OCCLUDER, OnnxProtobufInspector.matchSlotForFileName("dfl_xseg.onnx"))
        assertEquals(ModelSlot.SEGMENTATION, OnnxProtobufInspector.matchSlotForFileName("segformer_B5_ce.onnx"))
        assertEquals(null, OnnxProtobufInspector.matchSlotForFileName("crossface_simswap.onnx"))
    }

    @Test
    fun simswap512_usesUnsigned0To1DefaultAndSelfHealingCatchesSolarizedStats() {
        val spec512 = FaceFusionOptionDEngine.defaultSpecForSwapper(File("simswap_512_unofficial.onnx"))
        assertEquals(FaceFusionOptionDEngine.ChannelLayout.RGB, spec512.channelLayout)
        assertEquals(FaceFusionOptionDEngine.InputNormMode.UNSIGNED_0_TO_1, spec512.normMode)

        val spec256 = FaceFusionOptionDEngine.defaultSpecForSwapper(File("simswap_256.onnx"))
        assertEquals(FaceFusionOptionDEngine.ChannelLayout.RGB, spec256.channelLayout)
        assertEquals(FaceFusionOptionDEngine.InputNormMode.IMAGENET, spec256.normMode)

        val solarizedStats = com.example.onnx.InSwapperEngine.RawOutputStats(
            minVal = 0.00f,
            maxVal = 1.00f,
            negFraction = 0.0f,
            gtOneFraction = 0.0f,
            nanFraction = 0.0f,
            isSignedOutput = false,
            neonRedFraction = 0.005f,
            collapsedBlackFraction = 0.005f,
            lumaCorrelation = 0.16f,
            artifactScore = 0.03f,
            centerFaceDiff = 81.3f
        )
        assertTrue(FaceFusionOptionDEngine.shouldRetryCorruptedOrSolarizedOutput(solarizedStats))

        val cleanStats = com.example.onnx.InSwapperEngine.RawOutputStats(
            minVal = 0.02f,
            maxVal = 0.96f,
            negFraction = 0.0f,
            gtOneFraction = 0.0f,
            nanFraction = 0.0f,
            isSignedOutput = false,
            neonRedFraction = 0.0f,
            collapsedBlackFraction = 0.0f,
            lumaCorrelation = 0.84f,
            artifactScore = 0.0f,
            centerFaceDiff = 16.5f
        )
        assertFalse(FaceFusionOptionDEngine.shouldRetryCorruptedOrSolarizedOutput(cleanStats))
        assertTrue(
            FaceFusionOptionDEngine.computeSpecQualityCost(cleanStats) <
                FaceFusionOptionDEngine.computeSpecQualityCost(solarizedStats)
        )
    }

    @Test
    fun optionDChoiceMatchers_identifyExpectedVariants() {
        assertTrue(FaceFusionOptionDEngine.isFileMatchingSwapperChoice(File("simswap_512_unofficial.onnx"), OptionDSwapperChoice.SIMSWAP_512))
        assertFalse(FaceFusionOptionDEngine.isFileMatchingSwapperChoice(File("hyperswap_1b_256.onnx"), OptionDSwapperChoice.SIMSWAP_512))
        assertTrue(FaceFusionOptionDEngine.isFileMatchingSwapperChoice(File("reswapper_256.onnx"), OptionDSwapperChoice.RESWAPPER_256))
        assertTrue(FaceFusionOptionDEngine.isFileMatchingSwapperChoice(File("blendswap_256.onnx"), OptionDSwapperChoice.RESWAPPER_256))
        assertTrue(FaceFusionOptionDEngine.isFileMatchingSwapperChoice(File("ghost_2_256.onnx"), OptionDSwapperChoice.RESWAPPER_256))
        assertTrue(FaceFusionOptionDEngine.isFileMatchingSwapperChoice(File("inswapper_128.onnx"), OptionDSwapperChoice.INSWAPPER_128))

        assertTrue(FaceFusionOptionDEngine.isFileMatchingEnhancerChoice(File("codeformer.onnx"), OptionDEnhancerChoice.CODEFORMER))
        assertTrue(FaceFusionOptionDEngine.isFileMatchingEnhancerChoice(File("gpen_bfr_512.onnx"), OptionDEnhancerChoice.GPEN_BFR))
        assertTrue(FaceFusionOptionDEngine.isFileMatchingEnhancerChoice(File("restoreformer_plus_plus.onnx"), OptionDEnhancerChoice.RESTOREFORMER))
        assertTrue(FaceFusionOptionDEngine.isFileMatchingEnhancerChoice(File("gfpgan_1.4.onnx"), OptionDEnhancerChoice.GFPGAN))
        assertFalse(FaceFusionOptionDEngine.isFileMatchingEnhancerChoice(File("codeformer.onnx"), OptionDEnhancerChoice.GFPGAN))
    }

    @Test
    fun balanceSourceEmbedding_appliesTargetRepulsionAndPreservesUnitNorm() {
        val cfg = com.example.onnx.OptionDFaceFusionConfig()
        assertTrue(cfg.pixelBoostEnabled)
        assertEquals(0.80f, cfg.codeformerFidelity, 0.001f)
        assertEquals(0.85f, cfg.swapperWeight, 0.001f)

        val src = FloatArray(512) { idx -> if (idx == 0) 1.0f else 0.0f }
        val tgt = FloatArray(512) { idx -> if (idx == 1) 1.0f else 0.0f }

        // w = 0.50 -> coeff = 0.0 -> pure normalized source
        val pure = FaceFusionOptionDEngine.balanceSourceEmbedding(src, tgt, swapperWeight = 0.50f)
        assertEquals(1.0f, pure[0], 1e-4f)
        assertEquals(0.0f, pure[1], 1e-4f)

        // w = 1.00 -> coeff = -0.35 -> repels target component (dim[1] < 0) & stays unit norm
        val repelled = FaceFusionOptionDEngine.balanceSourceEmbedding(src, tgt, swapperWeight = 1.0f)
        assertTrue(repelled[1] < -0.30f)
        val norm = com.example.onnx.ArcFaceRecognizer.computeL2Norm(repelled)
        assertEquals(1.0f, norm, 1e-4f)

        // w = 0.00 -> coeff = +0.35 -> blends positively toward target component
        val softened = FaceFusionOptionDEngine.balanceSourceEmbedding(src, tgt, swapperWeight = 0.0f)
        assertTrue(softened[1] > 0.30f)
    }
}
