package com.example

import com.example.onnx.NextGenBcdHelper
import com.example.onnx.NextGenBcdSlot
import com.example.onnx.NextGenInputNormMode
import com.example.onnx.NextGenSwapModelFamily
import com.example.onnx.OnnxProtobufInspector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NextGenBcdModelsTest {

    @Test
    fun `classifySlotForFileName maps ReSwapper SimSwap CodeFormer and XSeg cleanly`() {
        assertEquals(NextGenBcdSlot.NEXTGEN_SWAPPER, NextGenBcdHelper.classifySlotForFileName("reswapper_256.onnx"))
        assertEquals(NextGenBcdSlot.NEXTGEN_SWAPPER, NextGenBcdHelper.classifySlotForFileName("simswap_512_unofficial.onnx"))
        assertEquals(NextGenBcdSlot.CODEFORMER, NextGenBcdHelper.classifySlotForFileName("codeformer.onnx"))
        assertEquals(NextGenBcdSlot.CODEFORMER, NextGenBcdHelper.classifySlotForFileName("CodeFormer_fp16.onnx"))
        assertEquals(NextGenBcdSlot.XSEG_OCCLUDER, NextGenBcdHelper.classifySlotForFileName("face_occluder.onnx"))
        assertEquals(NextGenBcdSlot.XSEG_OCCLUDER, NextGenBcdHelper.classifySlotForFileName("dfl_xseg.onnx"))
        assertEquals(NextGenBcdSlot.XSEG_OCCLUDER, NextGenBcdHelper.classifySlotForFileName("occluder_256.onnx"))

        // Option A & B models MUST NOT be misclassified into NextGen slots
        assertNull(NextGenBcdHelper.classifySlotForFileName("hyperswap_1b_256.onnx"))
        assertNull(NextGenBcdHelper.classifySlotForFileName("inswapper_128.onnx"))
        assertNull(NextGenBcdHelper.classifySlotForFileName("gfpgan_1.4.onnx"))
        assertNull(NextGenBcdHelper.classifySlotForFileName("segformer_B5_ce.onnx"))
    }

    @Test
    fun `OnnxProtobufInspector keeps Option A and B slots isolated from NextGen BCD models`() {
        assertNull(OnnxProtobufInspector.matchSlotForFileName("reswapper_256.onnx"))
        assertNull(OnnxProtobufInspector.matchSlotForFileName("simswap_512.onnx"))
        assertNull(OnnxProtobufInspector.matchSlotForFileName("codeformer.onnx"))
        assertNull(OnnxProtobufInspector.matchSlotForFileName("face_occluder.onnx"))
        assertNull(OnnxProtobufInspector.matchSlotForFileName("dfl_xseg.onnx"))

        // Original Option A/B models remain mapped identically
        assertEquals(com.example.onnx.ModelSlot.SWAPPER, OnnxProtobufInspector.matchSlotForFileName("hyperswap_1b_256.onnx"))
        assertEquals(com.example.onnx.ModelSlot.ENHANCEMENT, OnnxProtobufInspector.matchSlotForFileName("gfpgan_1.4.onnx"))
        assertEquals(com.example.onnx.ModelSlot.SEGMENTATION, OnnxProtobufInspector.matchSlotForFileName("segformer_B5_ce.onnx"))
    }

    @Test
    fun `swapper family and resolution inference handle reswapper 256 and simswap 512`() {
        assertEquals(NextGenSwapModelFamily.RESWAPPER, NextGenBcdHelper.classifySwapperFamily("reswapper_256.onnx"))
        assertEquals(256, NextGenBcdHelper.inferSwapperNominalResolution("reswapper_256.onnx"))

        assertEquals(NextGenSwapModelFamily.SIMSWAP, NextGenBcdHelper.classifySwapperFamily("simswap_512.onnx"))
        assertEquals(512, NextGenBcdHelper.inferSwapperNominalResolution("simswap_512.onnx"))
        assertEquals(512, NextGenBcdHelper.inferSwapperNominalResolution("simswap.onnx"))
    }

    @Test
    fun `computeCodeFormerFidelityWeight stays clamped in identity-safe range`() {
        val wLowEnhance = NextGenBcdHelper.computeCodeFormerFidelityWeight(0.0f)
        val wHighEnhance = NextGenBcdHelper.computeCodeFormerFidelityWeight(1.0f)
        assertTrue(wLowEnhance in 0.48..0.82)
        assertTrue(wHighEnhance in 0.48..0.82)
        assertEquals(0.82, wLowEnhance, 0.001)
        assertEquals(0.52, wHighEnhance, 0.001)
    }

    @Test
    fun `upscaleXsegMaskTo512 handles logits 0-1 and 2-channel outputs accurately`() {
        val h = 64
        val w = 64
        val logits = FloatArray(h * w) { idx ->
            val y = idx / w
            if (y < 32) 8.0f else -8.0f // top half foreground (face), bottom half occluded
        }
        val gate512 = NextGenBcdHelper.upscaleXsegMaskTo512(logits, maskH = h, maskW = w)
        assertEquals(512 * 512, gate512.size)
        // Top region should remain open near 1.0, bottom region should suppress near 0.0
        assertTrue("Expected open top mask, got ${gate512[40 * 512 + 256]}", gate512[40 * 512 + 256] > 0.90f)
        assertTrue("Expected occluded bottom mask, got ${gate512[460 * 512 + 256]}", gate512[460 * 512 + 256] < 0.10f)

        // 2-channel softmax test
        val twoCh = FloatArray(2 * h * w)
        for (i in 0 until h * w) {
            twoCh[i] = -3.0f
            twoCh[h * w + i] = 3.0f
        }
        val softGate = NextGenBcdHelper.upscaleXsegMaskTo512(twoCh, maskH = h, maskW = w, channels = 2)
        assertEquals(512 * 512, softGate.size)
        assertTrue(softGate[256 * 512 + 256] > 0.95f)
    }

    @Test
    fun `encodeImageFloats supports ImageNet signed m1p1 and unsigned 0p1 layouts`() {
        val hw = 4
        val pixels = IntArray(hw) { (0xFF shl 24) or (255 shl 16) or (0 shl 8) or 128 }
        val signedNchw = NextGenBcdHelper.encodeImageFloats(
            pixels = pixels,
            hw = hw,
            isNhwc = false,
            useBgr = false,
            normMode = NextGenInputNormMode.SIGNED_MINUS1_TO_1
        )
        assertEquals(1.0f, signedNchw[0], 0.01f)
        assertEquals(-1.0f, signedNchw[hw], 0.01f)

        val imgnetNhwc = NextGenBcdHelper.encodeImageFloats(
            pixels = pixels,
            hw = hw,
            isNhwc = true,
            useBgr = false,
            normMode = NextGenInputNormMode.IMAGENET
        )
        // R=1.0 -> (1.0 - 0.485) / 0.229 ~= 2.2489
        assertEquals((1.0f - 0.485f) / 0.229f, imgnetNhwc[0], 0.01f)
        // G=0.0 -> (0.0 - 0.456) / 0.224 ~= -2.0357
        assertEquals((0.0f - 0.456f) / 0.224f, imgnetNhwc[1], 0.01f)
    }
}
