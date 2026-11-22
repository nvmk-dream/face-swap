package com.example

import com.example.onnx.ArcFaceRecognizer
import com.example.onnx.FaceBlender
import kotlin.math.abs
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExampleUnitTest {
    @Test
    fun l2Normalize_producesUnitLength512Vector() {
        val raw = FloatArray(512) { idx -> (idx + 1).toFloat() * 0.1f }
        val normed = ArcFaceRecognizer.l2Normalize(raw)
        var sumSq = 0.0
        for (v in normed) sumSq += (v * v).toDouble()
        assertEquals(1.0, sqrt(sumSq), 1e-4)
    }

    @Test
    fun featheredFaceMask128_hasZeroBordersAndUnitCenter() {
        val mask = FaceBlender.createFeatheredFaceMask128()
        assertEquals(128 * 128, mask.size)
        // Center of face (64, 68) must be 1.0f
        assertEquals(1.0f, mask[68 * 128 + 64], 1e-4f)
        // Outer corners/edges must be 0.0f so no box seam appears
        assertEquals(0.0f, mask[0], 1e-5f)
        assertEquals(0.0f, mask[127], 1e-5f)
        assertEquals(0.0f, mask[127 * 128 + 127], 1e-5f)
        assertTrue(abs(mask[40 * 128 + 40]) > 0f)
    }

    @Test
    fun projectWithEmap_transformsAndL2Normalizes512Vector() {
        val normed = FloatArray(512) { if (it == 0) 1.0f else 0.0f }
        val emap = FloatArray(512 * 512) { 0.0f }
        emap[0 * 512 + 7] = 3.0f
        emap[0 * 512 + 9] = 4.0f
        val projected = ArcFaceRecognizer.projectWithEmap(normed, emap)
        assertEquals(512, projected.size)
        assertEquals(0.6f, projected[7], 1e-4f)
        assertEquals(0.8f, projected[9], 1e-4f)
        assertEquals(1.0f, ArcFaceRecognizer.computeL2Norm(projected), 1e-4f)
    }

    @Test
    fun analyzeRawSwapOutput_computesCenterFaceDiffForNearIdentityCheck() {
        val side = 64
        val hw = side * side
        val tgt = IntArray(hw) { (0xFF shl 24) or (120 shl 16) or (100 shl 8) or 80 }
        val identicalFloats = FloatArray(3 * hw)
        for (i in 0 until hw) {
            identicalFloats[i] = 120f / 255f
            identicalFloats[hw + i] = 100f / 255f
            identicalFloats[2 * hw + i] = 80f / 255f
        }
        val identicalStats = com.example.onnx.InSwapperEngine.analyzeRawSwapOutput(
            outFloats = identicalFloats,
            outPixels = tgt.copyOf(),
            targetPixels = tgt,
            isSignedOutput = false
        )
        assertEquals(0.0f, identicalStats.centerFaceDiff, 1e-3f)

        val swappedPx = tgt.copyOf()
        for (y in 16 until 48) {
            for (x in 16 until 48) {
                swappedPx[y * side + x] = (0xFF shl 24) or (180 shl 16) or (150 shl 8) or 120
            }
        }
        val swappedStats = com.example.onnx.InSwapperEngine.analyzeRawSwapOutput(
            outFloats = identicalFloats,
            outPixels = swappedPx,
            targetPixels = tgt,
            isSignedOutput = false
        )
        assertTrue("Swapped center diff must reflect identity shift (${swappedStats.centerFaceDiff})", swappedStats.centerFaceDiff > 40f)
    }

    @Test
    fun reshapePixels_protectsHeadAndBordersWhileWideningHipContour() {
        val w = 120
        val h = 160
        val white = -0x1
        val darkGray = (0xFF shl 24) or (68 shl 16) or (68 shl 8) or 68
        val blue = (0xFF shl 24) or 255
        val src = IntArray(w * h) { white }
        for (y in 20 until h) {
            for (x in 46 until 74) src[y * w + x] = darkGray
        }
        for (y in 10 until 28) {
            for (x in 52 until 68) src[y * w + x] = blue
        }

        val reshaped = com.example.onnx.BodyReshapeAndOutpaintEngine.reshapePixels(
            srcPx = src,
            w = w,
            h = h,
            curvyStrength = 0.85f
        )
        assertEquals(src.size, reshaped.size)
        // Head block stays blue
        assertEquals(blue, reshaped[18 * w + 60])
        // Borders stay white
        val hipY = (h * 0.64f).toInt()
        assertEquals(white, reshaped[hipY * w + 1])
        assertEquals(white, reshaped[hipY * w + (w - 2)])

        var origDark = 0
        var curvyDark = 0
        for (x in 0 until w) {
            if (((src[hipY * w + x] ushr 16) and 0xFF) < 128) origDark++
            if (((reshaped[hipY * w + x] ushr 16) and 0xFF) < 128) curvyDark++
        }
        assertTrue("Curvy hip contour ($curvyDark) must widen original ($origDark)", curvyDark > origDark)
    }

    @Test
    fun extendPixels_expandsCanvasDownwardAndBuildsFeatheredMask() {
        val w = 80
        val h = 100
        val red = (0xFF shl 24) or (255 shl 16)
        val src = IntArray(w * h) { red }
        val (extPx, maskPx) = com.example.onnx.BodyReshapeAndOutpaintEngine.extendPixels(
            srcPx = src,
            origW = w,
            origH = h,
            extendFraction = com.example.onnx.OutpaintExtendPreset.FULL_BODY.fraction
        )
        assertEquals(w * 170, extPx.size)
        assertEquals(w * 170, maskPx.size)
        // Top row preserved & unmasked
        assertEquals(red, extPx[10 * w + 40])
        assertEquals(0, (maskPx[10 * w + 40] ushr 24) and 0xFF)
        // Extended bottom row fully masked white (alpha 255)
        assertEquals(255, (maskPx[140 * w + 40] ushr 24) and 0xFF)
    }

    @Test
    fun buildOutpaintPrompt_appendsFullBodyTermsCleanly() {
        val updated = com.example.onnx.BodyReshapeAndOutpaintEngine.buildOutpaintPrompt(
            "red silk saree",
            com.example.onnx.OutpaintExtendPreset.FULL_BODY
        )
        assertTrue(updated.contains("red silk saree"))
        assertTrue(updated.contains("feet"))
    }

    @Test
    fun classifyPackType_separatesRealisticVisionFp16FromSd15Fp16AndRealVisInt8() {
        assertEquals(
            com.example.onnx.SdInpaintPackType.REALVIS_FP16,
            com.example.onnx.SdInpaintPackManager.classifyPackType("Realistic_Vision_V5.1-inpainting-onnx-fp16")
        )
        assertEquals(
            com.example.onnx.SdInpaintPackType.SD15_FP16,
            com.example.onnx.SdInpaintPackManager.classifyPackType("stable-diffusion-v1-5-inpainting-onnx-fp16")
        )
        assertEquals(
            com.example.onnx.SdInpaintPackType.REALVIS_INT8,
            com.example.onnx.SdInpaintPackManager.classifyPackType("realvis51-inpaint-int8-onnx")
        )
        assertEquals(
            com.example.onnx.SdInpaintPackType.SD15_FP32,
            com.example.onnx.SdInpaintPackManager.classifyPackType("sd15-inpainting-onnx-fp32")
        )
    }

    @Test
    fun dedicatedSdSlots_haveDistinctDirectoriesAndOptions() {
        assertEquals("realvis51_fp16", com.example.onnx.SdInpaintPackType.REALVIS_FP16.id)
        assertEquals("sd15_fp16", com.example.onnx.SdInpaintPackType.SD15_FP16.id)
        assertTrue(
            com.example.onnx.SdInpaintPackType.REALVIS_FP16.id !=
                com.example.onnx.SdInpaintPackType.SD15_FP16.id
        )
        assertTrue(
            com.example.onnx.InpaintModelOption.REALVIS_FP16 !=
                com.example.onnx.InpaintModelOption.SD15_FP16
        )
    }
}
