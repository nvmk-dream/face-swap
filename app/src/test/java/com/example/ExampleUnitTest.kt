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
}
