package com.example

import com.example.onnx.GhostHeadReplacementEngine
import com.example.onnx.SkinToneSourceMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class GhostCopyPasteRetouchTest {

    private fun argb(r: Int, g: Int, b: Int): Int {
        return (0xFF shl 24) or ((r and 0xFF) shl 16) or ((g and 0xFF) shl 8) or (b and 0xFF)
    }

    @Test
    fun `sanitizeCopyPasteHeadAlphaPixels deletes sideways shoulder wings and horizontal neck bar`() {
        val size = 256
        val total = size * size
        val rawAlpha = FloatArray(total)
        val pixels = IntArray(total) { argb(30, 30, 30) }

        // Simulate MODNet output: full 1.0 across entire lower body/shoulders down to ny = 0.97
        for (y in 0 until size) {
            val ny = y.toFloat() / size
            for (x in 0 until size) {
                val nx = x.toFloat() / size
                val i = y * size + x
                if (ny in 0.15f..0.97f && nx in 0.15f..0.85f) {
                    rawAlpha[i] = 1.0f
                }
                // Warm skin on face & lower neck/shoulders
                if (ny > 0.35f) {
                    pixels[i] = argb(205, 160, 130)
                }
            }
        }

        val cleaned = GhostHeadReplacementEngine.sanitizeCopyPasteHeadAlphaPixels(
            rawAlpha = rawAlpha,
            pixels = pixels,
            cropSize = size,
            centerNx = 0.50f,
            isSourceMask = true
        )

        // Core face center must remain solid 1.0
        val faceIdx = (size * 0.55f).toInt() * size + (size * 0.50f).toInt()
        assertTrue("Face core must stay solid", cleaned[faceIdx] > 0.95f)

        // Left & right sideways shoulder wings (ny=0.84, nx=0.22 and nx=0.78) must be deleted (0.0)
        val leftWingIdx = (size * 0.84f).toInt() * size + (size * 0.22f).toInt()
        val rightWingIdx = (size * 0.84f).toInt() * size + (size * 0.78f).toInt()
        assertEquals(0f, cleaned[leftWingIdx], 0.001f)
        assertEquals(0f, cleaned[rightWingIdx], 0.001f)

        // Bottom horizontal crop edge (ny=0.94) must be completely faded to 0.0 (no horizontal bar)
        val bottomNeckIdx = (size * 0.94f).toInt() * size + (size * 0.50f).toInt()
        assertEquals(0f, cleaned[bottomNeckIdx], 0.001f)
    }

    @Test
    fun `sanitizeCopyPasteHeadAlphaPixels and envelope suppress lateral ghost earrings`() {
        val size = 256
        val total = size * size
        val rawSrcAlpha = FloatArray(total)
        val rawTgtAlpha = FloatArray(total)
        val srcPixels = IntArray(total) { argb(24, 22, 22) } // dark hair background
        val tgtPixels = IntArray(total) { argb(24, 22, 22) }

        // Place bright metallic dangling earrings on left & right lateral ear-drop zones (ny=0.66, nx=0.25 / 0.75)
        for (y in (size * 0.25f).toInt()..(size * 0.78f).toInt()) {
            val ny = y.toFloat() / size
            for (x in (size * 0.22f).toInt()..(size * 0.78f).toInt()) {
                val nx = x.toFloat() / size
                val i = y * size + x
                rawSrcAlpha[i] = 1.0f
                rawTgtAlpha[i] = 1.0f
                if (ny in 0.58f..0.72f && (abs(nx - 0.25f) < 0.03f || abs(nx - 0.75f) < 0.03f)) {
                    srcPixels[i] = argb(225, 225, 228) // silver earring in source
                    tgtPixels[i] = argb(220, 220, 224) // silver earring in target
                }
            }
        }

        val cleanSrc = GhostHeadReplacementEngine.sanitizeCopyPasteHeadAlphaPixels(
            rawAlpha = rawSrcAlpha,
            pixels = srcPixels,
            cropSize = size,
            centerNx = 0.50f,
            isSourceMask = true
        )
        val cleanTgt = GhostHeadReplacementEngine.sanitizeCopyPasteHeadAlphaPixels(
            rawAlpha = rawTgtAlpha,
            pixels = tgtPixels,
            cropSize = size,
            centerNx = 0.50f,
            isSourceMask = false
        )
        val env = GhostHeadReplacementEngine.buildCleanCopyPasteEnvelopeAlpha(
            sourceHeadAlpha = cleanSrc,
            targetHeadAlpha = cleanTgt,
            cropSize = size,
            centerNx = 0.50f
        )

        val leftEarringIdx = (size * 0.65f).toInt() * size + (size * 0.25f).toInt()
        val rightEarringIdx = (size * 0.65f).toInt() * size + (size * 0.75f).toInt()
        assertTrue("Left dangling source earring must be suppressed", cleanSrc[leftEarringIdx] < 0.08f)
        assertTrue("Right dangling source earring must be suppressed", cleanSrc[rightEarringIdx] < 0.08f)
        assertTrue("Envelope must not stamp lateral lower-ear ghost patch", env[rightEarringIdx] < 0.15f)
    }

    @Test
    fun `applyFinalNeckToFaceColorRetouchRoi matches face skin colour to target neck while keeping hair and eyes intact`() {
        val size = 256
        val total = size * size
        val compositePixels = IntArray(total)
        val origTargetPixels = IntArray(total)

        val neckColor = argb(150, 108, 84)      // Darker warm target neck
        val brightFaceColor = argb(228, 185, 155) // Much brighter swapped face before retouch
        val darkHairColor = argb(20, 18, 18)
        val eyeWhiteColor = argb(242, 240, 240)

        val envelope = FloatArray(total) { 1.0f }
        val skinWeight = FloatArray(total)

        for (y in 0 until size) {
            val ny = y.toFloat() / size
            for (x in 0 until size) {
                val nx = x.toFloat() / size
                val i = y * size + x
                origTargetPixels[i] = neckColor
                if (ny < 0.28f) {
                    compositePixels[i] = darkHairColor
                    skinWeight[i] = 0f
                } else if (ny in 0.41f..0.48f && (abs(nx - 0.383f) < 0.03f || abs(nx - 0.617f) < 0.03f)) {
                    compositePixels[i] = eyeWhiteColor
                    skinWeight[i] = 0.9f
                } else if (ny <= 0.85f && abs(nx - 0.50f) < 0.18f) {
                    compositePixels[i] = brightFaceColor
                    skinWeight[i] = 1.0f
                } else {
                    compositePixels[i] = neckColor
                    skinWeight[i] = 0f
                }
            }
        }

        // Identity 2x3 matrix mapping target [0..255] -> crop [0..255]
        val identity2x3 = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        GhostHeadReplacementEngine.applyFinalNeckToFaceColorRetouchRoi(
            compositePixels = compositePixels,
            origTargetRoiPx = origTargetPixels,
            minX = 0,
            maxX = size - 1,
            minY = 0,
            maxY = size - 1,
            targetWidth = size,
            forwardHeadMatrix2x3 = identity2x3,
            headCropSize = size,
            envelopeAlpha = envelope,
            skinAndNeckWeight = skinWeight,
            centerNx = 0.50f,
            skinToneMode = SkinToneSourceMode.TARGET_SCENE
        )

        // Cheek pixel must have shifted close to Target neck luminance/color
        val cheekPx = compositePixels[(size * 0.58f).toInt() * size + (size * 0.42f).toInt()]
        val cheekR = (cheekPx ushr 16) and 0xFF
        val cheekG = (cheekPx ushr 8) and 0xFF
        val cheekB = cheekPx and 0xFF
        assertTrue("Face R should move close to Target neck R=150 (was 228, got $cheekR)", abs(cheekR - 150) < 20)
        assertTrue("Face G should move close to Target neck G=108 (was 185, got $cheekG)", abs(cheekG - 108) < 20)
        assertTrue("Face B should move close to Target neck B=84 (was 155, got $cheekB)", abs(cheekB - 84) < 20)

        // Dark hair must remain untouched
        val hairPx = compositePixels[(size * 0.20f).toInt() * size + (size * 0.50f).toInt()]
        assertEquals(darkHairColor, hairPx)

        // Eye sclera must stay bright white (not darkened to neck color)
        val eyePx = compositePixels[(size * 0.445f).toInt() * size + (size * 0.383f).toInt()]
        val eyeR = (eyePx ushr 16) and 0xFF
        assertTrue("Eye sclera must stay protected bright white (got R=$eyeR)", eyeR > 225)
    }
}
