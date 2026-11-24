package com.example

import com.example.onnx.CleanEraseHeadReplacementEngine
import com.example.onnx.HeadSegmentationAndInpainting
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-JVM repeated verification suite exercising every Option B+C neck-to-face seam failure path:
 *  - Stage 3 erase vs Stage 4 fade zero-gap cross-dissolve across all vertical/horizontal positions
 *  - Shadow-safe skin chrominance detection under chin shadows vs cloth/wall backgrounds
 *  - Multi-scenario Chin-to-Neck seam repair (dark troughs, cool wall gaps, two-tone rings,
 *    deep skin, rolled +/-30 deg heads, micro-texture preservation, collar/necklace protection).
 */
class NeckFaceSeamRepeatedTest {

    private fun rgb(r: Int, g: Int, b: Int): Int =
        (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)

    private fun lum(c: Int): Float =
        0.299f * ((c ushr 16) and 0xFF) + 0.587f * ((c ushr 8) and 0xFF) + 0.114f * (c and 0xFF)

    @Test
    fun stage3AndStage4NeckCrossDissolve_neverExposesEmptyBackgroundGap() {
        val eyeDist = 60f
        val skinCases = listOf(
            0.88f, // clean lit neck
            0.42f, // moderate chin shadow
            0.22f, // deep chin shadow
            0.05f  // segmentation missed Target neck -> Stage 3 erased it!
        )

        for (tgtSkinW in skinCases) {
            for (dyStep in 0..25) {
                val neckDy = eyeDist * (dyStep * 0.02f) // 0.00 .. 0.50 * eyeDist
                for (dxStep in 0..8) {
                    val neckDx = eyeDist * (dxStep * 0.05f) // 0.00 .. 0.40 * eyeDist
                    val (stage3Erase, stage4Drop) = CleanEraseHeadReplacementEngine.computeZeroGapNeckOverlapPair(
                        neckDy = neckDy,
                        neckDxAbs = neckDx,
                        eyeDist = eyeDist,
                        tgtSkinWeight = tgtSkinW,
                        tgtLum = 155f,
                        tgtWarmRGeBMinus4 = true,
                        clothProtect = 0f,
                        decorProtect = 0f,
                        srcSkinWeight = 0.85f,
                        dilatedOrEarEraseBase = 0.92f
                    )
                    val targetPreserved = 1f - stage3Erase
                    val sourceKeptOverErased = (1f - stage4Drop) * stage3Erase
                    val totalCoverage = targetPreserved + sourceKeptOverErased
                    assertTrue(
                        "Zero-gap coverage dropped to $totalCoverage (erase=$stage3Erase, drop=$stage4Drop) at dy=${neckDy / eyeDist}, tgtSkin=$tgtSkinW",
                        totalCoverage >= 0.88f
                    )
                }
            }
        }
    }

    @Test
    fun stage3AndStage4NeckOverlap_protectsCollarAndDupattaNearChin() {
        val eyeDist = 60f
        val (stage3Erase, stage4Drop) = CleanEraseHeadReplacementEngine.computeZeroGapNeckOverlapPair(
            neckDy = eyeDist * 0.25f,
            neckDxAbs = 0f,
            eyeDist = eyeDist,
            tgtSkinWeight = 0.08f,
            tgtLum = 140f,
            tgtWarmRGeBMinus4 = false,
            clothProtect = 0.94f,
            decorProtect = 0f,
            srcSkinWeight = 0.12f,
            dilatedOrEarEraseBase = 0.90f
        )
        assertTrue("Expected Target collar untouched (erase=$stage3Erase)", stage3Erase < 0.08f)
        assertTrue("Expected Source non-skin blouse faded out over collar (drop=$stage4Drop)", stage4Drop > 0.65f)
    }

    @Test
    fun shadowAwareSkinLikelihood_keepsShadowedNeckAndRejectsCoolCollarAndSky() {
        val baseFaces = listOf(
            floatArrayOf(235f, 192f, 165f), // fair/medium skin
            floatArrayOf(195f, 150f, 120f), // wheatish skin
            floatArrayOf(132f, 94f, 70f)    // deeper brown skin
        )
        val shadowFactors = floatArrayOf(0.85f, 0.68f, 0.54f)

        for (face in baseFaces) {
            for (sh in shadowFactors) {
                val score = HeadSegmentationAndInpainting.computeShadowAwareSkinLikelihood(
                    r = face[0] * sh,
                    g = face[1] * sh,
                    b = face[2] * sh,
                    skinR = face[0],
                    skinG = face[1],
                    skinB = face[2]
                )
                assertTrue(
                    "Shadowed neck (factor=$sh on (${face[0]},${face[1]},${face[2]})) dropped to $score",
                    score >= 0.62f
                )
            }
        }

        val skyScore = HeadSegmentationAndInpainting.computeShadowAwareSkinLikelihood(
            r = 120f, g = 185f, b = 235f,
            skinR = 220f, skinG = 175f, skinB = 148f
        )
        val emeraldCollarScore = HeadSegmentationAndInpainting.computeShadowAwareSkinLikelihood(
            r = 22f, g = 155f, b = 70f,
            skinR = 220f, skinG = 175f, skinB = 148f
        )
        assertTrue("Cool sky leaked skin score $skyScore", skyScore < 0.15f)
        assertTrue("Emerald collar leaked skin score $emeraldCollarScore", emeraldCollarScore < 0.15f)
    }

    @Test
    fun repairNeckChinSeamPixels_passesRepeatedMultiScenarioStressSuite() {
        val w = 240
        val h = 240
        val eyeDist = 48f
        val eyeMidX = 120f
        val eyeMidY = 92f
        val chinY = eyeMidY + eyeDist * 1.65f // ~171.2

        // Repeated Scenario 1: Dark black U-trough + background grey wall gap under chin
        run {
            val orig = IntArray(w * h) { rgb(200, 215, 235) }
            val out = IntArray(w * h) { rgb(200, 215, 235) }
            for (y in 60..215) {
                for (x in 82..158) {
                    orig[y * w + x] = rgb(215, 170, 142)
                    out[y * w + x] = rgb(215, 170, 142)
                }
            }
            // Inject both a dark U-trough AND a cool blue-gray wall gap right below chin
            for (y in 174..186) {
                for (x in 98..142) {
                    out[y * w + x] = if (y < 180) rgb(52, 42, 38) else rgb(155, 172, 198)
                }
            }
            val healed = CleanEraseHeadReplacementEngine.repairNeckChinSeamPixels(
                outPx = out, origPx = orig, w = w, h = h,
                eyeMidX = eyeMidX, eyeMidY = eyeMidY, eyeDist = eyeDist,
                chinEstX = eyeMidX, chinEstY = chinY
            )
            assertTrue("Expected significant neck seam repair, got $healed", healed > 250)
            val centerTrough = out[177 * w + 120]
            val centerWallGap = out[183 * w + 120]
            assertTrue("Dark trough not healed: r=${(centerTrough ushr 16) and 0xFF}", ((centerTrough ushr 16) and 0xFF) > 165)
            assertTrue("Cool wall gap not healed: r-b=${((centerWallGap ushr 16) and 0xFF) - (centerWallGap and 0xFF)}", ((centerWallGap ushr 16) and 0xFF) - (centerWallGap and 0xFF) > 35)
        }

        // Repeated Scenario 2: Sharp Two-Tone Chin Step (SOURCE_IDENTITY warm bright vs TARGET_SCENE tan neck) + pore texture check
        run {
            val orig = IntArray(w * h) { rgb(210, 220, 230) }
            val out = IntArray(w * h) { rgb(210, 220, 230) }
            for (y in 60..220) {
                for (x in 82..158) {
                    // Fine +/-5 skin micro-pores texture
                    val pore = if (((x + y) and 1) == 0) 4 else -4
                    orig[y * w + x] = rgb(196 + pore, 152 + pore, 126 + pore)
                    out[y * w + x] = if (y <= 176) {
                        rgb(236 + pore, 194 + pore, 166 + pore)
                    } else {
                        rgb(196 + pore, 152 + pore, 126 + pore)
                    }
                }
            }
            // Before repair: row 176 vs 177 jump is ~40 red units
            val rawJump = abs(((out[176 * w + 120] ushr 16) and 0xFF) - ((out[177 * w + 120] ushr 16) and 0xFF))
            assertTrue(rawJump >= 32)

            CleanEraseHeadReplacementEngine.repairNeckChinSeamPixels(
                outPx = out, origPx = orig, w = w, h = h,
                eyeMidX = eyeMidX, eyeMidY = eyeMidY, eyeDist = eyeDist,
                chinEstX = eyeMidX, chinEstY = chinY
            )
            // After repair: measure smoothed 5px moving average row-to-row red step across seam (filtering +/-4 pore grain)
            var maxSmoothedRowStep = 0f
            var prevSmoothR = 0f
            for (y in 168..194) {
                var sumR = 0f
                for (dx in -4..4) {
                    sumR += ((out[y * w + 120 + dx] ushr 16) and 0xFF)
                }
                val smoothR = sumR / 9f
                if (y > 168) {
                    maxSmoothedRowStep = max(maxSmoothedRowStep, abs(smoothR - prevSmoothR))
                }
                prevSmoothR = smoothR
            }
            assertTrue("Expected smooth two-tone transition across chin-neck, maxSmoothedRowStep=$maxSmoothedRowStep", maxSmoothedRowStep < 9.5f)

            // Verify skin micro-pore grain survived cleanly (neighbor delta > 2)
            val pA = lum(out[180 * w + 120])
            val pB = lum(out[180 * w + 121])
            assertTrue("Expected skin micro-pores preserved (diff=${abs(pA - pB)})", abs(pA - pB) >= 2.0f)
        }

        // Repeated Scenario 3: Rolled +32 deg head axis alignment + Deeper skin tone + Collar/Olive-Dupatta protection
        run {
            val rollRad = Math.toRadians(32.0)
            val ux = cos(rollRad).toFloat()
            val uy = sin(rollRad).toFloat()
            val vx = -uy
            val vy = ux
            val cx = 120f
            val cy = 96f

            val orig = IntArray(w * h) { rgb(180, 200, 220) }
            val out = IntArray(w * h) { rgb(180, 200, 220) }
            val chinLocalV = eyeDist * 1.65f
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val dx = x - cx
                    val dy = y - cy
                    val lu = dx * ux + dy * uy
                    val lv = dx * vx + dy * vy
                    if (abs(lu) <= 40f && lv in -10f..(eyeDist * 2.4f)) {
                        val c = if (lv > eyeDist * 2.02f) {
                            rgb(165, 148, 52) // Olive-gold dupatta collar along neck base
                        } else {
                            rgb(138, 98, 74) // Deep brown healthy neck/face skin
                        }
                        orig[y * w + x] = c
                        out[y * w + x] = c
                        if (lv in (chinLocalV + 2f)..(chinLocalV + 11f) && abs(lu) <= 22f) {
                            out[y * w + x] = rgb(38, 30, 28) // Dark oblique U-trough along rolled chin!
                        }
                    }
                }
            }

            CleanEraseHeadReplacementEngine.repairNeckChinSeamPixels(
                outPx = out, origPx = orig, w = w, h = h,
                eyeMidX = cx, eyeMidY = cy, eyeDist = eyeDist,
                chinEstX = cx + vx * chinLocalV,
                chinEstY = cy + vy * chinLocalV,
                ux = ux, uy = uy, vx = vx, vy = vy
            )

            // Check oblique trough midpoint along rolled neck axis
            val probeV = chinLocalV + 6.5f
            val px = (cx + vx * probeV).toInt()
            val py = (cy + vy * probeV).toInt()
            val healedOblique = out[py * w + px]
            assertTrue(
                "Expected oblique rolled chin trough healed, got RGB=(${(healedOblique ushr 16) and 0xFF},${(healedOblique ushr 8) and 0xFF},${healedOblique and 0xFF})",
                ((healedOblique ushr 16) and 0xFF) > 102
            )

            // Check olive-gold dupatta collar directly below remains exact!
            val clothV = eyeDist * 2.18f
            val cpx = (cx + vx * clothV).toInt()
            val cpy = (cy + vy * clothV).toInt()
            assertEquals(rgb(165, 148, 52), out[cpy * w + cpx])
        }
    }

    @Test
    fun repairNeckChinSeamPixels_neverSmudgesEarringsOrShoulderHairCurlsUnderJaw() {
        val w = 260
        val h = 260
        val eyeDist = 48f
        val eyeMidX = 130f
        val eyeMidY = 88f
        val chinY = eyeMidY + eyeDist * 1.65f // ~167.2

        // Scenario 4 (Result #79): Draped shoulder curls (lum 48..62, warm brown-black) + gold earrings
        // flanking a central neck with a dark U-trough & slight two-tone chin step
        val orig = IntArray(w * h) { rgb(190, 200, 210) }
        val out = IntArray(w * h) { rgb(190, 200, 210) }

        // Central healthy neck skin (|dx| <= 0.72 * eyeDist initially in background around earrings/curls)
        for (y in 60..235) {
            for (x in 90..170) {
                val skin = if (y <= 170) rgb(224, 178, 146) else rgb(202, 158, 128)
                orig[y * w + x] = rgb(202, 158, 128)
                out[y * w + x] = skin
            }
        }
        // Inject center U-trough in neck (|dx| <= 0.34 * eyeDist, y = 171..177)
        for (y in 171..177) {
            for (x in 114..146) {
                out[y * w + x] = rgb(44, 34, 30)
            }
        }
        // Inject wavy brown-black shoulder curls (|dx| in 0.44..0.82 * eyeDist, y = 165..220)
        // Crucially: on outPx they overlap neck skin (where origPx had skin at |dx|=0.44..0.58)
        val curlColor = rgb(62, 48, 42) // lum ~ 51.5, r-b = 20 (caught by new lum<86 & r-b<26 + belowIsDarkHair!)
        for (y in 165..220) {
            for (x in 92..108) {
                out[y * w + x] = curlColor
            }
            for (x in 152..168) {
                out[y * w + x] = curlColor
            }
        }
        // Gold dangling earrings at |dx| ~ 0.62..0.75 * eyeDist, y = 158..172
        val goldEarring = rgb(218, 172, 72)
        for (y in 158..172) {
            for (x in 94..101) {
                orig[y * w + x] = goldEarring
                out[y * w + x] = goldEarring
            }
            for (x in 159..166) {
                orig[y * w + x] = goldEarring
                out[y * w + x] = goldEarring
            }
        }

        val healed = CleanEraseHeadReplacementEngine.repairNeckChinSeamPixels(
            outPx = out, origPx = orig, w = w, h = h,
            eyeMidX = eyeMidX, eyeMidY = eyeMidY, eyeDist = eyeDist,
            chinEstX = eyeMidX, chinEstY = chinY
        )
        assertTrue("Center U-trough should heal cleanly, healed=$healed", healed > 80)

        // 1. Center trough must be healed to healthy warm skin
        val centerTrough = out[174 * w + 130]
        assertTrue(
            "Center trough wasn't healed: R=${(centerTrough ushr 16) and 0xFF}",
            ((centerTrough ushr 16) and 0xFF) > 165
        )

        // 2. ZERO smudging on left or right dark shoulder curls (both inner 0.46*eyeDist and outer 0.68*eyeDist)!
        for (y in 166..205) {
            for (x in 92..108) {
                if (x !in 94..101 || y !in 158..172) {
                    assertEquals("Left shoulder curl smudged at ($x,$y)", curlColor, out[y * w + x])
                }
            }
            for (x in 152..168) {
                if (x !in 159..166 || y !in 158..172) {
                    assertEquals("Right shoulder curl smudged at ($x,$y)", curlColor, out[y * w + x])
                }
            }
        }

        // 3. ZERO smudging on gold earrings!
        for (y in 158..172) {
            for (x in 94..101) {
                assertEquals("Left earring smudged at ($x,$y)", goldEarring, out[y * w + x])
            }
            for (x in 159..166) {
                assertEquals("Right earring smudged at ($x,$y)", goldEarring, out[y * w + x])
            }
        }

        // Scenario 5: Healthy uniform neck with no seam or drift -> ZERO unnecessary smoothing/blur!
        val cleanOrig = IntArray(w * h) { rgb(208, 164, 134) }
        val cleanOut = IntArray(w * h) { rgb(208, 164, 134) }
        val untouched = CleanEraseHeadReplacementEngine.repairNeckChinSeamPixels(
            outPx = cleanOut, origPx = cleanOrig, w = w, h = h,
            eyeMidX = eyeMidX, eyeMidY = eyeMidY, eyeDist = eyeDist,
            chinEstX = eyeMidX, chinEstY = chinY
        )
        assertEquals("Healthy uniform neck should have 0 pixels modified", 0, untouched)
    }
}
