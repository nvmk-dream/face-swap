package com.example

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.test.core.app.ApplicationProvider
import com.example.onnx.BodyReshapeAndOutpaintEngine
import com.example.onnx.InteractiveInpaintEngine
import com.example.onnx.OutpaintExtendPreset
import com.example.onnx.SdInpaintPackManager
import com.example.ui.components.InpaintBodyAndOutpaintCard
import com.example.ui.components.InpaintFolderAutoPickButton
import com.example.ui.theme.MyApplicationTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class BodyReshapeAndOutpaintTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun applyCurvyBodyReshape_preservesHeadAndBordersWhileWideningHipContour() {
        val w = 120
        val h = 160
        val src = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        src.eraseColor(Color.WHITE)

        // Draw a vertical dark torso/hip silhouette centered at x=60 (width 28px: x=46..74)
        for (y in 20 until h) {
            for (x in 46 until 74) {
                src.setPixel(x, y, Color.DKGRAY)
            }
        }
        // Mark a distinct blue face block at top (y=10..28) that MUST stay untouched
        for (y in 10 until 28) {
            for (x in 52 until 68) {
                src.setPixel(x, y, Color.BLUE)
            }
        }

        val reshaped = BodyReshapeAndOutpaintEngine.applyCurvyBodyReshape(
            sourceBitmap = src,
            curvyStrength = 0.85f,
            waistYRatio = 0.44f,
            hipYRatio = 0.64f
        )

        assertEquals(w, reshaped.width)
        assertEquals(h, reshaped.height)

        // Head/face region must remain unchanged
        assertEquals(Color.BLUE, reshaped.getPixel(60, 18))
        // Outer canvas border pixels must remain white/anchored
        assertEquals(Color.WHITE, reshaped.getPixel(1, (h * 0.64f).toInt()))
        assertEquals(Color.WHITE, reshaped.getPixel(w - 2, (h * 0.64f).toInt()))

        // Hip row dark silhouette width should be wider than original 28px
        val hipY = (h * 0.64f).toInt()
        var origDarkWidth = 0
        var curvyDarkWidth = 0
        for (x in 0 until w) {
            if (Color.red(src.getPixel(x, hipY)) < 128) origDarkWidth++
            if (Color.red(reshaped.getPixel(x, hipY)) < 128) curvyDarkWidth++
        }
        assertTrue(
            "Expected hip silhouette to widen (orig=$origDarkWidth, curvy=$curvyDarkWidth)",
            curvyDarkWidth > origDarkWidth
        )
    }

    @Test
    fun extendCanvasForFullBodyOutpaint_expandsHeightPreservesTopAndCreatesSeamMask() {
        val w = 100
        val h = 120
        val src = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        src.eraseColor(Color.rgb(40, 90, 170))
        // Distinctive top-middle red marker
        for (y in 20 until 40) {
            for (x in 35 until 65) {
                src.setPixel(x, y, Color.RED)
            }
        }

        val extended = BodyReshapeAndOutpaintEngine.extendCanvasForFullBodyOutpaint(
            sourceBitmap = src,
            extendFraction = OutpaintExtendPreset.FULL_BODY.fraction
        )

        assertEquals(w, extended.originalWidth)
        assertEquals(h, extended.originalHeight)
        assertEquals(w, extended.extendedBitmap.width)
        assertTrue(extended.extraHeight >= 80)
        assertEquals(h + extended.extraHeight, extended.extendedBitmap.height)
        assertEquals(extended.extendedBitmap.width, extended.autoMaskBitmap.width)
        assertEquals(extended.extendedBitmap.height, extended.autoMaskBitmap.height)

        // Top region pixels preserved untouched & mask alpha is 0 in top face/chest region
        assertEquals(Color.RED, extended.extendedBitmap.getPixel(50, 30))
        assertEquals(0, Color.alpha(extended.autoMaskBitmap.getPixel(50, 30)))

        // Bottom extended region is fully masked white & mask passes InteractiveInpaintEngine validation
        assertEquals(255, Color.alpha(extended.autoMaskBitmap.getPixel(50, h + 20)))
        assertTrue(InteractiveInpaintEngine.hasPaintedMaskPixels(extended.autoMaskBitmap))
    }

    @Test
    fun isMissingCompanionExternalData_recognizesPbAndOnnxDataCompanions() {
        val tmpDir = File(context.cacheDir, "sd_ext_test").apply {
            deleteRecursively()
            mkdirs()
        }
        val onnxStub = File(tmpDir, "model.onnx")
        val stubContent = ByteArray(4096) { 1 }
        val refBytes = "external_data:weights.pb".toByteArray()
        System.arraycopy(refBytes, 0, stubContent, 64, refBytes.size)
        onnxStub.writeBytes(stubContent)

        // Missing weights.pb -> should report true
        assertTrue(SdInpaintPackManager.isMissingCompanionExternalData(onnxStub))

        // Once weights.pb exists (>1KB) -> should report false
        File(tmpDir, "weights.pb").writeBytes(ByteArray(2048) { 2 })
        assertFalse(SdInpaintPackManager.isMissingCompanionExternalData(onnxStub))
        tmpDir.deleteRecursively()
    }

    @Test
    fun uiComponents_autoPickFolderAndBodyOutpaintCardTriggerCallbacks() {
        var autoPickClicked = false
        var selectedPreset = OutpaintExtendPreset.NONE
        var resetClicked = false

        composeTestRule.setContent {
            MyApplicationTheme {
                androidx.compose.foundation.layout.Column {
                    InpaintFolderAutoPickButton(
                        isBusy = false,
                        installedPacksCount = 0,
                        onAutoPickFolder = { autoPickClicked = true }
                    )
                    InpaintBodyAndOutpaintCard(
                        hasSourceImage = true,
                        curvyAmount = 0.5f,
                        hipPositionRatio = 0.64f,
                        outpaintPreset = OutpaintExtendPreset.FULL_BODY,
                        hasSdPackInstalled = true,
                        isAdjusting = false,
                        onCurvyAmountChanged = {},
                        onHipPositionChanged = {},
                        onSelectOutpaintPreset = { selectedPreset = it },
                        onResetEdits = { resetClicked = true }
                    )
                }
            }
        }

        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("inpaint_folder_autopick_btn")
            .assertIsDisplayed()
            .performClick()
        assertTrue(autoPickClicked)

        composeTestRule.onNodeWithTag("inpaint_outpaint_preset_knee")
            .assertIsDisplayed()
            .performClick()
        assertEquals(OutpaintExtendPreset.KNEE, selectedPreset)

        composeTestRule.onNodeWithTag("inpaint_body_reset_btn")
            .assertIsDisplayed()
            .performClick()
        assertTrue(resetClicked)
    }
}
