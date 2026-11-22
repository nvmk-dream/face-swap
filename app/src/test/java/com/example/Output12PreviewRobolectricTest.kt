package com.example

import android.graphics.Bitmap
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import com.example.onnx.FaceSwapExecutionResult
import com.example.ui.FaceSwapUiState
import com.example.ui.tabs.FrontResultShowcaseCard
import com.example.ui.theme.MyApplicationTheme
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class Output12PreviewRobolectricTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `Output 1 Output 2 and Final tiles render and trigger fullscreen preview callbacks`() {
        val dummyBmp = { Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888) }
        val out1 = dummyBmp()
        val out2 = dummyBmp()
        val finalBmp = dummyBmp()
        val execResult = FaceSwapExecutionResult(
            outputBitmap = finalBmp,
            alignedSource112 = dummyBmp(),
            alignedTarget128 = dummyBmp(),
            rawSwapped128 = dummyBmp(),
            swappedFacesCount = 1,
            detectionMs = 10L,
            embeddingMs = 10L,
            inswapperMs = 10L,
            blendingMs = 10L,
            totalMs = 40L,
            pipelineSummary = "3-Step Portrait-First | engineUsed=NEURAL",
            headlessTargetPreview = out1,
            sourceHeadCutoutPreview = out2
        )

        val clicks = mutableListOf<Pair<Bitmap, String>>()

        composeTestRule.setContent {
            MyApplicationTheme {
                FrontResultShowcaseCard(
                    uiState = FaceSwapUiState(swapResult = execResult),
                    onDownloadHd = {},
                    onSaveToGallery = {},
                    onInspectResultDetails = {},
                    onViewStepFullScreen = { bmp, title -> clicks.add(bmp to title) }
                )
            }
        }

        composeTestRule.onNodeWithTag("btn_toggle_output12_preview").assertIsDisplayed()
        composeTestRule.onNodeWithTag("preview_tile_output1").assertIsDisplayed()
        composeTestRule.onNodeWithTag("preview_tile_output2").assertIsDisplayed()
        composeTestRule.onNodeWithTag("preview_tile_final").assertIsDisplayed()

        composeTestRule.onNodeWithTag("preview_tile_output1").performClick()
        composeTestRule.onNodeWithTag("preview_tile_output2").performClick()
        composeTestRule.onNodeWithTag("preview_tile_final").performClick()

        assertEquals(3, clicks.size)
        assertEquals("Output 1: Headless Target", clicks[0].second)
        assertEquals(out1, clicks[0].first)
        assertEquals("Output 2: Source Head Cutout", clicks[1].second)
        assertEquals(out2, clicks[1].first)
        assertEquals("Final Merged Output", clicks[2].second)
        assertEquals(finalBmp, clicks[2].first)

        // Toggle Hide -> tiles collapse
        composeTestRule.onNodeWithTag("btn_toggle_output12_preview").performClick()
        composeTestRule.onNodeWithTag("preview_tile_output1").assertDoesNotExist()
    }
}
