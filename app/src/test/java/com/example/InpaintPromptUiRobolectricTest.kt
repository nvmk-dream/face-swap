package com.example

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextReplacement
import com.example.onnx.SdInpaintOnnxEngine
import com.example.ui.tabs.FullScreenPromptEditorDialog
import com.example.ui.tabs.InpaintProgressCancelContent
import com.example.ui.theme.MyApplicationTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class InpaintPromptUiRobolectricTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun positiveAndNegativePromptDialogs_showVisibleOkButtonAndApplyPrompt() {
        var appliedPositive = ""
        var appliedNegative = ""

        composeTestRule.mainClock.autoAdvance = false

        composeTestRule.setContent {
            MyApplicationTheme {
                var mode by remember { mutableStateOf("POSITIVE") }
                if (mode == "POSITIVE") {
                    FullScreenPromptEditorDialog(
                        title = "Positive Prompt",
                        subtitle = "Describe clearly what you want generated inside the painted mask",
                        accentColor = Color(0xFF0284C7),
                        initialText = "",
                        placeholderText = "e.g. modern dress",
                        quickTags = listOf("modern dress", "silk saree"),
                        showResetDefault = false,
                        defaultText = "",
                        dialogTag = "positive_prompt_fullscreen_dialog",
                        inputTag = "positive_prompt_fullscreen_input",
                        applyTag = "positive_prompt_apply_btn",
                        onDismiss = {},
                        onApply = {
                            appliedPositive = it
                            mode = "NEGATIVE"
                        }
                    )
                } else {
                    FullScreenPromptEditorDialog(
                        title = "Negative Prompt",
                        subtitle = "Things the AI should strictly avoid generating",
                        accentColor = Color(0xFFDC2626),
                        initialText = SdInpaintOnnxEngine.DEFAULT_NEGATIVE_PROMPT,
                        placeholderText = SdInpaintOnnxEngine.DEFAULT_NEGATIVE_PROMPT,
                        quickTags = listOf("blurry", "deformed hands"),
                        showResetDefault = true,
                        defaultText = SdInpaintOnnxEngine.DEFAULT_NEGATIVE_PROMPT,
                        dialogTag = "negative_prompt_fullscreen_dialog",
                        inputTag = "negative_prompt_fullscreen_input",
                        applyTag = "negative_prompt_apply_btn",
                        onDismiss = {},
                        onApply = { appliedNegative = it }
                    )
                }
            }
        }

        repeat(2) { composeTestRule.mainClock.advanceTimeByFrame() }

        // Positive Prompt Dialog: OK button visible & applies text
        composeTestRule.onNodeWithTag("positive_prompt_fullscreen_dialog").assertIsDisplayed()
        composeTestRule.onNodeWithTag("positive_prompt_fullscreen_input")
            .performTextReplacement("silk saree golden zari")
        repeat(2) { composeTestRule.mainClock.advanceTimeByFrame() }
        composeTestRule.onNodeWithTag("positive_prompt_apply_btn")
            .assertIsDisplayed()
            .performClick()
        repeat(2) { composeTestRule.mainClock.advanceTimeByFrame() }
        assertEquals("silk saree golden zari", appliedPositive)

        // Negative Prompt Dialog: OK button visible & applies text
        composeTestRule.onNodeWithTag("negative_prompt_fullscreen_dialog").assertIsDisplayed()
        composeTestRule.onNodeWithTag("negative_prompt_fullscreen_input")
            .performTextReplacement("blurry, extra fingers")
        repeat(2) { composeTestRule.mainClock.advanceTimeByFrame() }
        composeTestRule.onNodeWithTag("negative_prompt_apply_btn")
            .assertIsDisplayed()
            .performClick()
        repeat(2) { composeTestRule.mainClock.advanceTimeByFrame() }
        assertEquals("blurry, extra fingers", appliedNegative)
    }

    @Test
    fun inpaintProgressOverlay_showsCancelButtonAndInvokesCallback() {
        var cancelledClicked = false

        composeTestRule.setContent {
            MyApplicationTheme {
                var showOverlay by remember { mutableStateOf(true) }
                if (showOverlay) {
                    InpaintProgressCancelContent(
                        progressText = "Stage 3/4: Diffusion step 2/11...",
                        onCancelInpaint = {
                            cancelledClicked = true
                            showOverlay = false
                        },
                        animateProgress = false
                    )
                }
            }
        }

        composeTestRule.onNodeWithText("Stage 3/4: Diffusion step 2/11...").assertIsDisplayed()
        composeTestRule.onNodeWithTag("inpaint_cancel_btn")
            .assertIsDisplayed()
            .performClick()

        assertTrue(cancelledClicked)
    }
}
