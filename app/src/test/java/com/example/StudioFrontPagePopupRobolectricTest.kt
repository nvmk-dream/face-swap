package com.example

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.dp
import com.example.onnx.FaceSwapStrategyOption
import com.example.ui.tabs.FaceSwapStrategyConfirmDialog
import com.example.ui.tabs.FaceSwapStrategySelectorCard
import com.example.ui.theme.MyApplicationTheme
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StudioFrontPagePopupRobolectricTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `clicking A B B+C shows Tamil popup and OK vs Cancel updates selection`() {
        composeTestRule.setContent {
            MyApplicationTheme {
                var selected by remember {
                    mutableStateOf(FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY)
                }
                var pending by remember {
                    mutableStateOf<FaceSwapStrategyOption?>(null)
                }
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    FaceSwapStrategySelectorCard(
                        selectedOption = selected,
                        onSelectRequested = { pending = it }
                    )
                    val curPending = pending
                    if (curPending != null) {
                        FaceSwapStrategyConfirmDialog(
                            pendingOption = curPending,
                            onConfirm = {
                                selected = curPending
                                pending = null
                            },
                            onCancel = {
                                pending = null
                            }
                        )
                    }
                }
            }
        }

        val badgeB = "Active: B • ${FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY.badgeSummary}"
        val badgeBC = "Active: B+C • ${FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD.badgeSummary}"

        composeTestRule.onNodeWithTag("active_strategy_badge").assertExists()
        composeTestRule.onNodeWithText(badgeB).assertExists()

        // Tap Option B -> verifies exact clean 5-step Tamil description -> OK
        composeTestRule.onNodeWithTag("strategy_option_b").performClick()
        composeTestRule.onNodeWithText(FaceSwapStrategyOption.OPTION_B_PURE_IDENTITY.tamilPopupTitle)
            .assertIsDisplayed()
        composeTestRule.onNodeWithText(
            "5-pt Align → ArcFace → Swapper → Target Lighting Color Match → Tight-Feathered Blend (+ Controlled Enhancer) என்ற சுத்தமான முறையைப் பயன்படுத்துகிறது.",
            substring = true
        ).assertIsDisplayed()
        composeTestRule.onNodeWithTag("strategy_popup_ok").performClick()

        // Tap Option A -> Tamil popup opens with OK & Cancel
        composeTestRule.onNodeWithTag("strategy_option_a").performClick()

        composeTestRule.onNodeWithText(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY.tamilPopupTitle)
            .assertIsDisplayed()
        composeTestRule.onNodeWithText(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY.tamilPopupDescription)
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("strategy_popup_ok").assertExists()
        composeTestRule.onNodeWithTag("strategy_popup_cancel").assertExists()

        // Tap Cancel -> popup closes, active remains B
        composeTestRule.onNodeWithTag("strategy_popup_cancel").performClick()
        composeTestRule.onNodeWithText(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY.tamilPopupTitle)
            .assertDoesNotExist()
        composeTestRule.onNodeWithText(badgeB).assertExists()

        // Tap Option B+C -> Tap OK -> active switches to B+C
        composeTestRule.onNodeWithTag("strategy_option_bc").performClick()
        composeTestRule.onNodeWithText(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD.tamilPopupTitle)
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("strategy_popup_ok").performClick()

        composeTestRule.onNodeWithText(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD.tamilPopupTitle)
            .assertDoesNotExist()
        composeTestRule.onNodeWithText(badgeBC).assertExists()
    }
}
