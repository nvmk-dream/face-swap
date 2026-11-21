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
import com.example.onnx.CraniofacialPairReport
import com.example.onnx.CraniofacialProfile
import com.example.onnx.FaceSwapStrategyOption
import com.example.onnx.OptionDCraniofacialFitRatios
import com.example.ui.tabs.CraniofacialAnalysisDiagnosticCard
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
    fun `clicking A B B+C D shows Tamil popup and OK vs Cancel updates selection`() {
        composeTestRule.setContent {
            MyApplicationTheme {
                var selected by remember {
                    mutableStateOf(FaceSwapStrategyOption.OPTION_B_SHAPE_CONTOUR)
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

        val badgeB = "Active: B • ${FaceSwapStrategyOption.OPTION_B_SHAPE_CONTOUR.badgeSummary}"
        val badgeBC = "Active: B+C • ${FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD.badgeSummary}"
        val badgeD = "Active: D • ${FaceSwapStrategyOption.OPTION_D_CRANIOFACIAL_FIT.badgeSummary}"

        composeTestRule.onNodeWithTag("active_strategy_badge").assertExists()
        composeTestRule.onNodeWithText(badgeB).assertExists()

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

        // Tap Option D in 2x2 Grid -> Tap OK -> active switches to D
        composeTestRule.onNodeWithTag("strategy_option_d").performClick()
        composeTestRule.onNodeWithText(FaceSwapStrategyOption.OPTION_D_CRANIOFACIAL_FIT.tamilPopupTitle)
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("strategy_popup_ok").performClick()
        composeTestRule.onNodeWithText(badgeD).assertExists()
    }

    @Test
    fun `craniofacial diagnostics card displays match percentage and warning box`() {
        val dummyProfile = CraniofacialProfile(
            cheekWidthNormalized = 1.0f,
            squareJawRatio = 0.70f,
            lowerJawRatio = 0.40f,
            foreheadWidthRatio = 0.84f,
            upperThirdRatio = 1.0f,
            midThirdRatio = 1.0f,
            lowerThirdRatio = 0.90f,
            interpupillaryDistO = 64f,
            bizygomaticWidthB = 130f,
            bigonialJawWidthC = 91f,
            faceLengthD = 180f,
            lowerFaceLengthE = 58f,
            bitragionChinArcK = 3.1f,
            bitragionSubnasaleArcL = 2.5f,
            noseProtrusionM = 0.25f,
            faceDepthP = 0.95f,
            chinToMouthRatio = 0.20f,
            yawDeg = 2f,
            pitchDeg = 0f,
            rollDeg = 0f,
            mouthOpenRatio = 0.08f,
            smileCurvatureCoeff = 0.12f
        )
        val report = CraniofacialPairReport(
            sourceProfile = dummyProfile,
            targetProfile = dummyProfile.copy(yawDeg = 30f),
            overallCompatibilityPct = 76,
            jawMatchSummary = "Source 1 : 0.70 : 0.40 (Tgt 0.72)",
            foreheadThirdsSummary = "Forehead 8-9: 84% • Thirds 1 : 1.00 : 0.90",
            klArcSummary = "K-Arc 3.10 vs 3.25",
            poseDeltaSummary = "Yaw Δ 28°",
            reactionSyncSummary = "Target Reaction Sync",
            diagnosticWarnings = listOf("தலைத் திருப்பம் (Yaw Δ 28°) அதிகம்"),
            optionDFitRatios = OptionDCraniofacialFitRatios()
        )
        composeTestRule.setContent {
            MyApplicationTheme {
                CraniofacialAnalysisDiagnosticCard(
                    report = report,
                    selectedStrategy = FaceSwapStrategyOption.OPTION_D_CRANIOFACIAL_FIT
                )
            }
        }
        composeTestRule.onNodeWithTag("craniofacial_analysis_card").assertExists()
        composeTestRule.onNodeWithText("76% Match").assertIsDisplayed()
        composeTestRule.onNodeWithTag("craniofacial_mandibular_summary").assertExists()
        composeTestRule.onNodeWithTag("craniofacial_warning_box").assertExists()
    }
}
