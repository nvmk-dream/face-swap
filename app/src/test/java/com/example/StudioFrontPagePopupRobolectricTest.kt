package com.example

import android.os.Looper
import android.view.View
import android.view.ViewGroup
import androidx.activity.ComponentActivity
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewRootForTest
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.semantics.SemanticsNode
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.dp
import com.example.onnx.FaceSwapStrategyOption
import com.example.ui.tabs.FaceSwapStrategyConfirmDialog
import com.example.ui.tabs.FaceSwapStrategySelectorCard
import com.example.ui.theme.MyApplicationTheme
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StudioFrontPagePopupRobolectricTest {

    @OptIn(ExperimentalComposeUiApi::class)
    private fun collectAllSemanticsNodes(): List<SemanticsNode> {
        val result = mutableListOf<SemanticsNode>()
        fun walkView(v: View) {
            if (v is ViewRootForTest) {
                fun walkNode(node: SemanticsNode) {
                    result.add(node)
                    node.children.forEach { walkNode(it) }
                }
                walkNode(v.semanticsOwner.unmergedRootSemanticsNode)
            }
            if (v is ViewGroup) {
                for (i in 0 until v.childCount) walkView(v.getChildAt(i))
            }
        }
        val rootViews = android.view.inspector.WindowInspector.getGlobalWindowViews()
        rootViews.forEach { walkView(it) }
        return result
    }

    private fun findByTag(tag: String): SemanticsNode? =
        collectAllSemanticsNodes().firstOrNull {
            it.config.getOrNull(SemanticsProperties.TestTag) == tag
        }

    private fun findByText(substring: String): SemanticsNode? =
        collectAllSemanticsNodes().firstOrNull { node ->
            val texts = node.config.getOrNull(SemanticsProperties.Text).orEmpty()
            texts.any { it.text.contains(substring) }
        }

    private fun clickNode(node: SemanticsNode) {
        val action = node.config.getOrNull(SemanticsActions.OnClick)
        assertNotNull("Expected clickable SemanticsNode", action)
        action!!.action!!.invoke()
        Shadows.shadowOf(Looper.getMainLooper()).idle()
    }

    @Test
    fun `clicking A B B+C shows Tamil popup and OK vs Cancel updates selection`() {
        val controller = Robolectric.buildActivity(ComponentActivity::class.java).setup()
        val activity = controller.get()
        val composeView = ComposeView(activity)
        composeView.setContent {
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
        activity.setContentView(composeView)
        Shadows.shadowOf(Looper.getMainLooper()).idle()

        val badgeB = "Active: B • ${FaceSwapStrategyOption.OPTION_B_SHAPE_CONTOUR.badgeSummary}"
        val badgeBC = "Active: B+C • ${FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD.badgeSummary}"

        assertNotNull("Active strategy badge missing", findByTag("active_strategy_badge"))
        assertNotNull("Default active B badge missing", findByText(badgeB))

        // Tap Option A -> Tamil popup opens with OK & Cancel
        val optANode = findByTag("strategy_option_a")
        assertNotNull("Option A button missing", optANode)
        clickNode(optANode!!)

        assertNotNull(
            "Option A Tamil title missing in popup",
            findByText(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY.tamilPopupTitle)
        )
        assertNotNull(
            "Option A Tamil description missing in popup",
            findByText(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY.tamilPopupDescription)
        )
        val okNode = findByTag("strategy_popup_ok")
        val cancelNode = findByTag("strategy_popup_cancel")
        assertNotNull("Popup OK button missing", okNode)
        assertNotNull("Popup Cancel button missing", cancelNode)

        // Tap Cancel -> popup closes, active remains B
        clickNode(cancelNode!!)
        assertNull(
            "Popup should close after Cancel",
            findByText(FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY.tamilPopupTitle)
        )
        assertNotNull("Cancel must preserve Option B active", findByText(badgeB))

        // Tap Option B+C -> Tap OK -> active switches to B+C
        val optBcNode = findByTag("strategy_option_bc")
        assertNotNull("Option B+C button missing", optBcNode)
        clickNode(optBcNode!!)

        assertNotNull(
            "Option B+C Tamil title missing in popup",
            findByText(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD.tamilPopupTitle)
        )
        val okBcNode = findByTag("strategy_popup_ok")
        assertNotNull("Popup OK button missing for B+C", okBcNode)
        clickNode(okBcNode!!)

        assertNull(
            "Popup should close after OK",
            findByText(FaceSwapStrategyOption.OPTION_BC_FULL_HEAD_HD.tamilPopupTitle)
        )
        assertNotNull("OK must activate Option B+C", findByText(badgeBC))

        controller.pause().stop().destroy()
    }
}
