package com.example

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import androidx.test.core.app.ApplicationProvider
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import com.example.onnx.DetectedFace
import com.example.onnx.Face3DReconstruction
import com.example.onnx.HeadPoseEstimationEngine
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxProtobufInspector
import com.example.ui.tabs.ModelsInspectorTab
import com.example.ui.theme.MyApplicationTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.util.concurrent.CancellationException
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SixDRepNetIntegrationTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `matchSlotForFileName maps SixDRepNet and head pose filenames to ModelSlot HEAD_POSE`() {
        assertEquals(ModelSlot.HEAD_POSE, OnnxProtobufInspector.matchSlotForFileName("6DRepNet360_Slim.onnx"))
        assertEquals(ModelSlot.HEAD_POSE, OnnxProtobufInspector.matchSlotForFileName("SixDRepNet.onnx"))
        assertEquals(ModelSlot.HEAD_POSE, OnnxProtobufInspector.matchSlotForFileName("6DRepNet_360.onnx"))
        assertEquals(ModelSlot.HEAD_POSE, OnnxProtobufInspector.matchSlotForFileName("head_pose_resnet50.onnx"))
        assertEquals(ModelSlot.HEAD_POSE, OnnxProtobufInspector.matchSlotForFileName("whenet_mobilenet.onnx"))
    }

    @Test
    fun `computeRotationMatrixFrom6D and rotationMatrixToEulerDeg recover identity and tilted 6D pose accurately`() {
        val identityMat = HeadPoseEstimationEngine.computeRotationMatrixFrom6D(
            floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        )
        val (p0, y0, r0) = HeadPoseEstimationEngine.rotationMatrixToEulerDeg(identityMat)
        assertTrue("Expected ~0 pitch, got $p0", abs(p0) < 0.01f)
        assertTrue("Expected ~0 yaw, got $y0", abs(y0) < 0.01f)
        assertTrue("Expected ~0 roll, got $r0", abs(r0) < 0.01f)

        // Construct a 35-deg in-plane roll rotation where column 2 has slight non-orthogonality
        val rad = Math.toRadians(35.0)
        val raw6D = floatArrayOf(
            cos(rad).toFloat(), sin(rad).toFloat(), 0f,
            (-sin(rad) + 0.15 * cos(rad)).toFloat(), (cos(rad) + 0.15 * sin(rad)).toFloat(), 0f
        )
        val mat = HeadPoseEstimationEngine.computeRotationMatrixFrom6D(raw6D)
        val (_, _, roll) = HeadPoseEstimationEngine.rotationMatrixToEulerDeg(mat)
        assertTrue("Expected Gram-Schmidt roll ~35 deg, got $roll", abs(roll - 35f) < 0.5f)
    }

    @Test
    fun `parsePoseOutputs supports 3-elem, 6-elem, 3x3 matrix, and 3 named tensors`() {
        // 1) [1, 3] ypr
        val out3 = HeadPoseEstimationEngine.parsePoseOutputs(
            listOf(Triple("ypr", longArrayOf(1L, 3L), floatArrayOf(52f, -18f, 28f)))
        )
        assertNotNull(out3)
        assertEquals(52f, out3!!.yawDeg, 0.01f)
        assertEquals(-18f, out3.pitchDeg, 0.01f)
        assertEquals(28f, out3.rollDeg, 0.01f)

        // 2) [1, 3, 3] identity matrix
        val outMat = HeadPoseEstimationEngine.parsePoseOutputs(
            listOf(
                Triple(
                    "rot_mat",
                    longArrayOf(1L, 3L, 3L),
                    floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
                )
            )
        )
        assertNotNull(outMat)
        assertEquals(0f, outMat!!.yawDeg, 0.01f)

        // 3) Named scalar yaw/pitch/roll outputs
        val outNamed = HeadPoseEstimationEngine.parsePoseOutputs(
            listOf(
                Triple("yaw", longArrayOf(1L, 1L), floatArrayOf(-61f)),
                Triple("pitch", longArrayOf(1L, 1L), floatArrayOf(14f)),
                Triple("roll", longArrayOf(1L, 1L), floatArrayOf(-32f))
            )
        )
        assertNotNull(outNamed)
        assertEquals(-61f, outNamed!!.yawDeg, 0.01f)
        assertEquals(14f, outNamed.pitchDeg, 0.01f)
        assertEquals(-32f, outNamed.rollDeg, 0.01f)
    }

    @Test
    fun `Face3DReconstruction honors extreme SixDRepNet yaw when neuralPoseWeight is 0_85`() {
        val kps = listOf(
            PointF(100f, 95f),
            PointF(132f, 95f),
            PointF(130f, 120f),
            PointF(104f, 148f),
            PointF(126f, 148f)
        )
        val pose = Face3DReconstruction.estimate3DHeadPose(
            landmarks5 = kps,
            hintPitch = -12f,
            hintYaw = 64f,
            hintRoll = 31f,
            neuralPoseWeight = 0.85f
        )
        assertTrue("Expected extreme profile yaw > 52 deg, got ${pose.yawDeg}", pose.yawDeg > 52f)
        assertTrue("Expected pitch near -12 deg, got ${pose.pitchDeg}", pose.pitchDeg < -5f)
        assertTrue("Expected roll near 31 deg, got ${pose.rollDeg}", pose.rollDeg > 20f)
    }

    @Test
    fun `estimateHeadPoseOnnx throws CancellationException immediately when cancelled`() {
        val bmp = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        val face = DetectedFace(
            index = 0,
            boundingBox = RectF(24f, 24f, 96f, 108f),
            score = 0.99f,
            landmarks5 = listOf(
                PointF(48f, 54f),
                PointF(76f, 54f),
                PointF(62f, 70f),
                PointF(50f, 88f),
                PointF(74f, 88f)
            ),
            detectorSource = "test"
        )
        try {
            HeadPoseEstimationEngine.estimateHeadPoseOnnx(
                ortEnv = null,
                headPoseFile = File("/nonexistent/SixDRepNet.onnx"),
                bitmap = bmp,
                face = face,
                isCancelled = { true }
            )
            fail("Expected CancellationException when isCancelled returns true")
        } catch (_: CancellationException) {
            // Expected
        }
    }

    @Test
    fun `inspectAllModels includes HEAD_POSE slot with SixDRepNet canonical metadata`() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val inspections = OnnxProtobufInspector.inspectAllModels(ctx, null)
        val headPoseInspection = inspections.firstOrNull { it.slot == ModelSlot.HEAD_POSE }
        assertNotNull(headPoseInspection)
        assertEquals("SixDRepNet.onnx", headPoseInspection!!.slot.canonicalFileName)
        assertEquals("head_pose", headPoseInspection.slot.subdirectory)
    }

    @Test
    fun `settings Models tab displays 6DRepNet360_Slim upload card and triggers HEAD_POSE import`() {
        var importedSlot: ModelSlot? = null
        composeTestRule.setContent {
            MyApplicationTheme {
                ModelsInspectorTab(
                    inspections = emptyList(),
                    isInspecting = false,
                    onImportModelForSlot = { importedSlot = it },
                    onRefreshModels = {}
                )
            }
        }

        composeTestRule.onNodeWithTag("models_sixdrepnet_slim_card")
            .performScrollTo()
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("models_import_sixdrepnet_slim_button")
            .performScrollTo()
            .assertIsDisplayed()
            .performClick()

        assertEquals(ModelSlot.HEAD_POSE, importedSlot)
    }
}
