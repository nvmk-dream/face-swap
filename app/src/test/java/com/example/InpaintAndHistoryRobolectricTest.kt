package com.example

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.example.data.SwapAuditDatabase
import com.example.data.SwapAuditLog
import com.example.onnx.InteractiveInpaintEngine
import com.example.util.ImageGalleryHelper
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class InpaintAndHistoryRobolectricTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun swapAuditDatabase_retainsOnlyLatestThreeRecordsAndDeletesOldImages() = runBlocking {
        val db = SwapAuditDatabase.getInstance(context)
        db.clearAll()

        val savedPaths = mutableListOf<String>()
        for (i in 1..5) {
            val bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.rgb(20 * i, 40 * i, 180))
            }
            val path = ImageGalleryHelper.saveBitmapToHistoryCache(context, bmp)!!
            savedPaths.add(path)
            db.insertLog(
                SwapAuditLog(
                    timestampMillis = 1_000_000L + i * 100L,
                    sourceResolution = "64x64",
                    targetResolution = "64x64",
                    swappedFacesCount = 1,
                    pipelineSummary = "Test #$i",
                    detectionMs = 10L,
                    embeddingMs = 10L,
                    inswapperMs = 10L,
                    blendingMs = 10L,
                    totalMs = 40L,
                    colorTransferEnabled = false,
                    watermarkEnabled = false,
                    resultImagePath = path
                )
            )
            Thread.sleep(5)
        }

        val logs = db.observeAllLogs().first()
        assertEquals(3, logs.size)
        assertEquals("Test #5", logs[0].pipelineSummary)
        assertEquals("Test #4", logs[1].pipelineSummary)
        assertEquals("Test #3", logs[2].pipelineSummary)

        // Older files (1 and 2) should be auto-deleted; latest 3 must still exist
        assertFalse(File(savedPaths[0]).exists())
        assertFalse(File(savedPaths[1]).exists())
        assertTrue(File(savedPaths[2]).exists())
        assertTrue(File(savedPaths[3]).exists())
        assertTrue(File(savedPaths[4]).exists())
    }

    @Test
    fun interactiveInpaintEngine_fillsMaskedRegionAndPreservesExteriorPixels() {
        val src = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        src.eraseColor(Color.rgb(30, 120, 210))
        // Place a distinct red block at the center to remove via mask
        for (y in 48 until 80) {
            for (x in 48 until 80) {
                src.setPixel(x, y, Color.RED)
            }
        }

        val mask = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        mask.eraseColor(Color.TRANSPARENT)
        for (y in 44 until 84) {
            for (x in 44 until 84) {
                mask.setPixel(x, y, Color.WHITE)
            }
        }

        val result = InteractiveInpaintEngine.inpaint(
            ortEnv = null,
            sourceBitmap = src,
            maskBitmap = mask,
            prompt = "remove red square",
            fastMode = true,
            lamaModelFile = null,
            preferHardwareAccel = false
        )

        assertEquals(128, result.outputBitmap.width)
        assertEquals(128, result.outputBitmap.height)

        // Exterior unmasked pixel must remain 100% unchanged
        assertEquals(Color.rgb(30, 120, 210), result.outputBitmap.getPixel(10, 10))

        // Center pixel inside masked region must now resemble the blue background rather than red
        val centerColor = result.outputBitmap.getPixel(64, 64)
        assertTrue(Color.blue(centerColor) > Color.red(centerColor))
    }
}
