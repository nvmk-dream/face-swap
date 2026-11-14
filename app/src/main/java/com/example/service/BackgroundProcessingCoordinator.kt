package com.example.service

import android.graphics.Bitmap
import com.example.onnx.FaceSwapExecutionResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class BackgroundInpaintSync(
    val isRunning: Boolean = false,
    val progressText: String? = null,
    val resultBitmap: Bitmap? = null,
    val elapsedSec: Float? = null,
    val engineLabel: String? = null,
    val errorMessage: String? = null,
    val statusMessage: String? = null
)

data class BackgroundSwapSync(
    val isRunning: Boolean = false,
    val stageTitle: String? = null,
    val detailMessage: String? = null,
    val result: FaceSwapExecutionResult? = null,
    val auditLogId: Int? = null,
    val errorMessage: String? = null,
    val statusMessage: String? = null
)

/**
 * Application-lifetime coroutine coordinator for Inpaint and Face Swap / Head Swap jobs.
 * Works together with [ProcessingForegroundService] so jobs survive app-switching
 * (e.g., Chrome / Gemini AI), screen locks, and Activity recreations.
 */
object BackgroundProcessingCoordinator {
    val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _inpaintSync = MutableStateFlow(BackgroundInpaintSync())
    val inpaintSync = _inpaintSync.asStateFlow()

    private val _swapSync = MutableStateFlow(BackgroundSwapSync())
    val swapSync = _swapSync.asStateFlow()

    @Volatile
    var activeInpaintJob: Job? = null

    fun isAnyJobRunning(): Boolean = _inpaintSync.value.isRunning || _swapSync.value.isRunning

    fun updateInpaint(transform: (BackgroundInpaintSync) -> BackgroundInpaintSync) {
        _inpaintSync.update(transform)
    }

    fun updateSwap(transform: (BackgroundSwapSync) -> BackgroundSwapSync) {
        _swapSync.update(transform)
    }

    fun cancelInpaintJob() {
        runCatching {
            activeInpaintJob?.cancel(CancellationException("Inpaint cancelled by user."))
        }
        activeInpaintJob = null
        _inpaintSync.update {
            it.copy(
                isRunning = false,
                progressText = null,
                errorMessage = null,
                statusMessage = "Inpaint generation cancelled."
            )
        }
    }

    fun clearInpaintResult() {
        _inpaintSync.value = BackgroundInpaintSync()
    }
}
