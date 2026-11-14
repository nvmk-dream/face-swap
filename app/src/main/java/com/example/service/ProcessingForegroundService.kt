package com.example.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.example.MainActivity
import com.example.R

/**
 * Foreground Service + Partial WakeLock controller that keeps long-running
 * ONNX inference (Inpaint & Face Swap / Head Swap) running smoothly when the user
 * switches to other apps (Chrome, Gemini AI, etc.) or locks the phone screen.
 */
class ProcessingForegroundService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        activeInstance = this
        ensureChannels(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        activeInstance = this
        serviceActive = true
        ensureChannels(this)
        val title = intent?.getStringExtra(EXTRA_TITLE) ?: "FaceSwap Studio Running"
        val detail = intent?.getStringExtra(EXTRA_DETAIL) ?: "Processing image offline in background..."
        val progress = intent?.getIntExtra(EXTRA_PROGRESS, -1) ?: -1

        acquireOrRenewCpuWakeLock(forceRenew = false)
        val notification = buildOngoingNotification(this, title, detail, progress)
        val serviceType = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        } else {
            0
        }
        runCatching {
            ServiceCompat.startForeground(this, ONGOING_NOTIFICATION_ID, notification, serviceType)
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        if (activeInstance === this) {
            activeInstance = null
        }
        serviceActive = false
        startRequestedAtMs = 0L
        releaseCpuWakeLock()
        runCatching {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        }
        super.onDestroy()
    }

    private fun acquireOrRenewCpuWakeLock(forceRenew: Boolean) {
        val now = android.os.SystemClock.elapsedRealtime()
        if (!forceRenew && wakeLock?.isHeld == true && (now - lastWakeLockRenewMs) < WAKELOCK_RENEW_INTERVAL_MS) {
            return
        }
        runCatching {
            val pm = getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return
            val lock = wakeLock ?: pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKELOCK_TAG).apply {
                setReferenceCounted(false)
            }
            // Renew 30-minute safety window while job is actively making progress
            lock.acquire(WAKELOCK_TIMEOUT_MS)
            wakeLock = lock
            lastWakeLockRenewMs = now
        }
    }

    private fun releaseCpuWakeLock() {
        runCatching {
            wakeLock?.let { lock ->
                if (lock.isHeld) lock.release()
            }
            wakeLock = null
            lastWakeLockRenewMs = 0L
        }
    }

    companion object {
        private const val CHANNEL_ONGOING = "faceswap_processing_ongoing"
        private const val CHANNEL_DONE = "faceswap_processing_completed"
        private const val ONGOING_NOTIFICATION_ID = 4201
        private const val DONE_NOTIFICATION_ID = 4202
        private const val WAKELOCK_TAG = "FaceSwapStudio:OnnxBackgroundWakeLock"
        private const val WAKELOCK_TIMEOUT_MS = 30L * 60L * 1000L
        private const val WAKELOCK_RENEW_INTERVAL_MS = 5L * 60L * 1000L
        private const val NOTIFY_THROTTLE_MS = 400L

        private const val EXTRA_TITLE = "extra_title"
        private const val EXTRA_DETAIL = "extra_detail"
        private const val EXTRA_PROGRESS = "extra_progress"

        @Volatile
        private var serviceActive = false

        @Volatile
        private var activeInstance: ProcessingForegroundService? = null

        @Volatile
        private var startRequestedAtMs = 0L

        @Volatile
        private var lastWakeLockRenewMs = 0L

        @Volatile
        private var lastOngoingNotifyAtMs = 0L

        fun startOrUpdate(
            context: Context,
            title: String,
            detail: String,
            progressPercent: Int = -1
        ) {
            val appCtx = context.applicationContext
            ensureChannels(appCtx)
            val now = android.os.SystemClock.elapsedRealtime()
            activeInstance?.acquireOrRenewCpuWakeLock(forceRenew = false)

            val shouldAttemptStart = !serviceActive && (now - startRequestedAtMs > 2000L)
            if (shouldAttemptStart) {
                startRequestedAtMs = now
                val intent = Intent(appCtx, ProcessingForegroundService::class.java).apply {
                    putExtra(EXTRA_TITLE, title)
                    putExtra(EXTRA_DETAIL, detail)
                    putExtra(EXTRA_PROGRESS, progressPercent)
                }
                val launched = runCatching {
                    ContextCompat.startForegroundService(appCtx, intent)
                }.isSuccess
                if (launched) {
                    lastOngoingNotifyAtMs = now
                    return
                }
            }

            // Throttle intermediate notification ticks to avoid Android OS rate-limiting
            val isNearCompletion = progressPercent >= 95
            if (!isNearCompletion && (now - lastOngoingNotifyAtMs) < NOTIFY_THROTTLE_MS) {
                return
            }
            if (canPostNotifications(appCtx)) {
                runCatching {
                    val notification = buildOngoingNotification(appCtx, title, detail, progressPercent)
                    NotificationManagerCompat.from(appCtx).notify(ONGOING_NOTIFICATION_ID, notification)
                    lastOngoingNotifyAtMs = now
                }
            }
        }

        fun finishAndNotify(
            context: Context,
            showCompletionNotification: Boolean,
            completedTitle: String,
            completedBody: String
        ) {
            val appCtx = context.applicationContext
            serviceActive = false
            startRequestedAtMs = 0L
            lastOngoingNotifyAtMs = 0L
            activeInstance?.releaseCpuWakeLock()
            runCatching {
                appCtx.stopService(Intent(appCtx, ProcessingForegroundService::class.java))
            }
            runCatching {
                NotificationManagerCompat.from(appCtx).cancel(ONGOING_NOTIFICATION_ID)
            }
            if (showCompletionNotification && canPostNotifications(appCtx)) {
                ensureChannels(appCtx)
                val openIntent = Intent(appCtx, MainActivity::class.java).apply {
                    flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
                }
                val pendingFlags = PendingIntent.FLAG_UPDATE_CURRENT or
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
                val pendingIntent = PendingIntent.getActivity(appCtx, 102, openIntent, pendingFlags)

                val doneNotification = NotificationCompat.Builder(appCtx, CHANNEL_DONE)
                    .setSmallIcon(R.drawable.ic_stat_studio)
                    .setContentTitle(completedTitle)
                    .setContentText(completedBody)
                    .setStyle(NotificationCompat.BigTextStyle().bigText(completedBody))
                    .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                    .setAutoCancel(true)
                    .setContentIntent(pendingIntent)
                    .build()

                runCatching {
                    NotificationManagerCompat.from(appCtx).notify(DONE_NOTIFICATION_ID, doneNotification)
                }
            }
        }

        private fun canPostNotifications(context: Context): Boolean {
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.POST_NOTIFICATIONS
                ) == PackageManager.PERMISSION_GRANTED
            } else {
                true
            }
        }

        private fun ensureChannels(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
            val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager ?: return
            if (nm.getNotificationChannel(CHANNEL_ONGOING) == null) {
                val ongoing = NotificationChannel(
                    CHANNEL_ONGOING,
                    "Offline AI Processing",
                    NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Keeps Inpaint and Face Swap running while using other apps or screen lock."
                    setShowBadge(false)
                }
                nm.createNotificationChannel(ongoing)
            }
            if (nm.getNotificationChannel(CHANNEL_DONE) == null) {
                val done = NotificationChannel(
                    CHANNEL_DONE,
                    "Completed Images",
                    NotificationManager.IMPORTANCE_DEFAULT
                ).apply {
                    description = "Notifies when background Inpaint or Face Swap finishes."
                }
                nm.createNotificationChannel(done)
            }
        }

        private fun buildOngoingNotification(
            context: Context,
            title: String,
            detail: String,
            progressPercent: Int
        ): Notification {
            val openIntent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pendingFlags = PendingIntent.FLAG_UPDATE_CURRENT or
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE else 0
            val pendingIntent = PendingIntent.getActivity(context, 101, openIntent, pendingFlags)

            val builder = NotificationCompat.Builder(context, CHANNEL_ONGOING)
                .setSmallIcon(R.drawable.ic_stat_studio)
                .setContentTitle(title)
                .setContentText(detail)
                .setStyle(NotificationCompat.BigTextStyle().bigText(detail))
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .setCategory(NotificationCompat.CATEGORY_PROGRESS)
                .setContentIntent(pendingIntent)

            if (progressPercent in 0..100) {
                builder.setProgress(100, progressPercent, false)
            } else {
                builder.setProgress(100, 0, true)
            }
            return builder.build()
        }
    }
}
