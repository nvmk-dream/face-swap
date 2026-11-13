package com.example.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.example.util.ImageGalleryHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File

data class SwapAuditLog(
    val id: Int = 0,
    val timestampMillis: Long = System.currentTimeMillis(),
    val sourceResolution: String,
    val targetResolution: String,
    val swappedFacesCount: Int,
    val pipelineSummary: String,
    val detectionMs: Long,
    val embeddingMs: Long,
    val inswapperMs: Long,
    val blendingMs: Long,
    val totalMs: Long,
    val colorTransferEnabled: Boolean,
    val watermarkEnabled: Boolean,
    val savedGalleryPath: String? = null,
    val resultImagePath: String? = null
)

interface SwapAuditDao {
    fun observeAllLogs(): Flow<List<SwapAuditLog>>
    suspend fun insertLog(log: SwapAuditLog): Long
    suspend fun updateSavedPath(id: Int, galleryPath: String)
    suspend fun deleteById(id: Int)
    suspend fun clearAll()
}

class SwapAuditDatabase private constructor(context: Context) :
    SQLiteOpenHelper(context.applicationContext, "faceswap_studio_audit.db", null, 2),
    SwapAuditDao {

    private val appContext: Context = context.applicationContext
    private val logsFlow = MutableStateFlow<List<SwapAuditLog>>(emptyList())

    init {
        enforceMaxHistoryLimitSync()
        refreshLogsSync()
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS swap_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestampMillis INTEGER NOT NULL,
                sourceResolution TEXT NOT NULL,
                targetResolution TEXT NOT NULL,
                swappedFacesCount INTEGER NOT NULL,
                pipelineSummary TEXT NOT NULL,
                detectionMs INTEGER NOT NULL,
                embeddingMs INTEGER NOT NULL,
                inswapperMs INTEGER NOT NULL,
                blendingMs INTEGER NOT NULL,
                totalMs INTEGER NOT NULL,
                colorTransferEnabled INTEGER NOT NULL,
                watermarkEnabled INTEGER NOT NULL,
                savedGalleryPath TEXT,
                resultImagePath TEXT
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            val altered = runCatching {
                db.execSQL("ALTER TABLE swap_audit_logs ADD COLUMN resultImagePath TEXT")
            }.isSuccess
            if (!altered) {
                db.execSQL("DROP TABLE IF EXISTS swap_audit_logs")
                onCreate(db)
            }
        }
    }

    fun swapAuditDao(): SwapAuditDao = this

    override fun observeAllLogs(): Flow<List<SwapAuditLog>> = logsFlow.asStateFlow()

    override suspend fun insertLog(log: SwapAuditLog): Long = withContext(Dispatchers.IO) {
        val values = ContentValues().apply {
            if (log.id != 0) put("id", log.id)
            put("timestampMillis", log.timestampMillis)
            put("sourceResolution", log.sourceResolution)
            put("targetResolution", log.targetResolution)
            put("swappedFacesCount", log.swappedFacesCount)
            put("pipelineSummary", log.pipelineSummary)
            put("detectionMs", log.detectionMs)
            put("embeddingMs", log.embeddingMs)
            put("inswapperMs", log.inswapperMs)
            put("blendingMs", log.blendingMs)
            put("totalMs", log.totalMs)
            put("colorTransferEnabled", if (log.colorTransferEnabled) 1 else 0)
            put("watermarkEnabled", if (log.watermarkEnabled) 1 else 0)
            put("savedGalleryPath", log.savedGalleryPath)
            put("resultImagePath", log.resultImagePath)
        }
        val rowId = writableDatabase.insertWithOnConflict(
            "swap_audit_logs",
            null,
            values,
            SQLiteDatabase.CONFLICT_REPLACE
        )
        enforceMaxHistoryLimitSync()
        refreshLogsSync()
        rowId
    }

    override suspend fun updateSavedPath(id: Int, galleryPath: String) = withContext(Dispatchers.IO) {
        val values = ContentValues().apply {
            put("savedGalleryPath", galleryPath)
        }
        writableDatabase.update("swap_audit_logs", values, "id = ?", arrayOf(id.toString()))
        refreshLogsSync()
    }

    override suspend fun deleteById(id: Int) = withContext(Dispatchers.IO) {
        runCatching {
            readableDatabase.rawQuery(
                "SELECT resultImagePath FROM swap_audit_logs WHERE id = ?",
                arrayOf(id.toString())
            ).use { cursor ->
                if (cursor.moveToFirst() && !cursor.isNull(0)) {
                    val path = cursor.getString(0)
                    if (!path.isNullOrBlank()) {
                        runCatching { File(path).delete() }
                    }
                }
            }
        }
        writableDatabase.delete("swap_audit_logs", "id = ?", arrayOf(id.toString()))
        refreshLogsSync()
    }

    override suspend fun clearAll() = withContext(Dispatchers.IO) {
        writableDatabase.delete("swap_audit_logs", null, null)
        ImageGalleryHelper.pruneOrphanHistoryFiles(appContext, emptySet())
        refreshLogsSync()
    }

    /**
     * Strictly retains only the latest [MAX_HISTORY_RECORDS] (3) results and
     * auto-deletes older records plus their cached thumbnail image files.
     */
    private fun enforceMaxHistoryLimitSync() {
        runCatching {
            writableDatabase.execSQL(
                """
                DELETE FROM swap_audit_logs
                WHERE id NOT IN (
                    SELECT id FROM swap_audit_logs
                    ORDER BY timestampMillis DESC, id DESC
                    LIMIT $MAX_HISTORY_RECORDS
                )
                """.trimIndent()
            )
            val keepPaths = HashSet<String>()
            readableDatabase.rawQuery(
                "SELECT resultImagePath FROM swap_audit_logs WHERE resultImagePath IS NOT NULL",
                null
            ).use { cursor ->
                while (cursor.moveToNext()) {
                    val p = cursor.getString(0)
                    if (!p.isNullOrBlank()) keepPaths.add(p)
                }
            }
            ImageGalleryHelper.pruneOrphanHistoryFiles(appContext, keepPaths)
        }
    }

    private fun refreshLogsSync() {
        val result = mutableListOf<SwapAuditLog>()
        try {
            readableDatabase.rawQuery(
                "SELECT id, timestampMillis, sourceResolution, targetResolution, swappedFacesCount, " +
                    "pipelineSummary, detectionMs, embeddingMs, inswapperMs, blendingMs, totalMs, " +
                    "colorTransferEnabled, watermarkEnabled, savedGalleryPath, resultImagePath " +
                    "FROM swap_audit_logs ORDER BY timestampMillis DESC, id DESC LIMIT $MAX_HISTORY_RECORDS",
                null
            ).use { cursor ->
                while (cursor.moveToNext()) {
                    result.add(
                        SwapAuditLog(
                            id = cursor.getInt(0),
                            timestampMillis = cursor.getLong(1),
                            sourceResolution = cursor.getString(2) ?: "",
                            targetResolution = cursor.getString(3) ?: "",
                            swappedFacesCount = cursor.getInt(4),
                            pipelineSummary = cursor.getString(5) ?: "",
                            detectionMs = cursor.getLong(6),
                            embeddingMs = cursor.getLong(7),
                            inswapperMs = cursor.getLong(8),
                            blendingMs = cursor.getLong(9),
                            totalMs = cursor.getLong(10),
                            colorTransferEnabled = cursor.getInt(11) != 0,
                            watermarkEnabled = cursor.getInt(12) != 0,
                            savedGalleryPath = if (cursor.isNull(13)) null else cursor.getString(13),
                            resultImagePath = if (cursor.isNull(14)) null else cursor.getString(14)
                        )
                    )
                }
            }
        } catch (_: Throwable) {
        }
        logsFlow.value = result
    }

    companion object {
        const val MAX_HISTORY_RECORDS = 3

        @Volatile
        private var INSTANCE: SwapAuditDatabase? = null

        fun getInstance(context: Context): SwapAuditDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = SwapAuditDatabase(context)
                INSTANCE = instance
                instance
            }
        }
    }
}

class SwapAuditRepository(private val dao: SwapAuditDao) {
    val allLogs: Flow<List<SwapAuditLog>> = dao.observeAllLogs()

    suspend fun insert(log: SwapAuditLog): Int = dao.insertLog(log).toInt()

    suspend fun markSavedToGallery(id: Int, path: String) = dao.updateSavedPath(id, path)

    suspend fun deleteById(id: Int) = dao.deleteById(id)

    suspend fun clearAll() = dao.clearAll()
}
