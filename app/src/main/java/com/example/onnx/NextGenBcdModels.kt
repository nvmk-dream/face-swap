package com.example.onnx

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp

enum class NextGenBcdSlot(
    val id: String,
    val subdirName: String,
    val canonicalHint: String,
    val title: String,
    val tamilRoleSummary: String,
    val accentHex: Long
) {
    NEXTGEN_SWAPPER(
        id = "nextgen_swapper",
        subdirName = "swapper",
        canonicalHint = "reswapper_256.onnx / simswap_512.onnx",
        title = "1. ReSwapper 256 / SimSwap 512",
        tamilRoleSummary = "Option B+C & D-ல் மட்டும் பயன்படுத்தப்படும் உயர்-துல்லிய முகம் மாற்றும் மாடல் (Option A & B-ல் HyperSwap/InSwapper மாறாது).",
        accentHex = 0xFF0284C7L
    ),
    CODEFORMER(
        id = "nextgen_codeformer",
        subdirName = "codeformer",
        canonicalHint = "codeformer.onnx",
        title = "2. CodeFormer (512×512 Fidelity Restorer)",
        tamilRoleSummary = "Option B+C & D-ல் கண், உதடு மற்றும் சருமத் துளைகளை fidelity weight (w) உடன் HD தரத்தில் தெளிவாக்கும் (GFPGAN A/B-க்கு தனி).",
        accentHex = 0xFF7C3AEDL
    ),
    XSEG_OCCLUDER(
        id = "nextgen_occluder",
        subdirName = "occluder",
        canonicalHint = "face_occluder.onnx / dfl_xseg.onnx",
        title = "3. XSeg / Face Occluder (256×256 Mask)",
        tamilRoleSummary = "Option B+C & D-ல் முடி இழை, கண்ணாடி, கை மற்றும் தடைகளைத் துல்லியமாகப் பாதுகாக்கும் 256×256 Occlusion Mask.",
        accentHex = 0xFF059669L
    )
}

enum class NextGenSwapModelFamily {
    SIMSWAP,
    RESWAPPER,
    GENERIC
}

data class NextGenBcdSlotStatus(
    val slot: NextGenBcdSlot,
    val isInstalled: Boolean = false,
    val installedFile: File? = null,
    val installedFileName: String? = null,
    val fileSizeBytes: Long = 0L,
    val detailBadge: String = "Optional • Missing"
)

data class NextGenBcdModelsState(
    val swapper: NextGenBcdSlotStatus = NextGenBcdSlotStatus(NextGenBcdSlot.NEXTGEN_SWAPPER),
    val codeformer: NextGenBcdSlotStatus = NextGenBcdSlotStatus(NextGenBcdSlot.CODEFORMER),
    val occluder: NextGenBcdSlotStatus = NextGenBcdSlotStatus(NextGenBcdSlot.XSEG_OCCLUDER)
) {
    val allStatuses: List<NextGenBcdSlotStatus>
        get() = listOf(swapper, codeformer, occluder)

    val installedCount: Int
        get() = allStatuses.count { it.isInstalled }
}

/**
 * Pure-JVM tensor/classification/mask helpers for Option B+C & Option D Next-Gen ONNX models.
 * Kept pure (no Android framework dependencies) so every path can be tested on JVM directly.
 */
object NextGenBcdHelper {

    fun classifySlotForFileName(fileName: String): NextGenBcdSlot? {
        val lower = fileName.trim().lowercase()
        if (!lower.endsWith(".onnx")) return null
        return when {
            lower.contains("reswapper") || lower.contains("simswap") ->
                NextGenBcdSlot.NEXTGEN_SWAPPER
            lower.contains("codeformer") ->
                NextGenBcdSlot.CODEFORMER
            lower.contains("face_occluder") ||
                lower.contains("face-occluder") ||
                lower.contains("dfl_xseg") ||
                lower.contains("dfl-xseg") ||
                lower.contains("xseg") ||
                lower.contains("occluder") ->
                NextGenBcdSlot.XSEG_OCCLUDER
            else -> null
        }
    }

    fun classifySwapperFamily(fileName: String): NextGenSwapModelFamily {
        val lower = fileName.trim().lowercase()
        return when {
            lower.contains("simswap") -> NextGenSwapModelFamily.SIMSWAP
            lower.contains("reswapper") -> NextGenSwapModelFamily.RESWAPPER
            else -> NextGenSwapModelFamily.GENERIC
        }
    }

    fun inferSwapperNominalResolution(fileName: String): Int {
        val lower = fileName.trim().lowercase()
        return when {
            lower.contains("512") -> 512
            lower.contains("256") -> 256
            lower.contains("224") -> 224
            lower.contains("128") -> 128
            lower.contains("simswap") -> 512
            else -> 256
        }
    }

    fun computeSwapperCandidatePriority(
        fileName: String,
        sizeBytes: Long,
        hasCompanionData: Boolean
    ): Int {
        val lower = fileName.trim().lowercase()
        val base = when {
            lower.contains("simswap_512") || lower.contains("simswap-512") -> 120
            lower.contains("reswapper_256") || lower.contains("reswapper-256") -> 115
            lower.contains("simswap") -> 105
            lower.contains("reswapper") -> 100
            else -> 60
        }
        val completeBonus = if (sizeBytes >= 8L * 1024L * 1024L || hasCompanionData) 1000 else 0
        return base + completeBonus
    }

    fun computeOccluderCandidatePriority(
        fileName: String,
        sizeBytes: Long,
        hasCompanionData: Boolean
    ): Int {
        val lower = fileName.trim().lowercase()
        val base = when {
            lower.contains("dfl_xseg") || lower.contains("dfl-xseg") -> 120
            lower.contains("face_occluder") || lower.contains("face-occluder") -> 115
            lower.contains("xseg") -> 110
            else -> 90
        }
        val completeBonus = if (sizeBytes >= 1024L * 1024L || hasCompanionData) 1000 else 0
        return base + completeBonus
    }

    fun computeCodeFormerFidelityWeight(enhancementStrength: Float): Double {
        val clamped = enhancementStrength.coerceIn(0f, 1f)
        // CodeFormer 'w': 0.0 -> max restoration, 1.0 -> max identity preservation.
        // Keeping w in [0.48, 0.82] yields crisp 512px pores/lashes while preserving source identity.
        return (0.82 - 0.30 * clamped).coerceIn(0.48, 0.82)
    }

    fun buildXsegRgbInputFloats(
        pixels: IntArray,
        width: Int,
        height: Int,
        isNhwc: Boolean
    ): FloatArray {
        val hw = width * height
        val out = FloatArray(3 * hw)
        if (isNhwc) {
            for (i in 0 until hw) {
                val c = pixels[i]
                val base = i * 3
                out[base] = ((c ushr 16) and 0xFF) / 255.0f
                out[base + 1] = ((c ushr 8) and 0xFF) / 255.0f
                out[base + 2] = (c and 0xFF) / 255.0f
            }
        } else {
            for (i in 0 until hw) {
                val c = pixels[i]
                out[i] = ((c ushr 16) and 0xFF) / 255.0f
                out[hw + i] = ((c ushr 8) and 0xFF) / 255.0f
                out[2 * hw + i] = (c and 0xFF) / 255.0f
            }
        }
        return out
    }

    fun encodeImageFloats(
        pixels: IntArray,
        hw: Int,
        isNhwc: Boolean,
        useBgr: Boolean,
        normMode: NextGenInputNormMode
    ): FloatArray {
        val out = FloatArray(3 * hw)
        val meanR = 0.485f
        val meanG = 0.456f
        val meanB = 0.406f
        val stdR = 0.229f
        val stdG = 0.224f
        val stdB = 0.225f
        for (i in 0 until hw) {
            val c = pixels[i]
            val rByte = (c ushr 16) and 0xFF
            val gByte = (c ushr 8) and 0xFF
            val bByte = c and 0xFF
            val c0 = if (!useBgr) rByte else bByte
            val c1 = gByte
            val c2 = if (!useBgr) bByte else rByte

            val v0: Float
            val v1: Float
            val v2: Float
            when (normMode) {
                NextGenInputNormMode.SIGNED_MINUS1_TO_1 -> {
                    v0 = (c0 - 127.5f) / 127.5f
                    v1 = (c1 - 127.5f) / 127.5f
                    v2 = (c2 - 127.5f) / 127.5f
                }
                NextGenInputNormMode.UNSIGNED_0_TO_1 -> {
                    v0 = c0 / 255.0f
                    v1 = c1 / 255.0f
                    v2 = c2 / 255.0f
                }
                NextGenInputNormMode.IMAGENET -> {
                    val m0 = if (!useBgr) meanR else meanB
                    val s0 = if (!useBgr) stdR else stdB
                    val m2 = if (!useBgr) meanB else meanR
                    val s2 = if (!useBgr) stdB else stdR
                    v0 = ((c0 / 255.0f) - m0) / s0
                    v1 = ((c1 / 255.0f) - meanG) / stdG
                    v2 = ((c2 / 255.0f) - m2) / s2
                }
            }

            if (isNhwc) {
                val base = i * 3
                out[base] = v0
                out[base + 1] = v1
                out[base + 2] = v2
            } else {
                out[i] = v0
                out[hw + i] = v1
                out[2 * hw + i] = v2
            }
        }
        return out
    }

    /**
     * Extracts single-channel probability mask from raw model output (`[1,1,H,W]`, `[1,H,W,1]`,
     * `[1,H,W]`, or `[1,2,H,W]` logits), applies sigmoid if raw values are logits outside `[0,1]`,
     * bilinearly resizes to `512x512`, and applies a light 5x5 boundary feather.
     */
    fun upscaleXsegMaskTo512(
        outFloats: FloatArray,
        maskH: Int,
        maskW: Int,
        channels: Int = 1,
        isNhwc: Boolean = false
    ): FloatArray {
        val targetSize = 512
        val hw = maskH * maskW
        if (maskH <= 0 || maskW <= 0 || outFloats.size < hw) {
            return FloatArray(targetSize * targetSize) { 1.0f }
        }

        val probGrid = FloatArray(hw)
        if (channels == 2 && outFloats.size >= 2 * hw) {
            for (i in 0 until hw) {
                val l0 = if (isNhwc) outFloats[i * 2] else outFloats[i]
                val l1 = if (isNhwc) outFloats[i * 2 + 1] else outFloats[hw + i]
                val maxL = maxOf(l0, l1)
                val e0 = exp((l0 - maxL).toDouble())
                val e1 = exp((l1 - maxL).toDouble())
                val p = (e1 / (e0 + e1)).toFloat()
                probGrid[i] = if (p.isFinite()) p.coerceIn(0f, 1f) else 1f
            }
        } else {
            var minV = Float.MAX_VALUE
            var maxV = -Float.MAX_VALUE
            for (i in 0 until hw) {
                val raw = outFloats[i]
                if (raw.isFinite()) {
                    if (raw < minV) minV = raw
                    if (raw > maxV) maxV = raw
                }
            }
            val needsSigmoid = (minV < -0.10f || maxV > 1.10f)
            for (i in 0 until hw) {
                val raw = outFloats[i]
                val v = if (!raw.isFinite()) {
                    1f
                } else if (needsSigmoid) {
                    (1.0 / (1.0 + exp(-raw.coerceIn(-12f, 12f).toDouble()))).toFloat()
                } else {
                    raw.coerceIn(0f, 1f)
                }
                probGrid[i] = v
            }
        }

        // Bilinearly upscale HxW -> 512x512
        val upscaled = FloatArray(targetSize * targetSize)
        val scaleY = if (maskH > 1) (maskH - 1).toFloat() / (targetSize - 1).toFloat() else 0f
        val scaleX = if (maskW > 1) (maskW - 1).toFloat() / (targetSize - 1).toFloat() else 0f
        for (y in 0 until targetSize) {
            val fy = y * scaleY
            val y0 = fy.toInt().coerceIn(0, maskH - 1)
            val y1 = (y0 + 1).coerceAtMost(maskH - 1)
            val dy = fy - y0
            val row = y * targetSize
            val r0 = y0 * maskW
            val r1 = y1 * maskW
            for (x in 0 until targetSize) {
                val fx = x * scaleX
                val x0 = fx.toInt().coerceIn(0, maskW - 1)
                val x1 = (x0 + 1).coerceAtMost(maskW - 1)
                val dx = fx - x0
                val p00 = probGrid[r0 + x0]
                val p10 = probGrid[r0 + x1]
                val p01 = probGrid[r1 + x0]
                val p11 = probGrid[r1 + x1]
                val top = p00 + (p10 - p00) * dx
                val bot = p01 + (p11 - p01) * dx
                upscaled[row + x] = (top + (bot - top) * dy).coerceIn(0f, 1f)
            }
        }

        // Fast 5x5 edge feather for organic hair/hand/glasses transitions
        val smoothed = FloatArray(targetSize * targetSize)
        for (y in 0 until targetSize) {
            val y0 = (y - 2).coerceAtLeast(0)
            val y1 = (y + 2).coerceAtMost(targetSize - 1)
            val row = y * targetSize
            for (x in 0 until targetSize) {
                val x0 = (x - 2).coerceAtLeast(0)
                val x1 = (x + 2).coerceAtMost(targetSize - 1)
                var sum = 0f
                var count = 0
                for (ny in y0..y1) {
                    val nRow = ny * targetSize
                    for (nx in x0..x1) {
                        sum += upscaled[nRow + nx]
                        count++
                    }
                }
                smoothed[row + x] = (sum / count.coerceAtLeast(1)).coerceIn(0f, 1f)
            }
        }
        return smoothed
    }
}

enum class NextGenInputNormMode {
    SIGNED_MINUS1_TO_1,
    UNSIGNED_0_TO_1,
    IMAGENET
}

/**
 * Isolated storage & folder scanner for Option B+C and Option D Next-Gen ONNX models:
 * `filesDir/models/nextgen_bcd/{swapper,codeformer,occluder}/`.
 * Never touches or overwrites Option A & B's `face_swap/`, `enhancement/`, or `segmentation/` slots.
 */
object NextGenBcdModelManager {

    private const val NEXTGEN_ROOT_DIR = "nextgen_bcd"

    fun getRootDir(context: Context): File {
        val modelsRoot = OnnxProtobufInspector.getModelsRootDir(context)
        return File(modelsRoot, NEXTGEN_ROOT_DIR).apply { if (!exists()) mkdirs() }
    }

    fun getSlotDir(context: Context, slot: NextGenBcdSlot): File {
        return File(getRootDir(context), slot.subdirName).apply { if (!exists()) mkdirs() }
    }

    fun resolveActiveFile(context: Context, slot: NextGenBcdSlot): File? {
        val dir = getSlotDir(context, slot)
        val candidates = dir.listFiles()
            ?.filter { it.isFile && it.name.endsWith(".onnx", ignoreCase = true) && it.length() > 1024L }
            .orEmpty()
        if (candidates.isEmpty()) return null
        return candidates.maxByOrNull { f ->
            val companion = dir.listFiles()?.any { c ->
                c.isFile && c.name.startsWith(f.name, ignoreCase = true) &&
                    (c.name.endsWith(".data", ignoreCase = true) || c.name.endsWith(".onnx_data", ignoreCase = true)) &&
                    c.length() > 1024L
            } ?: false
            when (slot) {
                NextGenBcdSlot.NEXTGEN_SWAPPER -> NextGenBcdHelper.computeSwapperCandidatePriority(f.name, f.length(), companion)
                NextGenBcdSlot.XSEG_OCCLUDER -> NextGenBcdHelper.computeOccluderCandidatePriority(f.name, f.length(), companion)
                NextGenBcdSlot.CODEFORMER -> if (f.length() >= 8L * 1024L * 1024L || companion) 1100 else 100
            }
        }
    }

    fun inspectSlot(context: Context, slot: NextGenBcdSlot): NextGenBcdSlotStatus {
        val file = resolveActiveFile(context, slot)
        if (file == null || !file.exists() || file.length() <= 1024L) {
            return NextGenBcdSlotStatus(
                slot = slot,
                isInstalled = false,
                installedFile = null,
                installedFileName = null,
                fileSizeBytes = 0L,
                detailBadge = "Optional • Missing"
            )
        }
        val mb = "%.1f MB".format(file.length() / (1024.0 * 1024.0))
        val resBadge = when (slot) {
            NextGenBcdSlot.NEXTGEN_SWAPPER -> {
                val res = NextGenBcdHelper.inferSwapperNominalResolution(file.name)
                "B+C/D Ready • ${res}px ($mb)"
            }
            NextGenBcdSlot.CODEFORMER -> "B+C/D Ready • 512px ($mb)"
            NextGenBcdSlot.XSEG_OCCLUDER -> "B+C/D Ready • 256px ($mb)"
        }
        return NextGenBcdSlotStatus(
            slot = slot,
            isInstalled = true,
            installedFile = file,
            installedFileName = file.name,
            fileSizeBytes = file.length(),
            detailBadge = resBadge
        )
    }

    fun inspectAllSlots(context: Context): NextGenBcdModelsState {
        return NextGenBcdModelsState(
            swapper = inspectSlot(context, NextGenBcdSlot.NEXTGEN_SWAPPER),
            codeformer = inspectSlot(context, NextGenBcdSlot.CODEFORMER),
            occluder = inspectSlot(context, NextGenBcdSlot.XSEG_OCCLUDER)
        )
    }

    fun importSingleModelFromUri(
        context: Context,
        uri: Uri,
        slot: NextGenBcdSlot
    ): Result<File> {
        return try {
            val rawName = queryDisplayName(context, uri)?.trim().orEmpty()
            val sanitizedName = rawName
                .substringAfterLast('/')
                .substringAfterLast('\\')
                .ifBlank { defaultFileNameForSlot(slot) }
                .let { name -> if (name.endsWith(".onnx", ignoreCase = true)) name else "$name.onnx" }

            val slotDir = getSlotDir(context, slot)
            // Evict any cached ORT sessions inside this slot folder cleanly
            slotDir.listFiles()?.forEach { oldFile ->
                if (oldFile.isFile) {
                    OnnxProtobufInspector.evictCachedSession(oldFile)
                }
            }

            val tempFile = File(slotDir, "$sanitizedName.tmp")
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read <= 0) break
                        output.write(buffer, 0, read)
                    }
                    output.flush()
                }
            } ?: error("Unable to open selected ONNX file.")

            if (tempFile.length() < 1024L) {
                tempFile.delete()
                error("Selected file is too small (${tempFile.length()} bytes) to be a valid ONNX model.")
            }

            // Replace old files in this dedicated B+C/D sub-slot so resolveActiveFile picks the newly uploaded model
            val destFile = File(slotDir, sanitizedName)
            slotDir.listFiles()?.forEach { old ->
                if (old.isFile && old.name != tempFile.name && old.name != destFile.name) {
                    old.delete()
                }
            }
            if (destFile.exists()) destFile.delete()
            if (!tempFile.renameTo(destFile)) {
                tempFile.copyTo(destFile, overwrite = true)
                tempFile.delete()
            }
            OnnxProtobufInspector.clearCrashGuardForFile(destFile, context)
            Result.success(destFile)
        } catch (oom: OutOfMemoryError) {
            System.gc()
            Result.failure(IllegalStateException("Out of memory while importing ${slot.canonicalHint}"))
        } catch (t: Throwable) {
            Result.failure(t)
        }
    }

    fun deleteSlotModel(context: Context, slot: NextGenBcdSlot): Boolean {
        val dir = getSlotDir(context, slot)
        val files = dir.listFiles().orEmpty()
        var removed = false
        for (f in files) {
            if (f.isFile) {
                OnnxProtobufInspector.evictCachedSession(f)
                if (f.delete()) removed = true
            }
        }
        return removed
    }

    data class ScannedCandidate(
        val docId: String,
        val displayName: String,
        val sizeBytes: Long,
        val slot: NextGenBcdSlot,
        val priority: Int
    )

    internal fun importFromTreeScan(
        context: Context,
        treeUri: Uri,
        discovered: List<ScannedCandidate>,
        dataFilesByName: Map<String, Pair<String, Long>>,
        onProgress: (String) -> Unit = {}
    ): List<NextGenBcdSlot> {
        if (discovered.isEmpty()) return emptyList()
        val bestPerSlot = mutableMapOf<NextGenBcdSlot, ScannedCandidate>()
        for (entry in discovered) {
            val prev = bestPerSlot[entry.slot]
            if (prev == null || entry.priority > prev.priority) {
                bestPerSlot[entry.slot] = entry
            }
        }
        val imported = mutableListOf<NextGenBcdSlot>()
        val rootDir = getRootDir(context)
        for ((slot, entry) in bestPerSlot) {
            val slotDir = getSlotDir(context, slot)
            val destFile = File(slotDir, entry.displayName.trim())
            val companion = findCompanionData(entry.displayName, dataFilesByName)
            val compDest = companion?.first?.let { File(slotDir, it) }

            val mainMatches = entry.sizeBytes > 1024L && destFile.exists() && destFile.length() == entry.sizeBytes
            val compMatches = companion == null || (compDest != null && compDest.exists() &&
                (companion.second.second <= 0L || compDest.length() == companion.second.second))
            if (mainMatches && compMatches) {
                imported.add(slot)
                continue
            }

            val needBytes = entry.sizeBytes.coerceAtLeast(0L) + (companion?.second?.second ?: 0L)
            if (needBytes > 0L && rootDir.usableSpace < needBytes + 48L * 1024L * 1024L) {
                continue
            }

            onProgress("Importing B+C/D model: ${entry.displayName}...")
            val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, entry.docId)
            val res = importSingleModelFromUri(context, docUri, slot)
            if (res.isSuccess) {
                val copiedFile = res.getOrNull()
                if (companion != null && copiedFile != null) {
                    val compUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, companion.second.first)
                    val targetComp = File(slotDir, companion.first)
                    runCatching { copyStreamToFile(context, compUri, targetComp) }
                }
                imported.add(slot)
            }
        }
        return imported
    }

    private fun findCompanionData(
        displayName: String,
        dataFilesByName: Map<String, Pair<String, Long>>
    ): Pair<String, Pair<String, Long>>? {
        val lower = displayName.trim().lowercase()
        val stem = lower.removeSuffix(".onnx")
        val candidates = listOf(
            "$lower.data",
            "${lower}_data",
            "$stem.onnx.data",
            "$stem.onnx_data",
            "$stem.data"
        )
        for (c in candidates) {
            val found = dataFilesByName[c]
            if (found != null) return c to found
        }
        return null
    }

    private fun defaultFileNameForSlot(slot: NextGenBcdSlot): String {
        return when (slot) {
            NextGenBcdSlot.NEXTGEN_SWAPPER -> "reswapper_256.onnx"
            NextGenBcdSlot.CODEFORMER -> "codeformer.onnx"
            NextGenBcdSlot.XSEG_OCCLUDER -> "face_occluder.onnx"
        }
    }

    private fun copyStreamToFile(context: Context, uri: Uri, destFile: File) {
        val tmp = File(destFile.parentFile, "${destFile.name}.tmp")
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(tmp).use { output ->
                val buf = ByteArray(256 * 1024)
                while (true) {
                    val read = input.read(buf)
                    if (read <= 0) break
                    output.write(buf, 0, read)
                }
                output.flush()
            }
        } ?: error("Unable to open stream for ${destFile.name}")
        if (destFile.exists()) destFile.delete()
        if (!tmp.renameTo(destFile)) {
            tmp.copyTo(destFile, overwrite = true)
            tmp.delete()
        }
    }

    private fun queryDisplayName(context: Context, uri: Uri): String? {
        return runCatching {
            context.contentResolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        }.getOrNull()
    }
}
