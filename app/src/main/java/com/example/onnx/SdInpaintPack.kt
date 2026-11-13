package com.example.onnx

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import java.io.File
import java.io.FileOutputStream

enum class SdInpaintPackType(
    val id: String,
    val displayLabel: String,
    val priorityOrder: Int
) {
    SD15_FP16("sd15_fp16", "SD 1.5 Inpaint (FP16)", 1),
    REALVIS_INT8("realvis51_int8", "RealVis 5.1 (INT8)", 2),
    CUSTOM_SD("custom_sd", "SD 1.5 Inpaint (Custom)", 3)
}

enum class InpaintModelOption(
    val id: String,
    val displayName: String,
    val shortBadge: String
) {
    AUTO("auto", "Auto (FP16 > INT8)", "AUTO"),
    SD15_FP16("sd15_fp16", "SD 1.5 Inpaint (FP16)", "FP16"),
    REALVIS_INT8("realvis51_int8", "RealVis 5.1 (INT8)", "INT8"),
    LAMA_ONNX("lama_fp32", "LaMa (Object Remove)", "LAMA"),
    CV_FALLBACK("cv_fallback", "Fast Offline CV", "CV")
}

data class InstalledSdInpaintPack(
    val packType: SdInpaintPackType,
    val packDir: File,
    val textEncoderOnnx: File,
    val unetOnnx: File,
    val vaeEncoderOnnx: File,
    val vaeDecoderOnnx: File,
    val tokenizerDir: File?,
    val totalSizeBytes: Long
) {
    val isComplete: Boolean
        get() = textEncoderOnnx.exists() && textEncoderOnnx.length() > 1024L &&
            unetOnnx.exists() && unetOnnx.length() > 1024L &&
            vaeEncoderOnnx.exists() && vaeEncoderOnnx.length() > 1024L &&
            vaeDecoderOnnx.exists() && vaeDecoderOnnx.length() > 1024L
}

object SdInpaintPackManager {

    private const val SD_INPAINT_ROOT = "sd_inpaint"
    private val COMPONENT_DIRS = setOf("unet", "text_encoder", "vae_encoder", "vae_decoder")
    private val TOKENIZER_FILES = setOf(
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    )

    fun getSdInpaintRootDir(context: Context): File {
        val modelsRoot = OnnxProtobufInspector.getModelsRootDir(context)
        return File(modelsRoot, SD_INPAINT_ROOT).apply {
            if (!exists()) mkdirs()
        }
    }

    fun classifyPackType(folderName: String): SdInpaintPackType {
        val lower = folderName.trim().lowercase()
        return when {
            lower.contains("int8") ||
                lower.contains("qint8") ||
                lower.contains("quint8") ||
                lower.contains("q8") ||
                lower.contains("realvis") -> SdInpaintPackType.REALVIS_INT8

            lower.contains("fp16") ||
                lower.contains("f16") ||
                lower.contains("half") ||
                lower.contains("v1-5-inpainting") ||
                lower.contains("v1.5-inpainting") ||
                lower.contains("sd-1-5-inpaint") ||
                lower.contains("sd15") -> SdInpaintPackType.SD15_FP16

            else -> SdInpaintPackType.CUSTOM_SD
        }
    }

    /**
     * Returns all valid installed SD 1.5 Inpainting packs ordered by user priority:
     * 1) `SD15_FP16` (Primary)
     * 2) `REALVIS_INT8` (Second option)
     * 3) `CUSTOM_SD`
     */
    fun discoverInstalledPacks(context: Context): List<InstalledSdInpaintPack> {
        val root = getSdInpaintRootDir(context)
        val packs = mutableListOf<InstalledSdInpaintPack>()
        val subDirs = root.listFiles()?.filter { it.isDirectory } ?: return emptyList()

        for (dir in subDirs) {
            val packType = SdInpaintPackType.entries.firstOrNull { it.id == dir.name }
                ?: classifyPackType(dir.name)
            val textEnc = selectBestOnnxInComponentDir(File(dir, "text_encoder"), packType) ?: continue
            val unet = selectBestOnnxInComponentDir(File(dir, "unet"), packType) ?: continue
            val vaeEnc = selectBestOnnxInComponentDir(File(dir, "vae_encoder"), packType) ?: continue
            val vaeDec = selectBestOnnxInComponentDir(File(dir, "vae_decoder"), packType) ?: continue
            val tokDir = File(dir, "tokenizer").takeIf { it.exists() && it.isDirectory }
            val totalBytes = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }

            val candidate = InstalledSdInpaintPack(
                packType = packType,
                packDir = dir,
                textEncoderOnnx = textEnc,
                unetOnnx = unet,
                vaeEncoderOnnx = vaeEnc,
                vaeDecoderOnnx = vaeDec,
                tokenizerDir = tokDir,
                totalSizeBytes = totalBytes
            )
            if (candidate.isComplete) {
                packs.add(candidate)
            }
        }
        return packs.sortedBy { it.packType.priorityOrder }
    }

    fun resolvePackForOption(
        context: Context,
        option: InpaintModelOption
    ): InstalledSdInpaintPack? {
        val installed = discoverInstalledPacks(context)
        if (installed.isEmpty()) return null
        return when (option) {
            InpaintModelOption.SD15_FP16 ->
                installed.firstOrNull { it.packType == SdInpaintPackType.SD15_FP16 }
            InpaintModelOption.REALVIS_INT8 ->
                installed.firstOrNull { it.packType == SdInpaintPackType.REALVIS_INT8 }
            InpaintModelOption.AUTO ->
                installed.firstOrNull() // Already sorted: FP16 (1) > INT8 (2) > Custom (3)
            InpaintModelOption.LAMA_ONNX, InpaintModelOption.CV_FALLBACK -> null
        }
    }

    /**
     * Computes the default model option following user priority:
     * Priority 1: SD 1.5 FP16 -> Priority 2: RealVis 5.1 INT8 -> LaMa -> Coming Soon / CV
     */
    fun determineDefaultModelOption(
        context: Context,
        lamaFile: File?
    ): InpaintModelOption {
        val installed = discoverInstalledPacks(context)
        if (installed.any { it.packType == SdInpaintPackType.SD15_FP16 }) {
            return InpaintModelOption.SD15_FP16
        }
        if (installed.any { it.packType == SdInpaintPackType.REALVIS_INT8 }) {
            return InpaintModelOption.REALVIS_INT8
        }
        if (installed.isNotEmpty()) {
            return InpaintModelOption.AUTO
        }
        if (lamaFile != null && lamaFile.exists() && lamaFile.length() > 1024L) {
            return InpaintModelOption.LAMA_ONNX
        }
        return InpaintModelOption.AUTO
    }

    fun formatDropdownLabel(
        context: Context,
        option: InpaintModelOption,
        lamaFile: File?
    ): String {
        val installed = discoverInstalledPacks(context)
        val hasLama = lamaFile != null && lamaFile.exists() && lamaFile.length() > 1024L
        if (installed.isEmpty() && !hasLama && option == InpaintModelOption.AUTO) {
            return "Coming Soon"
        }
        return when (option) {
            InpaintModelOption.AUTO -> {
                installed.firstOrNull()?.packType?.displayLabel
                    ?: if (hasLama) InpaintModelOption.LAMA_ONNX.displayName else "Coming Soon"
            }
            else -> option.displayName
        }
    }

    /**
     * When a component folder (`unet/`, `text_encoder/`, `vae_encoder/`, `vae_decoder/`) contains
     * 2 or 3 `.onnx` / `.data` files (e.g., `model.onnx` + `model_int8.onnx.data`, or both
     * `model.onnx` and `model_fp16.onnx`), deterministically select the right valid `.onnx` file.
     */
    fun selectBestOnnxInComponentDir(
        compDir: File,
        packType: SdInpaintPackType
    ): File? {
        if (!compDir.exists() || !compDir.isDirectory) return null
        val files = compDir.listFiles()?.filter { it.isFile } ?: return null
        val onnxFiles = files.filter { it.name.endsWith(".onnx", ignoreCase = true) && it.length() > 512L }
        if (onnxFiles.isEmpty()) return null
        if (onnxFiles.size == 1) return onnxFiles.first()

        val dataFileNames = files
            .filter {
                val lower = it.name.lowercase()
                lower.endsWith(".data") || lower.endsWith(".onnx_data") || lower.endsWith(".pb")
            }
            .map { it.name }

        return onnxFiles.maxByOrNull { onnx ->
            scoreOnnxCandidate(onnx, dataFileNames, packType)
        }
    }

    internal fun scoreOnnxCandidate(
        onnxFile: File,
        siblingDataNames: List<String>,
        packType: SdInpaintPackType
    ): Int {
        val lower = onnxFile.name.lowercase()
        var score = 0

        // Check if this .onnx header/proto actually references any sibling external data file
        val referencesExternalData = siblingDataNames.isNotEmpty() &&
            fileContainsAnyAsciiToken(onnxFile, siblingDataNames)
        if (referencesExternalData) {
            score += 2500
        }

        // Check direct stem match (e.g. model.onnx <-> model.onnx.data, or model_fp16.onnx <-> model_fp16.onnx.data)
        val hasExactStemData = siblingDataNames.any { dataName ->
            dataName.startsWith(onnxFile.name, ignoreCase = true)
        }
        if (hasExactStemData) {
            score += 1800
        }

        // Self-contained larger graph (> 5 MB)
        if (onnxFile.length() >= 5L * 1024L * 1024L) {
            score += 800
        }

        when (packType) {
            SdInpaintPackType.SD15_FP16 -> {
                if (lower.contains("fp16") || lower.contains("f16")) score += 450
                if (lower == "model.onnx") score += 350
                if (lower.contains("int8") || lower.contains("quant")) score -= 300
            }
            SdInpaintPackType.REALVIS_INT8 -> {
                if (lower.contains("int8") || lower.contains("quant")) score += 450
                if (lower == "model.onnx") score += 400
                if (lower.contains("fp16")) score -= 200
            }
            SdInpaintPackType.CUSTOM_SD -> {
                if (lower == "model.onnx") score += 400
            }
        }

        // Prefer unoptimized/standard over broken tiny stubs
        if (onnxFile.length() < 16L * 1024L && siblingDataNames.isEmpty()) {
            score -= 1000
        }
        return score
    }

    private fun fileContainsAnyAsciiToken(file: File, tokens: List<String>): Boolean {
        if (tokens.isEmpty() || !file.exists()) return false
        return runCatching {
            val maxScanBytes = minOf(file.length(), 8L * 1024L * 1024L).toInt()
            val bytes = ByteArray(maxScanBytes)
            file.inputStream().use { input ->
                var offset = 0
                while (offset < maxScanBytes) {
                    val read = input.read(bytes, offset, maxScanBytes - offset)
                    if (read <= 0) break
                    offset += read
                }
            }
            val text = String(bytes, Charsets.ISO_8859_1)
            tokens.any { token -> text.contains(token, ignoreCase = false) }
        }.getOrDefault(false)
    }

    private data class TreeNodeFile(
        val docId: String,
        val displayName: String,
        val sizeBytes: Long,
        val parentSegments: List<String>
    )

    /**
     * Scans a user-selected SAF folder (e.g. `Phone > face swap/`) up to depth 4 for
     * Stable Diffusion 1.5 Inpainting ONNX packs (`stable-diffusion-v1-5-inpainting-onnx-fp16`
     * as Priority 1, `realvis51-inpaint-int8-onnx` as Priority 2).
     * Preserves exact external `.data` filenames alongside `.onnx` files inside each component folder
     * (`text_encoder/`, `unet/`, `vae_encoder/`, `vae_decoder/`, `tokenizer/`).
     */
    fun importSdInpaintPacksFromTreeUri(
        context: Context,
        treeUri: Uri,
        onProgress: (String) -> Unit = {}
    ): List<SdInpaintPackType> {
        val resolver = context.contentResolver
        val rootDocId = DocumentsContract.getTreeDocumentId(treeUri)
        val rootDocName = queryDocDisplayName(context, treeUri, rootDocId).orEmpty()

        val collectedFiles = mutableListOf<TreeNodeFile>()

        fun walk(parentDocId: String, pathSegments: List<String>, depth: Int) {
            if (depth > 4) return
            val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentDocId)
            val projection = arrayOf(
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
                DocumentsContract.Document.COLUMN_SIZE
            )
            runCatching {
                resolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
                    val idIdx = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
                    val nameIdx = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
                    val mimeIdx = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_MIME_TYPE)
                    val sizeIdx = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_SIZE)
                    while (cursor.moveToNext()) {
                        val docId = if (idIdx >= 0) cursor.getString(idIdx) else continue
                        val displayName = if (nameIdx >= 0) cursor.getString(nameIdx).orEmpty() else ""
                        val mimeType = if (mimeIdx >= 0) cursor.getString(mimeIdx).orEmpty() else ""
                        val docSize = if (sizeIdx >= 0 && !cursor.isNull(sizeIdx)) cursor.getLong(sizeIdx) else 0L

                        if (mimeType == DocumentsContract.Document.MIME_TYPE_DIR) {
                            val dirLower = displayName.trim().lowercase()
                            // Skip safety_checker / feature_extractor ONNX graphs to save >1GB of phone storage
                            if (dirLower == "safety_checker" || dirLower == "feature_extractor") {
                                continue
                            }
                            walk(docId, pathSegments + displayName.trim(), depth + 1)
                        } else {
                            collectedFiles.add(
                                TreeNodeFile(
                                    docId = docId,
                                    displayName = displayName.trim(),
                                    sizeBytes = docSize,
                                    parentSegments = pathSegments
                                )
                            )
                        }
                    }
                }
            }
        }

        walk(rootDocId, emptyList(), 0)
        if (collectedFiles.isEmpty()) return emptyList()

        // Group files belonging to component dirs (`unet`, `text_encoder`, `vae_encoder`, `vae_decoder`, `tokenizer`)
        // by their parent pack folder path.
        data class PackBucketKey(val packFolderPath: String, val packFolderName: String)
        val packBuckets = mutableMapOf<PackBucketKey, MutableList<Pair<String, TreeNodeFile>>>()

        for (file in collectedFiles) {
            val segs = file.parentSegments
            val compIdx = segs.indexOfLast {
                val l = it.lowercase()
                l in COMPONENT_DIRS || l == "tokenizer"
            }
            if (compIdx < 0) continue
            val compName = segs[compIdx].lowercase()
            val packSegs = segs.take(compIdx)
            val packFolderPath = if (packSegs.isEmpty()) "__root__" else packSegs.joinToString("/")
            val packFolderName = packSegs.lastOrNull() ?: rootDocName.ifBlank { "sd_inpaint" }
            val key = PackBucketKey(packFolderPath, packFolderName)
            packBuckets.getOrPut(key) { mutableListOf() }.add(compName to file)
        }

        if (packBuckets.isEmpty()) return emptyList()

        // Only consider buckets that contain at least unet + text_encoder + vae_encoder + vae_decoder
        val completeBuckets = packBuckets.entries
            .mapNotNull { (key, compFiles) ->
                val compsPresent = compFiles
                    .filter { (comp, f) -> comp in COMPONENT_DIRS && f.displayName.endsWith(".onnx", ignoreCase = true) }
                    .map { it.first }
                    .toSet()
                if (!compsPresent.containsAll(COMPONENT_DIRS)) {
                    null
                } else {
                    val packType = classifyPackType(key.packFolderName)
                    Triple(packType, key, compFiles)
                }
            }
            .sortedBy { it.first.priorityOrder } // Priority 1: FP16 first, Priority 2: INT8 second!

        if (completeBuckets.isEmpty()) return emptyList()

        val sdRoot = getSdInpaintRootDir(context)
        val importedTypes = mutableListOf<SdInpaintPackType>()

        for ((packType, _, compFiles) in completeBuckets) {
            // Skip duplicate packType if a higher-priority folder for the same type was already imported in this scan
            if (packType in importedTypes) continue

            val packDestDir = File(sdRoot, packType.id).apply { if (!exists()) mkdirs() }
            val filesToCopy = filterNeededComponentFiles(compFiles)
            val totalNeededBytes = filesToCopy.sumOf { (_, f) ->
                val targetFile = File(File(packDestDir, itCompDirName(f, compFiles)), f.displayName)
                if (targetFile.exists() && f.sizeBytes > 0L && targetFile.length() == f.sizeBytes) 0L else f.sizeBytes.coerceAtLeast(0L)
            }

            if (totalNeededBytes > 0L && sdRoot.usableSpace < totalNeededBytes + 80L * 1024L * 1024L) {
                onProgress("Skipped ${packType.displayLabel} (low storage space).")
                continue
            }

            var copySucceeded = true
            filesToCopy.forEachIndexed { index, (compName, node) ->
                val compDestDir = File(packDestDir, compName).apply { if (!exists()) mkdirs() }
                val destFile = File(compDestDir, node.displayName)
                if (destFile.exists() && node.sizeBytes > 512L && destFile.length() == node.sizeBytes) {
                    return@forEachIndexed
                }
                onProgress(
                    "Importing ${packType.displayLabel} (${index + 1}/${filesToCopy.size}): $compName/${node.displayName}..."
                )
                val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, node.docId)
                val ok = runCatching {
                    copyUriToFile(context, docUri, destFile)
                }.isSuccess
                if (!ok && node.displayName.endsWith(".onnx", ignoreCase = true)) {
                    copySucceeded = false
                }
            }

            if (copySucceeded) {
                // Verify pack is usable
                val textEnc = selectBestOnnxInComponentDir(File(packDestDir, "text_encoder"), packType)
                val unet = selectBestOnnxInComponentDir(File(packDestDir, "unet"), packType)
                val vaeEnc = selectBestOnnxInComponentDir(File(packDestDir, "vae_encoder"), packType)
                val vaeDec = selectBestOnnxInComponentDir(File(packDestDir, "vae_decoder"), packType)
                if (textEnc != null && unet != null && vaeEnc != null && vaeDec != null) {
                    importedTypes.add(packType)
                }
            }
        }

        return importedTypes
    }

    private fun itCompDirName(
        node: TreeNodeFile,
        compFiles: List<Pair<String, TreeNodeFile>>
    ): String {
        return compFiles.firstOrNull { it.second === node }?.first ?: "unet"
    }

    private fun filterNeededComponentFiles(
        compFiles: List<Pair<String, TreeNodeFile>>
    ): List<Pair<String, TreeNodeFile>> {
        val result = mutableListOf<Pair<String, TreeNodeFile>>()
        for ((comp, node) in compFiles) {
            val lower = node.displayName.lowercase()
            val keep = when (comp) {
                "tokenizer" -> lower in TOKENIZER_FILES
                in COMPONENT_DIRS ->
                    lower.endsWith(".onnx") ||
                        lower.endsWith(".onnx.data") ||
                        lower.endsWith(".data") ||
                        lower.endsWith(".onnx_data") ||
                        lower.endsWith(".pb") ||
                        lower.endsWith(".bin")
                else -> false
            }
            if (keep) {
                result.add(comp to node)
            }
        }
        return result
    }

    private fun queryDocDisplayName(context: Context, treeUri: Uri, docId: String): String? {
        return runCatching {
            val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, docId)
            context.contentResolver.query(
                docUri,
                arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME),
                null,
                null,
                null
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        }.getOrNull()
    }

    private fun copyUriToFile(context: Context, uri: Uri, destFile: File) {
        val tempFile = File(destFile.parentFile, "${destFile.name}.tmp")
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
        } ?: error("Unable to open stream for ${destFile.name}")
        if (destFile.exists()) destFile.delete()
        if (!tempFile.renameTo(destFile)) {
            tempFile.copyTo(destFile, overwrite = true)
            tempFile.delete()
        }
    }
}
