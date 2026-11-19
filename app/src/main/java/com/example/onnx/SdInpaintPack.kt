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
    SD15_FP32("sd15_fp32", "SD 1.5 Inpaint (FP32)", 1),
    REALVIS_INT8("realvis51_int8", "RealVis 5.1 (INT8)", 2),
    CUSTOM_SD("custom_sd", "SD 1.5 Inpaint (Custom)", 3),
    SD15_FP16("sd15_fp16", "SD 1.5 Inpaint (FP16)", 4)
}

enum class InpaintModelOption(
    val id: String,
    val displayName: String,
    val shortBadge: String
) {
    AUTO("auto", "Auto (FP32 / INT8 > FP16)", "AUTO"),
    SD15_FP32("sd15_fp32", "SD 1.5 Inpaint (FP32)", "FP32"),
    REALVIS_INT8("realvis51_int8", "RealVis 5.1 (INT8)", "INT8"),
    SD15_FP16("sd15_fp16", "SD 1.5 Inpaint (FP16)", "FP16"),
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
            vaeDecoderOnnx.exists() && vaeDecoderOnnx.length() > 1024L &&
            !hasMissingExternalWeights()

    fun hasMissingExternalWeights(): Boolean {
        val components = listOf(textEncoderOnnx, unetOnnx, vaeEncoderOnnx, vaeDecoderOnnx)
        return components.any { onnxFile ->
            SdInpaintPackManager.isMissingCompanionExternalData(onnxFile)
        }
    }

    val hasTokenizer: Boolean
        get() = SdInpaintPackManager.hasTokenizerFiles(tokenizerDir)

    val isReadyWithTokenizer: Boolean
        get() = isComplete && hasTokenizer
}

data class ClipTokenizerStatus(
    val hasVocabJson: Boolean = false,
    val hasMergesTxt: Boolean = false,
    val hasTokenizerJson: Boolean = false,
    val isReady: Boolean = false,
    val activeSourceLabel: String = "Missing"
)

data class ClipTokenizerImportResult(
    val importedFileNames: List<String>,
    val skippedFileNames: List<String>,
    val status: ClipTokenizerStatus,
    val tokenizerVerifiedLoaded: Boolean
)

object SdInpaintPackManager {

    private const val SD_INPAINT_ROOT = "sd_inpaint"
    private const val SHARED_TOKENIZER_DIR = "_shared_tokenizer"
    private val COMPONENT_DIRS = setOf("unet", "text_encoder", "vae_encoder", "vae_decoder")
    private val TOKENIZER_FOLDER_NAMES = setOf("tokenizer", "clip_tokenizer", "tokenizer_1", "tokenizer_2")
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

    fun getSharedTokenizerDir(context: Context): File {
        return File(getSdInpaintRootDir(context), SHARED_TOKENIZER_DIR).apply {
            if (!exists()) mkdirs()
        }
    }

    fun classifyPackType(folderName: String): SdInpaintPackType {
        val lower = folderName.trim().lowercase()
        return when {
            lower.contains("fp32") ||
                lower.contains("f32") ||
                lower.contains("float32") -> SdInpaintPackType.SD15_FP32

            lower.contains("int8") ||
                lower.contains("qint8") ||
                lower.contains("quint8") ||
                lower.contains("q8") ||
                lower.contains("realvis") -> SdInpaintPackType.REALVIS_INT8

            lower.contains("fp16") ||
                lower.contains("f16") ||
                lower.contains("half") -> SdInpaintPackType.SD15_FP16

            lower.contains("v1-5-inpainting") ||
                lower.contains("v1.5-inpainting") ||
                lower.contains("sd-1-5-inpaint") ||
                lower.contains("sd15") -> SdInpaintPackType.SD15_FP32

            else -> SdInpaintPackType.CUSTOM_SD
        }
    }

    /**
     * Returns all valid installed SD 1.5 Inpainting packs ordered by CPU speed & quality priority:
     * 1) `SD15_FP32` (Primary CPU)
     * 2) `REALVIS_INT8` (Fast Quantized CPU)
     * 3) `CUSTOM_SD`
     * 4) `SD15_FP16` (Secondary fallback on CPU)
     */
    fun discoverInstalledPacks(context: Context): List<InstalledSdInpaintPack> {
        val root = getSdInpaintRootDir(context)
        val sharedTokDir = getSharedTokenizerDir(context)
        val packs = mutableListOf<InstalledSdInpaintPack>()
        val subDirs = root.listFiles()
            ?.filter { it.isDirectory && !it.name.startsWith("_") }
            ?: return emptyList()

        val siblingTokenizerDir by lazy {
            findAnyValidTokenizerDirUnderRoot(root)
        }

        for (dir in subDirs) {
            runCatching {
                val packType = SdInpaintPackType.entries.firstOrNull { it.id == dir.name }
                    ?: classifyPackType(dir.name)
                val textEnc = selectBestOnnxInComponentDir(File(dir, "text_encoder"), packType) ?: return@runCatching
                val unet = selectBestOnnxInComponentDir(File(dir, "unet"), packType) ?: return@runCatching
                val vaeEnc = selectBestOnnxInComponentDir(File(dir, "vae_encoder"), packType) ?: return@runCatching
                val vaeDec = selectBestOnnxInComponentDir(File(dir, "vae_decoder"), packType) ?: return@runCatching
                val tokSub = File(dir, "tokenizer")
                val tokDir = when {
                    hasTokenizerFiles(tokSub) -> tokSub
                    hasTokenizerFiles(dir) -> dir
                    hasTokenizerFiles(sharedTokDir) -> sharedTokDir
                    siblingTokenizerDir != null -> siblingTokenizerDir
                    else -> tokSub.takeIf { it.exists() && it.isDirectory }
                }
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
        }
        return packs.sortedBy { it.packType.priorityOrder }
    }

    private fun findAnyValidTokenizerDirUnderRoot(root: File): File? {
        val dirs = root.listFiles()?.filter { it.isDirectory && !it.name.startsWith("_") }.orEmpty()
        for (d in dirs) {
            val sub = File(d, "tokenizer")
            if (hasTokenizerFiles(sub)) return sub
            if (hasTokenizerFiles(d)) return d
        }
        return null
    }

    private fun syncPackAndSharedTokenizers(context: Context): File {
        val root = getSdInpaintRootDir(context)
        val sharedDir = getSharedTokenizerDir(context)
        val packDirs = root.listFiles()
            ?.filter { it.isDirectory && !it.name.startsWith("_") }
            .orEmpty()

        // Step 1: Collect any tokenizer files found in pack directories into _shared_tokenizer
        for (packDir in packDirs) {
            val candidateSubDirs = listOf(File(packDir, "tokenizer"), packDir)
            for (srcDir in candidateSubDirs) {
                if (!srcDir.exists() || !srcDir.isDirectory) continue
                for (name in TOKENIZER_FILES) {
                    val srcFile = File(srcDir, name)
                    val sharedFile = File(sharedDir, name)
                    if (srcFile.exists() && srcFile.length() > 16L &&
                        (!sharedFile.exists() || sharedFile.length() <= 16L)
                    ) {
                        runCatching { srcFile.copyTo(sharedFile, overwrite = true) }
                    }
                }
            }
        }

        // Step 2: Mirror valid tokenizer files in _shared_tokenizer back to all pack tokenizer/ dirs
        for (name in TOKENIZER_FILES) {
            val sharedFile = File(sharedDir, name)
            if (!sharedFile.exists() || sharedFile.length() <= 16L) continue
            for (packDir in packDirs) {
                val tokDir = File(packDir, "tokenizer").apply { if (!exists()) mkdirs() }
                val target = File(tokDir, name)
                if (!target.exists() || target.length() != sharedFile.length()) {
                    runCatching { sharedFile.copyTo(target, overwrite = true) }
                }
            }
        }
        return sharedDir
    }

    fun inspectClipTokenizerStatus(context: Context): ClipTokenizerStatus {
        val root = getSdInpaintRootDir(context)
        val sharedDir = syncPackAndSharedTokenizers(context)
        val candidateDirs = buildList {
            add(sharedDir)
            root.listFiles()
                ?.filter { it.isDirectory && !it.name.startsWith("_") }
                ?.forEach { packDir ->
                    add(File(packDir, "tokenizer"))
                    add(packDir)
                }
        }

        // First look for a complete directory
        for (dir in candidateDirs) {
            if (hasTokenizerFiles(dir)) {
                val hasVocab = File(dir, "vocab.json").let { it.exists() && it.length() > 16L }
                val hasMerges = File(dir, "merges.txt").let { it.exists() && it.length() > 16L }
                val hasTokJson = File(dir, "tokenizer.json").let { it.exists() && it.length() > 16L }
                val source = when {
                    dir == sharedDir -> "Shared CLIP Tokenizer"
                    dir.name == "tokenizer" -> "${dir.parentFile?.name ?: "SD"}/tokenizer"
                    else -> dir.name
                }
                return ClipTokenizerStatus(
                    hasVocabJson = hasVocab,
                    hasMergesTxt = hasMerges,
                    hasTokenizerJson = hasTokJson,
                    isReady = true,
                    activeSourceLabel = source
                )
            }
        }

        // Aggregate partial files (e.g. vocab.json uploaded first, waiting for merges.txt)
        var hasVocab = false
        var hasMerges = false
        var hasTokJson = false
        for (dir in candidateDirs) {
            if (!dir.exists() || !dir.isDirectory) continue
            if (File(dir, "vocab.json").let { it.exists() && it.length() > 16L }) hasVocab = true
            if (File(dir, "merges.txt").let { it.exists() && it.length() > 16L }) hasMerges = true
            if (File(dir, "tokenizer.json").let { it.exists() && it.length() > 16L }) hasTokJson = true
        }
        val source = when {
            hasVocab && !hasMerges && !hasTokJson -> "vocab.json ready — add merges.txt"
            hasMerges && !hasVocab && !hasTokJson -> "merges.txt ready — add vocab.json"
            else -> "Missing (vocab.json + merges.txt or tokenizer.json)"
        }
        return ClipTokenizerStatus(
            hasVocabJson = hasVocab,
            hasMergesTxt = hasMerges,
            hasTokenizerJson = hasTokJson,
            isReady = false,
            activeSourceLabel = source
        )
    }

    fun importTokenizerFilesFromUris(
        context: Context,
        uris: List<Uri>,
        expectedCanonicalHint: String? = null
    ): Result<ClipTokenizerImportResult> {
        if (uris.isEmpty()) {
            return Result.failure(IllegalArgumentException("No files selected."))
        }
        return runCatching {
            val sharedDir = syncPackAndSharedTokenizers(context)
            val imported = mutableListOf<String>()
            val skipped = mutableListOf<String>()

            for (uri in uris) {
                val rawName = queryAnyDisplayName(context, uri).orEmpty().trim()
                val label = rawName.ifBlank { uri.lastPathSegment ?: "selected_file" }
                val canonicalName = resolveCanonicalTokenizerFileName(
                    context = context,
                    uri = uri,
                    rawDisplayName = rawName,
                    expectedCanonicalHint = if (uris.size == 1) expectedCanonicalHint else null
                )
                if (canonicalName == null) {
                    skipped.add("$label (unrecognized format)")
                    continue
                }
                val destShared = File(sharedDir, canonicalName)
                val stageTemp = File(sharedDir, "$canonicalName.validate.tmp")
                copyUriToFile(context, uri, stageTemp)

                // Quick format sanity check so invalid/empty JSON does not corrupt an existing valid file
                val validContent = validateTokenizerFileContent(canonicalName, stageTemp)
                if (validContent != null) {
                    stageTemp.delete()
                    skipped.add("$label ($validContent)")
                    continue
                }
                if (destShared.exists()) destShared.delete()
                if (!stageTemp.renameTo(destShared)) {
                    stageTemp.copyTo(destShared, overwrite = true)
                    stageTemp.delete()
                }
                imported.add(canonicalName)
            }

            if (imported.isEmpty()) {
                error(
                    "Could not import ${skipped.joinToString()}. " +
                        "Please select valid CLIP vocab.json + merges.txt (or tokenizer.json)."
                )
            }

            syncPackAndSharedTokenizers(context)
            val status = inspectClipTokenizerStatus(context)
            val verifiedLoaded = if (status.isReady) {
                verifyAnyTokenizerDirLoads(context, sharedDir)
            } else {
                false
            }
            ClipTokenizerImportResult(
                importedFileNames = imported.distinct(),
                skippedFileNames = skipped.distinct(),
                status = status,
                tokenizerVerifiedLoaded = verifiedLoaded
            )
        }
    }

    private fun verifyAnyTokenizerDirLoads(context: Context, sharedDir: File): Boolean {
        if (ClipBpeTokenizer.loadFromDirectory(sharedDir).isLoaded()) return true
        val root = getSdInpaintRootDir(context)
        val dirs = root.listFiles()?.filter { it.isDirectory && !it.name.startsWith("_") }.orEmpty()
        for (dir in dirs) {
            if (ClipBpeTokenizer.loadFromDirectory(File(dir, "tokenizer")).isLoaded()) return true
            if (ClipBpeTokenizer.loadFromDirectory(dir).isLoaded()) return true
        }
        return false
    }

    private fun validateTokenizerFileContent(canonicalName: String, file: File): String? {
        if (!file.exists() || file.length() <= 16L) {
            return "file is empty (${file.length()} bytes)"
        }
        return when (canonicalName) {
            "tokenizer.json" -> runCatching {
                val parsed = ClipBpeTokenizer.fromTokenizerJson(file.readText())
                if (!parsed.isLoaded()) "missing model.vocab or merges in tokenizer.json" else null
            }.getOrElse { "invalid JSON: ${it.message ?: "parse error"}" }
            "vocab.json" -> runCatching {
                val raw = file.readText().trimStart('\uFEFF').trim()
                val obj = org.json.JSONObject(raw)
                if (obj.length() < 16) "vocab.json has too few entries (${obj.length()})" else null
            }.getOrElse { "invalid JSON: ${it.message ?: "parse error"}" }
            "merges.txt" -> {
                val hasMergePairs = file.bufferedReader().useLines { lines ->
                    lines.dropWhile { it.trim().isEmpty() || it.trim().startsWith("#") }
                        .take(10)
                        .any { it.trim().contains(' ') }
                }
                if (!hasMergePairs) "merges.txt has no BPE merge pairs" else null
            }
            else -> null
        }
    }

    private fun readUtf8Preview(context: Context, uri: Uri, maxBytes: Int = 262_144): String {
        return runCatching {
            context.contentResolver.openInputStream(uri)?.use { input ->
                val out = java.io.ByteArrayOutputStream()
                val buf = ByteArray(16 * 1024)
                var remaining = maxBytes
                while (remaining > 0) {
                    val toRead = minOf(buf.size, remaining)
                    val n = input.read(buf, 0, toRead)
                    if (n <= 0) break
                    out.write(buf, 0, n)
                    remaining -= n
                }
                out.toString(Charsets.UTF_8.name())
            }.orEmpty()
        }.getOrDefault("")
    }

    private fun normalizeFileNameStemAndExt(rawDisplayName: String): Pair<String, String> {
        val clean = rawDisplayName.trim().lowercase()
        val dotIdx = clean.lastIndexOf('.')
        return if (dotIdx > 0 && dotIdx < clean.length - 1) {
            val stem = clean.substring(0, dotIdx).replace(Regex("\\s+"), "")
            val ext = clean.substring(dotIdx)
            stem to ext
        } else {
            clean.replace(Regex("\\s+"), "") to ""
        }
    }

    private fun matchKnownTokenizerFileName(rawDisplayName: String): String? {
        val lower = rawDisplayName.trim().lowercase()
        val (compactStem, ext) = normalizeFileNameStemAndExt(rawDisplayName)
        return when {
            lower == "tokenizer_config.json" || lower.startsWith("tokenizer_config") ->
                "tokenizer_config.json"
            lower == "special_tokens_map.json" || lower.startsWith("special_tokens_map") ->
                "special_tokens_map.json"
            lower == "vocab.json" ||
                Regex("^vocab(\\s*\\(\\d+\\)|[_-]\\d+)?\\.json(\\.txt)?$").matches(lower) ||
                ((compactStem == "vocab" || compactStem == "vcop" || compactStem.startsWith("vcop_") ||
                    compactStem.startsWith("clip_vocab") || compactStem.endsWith("_vocab")) &&
                    (ext == ".json" || ext == ".txt")) ->
                "vocab.json"
            lower == "merges.txt" ||
                Regex("^merges(\\s*\\(\\d+\\)|[_-]\\d+)?\\.txt$").matches(lower) ||
                ((compactStem == "merges" || compactStem.startsWith("clip_merges") || compactStem.endsWith("_merges")) &&
                    ext == ".txt") ->
                "merges.txt"
            lower == "tokenizer.json" ||
                Regex("^tokenizer(\\s*\\(\\d+\\)|[_-]\\d+)?\\.json(\\.txt)?$").matches(lower) ||
                ((compactStem == "tokenizer" || compactStem == "tokenizer_fast" || compactStem == "clip_tokenizer") &&
                    (ext == ".json" || ext == ".txt")) ->
                "tokenizer.json"
            else -> null
        }
    }

    internal fun resolveCanonicalTokenizerFileName(
        context: Context,
        uri: Uri,
        rawDisplayName: String,
        expectedCanonicalHint: String? = null
    ): String? {
        val nameMatch = matchKnownTokenizerFileName(rawDisplayName)
        // Inspect file content (up to 256 KB so HuggingFace added_tokens headers don't push "model" past preview)
        val previewRaw = readUtf8Preview(context, uri, maxBytes = 262_144)
        val trimmed = previewRaw.trimStart('\uFEFF').trimStart()

        if (trimmed.isNotEmpty()) {
            val isJsonObject = trimmed.startsWith("{")
            val looksLikeTokenizerJson = isJsonObject &&
                ((trimmed.contains("\"model\"") && trimmed.contains("\"vocab\"")) ||
                    (trimmed.contains("\"added_tokens\"") && trimmed.contains("\"merges\"")))
            if (looksLikeTokenizerJson) return "tokenizer.json"

            val looksLikeMerges = !isJsonObject && !trimmed.startsWith("[") &&
                (trimmed.startsWith("#version") ||
                    trimmed.lineSequence()
                        .filter { it.isNotBlank() && !it.trimStart().startsWith("#") }
                        .take(5)
                        .count { line ->
                            val parts = line.trim().split(' ')
                            parts.size == 2 && !parts[0].contains('{')
                        } >= 2)
            if (looksLikeMerges) return "merges.txt"

            val looksLikeVocabJson = isJsonObject && !trimmed.contains("\"model\"") && (
                trimmed.contains("</w>") ||
                    trimmed.contains("\"<|startoftext|>\"") ||
                    trimmed.contains("\"<|endoftext|>\"") ||
                    trimmed.contains("\"!\"") ||
                    Regex("\"[^\"]{1,32}\"\\s*:\\s*\\d+").containsMatchIn(trimmed)
                )
            if (looksLikeVocabJson) return "vocab.json"
        }

        if (nameMatch != null) return nameMatch
        if (!expectedCanonicalHint.isNullOrBlank() && trimmed.isNotEmpty()) {
            return when (expectedCanonicalHint) {
                "tokenizer.json" -> if (trimmed.startsWith("{")) "tokenizer.json" else null
                "vocab.json" -> if (trimmed.startsWith("{")) "vocab.json" else null
                "merges.txt" -> if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) "merges.txt" else null
                else -> expectedCanonicalHint
            }
        }
        return null
    }

    private fun queryAnyDisplayName(context: Context, uri: Uri): String? {
        val queried = runCatching {
            context.contentResolver.query(
                uri,
                arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        }.getOrNull()
        if (!queried.isNullOrBlank()) return queried
        return uri.lastPathSegment?.substringAfterLast('/')?.substringAfterLast(':')
    }

    fun resolvePackForOption(
        context: Context,
        option: InpaintModelOption
    ): InstalledSdInpaintPack? {
        val installed = discoverInstalledPacks(context)
        if (installed.isEmpty()) return null
        return when (option) {
            InpaintModelOption.SD15_FP32 ->
                installed.firstOrNull { it.packType == SdInpaintPackType.SD15_FP32 }
            InpaintModelOption.REALVIS_INT8 ->
                installed.firstOrNull { it.packType == SdInpaintPackType.REALVIS_INT8 }
            InpaintModelOption.SD15_FP16 ->
                installed.firstOrNull { it.packType == SdInpaintPackType.SD15_FP16 }
            InpaintModelOption.AUTO ->
                installed.firstOrNull() // Already sorted: FP32 (1) > INT8 (2) > Custom (3) > FP16 (4)
            InpaintModelOption.LAMA_ONNX, InpaintModelOption.CV_FALLBACK -> null
        }
    }

    /**
     * Resolves an installed SD 1.5 pack optimized for mobile Stage 5 Head/Neck Seam Bridge:
     * prioritizes smaller/faster `REALVIS_INT8` first, then `SD15_FP16`, `CUSTOM_SD`, and finally `SD15_FP32`.
     */
    fun resolvePackForSeamBridge(context: Context): InstalledSdInpaintPack? {
        val installed = discoverInstalledPacks(context)
        if (installed.isEmpty()) return null
        val seamPriority = mapOf(
            SdInpaintPackType.REALVIS_INT8 to 1,
            SdInpaintPackType.SD15_FP16 to 2,
            SdInpaintPackType.CUSTOM_SD to 3,
            SdInpaintPackType.SD15_FP32 to 4
        )
        return installed.minByOrNull { seamPriority[it.packType] ?: 99 }
    }

    /**
     * Computes the default model option prioritizing CPU speed & accuracy:
     * Priority 1: SD 1.5 FP32 -> Priority 2: RealVis 5.1 INT8 -> Custom -> Secondary: SD 1.5 FP16 -> LaMa
     */
    fun determineDefaultModelOption(
        context: Context,
        lamaFile: File?
    ): InpaintModelOption {
        val installed = discoverInstalledPacks(context)
        if (installed.any { it.packType == SdInpaintPackType.SD15_FP32 }) {
            return InpaintModelOption.SD15_FP32
        }
        if (installed.any { it.packType == SdInpaintPackType.REALVIS_INT8 }) {
            return InpaintModelOption.REALVIS_INT8
        }
        if (installed.any { it.packType == SdInpaintPackType.CUSTOM_SD }) {
            return InpaintModelOption.AUTO
        }
        if (installed.any { it.packType == SdInpaintPackType.SD15_FP16 }) {
            return InpaintModelOption.SD15_FP16
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
            SdInpaintPackType.SD15_FP32 -> {
                if (lower.contains("fp32") || lower.contains("f32")) score += 450
                if (lower == "model.onnx") score += 400
                if (lower.contains("fp16")) score -= 250
                if (lower.contains("int8") || lower.contains("quant")) score -= 200
            }
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
            val tokenBytesList = tokens.filter { it.isNotEmpty() }.map { it.toByteArray(Charsets.ISO_8859_1) }
            if (tokenBytesList.isEmpty()) return false
            val maxTokenLen = tokenBytesList.maxOf { it.size }
            val maxScanBytes = minOf(file.length(), 8L * 1024L * 1024L)
            val chunkSize = 64 * 1024
            val buffer = ByteArray(chunkSize + maxTokenLen)
            var overlap = 0
            var totalRead = 0L
            file.inputStream().use { input ->
                while (totalRead < maxScanBytes) {
                    val toRead = minOf(chunkSize.toLong(), maxScanBytes - totalRead).toInt()
                    val read = input.read(buffer, overlap, toRead)
                    if (read <= 0) break
                    totalRead += read
                    val validLen = overlap + read
                    for (needle in tokenBytesList) {
                        if (containsByteSequence(buffer, validLen, needle)) {
                            return true
                        }
                    }
                    overlap = minOf(maxTokenLen - 1, validLen).coerceAtLeast(0)
                    if (overlap > 0) {
                        System.arraycopy(buffer, validLen - overlap, buffer, 0, overlap)
                    }
                }
            }
            false
        }.getOrDefault(false)
    }

    private fun containsByteSequence(haystack: ByteArray, haystackLen: Int, needle: ByteArray): Boolean {
        if (needle.isEmpty() || haystackLen < needle.size) return false
        val first = needle[0]
        val limit = haystackLen - needle.size
        var i = 0
        while (i <= limit) {
            if (haystack[i] == first) {
                var match = true
                var j = 1
                while (j < needle.size) {
                    if (haystack[i + j] != needle[j]) {
                        match = false
                        break
                    }
                    j++
                }
                if (match) return true
            }
            i++
        }
        return false
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
            if (depth > 6) return
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

        val rootTokenizerFiles = mutableMapOf<String, MutableList<TreeNodeFile>>()
        val anyTokenizerFiles = mutableMapOf<String, TreeNodeFile>()

        for (file in collectedFiles) {
            val canonicalTokName = matchKnownTokenizerFileName(file.displayName)
            if (canonicalTokName != null) {
                anyTokenizerFiles.putIfAbsent(canonicalTokName, file)
            }
            val segs = file.parentSegments
            val compIdx = segs.indexOfLast {
                val l = it.lowercase()
                l in COMPONENT_DIRS || l in TOKENIZER_FOLDER_NAMES
            }
            if (compIdx >= 0) {
                val rawCompName = segs[compIdx].lowercase()
                val compName = if (rawCompName in TOKENIZER_FOLDER_NAMES) "tokenizer" else rawCompName
                val packSegs = segs.take(compIdx)
                val packFolderPath = if (packSegs.isEmpty()) "__root__" else packSegs.joinToString("/")
                val packFolderName = packSegs.lastOrNull() ?: rootDocName.ifBlank { "sd_inpaint" }
                if (compName == "tokenizer" && packSegs.isEmpty()) {
                    // Sibling tokenizer/ folder at selected root can serve packs in child folders too
                    if (canonicalTokName != null) {
                        rootTokenizerFiles.getOrPut("__root__") { mutableListOf() }.add(file)
                    }
                } else {
                    val key = PackBucketKey(packFolderPath, packFolderName)
                    packBuckets.getOrPut(key) { mutableListOf() }.add(compName to file)
                }
            } else if (canonicalTokName != null) {
                val packFolderPath = if (segs.isEmpty()) "__root__" else segs.joinToString("/")
                rootTokenizerFiles.getOrPut(packFolderPath) { mutableListOf() }.add(file)
            }
        }

        // Attach root/parent/fallback tokenizer files when a pack doesn't already have complete tokenizer files
        for ((key, list) in packBuckets) {
            val existingTokNames = list.filter { it.first == "tokenizer" }
                .mapNotNull { matchKnownTokenizerFileName(it.second.displayName) }
                .toSet()
            val hasCompleteTokAlready = "tokenizer.json" in existingTokNames ||
                ("vocab.json" in existingTokNames && "merges.txt" in existingTokNames)
            if (!hasCompleteTokAlready) {
                val attached = mutableSetOf<String>().apply { addAll(existingTokNames) }
                val candidateSources = buildList {
                    rootTokenizerFiles[key.packFolderPath]?.let { addAll(it) }
                    rootTokenizerFiles["__root__"]?.let { addAll(it) }
                    addAll(anyTokenizerFiles.values)
                }
                for (node in candidateSources) {
                    val canon = matchKnownTokenizerFileName(node.displayName)
                    if (canon != null && canon !in attached) {
                        list.add("tokenizer" to node)
                        attached.add(canon)
                    }
                }
            }
        }

        // Even if only tokenizer files were present in the picked folder, copy them canonically into _shared_tokenizer
        if (anyTokenizerFiles.isNotEmpty()) {
            val sharedDir = getSharedTokenizerDir(context)
            anyTokenizerFiles.forEach { (canonicalName, node) ->
                val destFile = File(sharedDir, canonicalName)
                if (!destFile.exists() || (node.sizeBytes > 16L && destFile.length() != node.sizeBytes)) {
                    val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, node.docId)
                    runCatching { copyUriToFile(context, docUri, destFile) }
                }
            }
            syncPackAndSharedTokenizers(context)
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
                val targetName = if (compName == "tokenizer") {
                    matchKnownTokenizerFileName(node.displayName) ?: node.displayName
                } else {
                    node.displayName
                }
                val destFile = File(compDestDir, targetName)
                if (destFile.exists() && node.sizeBytes > 512L && destFile.length() == node.sizeBytes) {
                    return@forEachIndexed
                }
                onProgress(
                    "Importing ${packType.displayLabel} (${index + 1}/${filesToCopy.size}): $compName/$targetName..."
                )
                val docUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, node.docId)
                val ok = runCatching {
                    copyUriToFile(context, docUri, destFile)
                }.isSuccess
                if (!ok && targetName.endsWith(".onnx", ignoreCase = true)) {
                    copySucceeded = false
                }
            }

            if (copySucceeded) {
                syncPackAndSharedTokenizers(context)
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

        if (importedTypes.isNotEmpty()) {
            OnnxProtobufInspector.clearRuntimeLowRamLatch(context)
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
                "tokenizer" -> matchKnownTokenizerFileName(node.displayName) != null
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

    internal fun isMissingCompanionExternalData(onnxFile: File): Boolean {
        if (!onnxFile.exists() || onnxFile.length() >= 16L * 1024L * 1024L) return false
        val compDir = onnxFile.parentFile ?: return false
        val dataFiles = compDir.listFiles()
            ?.filter {
                it.isFile && (it.name.endsWith(".data", ignoreCase = true) ||
                    it.name.endsWith(".onnx_data", ignoreCase = true))
            }
            .orEmpty()
        val referencesExternalData = fileContainsAnyAsciiToken(onnxFile, listOf(".onnx.data", ".onnx_data", "model.data"))
        if (!referencesExternalData) return false
        return dataFiles.none { it.length() > 1024L }
    }

    fun hasTokenizerFiles(dir: File?): Boolean {
        if (dir == null || !dir.exists() || !dir.isDirectory) return false
        val tokJson = File(dir, "tokenizer.json")
        if (tokJson.exists() && tokJson.length() > 16L) return true
        val vocabJson = File(dir, "vocab.json")
        val mergesTxt = File(dir, "merges.txt")
        return vocabJson.exists() && vocabJson.length() > 16L &&
            mergesTxt.exists() && mergesTxt.length() > 16L
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
