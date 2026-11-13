package com.example.util

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.FileDescriptor
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.max

object ImageGalleryHelper {

    private const val MAX_IMAGE_DIMENSION = 1600

    /**
     * Memory-safely decodes a gallery or document image Uri into a software ARGB_8888 Bitmap,
     * applying EXIF orientation and capping the maximum dimension at 1600px.
     *
     * Supports OEM Gallery providers (Vivo, Xiaomi, Samsung, Oppo), Android Photo Picker
     * virtual/typed media URIs, DocumentsProvider URIs, and direct file descriptors.
     */
    fun decodeUriToBitmap(
        context: Context,
        uri: Uri,
        maxDimensionPx: Int = MAX_IMAGE_DIMENSION
    ): Result<Bitmap> {
        val safeMaxDim = maxDimensionPx.coerceIn(512, 4096)
        return runCatching {
            // Preserve read permission across coroutine dispatches if the provider supports it
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }

            val candidateUris = buildCandidateUris(context, uri)
            var lastError: Throwable? = null

            for (candidateUri in candidateUris) {
                // Strategy 1: Native ImageDecoder on Android 9+ (handles Photo Picker, HEIC, WebP, EXIF)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val imageDecoderResult = runCatching {
                        decodeWithImageDecoder(context, candidateUri, safeMaxDim)
                    }
                    if (imageDecoderResult.isSuccess) {
                        return@runCatching imageDecoderResult.getOrThrow()
                    } else {
                        lastError = imageDecoderResult.exceptionOrNull()
                    }
                }

                // Strategy 2: Copy stream once to a temporary cache file using multi-descriptor fallback
                val tempFileResult = runCatching {
                    decodeViaSinglePassTempFile(context, candidateUri, safeMaxDim)
                }
                if (tempFileResult.isSuccess) {
                    return@runCatching tempFileResult.getOrThrow()
                } else {
                    lastError = tempFileResult.exceptionOrNull()
                }

                // Strategy 3: Direct ParcelFileDescriptor decoding
                val pfdResult = runCatching {
                    decodeViaParcelFileDescriptor(context, candidateUri, safeMaxDim)
                }
                if (pfdResult.isSuccess) {
                    return@runCatching pfdResult.getOrThrow()
                } else {
                    lastError = pfdResult.exceptionOrNull()
                }
            }

            throw IllegalStateException(
                lastError?.message ?: "Unable to open image stream from Gallery URI ($uri)."
            )
        }
    }

    private fun decodeWithImageDecoder(
        context: Context,
        uri: Uri,
        maxDimensionPx: Int = MAX_IMAGE_DIMENSION
    ): Bitmap {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
            error("ImageDecoder requires API 28+")
        }
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            decoder.isMutableRequired = false
            val w = info.size.width
            val h = info.size.height
            val longest = max(w, h)
            if (longest > maxDimensionPx) {
                val scale = maxDimensionPx.toFloat() / longest.toFloat()
                decoder.setTargetSize(
                    (w * scale).toInt().coerceAtLeast(1),
                    (h * scale).toInt().coerceAtLeast(1)
                )
            }
        }
        if (bitmap.width <= 0 || bitmap.height <= 0) {
            error("Decoded bitmap has invalid dimensions.")
        }
        return ensureArgb8888(bitmap)
    }

    private fun decodeViaSinglePassTempFile(
        context: Context,
        uri: Uri,
        maxDimensionPx: Int = MAX_IMAGE_DIMENSION
    ): Bitmap {
        val tempFile = File(context.cacheDir, "picked_photo_${System.nanoTime()}.tmp")
        try {
            val stream = openBestEffortInputStream(context, uri)
                ?: error("Unable to open image stream from Gallery URI.")

            var bytesCopied = 0L
            stream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    bytesCopied = input.copyTo(output, bufferSize = 64 * 1024)
                    output.flush()
                }
            }

            if (bytesCopied <= 0L || !tempFile.exists() || tempFile.length() <= 0L) {
                error("Selected image stream returned 0 bytes.")
            }

            val boundsOpts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFile(tempFile.absolutePath, boundsOpts)

            if (boundsOpts.outWidth <= 0 || boundsOpts.outHeight <= 0) {
                error("Unsupported or corrupted image file format.")
            }

            var sampleSize = 1
            val longestSide = max(boundsOpts.outWidth, boundsOpts.outHeight)
            while (longestSide / sampleSize > maxDimensionPx) {
                sampleSize *= 2
            }

            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inMutable = false
            }

            val rawBitmap = BitmapFactory.decodeFile(tempFile.absolutePath, decodeOpts)
                ?: error("Failed to decode bitmap pixels from temporary file.")

            val rotationDegrees = readExifRotationFromFile(tempFile)
            val oriented = applyRotationIfNeeded(rawBitmap, rotationDegrees)
            return ensureArgb8888(oriented)
        } finally {
            runCatching { tempFile.delete() }
        }
    }

    private fun decodeViaParcelFileDescriptor(
        context: Context,
        uri: Uri,
        maxDimensionPx: Int = MAX_IMAGE_DIMENSION
    ): Bitmap {
        val pfd = openBestEffortParcelFileDescriptor(context, uri)
            ?: error("Unable to open ParcelFileDescriptor for Gallery URI.")

        pfd.use { descriptor ->
            val fd: FileDescriptor = descriptor.fileDescriptor
            val boundsOpts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFileDescriptor(fd, null, boundsOpts)
            if (boundsOpts.outWidth <= 0 || boundsOpts.outHeight <= 0) {
                error("Unable to decode image bounds from FileDescriptor.")
            }

            var sampleSize = 1
            val longestSide = max(boundsOpts.outWidth, boundsOpts.outHeight)
            while (longestSide / sampleSize > maxDimensionPx) {
                sampleSize *= 2
            }

            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inMutable = false
            }

            val rawBitmap = BitmapFactory.decodeFileDescriptor(fd, null, decodeOpts)
                ?: error("Unable to decode bitmap pixels from FileDescriptor.")

            val rotationDegrees = runCatching {
                val exif = ExifInterface(fd)
                exifToDegrees(
                    exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                )
            }.getOrDefault(0f)

            val oriented = applyRotationIfNeeded(rawBitmap, rotationDegrees)
            return ensureArgb8888(oriented)
        }
    }

    private fun openBestEffortInputStream(context: Context, uri: Uri): InputStream? {
        val resolver = context.contentResolver

        // 1. Typed asset file descriptor ("image/*") — required by virtual/transcoded Photo Picker & OEM galleries
        runCatching {
            resolver.openTypedAssetFileDescriptor(uri, "image/*", null)?.createInputStream()
        }.getOrNull()?.let { return it }

        // 2. Standard ContentResolver inputStream
        runCatching {
            resolver.openInputStream(uri)
        }.getOrNull()?.let { return it }

        // 3. AssetFileDescriptor ("r")
        runCatching {
            resolver.openAssetFileDescriptor(uri, "r")?.createInputStream()
        }.getOrNull()?.let { return it }

        // 4. ParcelFileDescriptor AutoCloseInputStream
        runCatching {
            resolver.openFileDescriptor(uri, "r")?.let {
                ParcelFileDescriptor.AutoCloseInputStream(it)
            }
        }.getOrNull()?.let { return it }

        // 5. Wildcard typed asset file descriptor ("*/*")
        runCatching {
            resolver.openTypedAssetFileDescriptor(uri, "*/*", null)?.createInputStream()
        }.getOrNull()?.let { return it }

        // 6. Direct file:// path fallback
        if (uri.scheme.equals("file", ignoreCase = true)) {
            val path = uri.path
            if (!path.isNullOrBlank()) {
                val f = File(path)
                if (f.exists() && f.canRead()) {
                    return runCatching { FileInputStream(f) }.getOrNull()
                }
            }
        }

        return null
    }

    private fun openBestEffortParcelFileDescriptor(
        context: Context,
        uri: Uri
    ): ParcelFileDescriptor? {
        val resolver = context.contentResolver
        runCatching {
            resolver.openTypedAssetFileDescriptor(uri, "image/*", null)?.parcelFileDescriptor
        }.getOrNull()?.let { return it }

        runCatching {
            resolver.openFileDescriptor(uri, "r")
        }.getOrNull()?.let { return it }

        runCatching {
            resolver.openAssetFileDescriptor(uri, "r")?.parcelFileDescriptor
        }.getOrNull()?.let { return it }

        return null
    }

    private fun buildCandidateUris(context: Context, originalUri: Uri): List<Uri> {
        val candidates = LinkedHashSet<Uri>()
        candidates.add(originalUri)

        // Check DocumentsContract URI translations (e.g. image:12345 -> MediaStore EXTERNAL_CONTENT_URI/12345)
        runCatching {
            if (DocumentsContract.isDocumentUri(context, originalUri)) {
                val docId = DocumentsContract.getDocumentId(originalUri)
                if (docId.startsWith("raw:")) {
                    val rawPath = docId.removePrefix("raw:")
                    val rawFile = File(rawPath)
                    if (rawFile.exists()) {
                        candidates.add(Uri.fromFile(rawFile))
                    }
                } else if (docId.contains(":")) {
                    val parts = docId.split(":")
                    val type = parts.getOrNull(0).orEmpty()
                    val idStr = parts.getOrNull(1).orEmpty()
                    val numericId = idStr.toLongOrNull()
                    if (numericId != null && (type.equals("image", ignoreCase = true) ||
                            originalUri.authority?.contains("media") == true)
                    ) {
                        candidates.add(
                            ContentUris.withAppendedId(
                                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                numericId
                            )
                        )
                    }
                    if (type.equals("primary", ignoreCase = true) && idStr.isNotEmpty()) {
                        val extFile = File(Environment.getExternalStorageDirectory(), idStr)
                        if (extFile.exists()) {
                            candidates.add(Uri.fromFile(extFile))
                        }
                    }
                }
            }
        }

        // Check if lastPathSegment contains an embedded media ID (common in Photo Picker / OEM picker URIs)
        runCatching {
            val lastSegment = originalUri.lastPathSegment.orEmpty()
            val numericId = when {
                lastSegment.contains(":") -> lastSegment.substringAfterLast(":").toLongOrNull()
                else -> lastSegment.toLongOrNull()
            }
            if (numericId != null && numericId > 0L) {
                candidates.add(
                    ContentUris.withAppendedId(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                        numericId
                    )
                )
            }
        }

        // Query ContentResolver for _ID or _DATA column if provided by OEM gallery
        runCatching {
            val projection = arrayOf(
                MediaStore.Images.Media._ID,
                MediaStore.MediaColumns.DATA
            )
            context.contentResolver.query(originalUri, projection, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val idIdx = cursor.getColumnIndex(MediaStore.Images.Media._ID)
                    if (idIdx >= 0 && !cursor.isNull(idIdx)) {
                        val mediaId = cursor.getLong(idIdx)
                        if (mediaId > 0L) {
                            candidates.add(
                                ContentUris.withAppendedId(
                                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                    mediaId
                                )
                            )
                        }
                    }
                    val dataIdx = cursor.getColumnIndex(MediaStore.MediaColumns.DATA)
                    if (dataIdx >= 0 && !cursor.isNull(dataIdx)) {
                        val filePath = cursor.getString(dataIdx)
                        if (!filePath.isNullOrBlank()) {
                            val file = File(filePath)
                            if (file.exists() && file.canRead()) {
                                candidates.add(Uri.fromFile(file))
                            }
                        }
                    }
                }
            }
        }

        return candidates.toList()
    }

    private fun readExifRotationFromFile(file: File): Float {
        return runCatching {
            val exif = ExifInterface(file.absolutePath)
            exifToDegrees(
                exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            )
        }.getOrDefault(0f)
    }

    private fun exifToDegrees(exifOrientation: Int): Float {
        return when (exifOrientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
    }

    private fun applyRotationIfNeeded(rawBitmap: Bitmap, rotationDegrees: Float): Bitmap {
        if (rotationDegrees == 0f) return rawBitmap
        val matrix = Matrix().apply { postRotate(rotationDegrees) }
        val rotated = Bitmap.createBitmap(
            rawBitmap,
            0,
            0,
            rawBitmap.width,
            rawBitmap.height,
            matrix,
            true
        )
        if (rotated !== rawBitmap) rawBitmap.recycle()
        return rotated
    }

    private fun ensureArgb8888(bitmap: Bitmap): Bitmap {
        return if (bitmap.config != Bitmap.Config.ARGB_8888) {
            val argb = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            bitmap.recycle()
            argb
        } else {
            bitmap
        }
    }

    /**
     * Rotates a Bitmap clockwise by 90 degrees.
     */
    fun rotateBitmap90(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { postRotate(90f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Center-crops a Bitmap to the requested aspect ratio (`aspectW : aspectH`).
     */
    fun cropBitmapToRatio(bitmap: Bitmap, aspectW: Int, aspectH: Int): Bitmap {
        if (aspectW <= 0 || aspectH <= 0) return bitmap
        val srcW = bitmap.width
        val srcH = bitmap.height
        val targetRatio = aspectW.toFloat() / aspectH.toFloat()
        val srcRatio = srcW.toFloat() / srcH.toFloat()

        val cropW: Int
        val cropH: Int
        if (srcRatio > targetRatio) {
            cropH = srcH
            cropW = (srcH * targetRatio).toInt().coerceIn(1, srcW)
        } else {
            cropW = srcW
            cropH = (srcW / targetRatio).toInt().coerceIn(1, srcH)
        }

        val x = ((srcW - cropW) / 2).coerceAtLeast(0)
        val y = ((srcH - cropH) / 2).coerceAtLeast(0)
        return Bitmap.createBitmap(bitmap, x, y, cropW, cropH)
    }

    /**
     * Saves the swapped Bitmap to the Android device's Gallery (`Pictures/FaceSwapStudio`)
     * using MediaStore without requiring internet or legacy storage permissions on Android 10+.
     *
     * @param isHd When true, saves uncompressed 100% PNG at the requested `targetMaxDimension`.
     *             When false, saves standard 95% JPEG.
     */
    fun saveBitmapToGallery(
        context: Context,
        bitmap: Bitmap,
        isHd: Boolean = true,
        targetMaxDimension: Int = 0
    ): Result<String> {
        return runCatching {
            val exportBitmap = if (targetMaxDimension > 0) {
                val longest = max(bitmap.width, bitmap.height)
                if (longest != targetMaxDimension && longest > 0) {
                    val scale = targetMaxDimension.toFloat() / longest.toFloat()
                    val newW = (bitmap.width * scale).toInt().coerceIn(64, 4096)
                    val newH = (bitmap.height * scale).toInt().coerceIn(64, 4096)
                    Bitmap.createScaledBitmap(bitmap, newW, newH, true)
                } else {
                    bitmap
                }
            } else {
                bitmap
            }

            val ext = if (isHd) "png" else "jpg"
            val mime = if (isHd) "image/png" else "image/jpeg"
            val prefix = if (isHd) "FaceSwap_HD" else "FaceSwap"
            val fileName = "${prefix}_${System.currentTimeMillis()}.$ext"
            val resolver = context.contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, mime)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        "${Environment.DIRECTORY_PICTURES}/FaceSwapStudio"
                    )
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val imageUri = resolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            ) ?: error("Could not create MediaStore entry in Android Gallery.")

            val outputStream: OutputStream = resolver.openOutputStream(imageUri)
                ?: error("Could not open MediaStore output stream.")

            outputStream.use { out ->
                val format = if (isHd) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG
                val quality = if (isHd) 100 else 95
                if (!exportBitmap.compress(format, quality, out)) {
                    error("Failed to compress output bitmap.")
                }
            }

            if (exportBitmap !== bitmap) {
                exportBitmap.recycle()
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(imageUri, contentValues, null, null)
            }

            "Pictures/FaceSwapStudio/$fileName (${exportBitmap.width}×${exportBitmap.height})"
        }
    }

    /**
     * Shares the result Bitmap via Android's native system Share sheet (`Intent.ACTION_SEND`)
     * using a MediaStore URI so no external internet or cloud upload is needed.
     */
    fun shareBitmap(context: Context, bitmap: Bitmap): Result<String> {
        return runCatching {
            val fileName = "FaceSwap_Share_${System.currentTimeMillis()}.png"
            val resolver = context.contentResolver
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        "${Environment.DIRECTORY_PICTURES}/FaceSwapStudio"
                    )
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val imageUri = resolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            ) ?: error("Could not prepare shareable MediaStore URI.")

            resolver.openOutputStream(imageUri)?.use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            } ?: error("Could not write shareable image stream.")

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(imageUri, contentValues, null, null)
            }

            val sendIntent = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, imageUri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            val chooser = Intent.createChooser(sendIntent, "Share Offline Face Swap / Head Result").apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(chooser)
            "Opened Android Share Sheet ($fileName)"
        }
    }

    /**
     * Saves a Bitmap as uncompressed 100% PNG directly to a user-selected folder/document URI
     * obtained via `ActivityResultContracts.CreateDocument` (Storage Access Framework).
     */
    fun saveBitmapToUri(
        context: Context,
        uri: Uri,
        bitmap: Bitmap
    ): Result<String> {
        return runCatching {
            val resolver = context.contentResolver
            val outputStream: OutputStream = resolver.openOutputStream(uri)
                ?: error("Could not open selected destination file in mobile storage.")
            outputStream.use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                    error("Failed to write PNG image data.")
                }
                out.flush()
            }
            val displayName = runCatching {
                resolver.query(
                    uri,
                    arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                    null,
                    null,
                    null
                )?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getString(0) else null
                }
            }.getOrNull() ?: uri.lastPathSegment ?: "Saved_Image.png"
            "Saved to folder: $displayName (${bitmap.width}×${bitmap.height})"
        }
    }

    /**
     * Persists a high-quality JPEG copy of a Swap/Inpaint result into private app storage
     * (`filesDir/history_results/`) so the last 3 History items can be viewed, zoomed, and saved.
     */
    fun saveBitmapToHistoryCache(context: Context, bitmap: Bitmap): String? {
        return runCatching {
            val dir = File(context.filesDir, "history_results")
            if (!dir.exists()) dir.mkdirs()
            val file = File(dir, "history_${System.currentTimeMillis()}.jpg")
            val longest = max(bitmap.width, bitmap.height)
            val maxCacheDim = 1440
            val outBmp = if (longest > maxCacheDim) {
                val scale = maxCacheDim.toFloat() / longest.toFloat()
                Bitmap.createScaledBitmap(
                    bitmap,
                    (bitmap.width * scale).toInt().coerceAtLeast(64),
                    (bitmap.height * scale).toInt().coerceAtLeast(64),
                    true
                )
            } else {
                bitmap
            }
            FileOutputStream(file).use { out ->
                outBmp.compress(Bitmap.CompressFormat.JPEG, 94, out)
                out.flush()
            }
            if (outBmp !== bitmap) {
                outBmp.recycle()
            }
            file.absolutePath
        }.getOrNull()
    }

    fun loadBitmapFromFile(path: String?): Bitmap? {
        if (path.isNullOrBlank()) return null
        return runCatching {
            val f = File(path)
            if (f.exists() && f.isFile && f.length() > 0L) {
                BitmapFactory.decodeFile(f.absolutePath)
            } else {
                null
            }
        }.getOrNull()
    }

    fun pruneOrphanHistoryFiles(context: Context, keepPaths: Set<String>) {
        runCatching {
            val dir = File(context.filesDir, "history_results")
            if (!dir.exists()) return
            dir.listFiles()?.forEach { file ->
                if (file.isFile && file.absolutePath !in keepPaths) {
                    file.delete()
                }
            }
        }
    }
}
