package com.example.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import com.example.util.ImageGalleryHelper
import java.io.File
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 3-Step Portrait-First Clean-Erase Head Replacement Pipeline (shared by **Option B+C** & **Head Replacement**):
 *  Step 1 (`Output 1` — Clean Headless Target):
 *   - Stage 1: Extract 468-pt 3D landmarks (`facemesh-v2-v1.onnx` / PnP-106) for Source & Target.
 *   - Stage 2: Depth Anything V2 scene depth & SegFormer/MODNet head/hair/neck separation.
 *   - Stage 3: Remove 100% of Target's old face, skull, hair & ears using `lama_fp32.onnx` or
 *              2D Multi-Directional Exterior Anchor Inpainting while protecting collar/clothing/jewelry.
 *  Step 2 (`Intermediate Full Portrait` — Clean Head + Hair + Neck + Lighting):
 *   - Stage 4: Warp Source head/hair/neck via 31-pt 3D mesh with `inBoundsMask` + border feathering
 *              (NO zoomed-out center-oval HyperSwap overwrite) & adapt lighting/skin tone smoothly.
 *   - Stage 5: Diffusion Seam Bridge (`SdInpaintOnnxEngine`) or fast multi-band seam bridge on
 *              outer hair & neck-to-collar transitions -> yields full `intermediatePortraitBitmap`.
 *  Step 3 (`Final High-Res Identity Swap` — `Output 3`):
 *   - Stage 6: Release temporary Step 1/2 ONNX sessions, detect face(s) on `intermediatePortraitBitmap`
 *              via `det_10g.onnx` (with exact 3D pose landmark fallback), and run `InSwapperEngine`
 *              (`ArcFace w600k_r50` -> `HyperSwap 256px` -> `GFPGAN 512x512` + eye/brow/lip/skin fusion).
 */
object CleanEraseHeadReplacementEngine {

    private data class PlacedHeadPose(
        val targetFaceIndex: Int,
        val fallbackDetectedFace: DetectedFace,
        val eyeMidPoint: PointF,
        val eyeDist: Float
    )

    fun execute(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        sourceFace: DetectedFace,
        targetBitmap: Bitmap,
        targetFacesToReplace: List<DetectedFace>,
        pipelineMode: HeadReplacementPipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
        enableColorTransfer: Boolean,
        enableProvenanceWatermark: Boolean,
        allowTwoModelFallbackForTesting: Boolean,
        lowMemoryMode: Boolean = false,
        preferHardwareAccel: Boolean = false,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        faceReactionMode: FaceReactionSourceMode = FaceReactionSourceMode.TARGET_REACTION,
        portraitBlurStrength: Float = 0f,
        blendStrength: Float = 1.0f,
        enhancementStrength: Float = 0.85f,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f,
        preloadedArcFaceSession: OrtSession? = null,
        preloadedSwapSession: OrtSession? = null,
        preloadedSegformerSession: OrtSession? = null,
        preloadedGfpganSession: OrtSession? = null,
        preloadedEmap512x512: FloatArray? = null,
        autoBodyAgeMatch: Boolean = true,
        faceAgingFactor: Float = 0.0f,
        isCancelled: () -> Boolean = { false },
        onProgress: (SwapStageProgress) -> Unit
    ): FaceSwapExecutionResult {
        require(targetFacesToReplace.isNotEmpty()) {
            "Please select at least one detected target head to replace."
        }
        val safeMaxWorkingDim = if (lowMemoryMode) 1280 else 2048
        val cappedSource = ImageGalleryHelper.capBitmapMaxDimension(sourceBitmap, safeMaxWorkingDim)
        val cappedTarget = ImageGalleryHelper.capBitmapMaxDimension(targetBitmap, safeMaxWorkingDim)
        val procSourceFace = if (cappedSource.wasScaled) {
            sourceFace.scaled(cappedSource.scaleX, cappedSource.scaleY)
        } else {
            sourceFace
        }
        val procTargetFaces = if (cappedTarget.wasScaled) {
            targetFacesToReplace.map { it.scaled(cappedTarget.scaleX, cappedTarget.scaleY) }
        } else {
            targetFacesToReplace
        }
        var result: FaceSwapExecutionResult? = null
        try {
            result = executeCapped(
                context = context,
                ortEnv = ortEnv,
                sourceBitmap = cappedSource.bitmap,
                sourceFace = procSourceFace,
                targetBitmap = cappedTarget.bitmap,
                targetFacesToReplace = procTargetFaces,
                pipelineMode = pipelineMode,
                enableColorTransfer = enableColorTransfer,
                enableProvenanceWatermark = enableProvenanceWatermark,
                allowTwoModelFallbackForTesting = allowTwoModelFallbackForTesting,
                lowMemoryMode = lowMemoryMode,
                preferHardwareAccel = preferHardwareAccel,
                skinToneMode = skinToneMode,
                faceReactionMode = faceReactionMode,
                portraitBlurStrength = portraitBlurStrength,
                blendStrength = blendStrength,
                enhancementStrength = enhancementStrength,
                offsetX = offsetX,
                offsetY = offsetY,
                scaleAdjust = scaleAdjust,
                preloadedArcFaceSession = preloadedArcFaceSession,
                preloadedSwapSession = preloadedSwapSession,
                preloadedSegformerSession = preloadedSegformerSession,
                preloadedGfpganSession = preloadedGfpganSession,
                preloadedEmap512x512 = preloadedEmap512x512,
                autoBodyAgeMatch = autoBodyAgeMatch,
                faceAgingFactor = faceAgingFactor,
                isCancelled = isCancelled,
                onProgress = onProgress
            )
            return result
        } finally {
            if (cappedSource.bitmap !== sourceBitmap) {
                cappedSource.bitmap.recycle()
            }
            if (cappedTarget.bitmap !== targetBitmap && result?.outputBitmap !== cappedTarget.bitmap) {
                cappedTarget.bitmap.recycle()
            }
        }
    }

    private fun executeCapped(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        sourceFace: DetectedFace,
        targetBitmap: Bitmap,
        targetFacesToReplace: List<DetectedFace>,
        pipelineMode: HeadReplacementPipelineMode,
        enableColorTransfer: Boolean,
        enableProvenanceWatermark: Boolean,
        allowTwoModelFallbackForTesting: Boolean,
        lowMemoryMode: Boolean,
        preferHardwareAccel: Boolean,
        skinToneMode: SkinToneSourceMode,
        faceReactionMode: FaceReactionSourceMode,
        portraitBlurStrength: Float,
        blendStrength: Float,
        enhancementStrength: Float,
        offsetX: Float,
        offsetY: Float,
        scaleAdjust: Float,
        preloadedArcFaceSession: OrtSession?,
        preloadedSwapSession: OrtSession?,
        preloadedSegformerSession: OrtSession?,
        preloadedGfpganSession: OrtSession?,
        preloadedEmap512x512: FloatArray?,
        autoBodyAgeMatch: Boolean,
        faceAgingFactor: Float,
        isCancelled: () -> Boolean,
        onProgress: (SwapStageProgress) -> Unit
    ): FaceSwapExecutionResult {
        val tStart = System.currentTimeMillis()
        val canvasSize = if (lowMemoryMode) 320 else 768
        val detFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.DETECTOR)
        val segFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SEGMENTATION)
        val matFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.MATTING)
        val lamaFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.INPAINTING)
        val faceMeshFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.FACE_MESH_3D)

        // =========================================================================
        // STAGE 1/6: 3D FaceMesh V2 Extraction + Body Age Harmony Plan
        // =========================================================================
        if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
        val tDet0 = System.currentTimeMillis()
        onProgress(
            SwapStageProgress(
                stepIndex = 1,
                totalSteps = 6,
                stageTitle = "Stage 1/6: 3D FaceMesh & Pose",
                detailMessage = "Extracting 3D facial vertices (FaceMesh V2) and body-age harmony plan...",
                progressFraction = 0.14f
            )
        )

        val srcMesh3D = FaceMesh3DEngine.extractFaceMesh3D(
            ortEnv = ortEnv,
            bitmap = sourceBitmap,
            face = sourceFace,
            faceMeshFile = if (faceMeshFile.exists()) faceMeshFile else null,
            preferHardwareAccel = preferHardwareAccel
        )
        val tgtMeshes3D = targetFacesToReplace.map { tgtFace ->
            if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
            FaceMesh3DEngine.extractFaceMesh3D(
                ortEnv = ortEnv,
                bitmap = targetBitmap,
                face = tgtFace,
                faceMeshFile = if (faceMeshFile.exists()) faceMeshFile else null,
                preferHardwareAccel = preferHardwareAccel
            )
        }

        val ageConfig = AgeHarmonyEngine.AgeHarmonyConfig(
            autoBodyAgeMatch = autoBodyAgeMatch,
            manualAgingFactor = faceAgingFactor
        )
        val targetAgePlans = if (autoBodyAgeMatch || abs(faceAgingFactor) >= 0.05f) {
            targetFacesToReplace.map { tgtFace ->
                if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
                runCatching {
                    AgeHarmonyEngine.resolveAgePlan(
                        context = context,
                        ortEnv = ortEnv,
                        sourceBitmap = sourceBitmap,
                        sourceFace = sourceFace,
                        targetBitmap = targetBitmap,
                        targetFace = tgtFace,
                        config = ageConfig,
                        preferHardwareAccel = preferHardwareAccel
                    )
                }.getOrNull()
            }
        } else {
            List(targetFacesToReplace.size) { null }
        }
        val firstAgeHarmonySummary = targetAgePlans.firstOrNull()?.summaryLabel
        val detMs = (System.currentTimeMillis() - tDet0).coerceAtLeast(1L)

        // =========================================================================
        // STAGE 2/6: Depth Anything V2 + Seam Pack Plan
        // =========================================================================
        if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
        val tEmb0 = System.currentTimeMillis()
        onProgress(
            SwapStageProgress(
                stepIndex = 2,
                totalSteps = 6,
                stageTitle = "Stage 2/6: Scene Depth & Portrait Prep",
                detailMessage = "Computing Depth Anything V2 foreground/background map & preparing headless pass...",
                progressFraction = 0.28f
            )
        )

        val targetDepthMap = if (DepthEstimationEngine.hasInstalledDepthModel(context)) {
            val depthAttempt = runCatching {
                DepthEstimationEngine.estimateDepth(
                    context = context,
                    ortEnv = ortEnv,
                    sourceBitmap = targetBitmap,
                    preferHardwareAccel = preferHardwareAccel,
                    isCancelled = isCancelled
                )
            }
            val depthErr = depthAttempt.exceptionOrNull()
            if (depthErr is java.util.concurrent.CancellationException || isCancelled()) {
                throw (depthErr as? java.util.concurrent.CancellationException
                    ?: java.util.concurrent.CancellationException("Head replacement cancelled by user."))
            }
            depthAttempt.getOrNull()
        } else {
            null
        }
        if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

        val sdPack = if (pipelineMode == HeadReplacementPipelineMode.CLEAN_ERASE_3D_DIFFUSION) {
            SdInpaintPackManager.resolvePackForSeamBridge(context)
        } else {
            null
        }
        val embMs = (System.currentTimeMillis() - tEmb0).coerceAtLeast(1L)

        val targetW = targetBitmap.width
        val targetH = targetBitmap.height
        var compositePixels: IntArray? = IntArray(targetW * targetH)
        targetBitmap.getPixels(compositePixels!!, 0, targetW, 0, 0, targetW, targetH)
        val borderEnvelope = computeCanvasBorderEnvelope(canvasSize)

        var firstTargetHeadPreview: Bitmap? = null
        var firstHeadlessPreview512: Bitmap? = null
        var firstSourceCutoutPreview512: Bitmap? = null
        var firstIntermediatePortraitPreview512: Bitmap? = null
        var firstTgtCanvasMatrix: FloatArray? = null
        val placedHeadPoses = mutableListOf<PlacedHeadPose>()
        var totalGenMs = 0L
        var totalBlendMs = 0L
        var usedDiffusionBridge = false
        var usedLamaErasure = false
        var segLabelSummary = "3D-Clean-Erase"

        val workingTargetBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888).apply {
            setPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)
        }
        val placedHeadsTargetMask = if (targetFacesToReplace.size > 1) FloatArray(targetW * targetH) else null

        try {
            targetFacesToReplace.forEachIndexed { idx, targetFace ->
                if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
                val tgtMesh = tgtMeshes3D[idx]
                val tgtYawRatio = (tgtMesh.headPose.yawDeg / 55f).coerceIn(-1f, 1f)
                val tgtCanvasMatrix = estimateWideHeadCanvasMatrix(
                    srcLandmarks5 = targetFace.landmarks5,
                    cropSize = canvasSize,
                    yawRatio = tgtYawRatio
                )

                // Map target 5-point landmarks into canvas space for accurate cheek/eye & expression fusion
                val tgtCanvasPts5 = targetFace.landmarks5.take(5).map { p ->
                    PointF(
                        tgtCanvasMatrix[0] * p.x + tgtCanvasMatrix[1] * p.y + tgtCanvasMatrix[2],
                        tgtCanvasMatrix[3] * p.x + tgtCanvasMatrix[4] * p.y + tgtCanvasMatrix[5]
                    )
                }

                // Multi-face fix: always crop from workingTargetBitmap synced with compositePixels so face #2
                // sees face #1's newly placed head/hair and never samples stale pre-swap pixels.
                val alignedTargetHead = FaceAlignment.warpAffineCrop(
                    srcBitmap = workingTargetBitmap,
                    forwardMatrix2x3 = tgtCanvasMatrix,
                    dstSize = canvasSize
                )
                val alreadyPlacedCanvasMask = if (placedHeadsTargetMask != null && idx > 0) {
                    sampleTargetMaskToCanvas(
                        targetMask = placedHeadsTargetMask,
                        targetWidth = targetW,
                        targetHeight = targetH,
                        tgtCanvasMatrix = tgtCanvasMatrix,
                        canvasSize = canvasSize
                    )
                } else {
                    null
                }

                var cleanHeadlessTargetCanvas: Bitmap? = null
                var warpedSourceHead: Bitmap? = null
                var litSourceHead: Bitmap? = null
                var stage4Composited: Bitmap? = null
                var seamResolvedCanvas: Bitmap? = null
                var step2PortraitHeadCanvas: Bitmap? = null

                try {
                    // =========================================================================
                    // STAGE 3/6: Complete Target Head Removal (True Headless Background Canvas)
                    // =========================================================================
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
                    onProgress(
                        SwapStageProgress(
                            stepIndex = 3,
                            totalSteps = 6,
                            stageTitle = "Stage 3/6: Erase Target Head (${idx + 1}/${targetFacesToReplace.size})",
                            detailMessage = "Removing target face, skull & hair cleanly while protecting collar & shoulders...",
                            progressFraction = 0.44f
                        )
                    )

                    val rawTgtMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
                        ortEnv = ortEnv,
                        alignedHeadCrop = alignedTargetHead,
                        segModelFile = if (segFile.exists()) segFile else null,
                        mattingModelFile = if (matFile.exists()) matFile else null,
                        preferHardwareAccel = preferHardwareAccel,
                        preloadedSegformerSession = preloadedSegformerSession,
                        isCancelled = isCancelled
                    )
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
                    val tgtMasks = if (rawTgtMasks.usedSegmentationOnnx) {
                        rawTgtMasks
                    } else {
                        boostMatteForWideHeadCanvas(
                            alignedHeadCrop = alignedTargetHead,
                            baseMasks = rawTgtMasks,
                            canvasPts5 = tgtCanvasPts5,
                            suppressBgHalo = false,
                            yawRatio = tgtYawRatio
                        )
                    }
                    segLabelSummary = tgtMasks.segmentationSourceLabel

                    val tgtInBoundsMask = computeTargetInBoundsMask(
                        tgtCanvasMatrix = tgtCanvasMatrix,
                        canvasSize = canvasSize,
                        targetWidth = targetW,
                        targetHeight = targetH
                    )

                    val alignedTgtPx = IntArray(canvasSize * canvasSize)
                    alignedTargetHead.getPixels(alignedTgtPx, 0, canvasSize, 0, 0, canvasSize, canvasSize)

                    // Protect background high-chroma flower garlands/decor and target dupatta + draped shoulder hair
                    val decorProtectMask = buildBackgroundDecorProtectMask(
                        alignedHeadCrop = alignedTargetHead,
                        canvasPts5 = tgtCanvasPts5,
                        tgtMasks = tgtMasks
                    )
                    val clothProtectMask = buildClothingCollarProtectMask(
                        canvasSize = canvasSize,
                        tgtCanvasPts5 = tgtCanvasPts5,
                        tgtMasks = tgtMasks,
                        tgtCanvasMatrix = tgtCanvasMatrix,
                        targetDepthMap = targetDepthMap,
                        targetWidth = targetW,
                        targetHeight = targetH,
                        alignedTgtPx = alignedTgtPx,
                        decorProtectMask = decorProtectMask
                    )

                    // Broaden dilation (~9px on 768px; slightly wider when SegFormer is absent for hat brims & glasses frames)
                    // + yaw-shifted ear/earring/jaw coverage so Target's old ear, glasses temples, and outer fringe erase cleanly.
                    val dilateRadius = if (tgtMasks.usedSegmentationOnnx) {
                        max(6, canvasSize / 82)
                    } else {
                        max(8, canvasSize / 64)
                    }
                    val dilatedTargetEraseMask = dilateAlpha(tgtMasks.fullHeadAlpha, canvasSize, radiusPx = dilateRadius)
                    val lEyeT = tgtCanvasPts5.getOrElse(0) { PointF(canvasSize * 0.42f, canvasSize * 0.51f) }
                    val rEyeT = tgtCanvasPts5.getOrElse(1) { PointF(canvasSize * 0.58f, canvasSize * 0.51f) }
                    val tEyeMidX = (lEyeT.x + rEyeT.x) * 0.5f
                    val tEyeMidY = (lEyeT.y + rEyeT.y) * 0.5f
                    val rawEyeDistT = hypot((rEyeT.x - lEyeT.x).toDouble(), (rEyeT.y - lEyeT.y).toDouble())
                        .toFloat().coerceAtLeast(14f)
                    val mMidYT = if (tgtCanvasPts5.size >= 5) (tgtCanvasPts5[3].y + tgtCanvasPts5[4].y) * 0.5f else tEyeMidY + rawEyeDistT
                    val tEyeDist = max(rawEyeDistT, abs(mMidYT - tEyeMidY) * 0.88f)
                    val earCenterShiftX = -tgtYawRatio * tEyeDist * 0.42f
                    val earRxSpan = tEyeDist * (1.58f + abs(tgtYawRatio) * 0.40f)

                    val cleanEraseMask = FloatArray(canvasSize * canvasSize) { i ->
                        val y = i / canvasSize
                        val x = i % canvasSize
                        // Include Target ear/earring/glasses-temple envelope shifted toward visible ear/back-skull
                        val earDx = (x - (tEyeMidX + earCenterShiftX)) / earRxSpan
                        val earDy = (y - (tEyeMidY + tEyeDist * 0.45f)) / (tEyeDist * 0.94f)
                        val earR = sqrt(earDx * earDx + earDy * earDy)
                        val earBoost = when {
                            earR <= 0.85f -> 0.95f
                            earR >= 1.0f -> 0f
                            else -> ((1f - earR) / 0.15f).coerceIn(0f, 0.95f)
                        }
                        val placedProtect = alreadyPlacedCanvasMask?.get(i) ?: 0f

                        val baseErase = max(dilatedTargetEraseMask[i], earBoost)
                        (baseErase *
                            (1f - clothProtectMask[i]) *
                            (1f - decorProtectMask[i]) *
                            (1f - placedProtect * 0.96f) *
                            tgtInBoundsMask[i] *
                            borderEnvelope[i]).coerceIn(0f, 1f)
                    }

                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
                    val (erasedCanvas, lamaSuccess) = eraseTargetHeadCleanly(
                        ortEnv = ortEnv,
                        targetCanvas = alignedTargetHead,
                        eraseMask = cleanEraseMask,
                        clothProtectMask = clothProtectMask,
                        decorProtectMask = decorProtectMask,
                        tgtInBoundsMask = tgtInBoundsMask,
                        tgtCanvasMatrix = tgtCanvasMatrix,
                        targetDepthMap = targetDepthMap,
                        targetWidth = targetW,
                        targetHeight = targetH,
                        lamaModelFile = if (lamaFile.exists()) lamaFile else null,
                        preferHardwareAccel = preferHardwareAccel,
                        isCancelled = isCancelled
                    )
                    cleanHeadlessTargetCanvas = erasedCanvas
                    if (lamaSuccess) usedLamaErasure = true

                    // Crucial cancel guard immediately after Stage 3 (LaMa/Laplace) and BEFORE Stage 4 (3D Warp + SegFormer + Blend)
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

                    // =========================================================================
                    // STAGE 4/6 (Step 2): 3D Pose Source Portrait Reconstruction (No zoomed-out face paste)
                    // =========================================================================
                    val tGen0 = System.currentTimeMillis()
                    onProgress(
                        SwapStageProgress(
                            stepIndex = 4,
                            totalSteps = 6,
                            stageTitle = "Stage 4/6: Step 2 Full Portrait Assembly",
                            detailMessage = "Warping source head, hair & neck onto headless canvas (${tgtMesh.sourceLabel})...",
                            progressFraction = 0.58f
                        )
                    )

                    val warpedSrcResult = FaceMesh3DEngine.warpSourceHeadToTarget3DPoseWithBounds(
                        sourceBitmap = sourceBitmap,
                        sourceMesh = srcMesh3D,
                        targetMesh = tgtMesh,
                        targetCanvasMatrix2x3 = tgtCanvasMatrix,
                        canvasSize = canvasSize,
                        offsetX = offsetX,
                        offsetY = offsetY,
                        scaleAdjust = scaleAdjust
                    )
                    warpedSourceHead = warpedSrcResult.bitmap
                    val srcBoundsMask = warpedSrcResult.inBoundsMask
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

                    val rawWarpedSrcMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
                        ortEnv = ortEnv,
                        alignedHeadCrop = warpedSourceHead,
                        segModelFile = if (segFile.exists()) segFile else null,
                        mattingModelFile = if (matFile.exists()) matFile else null,
                        preferHardwareAccel = preferHardwareAccel,
                        preloadedSegformerSession = preloadedSegformerSession,
                        isCancelled = isCancelled
                    )
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
                    val adjPts5 = adjustLandmarks5(tgtCanvasPts5, canvasSize, offsetX, offsetY, scaleAdjust)
                    val rawOrBoostedSrcMasks = if (rawWarpedSrcMasks.usedSegmentationOnnx) {
                        rawWarpedSrcMasks
                    } else {
                        boostMatteForWideHeadCanvas(
                            alignedHeadCrop = warpedSourceHead,
                            baseMasks = rawWarpedSrcMasks,
                            canvasPts5 = adjPts5,
                            suppressBgHalo = true,
                            yawRatio = tgtYawRatio
                        )
                    }
                    // Complete cropped top/side hair crown cleanly & harmonize hair contrast with zero wall leaks
                    val (filteredFullHeadAlpha, filteredSkinAndNeckWeight) = completeCroppedCrownAndDecontaminateSource(
                        warpedSourceHead = warpedSourceHead,
                        alignedTargetHead = alignedTargetHead,
                        srcAlpha = rawOrBoostedSrcMasks.fullHeadAlpha,
                        srcSkinWeight = rawOrBoostedSrcMasks.skinAndNeckWeight,
                        tgtAlpha = tgtMasks.fullHeadAlpha,
                        tgtSkinWeight = tgtMasks.skinAndNeckWeight,
                        clothProtectMask = clothProtectMask,
                        decorProtectMask = decorProtectMask,
                        srcBoundsMask = srcBoundsMask,
                        tgtInBoundsMask = tgtInBoundsMask,
                        borderEnvelope = borderEnvelope,
                        canvasPts5 = adjPts5,
                        canvasSize = canvasSize
                    )
                    totalGenMs += (System.currentTimeMillis() - tGen0).coerceAtLeast(1L)
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

                    val tBlend0 = System.currentTimeMillis()
                    // Harmonize hair, neck & ambient scene lighting on the intermediate portrait cleanly;
                    // final exact facial skin tone & ArcFace identity come from Step 3 at high resolution.
                    val step2ToneMode = if (skinToneMode == SkinToneSourceMode.SOURCE_IDENTITY) {
                        SkinToneSourceMode.BALANCED_BLEND
                    } else {
                        skinToneMode
                    }
                    litSourceHead = if (enableColorTransfer) {
                        GhostHeadReplacementEngine.adaptHeadSkinAndLightingSelective(
                            sourceHeadCrop = warpedSourceHead,
                            targetHeadCrop = alignedTargetHead,
                            skinAndNeckWeight = filteredSkinAndNeckWeight,
                            skinToneMode = step2ToneMode,
                            overrideLandmarks5 = adjPts5
                        )
                    } else {
                        warpedSourceHead
                    }
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

                    // Composite warped Source head onto the completely headless Target canvas (protecting collar)
                    stage4Composited = compositeSourceOverHeadlessCanvas(
                        sourceHead = litSourceHead,
                        headlessTarget = erasedCanvas,
                        sourceAlpha = filteredFullHeadAlpha,
                        clothProtectMask = clothProtectMask
                    )

                    // Capture Step 1 & Output 2 UI previews BEFORE recycling Stage 1-4 bitmaps so Stage 5 SD 1.5 has max free RAM
                    if (firstTargetHeadPreview == null) {
                        firstTargetHeadPreview = alignedTargetHead.copy(Bitmap.Config.ARGB_8888, false)
                        firstTgtCanvasMatrix = tgtCanvasMatrix.copyOf()
                        firstHeadlessPreview512 = if (erasedCanvas.width == 512) {
                            erasedCanvas.copy(Bitmap.Config.ARGB_8888, false)
                        } else {
                            Bitmap.createScaledBitmap(erasedCanvas, 512, 512, true)
                        }
                        val cutoutFull = buildTransparentCutoutPreviewBitmap(
                            cropBitmap = warpedSourceHead,
                            alpha = filteredFullHeadAlpha
                        )
                        firstSourceCutoutPreview512 = if (cutoutFull.width == 512) {
                            cutoutFull
                        } else {
                            Bitmap.createScaledBitmap(cutoutFull, 512, 512, true).also {
                                cutoutFull.recycle()
                            }
                        }
                    }
                    if (litSourceHead !== warpedSourceHead) {
                        litSourceHead.recycle()
                    }
                    litSourceHead = null
                    warpedSourceHead.recycle()
                    warpedSourceHead = null
                    erasedCanvas.recycle()
                    cleanHeadlessTargetCanvas = null

                    // =========================================================================
                    // STAGE 5/6: Diffusion Seam & Hair/Neck Bridge (Strict Identity Protection)
                    // =========================================================================
                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
                    OnnxProtobufInspector.promoteStuckCrashBreadcrumbIfAny(context)
                    val ramStatusBeforeStage5 = SdInpaintOnnxEngine.queryDeviceRamStatus(context)
                    val isRuntimeCrashLatched = OnnxProtobufInspector.isRuntimeLowRamLatched(context)
                    val (canRunSdSafely, stage5SdMaxDimPx) = resolveStage5BridgePlan(
                        lowMemoryMode = lowMemoryMode,
                        ramStatus = ramStatusBeforeStage5,
                        isRuntimeCrashLatched = isRuntimeCrashLatched
                    )
                    val useSdBridgeNow = sdPack != null && ortEnv != null && canRunSdSafely

                    onProgress(
                        SwapStageProgress(
                            stepIndex = 5,
                            totalSteps = 6,
                            stageTitle = "Stage 5/6: ${if (useSdBridgeNow) "Diffusion Seam Bridge" else "Fast 3D Seam Bridge"}",
                            detailMessage = if (useSdBridgeNow) {
                                "Freeing Stage 1-4 sessions & preparing SD 1.5 (${sdPack!!.packType.displayLabel}, ${stage5SdMaxDimPx}px)..."
                            } else if (isRuntimeCrashLatched) {
                                "Low-RAM Safe Guard active after prior force-stop — applying multi-band neck & hair seam bridge..."
                            } else {
                                "Applying multi-band neck-collar & hair perimeter seam bridge..."
                            },
                            progressFraction = 0.72f
                        )
                    )

                    val innerFaceProtectMask = FaceMesh3DEngine.buildInnerFaceIdentityMask(
                        targetMesh = tgtMesh,
                        targetCanvasMatrix2x3 = tgtCanvasMatrix,
                        canvasSize = canvasSize,
                        offsetX = offsetX,
                        offsetY = offsetY,
                        scaleAdjust = scaleAdjust
                    )

                    seamResolvedCanvas = if (useSdBridgeNow) {
                        // Free all Stage 1-4 ONNX sessions (det_10g, facemesh, arcface, hyperswap, segformer, mivolo)
                        // so SD 1.5 UNet/VAE has maximum native RAM and cannot OOM/lmkd-kill on mobile devices.
                        OnnxProtobufInspector.clearCachedSessions()
                        SdInpaintOnnxEngine.clearSessionCache()
                        System.gc()

                        val sdMaxDimPx = stage5SdMaxDimPx
                        val sdAttempt = runCatching {
                            runDiffusionSeamBridge(
                                context = context,
                                ortEnv = ortEnv!!,
                                pack = sdPack!!,
                                stage4Canvas = stage4Composited!!,
                                sourceAlpha = filteredFullHeadAlpha,
                                targetEraseMask = cleanEraseMask,
                                innerFaceProtectMask = innerFaceProtectMask,
                                clothProtectMask = clothProtectMask,
                                preferHardwareAccel = preferHardwareAccel,
                                maxPreferredResolutionPx = sdMaxDimPx,
                                isCancelled = isCancelled,
                                onDetailProgress = { sdMsg ->
                                    val frac = when {
                                        sdMsg.startsWith("1/4") -> 0.72f
                                        sdMsg.startsWith("2/4") -> 0.74f
                                        sdMsg.startsWith("3/4 Diffusion step") -> {
                                            val m = Regex("step\\s+(\\d+)/(\\d+)", RegexOption.IGNORE_CASE).find(sdMsg)
                                            val cur = m?.groupValues?.getOrNull(1)?.toFloatOrNull() ?: 1f
                                            val tot = (m?.groupValues?.getOrNull(2)?.toFloatOrNull() ?: 4f).coerceAtLeast(1f)
                                            (0.75f + (cur / tot) * 0.08f).coerceIn(0.75f, 0.83f)
                                        }
                                        sdMsg.startsWith("3/4") -> 0.75f
                                        sdMsg.startsWith("4/4") -> 0.84f
                                        else -> 0.76f
                                    }
                                    onProgress(
                                        SwapStageProgress(
                                            stepIndex = 5,
                                            totalSteps = 6,
                                            stageTitle = "Stage 5/6: Diffusion Seam Bridge",
                                            detailMessage = sdMsg,
                                            progressFraction = frac
                                        )
                                    )
                                }
                            )
                        }
                        sdAttempt.exceptionOrNull()?.let { err ->
                            if (err is java.util.concurrent.CancellationException || isCancelled()) {
                                SdInpaintOnnxEngine.clearSessionCache()
                                throw err
                            }
                            SdInpaintOnnxEngine.clearSessionCache()
                            System.gc()
                        }
                        val sdOut = sdAttempt.getOrNull()
                        if (sdOut != null) {
                            usedDiffusionBridge = true
                            sdOut
                        } else {
                            applyFastSeamFeatherBridge(
                                composited = stage4Composited!!,
                                sourceAlpha = filteredFullHeadAlpha,
                                innerFaceProtectMask = innerFaceProtectMask,
                                clothProtectMask = clothProtectMask
                            )
                        }
                    } else {
                        applyFastSeamFeatherBridge(
                            composited = stage4Composited!!,
                            sourceAlpha = filteredFullHeadAlpha,
                            innerFaceProtectMask = innerFaceProtectMask,
                            clothProtectMask = clothProtectMask
                        )
                    }

                    if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

                    // Mild hair/neck crispness before warping into Intermediate Full Portrait (NO zoomed-out GFPGAN!)
                    step2PortraitHeadCanvas = GhostHeadReplacementEngine.applyTiledDetailEnhancement(
                        bitmap = seamResolvedCanvas!!,
                        amount = 0.16f
                    )

                    // Union envelope covers both the erased old target head & the new 3D source head while
                    // strictly feathering canvas borders, protecting decor garlands, target collar, and already-placed adjacent heads.
                    val baseEnvelopeAlpha = FloatArray(canvasSize * canvasSize) { i ->
                        val rawUnion = max(filteredFullHeadAlpha[i], cleanEraseMask[i] * 0.96f)
                        val placedProtect = alreadyPlacedCanvasMask?.get(i) ?: 0f
                        val keepPlacedScale = 1f - placedProtect * (1f - innerFaceProtectMask[i] * 0.65f) * 0.95f
                        val protected = rawUnion *
                            (1f - clothProtectMask[i] * 0.90f) *
                            keepPlacedScale.coerceIn(0f, 1f) *
                            tgtInBoundsMask[i] *
                            borderEnvelope[i]
                        protected.coerceIn(0f, 1f)
                    }

                    // Warp Step 2 intermediate portrait onto compositePixels
                    GhostHeadReplacementEngine.warpHeadCropBackToTarget(
                        targetPixels = compositePixels!!,
                        targetWidth = targetW,
                        targetHeight = targetH,
                        headCrop = step2PortraitHeadCanvas!!,
                        forwardHeadMatrix2x3 = tgtCanvasMatrix,
                        envelopeAlpha = baseEnvelopeAlpha,
                        blendStrength = blendStrength,
                        offsetX = 0f,
                        offsetY = 0f,
                        scaleAdjust = 1.0f
                    )

                    if (idx + 1 < targetFacesToReplace.size) {
                        workingTargetBitmap.setPixels(compositePixels!!, 0, targetW, 0, 0, targetW, targetH)
                        if (placedHeadsTargetMask != null) {
                            accumulateCanvasMaskOntoTarget(
                                targetMask = placedHeadsTargetMask,
                                targetWidth = targetW,
                                targetHeight = targetH,
                                canvasMask = filteredFullHeadAlpha,
                                forwardMatrix2x3 = tgtCanvasMatrix,
                                canvasSize = canvasSize
                            )
                        }
                    }

                    // Record exact facial landmarks of the placed Step 2 head in target image coordinates
                    val invTgtCanvas = FaceAlignment.invertAffine2x3(tgtCanvasMatrix)
                    val placedLandmarks5 = adjPts5.map { p ->
                        PointF(
                            invTgtCanvas[0] * p.x + invTgtCanvas[1] * p.y + invTgtCanvas[2],
                            invTgtCanvas[3] * p.x + invTgtCanvas[4] * p.y + invTgtCanvas[5]
                        )
                    }
                    val placedEyeMidX = (placedLandmarks5[0].x + placedLandmarks5[1].x) * 0.5f
                    val placedEyeMidY = (placedLandmarks5[0].y + placedLandmarks5[1].y) * 0.5f
                    val placedEyeDist = hypot(
                        (placedLandmarks5[1].x - placedLandmarks5[0].x).toDouble(),
                        (placedLandmarks5[1].y - placedLandmarks5[0].y).toDouble()
                    ).toFloat().coerceAtLeast(10f)
                    val fallbackBbox = RectF(
                        (placedEyeMidX - placedEyeDist * 1.15f).coerceIn(0f, targetW.toFloat()),
                        (placedEyeMidY - placedEyeDist * 0.95f).coerceIn(0f, targetH.toFloat()),
                        (placedEyeMidX + placedEyeDist * 1.15f).coerceIn(0f, targetW.toFloat()),
                        (placedEyeMidY + placedEyeDist * 1.45f).coerceIn(0f, targetH.toFloat())
                    )
                    placedHeadPoses.add(
                        PlacedHeadPose(
                            targetFaceIndex = targetFace.index,
                            fallbackDetectedFace = DetectedFace(
                                index = targetFace.index,
                                boundingBox = fallbackBbox,
                                score = 0.98f,
                                landmarks5 = placedLandmarks5,
                                detectorSource = "Step2PortraitPose"
                            ),
                            eyeMidPoint = PointF(placedEyeMidX, placedEyeMidY),
                            eyeDist = placedEyeDist
                        )
                    )

                    if (firstIntermediatePortraitPreview512 == null) {
                        val canvasRef = step2PortraitHeadCanvas!!
                        firstIntermediatePortraitPreview512 = if (canvasRef.width == 512) {
                            canvasRef.copy(Bitmap.Config.ARGB_8888, false)
                        } else {
                            Bitmap.createScaledBitmap(canvasRef, 512, 512, true)
                        }
                    }

                    totalBlendMs += (System.currentTimeMillis() - tBlend0).coerceAtLeast(1L)
                } finally {
                    alignedTargetHead.recycle()
                    cleanHeadlessTargetCanvas?.recycle()
                    if (litSourceHead != null && litSourceHead !== warpedSourceHead) litSourceHead.recycle()
                    warpedSourceHead?.recycle()
                    if (stage4Composited != null && stage4Composited !== seamResolvedCanvas) stage4Composited.recycle()
                    if (step2PortraitHeadCanvas != null && step2PortraitHeadCanvas !== seamResolvedCanvas) step2PortraitHeadCanvas.recycle()
                    seamResolvedCanvas?.recycle()
                }
            }
        } finally {
            workingTargetBitmap.recycle()
        }

        if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

        // =========================================================================
        // STAGE 6/6 (Step 3): High-Res Face Swap (`ArcFace` -> `HyperSwap 256px` -> `GFPGAN 512`)
        // on the Step 2 Intermediate Full Portrait
        // =========================================================================
        val intermediatePortraitBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        intermediatePortraitBitmap.setPixels(compositePixels!!, 0, targetW, 0, 0, targetW, targetH)
        // Release the multi-megapixel pixel buffer before Stage 6 loads ArcFace/HyperSwap/GFPGAN
        @Suppress("UNUSED_VALUE")
        compositePixels = null

        // Release heavy Step 1/2 temporary ONNX sessions (LaMa/Depth/FaceMesh/SD) so Step 3 has full RAM
        if (ortEnv != null && preloadedArcFaceSession == null && preloadedSwapSession == null && preloadedGfpganSession == null) {
            SdInpaintOnnxEngine.clearSessionCache()
            OnnxProtobufInspector.clearCachedSessions()
            if (lowMemoryMode) System.gc()
        }

        if (isCancelled()) {
            intermediatePortraitBitmap.recycle()
            throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
        }

        onProgress(
            SwapStageProgress(
                stepIndex = 6,
                totalSteps = 6,
                stageTitle = "Stage 6/6: Step 3 Final Identity Swap",
                detailMessage = "Running SCRFD -> ArcFace -> HyperSwap 256px -> GFPGAN 512px on Step 2 portrait...",
                progressFraction = 0.88f
            )
        )

        val detectedOnIntermediate = if (ortEnv != null && detFile.exists() && detFile.length() > 1024L) {
            runCatching {
                ScrfdFaceDetector.detectFacesOnnx(
                    ortEnv = ortEnv,
                    detModelFile = detFile,
                    bitmap = intermediatePortraitBitmap,
                    confThreshold = 0.22f,
                    nmsThreshold = 0.40f
                )
            }.getOrNull().orEmpty()
        } else {
            emptyList()
        }

        if (isCancelled()) {
            intermediatePortraitBitmap.recycle()
            throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
        }

        val resolvedFacesOnIntermediate = placedHeadPoses.map { pose ->
            val maxDist = pose.eyeDist * 1.85f
            val bestMatch = detectedOnIntermediate
                .filter { cand -> cand.landmarks5.size >= 5 }
                .minByOrNull { cand ->
                    val cx = (cand.landmarks5[0].x + cand.landmarks5[1].x) * 0.5f
                    val cy = (cand.landmarks5[0].y + cand.landmarks5[1].y) * 0.5f
                    hypot((cx - pose.eyeMidPoint.x).toDouble(), (cy - pose.eyeMidPoint.y).toDouble()).toFloat()
                }
            val matchedDist = bestMatch?.let { cand ->
                val cx = (cand.landmarks5[0].x + cand.landmarks5[1].x) * 0.5f
                val cy = (cand.landmarks5[0].y + cand.landmarks5[1].y) * 0.5f
                hypot((cx - pose.eyeMidPoint.x).toDouble(), (cy - pose.eyeMidPoint.y).toDouble()).toFloat()
            } ?: Float.MAX_VALUE
            if (bestMatch != null && matchedDist <= maxDist) {
                bestMatch.copy(index = pose.targetFaceIndex)
            } else {
                pose.fallbackDetectedFace
            }
        }

        val step3Res = InSwapperEngine.executeFaceSwap(
            context = context,
            ortEnv = ortEnv,
            sourceBitmap = sourceBitmap,
            sourceFace = sourceFace,
            targetBitmap = intermediatePortraitBitmap,
            targetFacesToReplace = resolvedFacesOnIntermediate,
            enableColorTransfer = enableColorTransfer,
            enableProvenanceWatermark = enableProvenanceWatermark,
            allowTwoModelFallbackForTesting = allowTwoModelFallbackForTesting,
            skinToneMode = skinToneMode,
            faceReactionMode = faceReactionMode,
            enableOcclusionProtection = true,
            portraitBlurStrength = portraitBlurStrength,
            blendStrength = blendStrength,
            enhancementStrength = enhancementStrength.coerceAtLeast(0.85f),
            offsetX = 0f,
            offsetY = 0f,
            scaleAdjust = 1.0f,
            preferHardwareAccel = preferHardwareAccel,
            identityPreservationMode = false,
            preloadedArcFaceSession = preloadedArcFaceSession,
            preloadedSwapSession = preloadedSwapSession,
            preloadedGfpganSession = preloadedGfpganSession,
            preloadedSegformerSession = preloadedSegformerSession,
            preloadedEmap512x512 = preloadedEmap512x512,
            captureDiagnostics = true,
            autoBodyAgeMatch = autoBodyAgeMatch,
            faceAgingFactor = faceAgingFactor,
            swapStrategyOption = FaceSwapStrategyOption.OPTION_A_MASK_FIDELITY,
            precomputedTargetDepthMap = targetDepthMap,
            isCancelled = isCancelled,
            onProgress = { sub ->
                if (ortEnv != null) {
                    onProgress(
                        SwapStageProgress(
                            stepIndex = 6,
                            totalSteps = 6,
                            stageTitle = "Stage 6/6: ${sub.stageTitle}",
                            detailMessage = sub.detailMessage,
                            progressFraction = (0.88f + sub.progressFraction * 0.11f).coerceIn(0.88f, 0.99f)
                        )
                    )
                }
            }
        )

        // When GFPGAN neural super-resolution is not installed on device, HyperSwap runs at 256px
        // on intermediatePortraitBitmap. Transfer high-frequency crispness (irises, eyelashes, teeth,
        // skin micro-pores, bindi) from Step 2 portrait's high-res features onto step3Res without altering identity.
        if (!step3Res.usedNeuralGfpgan && !lowMemoryMode) {
            restoreMicroDetailFromIntermediatePortrait(
                step3Output = step3Res.outputBitmap,
                intermediatePortrait = intermediatePortraitBitmap,
                faces = resolvedFacesOnIntermediate,
                amount = 0.44f
            )
        }

        val step3Canvas512 = firstTgtCanvasMatrix?.let { mCanvas ->
            val m512 = FloatArray(6) { i -> mCanvas[i] * (512f / canvasSize.toFloat()) }
            FaceAlignment.warpAffineCrop(step3Res.outputBitmap, m512, 512)
        } ?: step3Res.stage3FinalBlend512?.let { it.copy(Bitmap.Config.ARGB_8888, false) }

        intermediatePortraitBitmap.recycle()
        step3Res.alignedTarget128.recycle()
        step3Res.stage1SwapOnly512?.takeIf { it !== step3Res.rawSwapped128 }?.recycle()
        step3Res.stage2SwapRestore512?.takeIf { it !== step3Res.rawSwapped128 }?.recycle()
        step3Res.stage3FinalBlend512?.takeIf { it !== step3Res.rawSwapped128 }?.recycle()

        val totalMs = (System.currentTimeMillis() - tStart).coerceAtLeast(1L)
        val summary = buildString {
            append("3-Step Portrait-First (${pipelineMode.title}): ")
            append("Step1(")
            append(if (usedLamaErasure) "LaMa Headless" else "2D Exterior Headless")
            append("+$segLabelSummary")
            if (targetDepthMap != null) append("+${targetDepthMap.modelUsedLabel}")
            append(") -> Step2(3D Portrait")
            if (usedDiffusionBridge) append("+SD1.5 Seam") else append("+SeamBridge")
            append(") -> Step3(${step3Res.engineUsed} ArcFace+HyperSwap+GFPGAN512)")
            if (portraitBlurStrength > 0.03f) append(" + Bokeh ${(portraitBlurStrength * 100).toInt()}%")
            append(" | engineUsed=${step3Res.engineUsed}")
        }

        return FaceSwapExecutionResult(
            outputBitmap = step3Res.outputBitmap,
            alignedSource112 = step3Res.alignedSource112,
            alignedTarget128 = firstTargetHeadPreview ?: Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888),
            rawSwapped128 = step3Res.rawSwapped128,
            swappedFacesCount = step3Res.swappedFacesCount,
            detectionMs = (detMs + step3Res.detectionMs).coerceAtLeast(1L),
            embeddingMs = (embMs + step3Res.embeddingMs).coerceAtLeast(1L),
            inswapperMs = (totalGenMs + step3Res.inswapperMs).coerceAtLeast(1L),
            blendingMs = (totalBlendMs + step3Res.blendingMs).coerceAtLeast(1L),
            totalMs = totalMs,
            pipelineSummary = summary,
            stage1SwapOnly512 = firstHeadlessPreview512,
            stage2SwapRestore512 = firstIntermediatePortraitPreview512,
            stage3FinalBlend512 = step3Canvas512,
            stageTimings = step3Res.stageTimings.copy(totalMs = totalMs),
            benchmarkReports = step3Res.benchmarkReports,
            engineUsed = step3Res.engineUsed,
            ageHarmonySummary = step3Res.ageHarmonySummary ?: firstAgeHarmonySummary,
            headlessTargetPreview = firstHeadlessPreview512,
            sourceHeadCutoutPreview = firstSourceCutoutPreview512
        )
    }

    private fun buildTransparentCutoutPreviewBitmap(cropBitmap: Bitmap, alpha: FloatArray): Bitmap {
        val w = cropBitmap.width
        val h = cropBitmap.height
        val total = w * h
        val px = IntArray(total)
        cropBitmap.getPixels(px, 0, w, 0, 0, w, h)
        for (i in 0 until total) {
            val aByte = (alpha[i].coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
            px[i] = (aByte shl 24) or (px[i] and 0x00FFFFFF)
        }
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
            setPixels(px, 0, w, 0, 0, w, h)
        }
    }

    private fun computeCanvasBorderEnvelope(canvasSize: Int): FloatArray {
        val margin = max(4, (canvasSize * 0.02f).toInt())
        val featherSpan = max(12, (canvasSize * 0.055f).toInt()).toFloat()
        val env = FloatArray(canvasSize * canvasSize)
        for (y in 0 until canvasSize) {
            val dy = min(y, canvasSize - 1 - y)
            val row = y * canvasSize
            for (x in 0 until canvasSize) {
                val dx = min(x, canvasSize - 1 - x)
                val edgeDist = min(dx, dy)
                env[row + x] = when {
                    edgeDist < margin -> 0f
                    edgeDist >= margin + featherSpan -> 1f
                    else -> {
                        val t = ((edgeDist - margin) / featherSpan).coerceIn(0f, 1f)
                        t * t * (3f - 2f * t)
                    }
                }
            }
        }
        return env
    }

    private fun sampleTargetMaskToCanvas(
        targetMask: FloatArray,
        targetWidth: Int,
        targetHeight: Int,
        tgtCanvasMatrix: FloatArray,
        canvasSize: Int
    ): FloatArray {
        val out = FloatArray(canvasSize * canvasSize)
        val inv = FaceAlignment.invertAffine2x3(tgtCanvasMatrix)
        for (y in 0 until canvasSize) {
            val row = y * canvasSize
            val baseTx = inv[1] * y + inv[2]
            val baseTy = inv[4] * y + inv[5]
            for (x in 0 until canvasSize) {
                val tx = (inv[0] * x + baseTx).toInt()
                val ty = (inv[3] * x + baseTy).toInt()
                if (tx in 0 until targetWidth && ty in 0 until targetHeight) {
                    out[row + x] = targetMask[ty * targetWidth + tx]
                }
            }
        }
        return out
    }

    private fun accumulateCanvasMaskOntoTarget(
        targetMask: FloatArray,
        targetWidth: Int,
        targetHeight: Int,
        canvasMask: FloatArray,
        forwardMatrix2x3: FloatArray,
        canvasSize: Int
    ) {
        val m00 = forwardMatrix2x3[0]
        val m01 = forwardMatrix2x3[1]
        val m02 = forwardMatrix2x3[2]
        val m10 = forwardMatrix2x3[3]
        val m11 = forwardMatrix2x3[4]
        val m12 = forwardMatrix2x3[5]
        for (y in 0 until targetHeight) {
            val rowOff = y * targetWidth
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            for (x in 0 until targetWidth) {
                val u = (m00 * x + baseU).toInt()
                val v = (m10 * x + baseV).toInt()
                if (u in 0 until canvasSize && v in 0 until canvasSize) {
                    val a = canvasMask[v * canvasSize + u]
                    if (a > targetMask[rowOff + x]) {
                        targetMask[rowOff + x] = a
                    }
                }
            }
        }
    }

    private fun estimateWideHeadCanvasMatrix(
        srcLandmarks5: List<PointF>,
        cropSize: Int,
        yawRatio: Float = 0f
    ): FloatArray {
        val s = cropSize / 256.0f
        val absYaw = abs(yawRatio)
        val yawShiftX = (-yawRatio * (7.0f + absYaw * 7.5f) * s).coerceIn(-22f * s, 22f * s)

        // Foreshortening guard for profile poses (|yaw| > 45 deg): 2D eye distance shrinks while
        // eye-to-mouth height remains stable; keep horizontal spacing proportional to avoid over-zooming.
        val rawEyeDist = hypot(
            (srcLandmarks5[1].x - srcLandmarks5[0].x).toDouble(),
            (srcLandmarks5[1].y - srcLandmarks5[0].y).toDouble()
        ).toFloat().coerceAtLeast(4f)
        val eyeMidX = (srcLandmarks5[0].x + srcLandmarks5[1].x) * 0.5f
        val eyeMidY = (srcLandmarks5[0].y + srcLandmarks5[1].y) * 0.5f
        val mouthMidX = (srcLandmarks5[3].x + srcLandmarks5[4].x) * 0.5f
        val mouthMidY = (srcLandmarks5[3].y + srcLandmarks5[4].y) * 0.5f
        val eyeToMouthDist = hypot((mouthMidX - eyeMidX).toDouble(), (mouthMidY - eyeMidY).toDouble())
            .toFloat().coerceAtLeast(4f)
        val aspectRatio = (rawEyeDist / (eyeToMouthDist * 0.95f)).coerceIn(0.45f, 1.0f)
        val horzEyeScale = if (absYaw > 0.30f) aspectRatio else 1.0f
        val halfEyeX = 19.0f * horzEyeScale
        val halfMouthX = 16.0f * (0.45f + 0.55f * horzEyeScale)

        // Map inter-ocular distance to 38/256 (~0.148 * cropSize) centered at y=132/256
        // leaving ~3.47x eyeDist headroom above eyes for tall/voluminous hair/hats and ~3.26x below for neck/collar.
        val dstPoints = arrayOf(
            PointF((128.0f - halfEyeX) * s + yawShiftX, 132.0f * s),
            PointF((128.0f + halfEyeX) * s + yawShiftX, 132.0f * s),
            PointF(128.0f * s + yawShiftX, 153.0f * s),
            PointF((128.0f - halfMouthX) * s + yawShiftX, 172.0f * s),
            PointF((128.0f + halfMouthX) * s + yawShiftX, 172.0f * s)
        )
        var srcMeanX = 0f
        var srcMeanY = 0f
        var dstMeanX = 0f
        var dstMeanY = 0f
        for (i in 0 until 5) {
            srcMeanX += srcLandmarks5[i].x
            srcMeanY += srcLandmarks5[i].y
            dstMeanX += dstPoints[i].x
            dstMeanY += dstPoints[i].y
        }
        srcMeanX /= 5f
        srcMeanY /= 5f
        dstMeanX /= 5f
        dstMeanY /= 5f

        var numA = 0f
        var numB = 0f
        var denom = 0f
        for (i in 0 until 5) {
            val u = srcLandmarks5[i].x - srcMeanX
            val v = srcLandmarks5[i].y - srcMeanY
            val xp = dstPoints[i].x - dstMeanX
            val yp = dstPoints[i].y - dstMeanY
            numA += u * xp + v * yp
            numB += u * yp - v * xp
            denom += u * u + v * v
        }
        if (denom < 1e-6f) return floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
        val a = numA / denom
        val b = numB / denom
        val tx = dstMeanX - (a * srcMeanX - b * srcMeanY)
        val ty = dstMeanY - (b * srcMeanX + a * srcMeanY)
        return floatArrayOf(a, -b, tx, b, a, ty)
    }

    /**
     * Enhances the Chromatic-Geodesic fallback matte in wide-headroom canvases so large target hairstyles,
     * wide hat brims, and big sunglasses frames/temples are cleanly detected via exterior background
     * chromatic divergence while preserving untouched borders.
     */
    private fun boostMatteForWideHeadCanvas(
        alignedHeadCrop: Bitmap,
        baseMasks: HeadSegmentationMasks,
        canvasPts5: List<PointF>,
        suppressBgHalo: Boolean = false,
        yawRatio: Float = 0f
    ): HeadSegmentationMasks {
        if (canvasPts5.size < 5) return baseMasks
        val size = alignedHeadCrop.width
        val total = size * size
        val pixels = IntArray(total)
        alignedHeadCrop.getPixels(pixels, 0, size, 0, 0, size, size)

        val lEye = canvasPts5[0]
        val rEye = canvasPts5[1]
        val eyeMidX = (lEye.x + rEye.x) * 0.5f
        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val rawEyeDist = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble()).toFloat().coerceAtLeast(10f)
        val mouthMidY = (canvasPts5[3].y + canvasPts5[4].y) * 0.5f
        val eyeDist = max(rawEyeDist, abs(mouthMidY - eyeMidY) * 0.88f)
        val absYaw = abs(yawRatio)

        // Sample exterior background colors ONLY from top-left/top-right corners and left/right borders
        // (never center top row where tall hair crowns or hats may reach)
        val bgR = mutableListOf<Float>()
        val bgG = mutableListOf<Float>()
        val bgB = mutableListOf<Float>()
        val step = (size / 14).coerceAtLeast(2)
        val cornerEnd = (size * 0.18f).toInt().coerceAtLeast(8)
        for (x in 3..cornerEnd step step) {
            val cL = pixels[3 * size + x]
            bgR.add(((cL ushr 16) and 0xFF).toFloat())
            bgG.add(((cL ushr 8) and 0xFF).toFloat())
            bgB.add((cL and 0xFF).toFloat())
            val cR = pixels[3 * size + (size - 1 - x)]
            bgR.add(((cR ushr 16) and 0xFF).toFloat())
            bgG.add(((cR ushr 8) and 0xFF).toFloat())
            bgB.add((cR and 0xFF).toFloat())
        }
        for (y in 4 until (size * 0.10f).toInt().coerceAtLeast(8) step step) {
            for (x in listOf(3, size - 4)) {
                val c = pixels[y * size + x]
                bgR.add(((c ushr 16) and 0xFF).toFloat())
                bgG.add(((c ushr 8) and 0xFF).toFloat())
                bgB.add((c and 0xFF).toFloat())
            }
        }

        val boostedAlpha = baseMasks.fullHeadAlpha.copyOf()
        val hairCenterX = eyeMidX - yawRatio * eyeDist * 0.40f
        val hairCenterY = eyeMidY - eyeDist * 0.60f
        val hairRx = eyeDist * (if (!suppressBgHalo) (2.75f + absYaw * 0.35f) else 2.45f)
        val hairRy = eyeDist * (if (!suppressBgHalo) 2.75f else 2.45f)
        val faceCoreRx = eyeDist * 1.15f
        val faceCoreRy = eyeDist * 1.35f
        val borderMargin = (size * 0.025f).toInt().coerceAtLeast(3)

        for (y in borderMargin until size - borderMargin) {
            val rowOffset = y * size
            val dy = (y - hairCenterY) / hairRy
            val fDy = (y - (eyeMidY + eyeDist * 0.35f)) / faceCoreRy
            val inGlassesBandY = !suppressBgHalo && y >= (eyeMidY - eyeDist * 0.45f) && y <= (eyeMidY + eyeDist * 0.52f)
            for (x in borderMargin until size - borderMargin) {
                val dx = (x - hairCenterX) / hairRx
                val fDx = (x - eyeMidX) / faceCoreRx
                val faceCoreR = sqrt(fDx * fDx + fDy * fDy)
                val ellR = sqrt(dx * dx + dy * dy)
                val idx = rowOffset + x

                val c = pixels[idx]
                val r = ((c ushr 16) and 0xFF).toFloat()
                val g = ((c ushr 8) and 0xFF).toFloat()
                val b = (c and 0xFF).toFloat()

                var minDistSq = Float.MAX_VALUE
                for (k in bgR.indices) {
                    val dr = r - bgR[k]
                    val dg = g - bgG[k]
                    val db = b - bgB[k]
                    val dSq = dr * dr * 0.30f + dg * dg * 0.59f + db * db * 0.11f
                    if (dSq < minDistSq) minDistSq = dSq
                }
                val fgSignal = (sqrt(minDistSq) / 28.0f).coerceIn(0f, 1f)

                if (ellR <= 1.0f) {
                    val envelope = if (ellR <= 0.85f) 1.0f else ((1.0f - ellR) / 0.15f).coerceIn(0f, 1f)
                    val wideHairA = fgSignal * envelope
                    if (wideHairA > boostedAlpha[idx]) {
                        boostedAlpha[idx] = wideHairA
                    }
                }
                if (inGlassesBandY && abs(x - eyeMidX) <= eyeDist * 1.38f && fgSignal > 0.22f) {
                    val templeFade = (1f - ((abs(x - eyeMidX) - eyeDist * 1.10f) / (eyeDist * 0.28f)).coerceIn(0f, 1f))
                    val glassesA = (fgSignal * 1.15f).coerceIn(0f, 1f) * templeFade
                    if (glassesA > boostedAlpha[idx]) {
                        boostedAlpha[idx] = glassesA
                    }
                }
                if (suppressBgHalo && faceCoreR > 1.05f && fgSignal < 0.25f) {
                    boostedAlpha[idx] = (boostedAlpha[idx] * (fgSignal / 0.25f)).coerceIn(0f, 1f)
                }
            }
        }

        return baseMasks.copy(fullHeadAlpha = boostedAlpha)
    }

    private fun adjustLandmarks5(
        canvasPts5: List<PointF>,
        canvasSize: Int,
        offsetX: Float,
        offsetY: Float,
        scaleAdjust: Float
    ): List<PointF> {
        if (canvasPts5.size < 5) return canvasPts5
        val eyeMidX = (canvasPts5[0].x + canvasPts5[1].x) * 0.5f
        val eyeMidY = (canvasPts5[0].y + canvasPts5[1].y) * 0.5f
        val safeScale = scaleAdjust.coerceIn(0.80f, 1.25f)
        val dx = offsetX * (canvasSize / 320f)
        val dy = offsetY * (canvasSize / 320f)
        return canvasPts5.map { p ->
            PointF(
                eyeMidX + (p.x - eyeMidX) * safeScale + dx,
                eyeMidY + (p.y - eyeMidY) * safeScale + dy
            )
        }
    }

    private fun computeTargetInBoundsMask(
        tgtCanvasMatrix: FloatArray,
        canvasSize: Int,
        targetWidth: Int,
        targetHeight: Int
    ): FloatArray {
        val inv = FaceAlignment.invertAffine2x3(tgtCanvasMatrix)
        val mask = FloatArray(canvasSize * canvasSize)
        val featherPx = max(6f, targetWidth * 0.01f)
        for (y in 0 until canvasSize) {
            val row = y * canvasSize
            val baseTx = inv[1] * y + inv[2]
            val baseTy = inv[4] * y + inv[5]
            for (x in 0 until canvasSize) {
                val tx = inv[0] * x + baseTx
                val ty = inv[3] * x + baseTy
                val dEdge = min(
                    min(tx - 1f, (targetWidth - 2f) - tx),
                    min(ty - 1f, (targetHeight - 2f) - ty)
                )
                mask[row + x] = when {
                    dEdge <= 0f -> 0f
                    dEdge >= featherPx -> 1f
                    else -> (dEdge / featherPx).coerceIn(0f, 1f)
                }
            }
        }
        return mask
    }

    private fun restoreMicroDetailFromIntermediatePortrait(
        step3Output: Bitmap,
        intermediatePortrait: Bitmap,
        faces: List<DetectedFace>,
        amount: Float = 0.44f
    ) {
        val w = step3Output.width
        val h = step3Output.height
        if (intermediatePortrait.width != w || intermediatePortrait.height != h) return
        val outPx = IntArray(w * h)
        val intPx = IntArray(w * h)
        step3Output.getPixels(outPx, 0, w, 0, 0, w, h)
        intermediatePortrait.getPixels(intPx, 0, w, 0, 0, w, h)

        for (face in faces) {
            if (face.landmarks5.size < 5) continue
            val lEye = face.landmarks5[0]
            val rEye = face.landmarks5[1]
            val eyeMidX = (lEye.x + rEye.x) * 0.5f
            val eyeMidY = (lEye.y + rEye.y) * 0.5f
            val eyeDist = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble())
                .toFloat().coerceAtLeast(16f)
            val rx = eyeDist * 1.10f
            val ry = eyeDist * 1.42f
            val cy = eyeMidY + eyeDist * 0.42f

            val rad = (eyeDist * 0.018f).toInt().coerceIn(2, 6)
            val x0 = (eyeMidX - rx).toInt().coerceIn(rad + 1, w - rad - 2)
            val x1 = (eyeMidX + rx).toInt().coerceIn(x0, w - rad - 2)
            val y0 = (cy - ry).toInt().coerceIn(rad + 1, h - rad - 2)
            val y1 = (cy + ry).toInt().coerceIn(y0, h - rad - 2)

            for (y in y0..y1) {
                val dy = (y - cy) / ry
                val row = y * w
                for (x in x0..x1) {
                    val dx = (x - eyeMidX) / rx
                    val rSq = dx * dx + dy * dy
                    if (rSq >= 1.0f) continue
                    val env = ((1.0f - sqrt(rSq)) / 0.40f).coerceIn(0f, 1f) * amount

                    val idx = row + x
                    val ic = intPx[idx]
                    val ir = (ic ushr 16) and 0xFF
                    val ig = (ic ushr 8) and 0xFF
                    val ib = ic and 0xFF
                    val iLum = 0.299f * ir + 0.587f * ig + 0.114f * ib

                    val n = intPx[(y - rad) * w + x]
                    val s = intPx[(y + rad) * w + x]
                    val l = intPx[row + x - rad]
                    val rr = intPx[row + x + rad]
                    val blurLum = (
                        (0.299f * (n ushr 16 and 0xFF) + 0.587f * (n ushr 8 and 0xFF) + 0.114f * (n and 0xFF)) +
                            (0.299f * (s ushr 16 and 0xFF) + 0.587f * (s ushr 8 and 0xFF) + 0.114f * (s and 0xFF)) +
                            (0.299f * (l ushr 16 and 0xFF) + 0.587f * (l ushr 8 and 0xFF) + 0.114f * (l and 0xFF)) +
                            (0.299f * (rr ushr 16 and 0xFF) + 0.587f * (rr ushr 8 and 0xFF) + 0.114f * (rr and 0xFF))
                        ) * 0.25f

                    val hpLum = ((iLum - blurLum) * env).coerceIn(-24f, 24f)
                    val oc = outPx[idx]
                    val or = (((oc ushr 16) and 0xFF) + hpLum).toInt().coerceIn(0, 255)
                    val og = (((oc ushr 8) and 0xFF) + hpLum).toInt().coerceIn(0, 255)
                    val ob = ((oc and 0xFF) + hpLum).toInt().coerceIn(0, 255)
                    outPx[idx] = (0xFF shl 24) or (or shl 16) or (og shl 8) or ob
                }
            }
        }
        step3Output.setPixels(outPx, 0, w, 0, 0, w, h)
    }

    private fun buildBackgroundDecorProtectMask(
        alignedHeadCrop: Bitmap,
        canvasPts5: List<PointF>,
        tgtMasks: HeadSegmentationMasks? = null
    ): FloatArray {
        val size = alignedHeadCrop.width
        val total = size * size
        val raw = FloatArray(total)
        if (canvasPts5.size < 5) return raw

        val pixels = IntArray(total)
        alignedHeadCrop.getPixels(pixels, 0, size, 0, 0, size, size)
        val eyeMidX = (canvasPts5[0].x + canvasPts5[1].x) * 0.5f
        val eyeMidY = (canvasPts5[0].y + canvasPts5[1].y) * 0.5f
        val rawEyeDist = hypot((canvasPts5[1].x - canvasPts5[0].x).toDouble(), (canvasPts5[1].y - canvasPts5[0].y).toDouble())
            .toFloat().coerceAtLeast(10f)
        val mouthMidY = (canvasPts5[3].y + canvasPts5[4].y) * 0.5f
        val eyeDist = max(rawEyeDist, abs(mouthMidY - eyeMidY) * 0.88f).coerceAtLeast(12f)
        // Cover full head + ears + tall dyed hair crowns / wide hats in fallback mode so skin/ears/hats aren't protected as decor
        val coreRx = eyeDist * 1.90f
        val coreRy = eyeDist * 2.55f
        val hasNeuralSeg = tgtMasks?.usedSegmentationOnnx == true

        for (y in 0 until size) {
            val row = y * size
            val fdy = (y - (eyeMidY - eyeDist * 0.30f)) / coreRy
            for (x in 0 until size) {
                val idx = row + x
                val headAlphaHere = tgtMasks?.fullHeadAlpha?.getOrNull(idx) ?: 0f
                if (headAlphaHere >= (if (hasNeuralSeg) 0.45f else 0.55f)) continue
                val fdx = (x - eyeMidX) / coreRx
                if (!hasNeuralSeg && fdx * fdx + fdy * fdy <= 1.0f) continue
                // Never protect face-level ear/skin band as flower decor
                if (abs(x - eyeMidX) <= eyeDist * 1.62f && y in (eyeMidY - eyeDist * 0.35f).toInt()..(eyeMidY + eyeDist * 1.45f).toInt()) {
                    val cSkin = pixels[idx]
                    val sr = (cSkin ushr 16) and 0xFF
                    val sg = (cSkin ushr 8) and 0xFF
                    val sb = cSkin and 0xFF
                    if (sr - sb < 85 && sr < 220) continue
                }

                val c = pixels[idx]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val maxCh = max(r, max(g, b))
                val minCh = min(r, min(g, b))
                val chroma = maxCh - minCh
                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                // Strictly match vivid yellow/red/orange marigold flowers & decor (never dull skin, hair, or wood door)
                val isVividDecor = (chroma >= 84 && maxCh >= 140) ||
                    (r >= 158 && g >= 118 && b <= 68 && chroma >= 92 && lum >= 115f) ||
                    (r >= 152 && r > g * 1.75f && r > b * 1.85f)
                if (isVividDecor) {
                    raw[idx] = 1.0f
                }
            }
        }
        val dilated = dilateAlpha(raw, size, radiusPx = max(2, size / 240))
        // Do not let decor dilation spill into dark black Target hair strands next to garlands
        for (i in 0 until total) {
            if (raw[i] < 0.5f && dilated[i] > 0f) {
                val c = pixels[i]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                if (lum < 58f) dilated[i] = 0f
            }
        }
        return dilated
    }

    private fun buildClothingCollarProtectMask(
        canvasSize: Int,
        tgtCanvasPts5: List<PointF>,
        tgtMasks: HeadSegmentationMasks,
        tgtCanvasMatrix: FloatArray,
        targetDepthMap: DepthMapResult?,
        targetWidth: Int,
        targetHeight: Int,
        alignedTgtPx: IntArray,
        decorProtectMask: FloatArray
    ): FloatArray {
        val total = canvasSize * canvasSize
        val protect = FloatArray(total)
        val rawEyeDist = if (tgtCanvasPts5.size >= 2) {
            hypot((tgtCanvasPts5[1].x - tgtCanvasPts5[0].x).toDouble(), (tgtCanvasPts5[1].y - tgtCanvasPts5[0].y).toDouble())
                .toFloat().coerceAtLeast(10f)
        } else {
            canvasSize * 0.15f
        }
        val eyeMidX = if (tgtCanvasPts5.size >= 2) (tgtCanvasPts5[0].x + tgtCanvasPts5[1].x) * 0.5f else canvasSize * 0.50f
        val eyeY = if (tgtCanvasPts5.size >= 5) (tgtCanvasPts5[0].y + tgtCanvasPts5[1].y) * 0.5f else canvasSize * 0.50f
        val mouthY = if (tgtCanvasPts5.size >= 5) (tgtCanvasPts5[3].y + tgtCanvasPts5[4].y) * 0.5f else canvasSize * 0.67f
        val eyeDist = max(rawEyeDist, abs(mouthY - eyeY) * 0.88f).coerceAtLeast(14f)
        val chinEstY = mouthY + (mouthY - eyeY) * 0.62f

        val baseCollarY = (chinEstY + eyeDist * 0.17f).coerceIn(canvasSize * 0.62f, canvasSize * 0.88f)
        val collarOffsetFromEyeY = baseCollarY - eyeY
        val rampSpan = canvasSize * 0.075f
        val inv = FaceAlignment.invertAffine2x3(tgtCanvasMatrix)
        val torsoCenterDepth = targetDepthMap?.let { dm ->
            val cx = ((inv[0] * (canvasSize * 0.5f) + inv[1] * (canvasSize * 0.92f) + inv[2]) / targetWidth).coerceIn(0f, 1f)
            val cy = ((inv[3] * (canvasSize * 0.5f) + inv[4] * (canvasSize * 0.92f) + inv[5]) / targetHeight).coerceIn(0f, 1f)
            dm.sampleBilinear(cx, cy)
        } ?: 0.5f

        // Rotate collar shoulder axis in canvas space to account for target head roll (> 25 deg tilted heads).
        // EstimateCanvasMatrix maps tilted face into eye-horizontal canvas space, tilting the torso by atan2(b, a).
        val canvasRollRad = atan2(tgtCanvasMatrix[3].toDouble(), tgtCanvasMatrix[0].toDouble()).toFloat()
        val effectiveCollarAngle = (canvasRollRad * 0.82f).coerceIn(-0.78f, 0.78f)
        val bodyUx = cos(effectiveCollarAngle)
        val bodyUy = sin(effectiveCollarAngle)
        val bodyVx = -bodyUy
        val bodyVy = bodyUx

        for (y in 0 until canvasSize) {
            val rowOffset = y * canvasSize
            val dyEye = y - eyeY
            for (x in 0 until canvasSize) {
                val dxEye = x - eyeMidX
                val localX = dxEye * bodyUx + dyEye * bodyUy
                val localY = dxEye * bodyVx + dyEye * bodyVy
                val uNorm = (localX / (eyeDist * 1.42f)).coerceIn(-1f, 1f)
                // Smooth curved shoulder scoop aligned with body roll angle so tilted collars are never sliced
                val curvedStartLocalY = collarOffsetFromEyeY - (uNorm * uNorm) * eyeDist * 0.30f
                if (localY < curvedStartLocalY) continue
                val idx = rowOffset + x

                val vWeight = ((localY - curvedStartLocalY) / rampSpan).coerceIn(0f, 1f)
                val smoothV = vWeight * vWeight * (3f - 2f * vWeight)

                val isNotSkin = (1f - tgtMasks.skinAndNeckWeight[idx]).coerceIn(0f, 1f)
                val lowHairW = ((localY - (curvedStartLocalY + eyeDist * 0.22f)) / (canvasSize * 0.12f)).coerceIn(0f, 1f)
                val baseProtect = max(isNotSkin, lowHairW)
                val depthFgBoost = if (targetDepthMap != null) {
                    val tx = inv[0] * x + inv[1] * y + inv[2]
                    val ty = inv[3] * x + inv[4] * y + inv[5]
                    val d = targetDepthMap.sampleBilinear(tx / targetWidth, ty / targetHeight)
                    if (abs(d - torsoCenterDepth) < 0.22f) 1.0f else 0.55f
                } else {
                    1.0f
                }
                protect[idx] = (smoothV * baseProtect * depthFgBoost).coerceIn(0f, 1f)
            }
        }
        return protect
    }

    private fun completeCroppedCrownAndDecontaminateSource(
        warpedSourceHead: Bitmap,
        alignedTargetHead: Bitmap,
        srcAlpha: FloatArray,
        srcSkinWeight: FloatArray,
        tgtAlpha: FloatArray,
        tgtSkinWeight: FloatArray,
        clothProtectMask: FloatArray,
        decorProtectMask: FloatArray,
        srcBoundsMask: FloatArray,
        tgtInBoundsMask: FloatArray,
        borderEnvelope: FloatArray,
        canvasPts5: List<PointF>,
        canvasSize: Int
    ): Pair<FloatArray, FloatArray> {
        val total = canvasSize * canvasSize
        val srcPx = IntArray(total)
        val tgtPx = IntArray(total)
        warpedSourceHead.getPixels(srcPx, 0, canvasSize, 0, 0, canvasSize, canvasSize)
        alignedTargetHead.getPixels(tgtPx, 0, canvasSize, 0, 0, canvasSize, canvasSize)

        val lEye = canvasPts5.getOrElse(0) { PointF(canvasSize * 0.42f, canvasSize * 0.51f) }
        val rEye = canvasPts5.getOrElse(1) { PointF(canvasSize * 0.58f, canvasSize * 0.51f) }
        val eyeMidX = (lEye.x + rEye.x) * 0.5f
        val eyeMidY = (lEye.y + rEye.y) * 0.5f
        val eyeDist = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble())
            .toFloat().coerceAtLeast(14f)

        // 1. Sample Source interior hair color AND Target hair color (for rich black contrast harmony)
        var sHR = 0L
        var sHG = 0L
        var sHB = 0L
        var sHCnt = 0L
        var tHR = 0L
        var tHG = 0L
        var tHB = 0L
        var tHCnt = 0L

        val yHairSampleMin = (eyeMidY - eyeDist * 1.85f).toInt().coerceIn(4, canvasSize - 10)
        val yHairSampleMax = (eyeMidY - eyeDist * 0.45f).toInt().coerceIn(yHairSampleMin, canvasSize - 10)
        val xHairSampleMin = (eyeMidX - eyeDist * 1.45f).toInt().coerceIn(4, canvasSize - 10)
        val xHairSampleMax = (eyeMidX + eyeDist * 1.45f).toInt().coerceIn(xHairSampleMin, canvasSize - 10)
        for (y in yHairSampleMin..yHairSampleMax step 2) {
            val row = y * canvasSize
            for (x in xHairSampleMin..xHairSampleMax step 2) {
                val idx = row + x
                if (srcBoundsMask[idx] >= 0.90f && srcAlpha[idx] >= 0.78f && srcSkinWeight[idx] < 0.25f) {
                    val c = srcPx[idx]
                    val r = (c ushr 16) and 0xFF
                    val g = (c ushr 8) and 0xFF
                    val b = c and 0xFF
                    val lum = 0.299f * r + 0.587f * g + 0.114f * b
                    if (lum in 8f..88f) {
                        sHR += r
                        sHG += g
                        sHB += b
                        sHCnt++
                    }
                }
                if (tgtInBoundsMask[idx] >= 0.90f && tgtAlpha[idx] >= 0.78f && tgtSkinWeight[idx] < 0.25f && decorProtectMask[idx] < 0.1f) {
                    val tc = tgtPx[idx]
                    val r = (tc ushr 16) and 0xFF
                    val g = (tc ushr 8) and 0xFF
                    val b = tc and 0xFF
                    val lum = 0.299f * r + 0.587f * g + 0.114f * b
                    if (lum in 6f..78f) {
                        tHR += r
                        tHG += g
                        tHB += b
                        tHCnt++
                    }
                }
            }
        }
        val rawSrcHairR = if (sHCnt > 8L) (sHR / sHCnt).toFloat() else 28f
        val rawSrcHairG = if (sHCnt > 8L) (sHG / sHCnt).toFloat() else 26f
        val rawSrcHairB = if (sHCnt > 8L) (sHB / sHCnt).toFloat() else 25f
        val rawSrcHairLum = 0.299f * rawSrcHairR + 0.587f * rawSrcHairG + 0.114f * rawSrcHairB

        val tgtHairLum = if (tHCnt > 8L) {
            0.299f * (tHR / tHCnt) + 0.587f * (tHG / tHCnt) + 0.114f * (tHB / tHCnt)
        } else {
            rawSrcHairLum
        }
        val harmonizeDarkHair = rawSrcHairLum < 92f && tgtHairLum < 82f
        val hairShadowScale = if (harmonizeDarkHair && rawSrcHairLum > tgtHairLum + 4f) {
            ((rawSrcHairLum * 0.18f + tgtHairLum * 0.82f) / rawSrcHairLum.coerceAtLeast(12f)).coerceIn(0.36f, 1.0f)
        } else {
            1.0f
        }
        val hairR = (rawSrcHairR * hairShadowScale).coerceIn(10f, 88f)
        val hairG = (rawSrcHairG * hairShadowScale).coerceIn(10f, 88f)
        val hairB = (rawSrcHairB * hairShadowScale).coerceIn(10f, 88f)
        val hairLum = 0.299f * hairR + 0.587f * hairG + 0.114f * hairB

        // 2. Clean exterior wall/window cyan reflections, neutralize cool tint on dark hair to match Target jet-black hair,
        //    and smoothly strip Source's bottom-edge blouse/necklace below chin without creating a crescent cheek seam!
        val outAlpha = FloatArray(total)
        val outSkinWeight = FloatArray(total)
        val skinFaceRx = eyeDist * 1.18f
        val skinFaceRy = eyeDist * 1.40f
        val mouthMidY = if (canvasPts5.size >= 5) (canvasPts5[3].y + canvasPts5[4].y) * 0.5f else eyeMidY + eyeDist * 1.15f
        val chinApproxY = mouthMidY + (mouthMidY - eyeMidY) * 0.60f

        for (y in 0 until canvasSize) {
            val row = y * canvasSize
            val sdy = (y - (eyeMidY + eyeDist * 0.28f)) / skinFaceRy
            for (x in 0 until canvasSize) {
                val idx = row + x
                val sdx = (x - eyeMidX) / skinFaceRx
                val skinCoreR = sqrt(sdx * sdx + sdy * sdy)
                val rawOutsideCore = ((skinCoreR - 0.90f) / 0.36f).coerceIn(0f, 1f)
                val outsideFaceCore = rawOutsideCore * rawOutsideCore * (3f - 2f * rawOutsideCore)
                val rawCyanCore = ((skinCoreR - 0.74f) / 0.28f).coerceIn(0f, 1f)
                val cyanOutsideCore = rawCyanCore * rawCyanCore * (3f - 2f * rawCyanCore)

                var a = (srcAlpha[idx] * srcBoundsMask[idx] * borderEnvelope[idx]).coerceIn(0f, 1f)
                val sw = (srcSkinWeight[idx] * srcBoundsMask[idx]).coerceIn(0f, 1f)
                val c = srcPx[idx]
                var r = ((c ushr 16) and 0xFF).toFloat()
                var g = ((c ushr 8) and 0xFF).toFloat()
                var b = (c and 0xFF).toFloat()
                val lum = 0.299f * r + 0.587f * g + 0.114f * b

                // Smoothly fade Source's lower neck & strip blouse/necklace below chin into Target's collar/dupatta
                if (y > chinApproxY + eyeDist * 0.10f && abs(x - eyeMidX) <= eyeDist * 0.76f) {
                    val neckFadeT = ((y - (chinApproxY + eyeDist * 0.10f)) / (eyeDist * 0.30f)).coerceIn(0f, 1f)
                    val smoothNeckFade = neckFadeT * neckFadeT * (3f - 2f * neckFadeT)
                    val colCenter = (1f - ((abs(x - eyeMidX) - eyeDist * 0.42f) / (eyeDist * 0.34f)).coerceIn(0f, 1f))
                    val lowBlouseT = ((y - (chinApproxY + eyeDist * 0.20f)) / (eyeDist * 0.18f)).coerceIn(0f, 1f)
                    val isNonSkinFabric = ((0.30f - sw) / 0.20f).coerceIn(0f, 1f)
                    val dropFactor = max(smoothNeckFade * 0.96f, lowBlouseT * isNonSkinFabric * 0.98f) * colCenter
                    a *= (1f - dropFactor).coerceIn(0f, 1f)
                }

                val pureHairRejectSkin = ((0.22f - sw) / 0.16f).coerceIn(0f, 1f)
                val hairEffect = outsideFaceCore * pureHairRejectSkin
                val cyanEffect = cyanOutsideCore * pureHairRejectSkin
                if ((hairEffect > 0.01f || cyanEffect > 0.01f) && a > 0.01f) {
                    val isCoolWindowGlare = sw < 0.18f &&
                        ((b - r >= 18f || g - r >= 18f) && lum >= 78f)
                    val isBrightExteriorWall = sw < 0.14f &&
                        ((lum > 122f && (b - r >= 12f || g - r >= 12f)) ||
                            (lum > 152f && abs(y - eyeMidY) < eyeDist * 2.2f))
                    if (isCoolWindowGlare || isBrightExteriorWall) {
                        a *= (1f - cyanOutsideCore).coerceIn(0f, 1f)
                    } else {
                        // Smoothly neutralize cool cyan/green cast on dark black hair to rich neutral black (no temple cyan or ellipse seam)
                        if (b > r + 2f || g > r + 2f) {
                            val neutralR = r.coerceAtMost(76f)
                            val coolShift = ((max(b - r, g - r) - 1f) / 10f).coerceIn(0f, 0.96f) * max(hairEffect, cyanEffect)
                            r = r * (1f - coolShift * 0.45f) + min(neutralR, hairR * 1.25f) * (coolShift * 0.45f)
                            g = g * (1f - coolShift * 0.94f) + min(g, r + 1f) * (coolShift * 0.94f)
                            b = b * (1f - coolShift * 0.96f) + min(b, r + 1f) * (coolShift * 0.96f)
                        }
                        if (hairShadowScale < 0.98f && lum < 104f && hairEffect > 0.01f) {
                            val hairWeight = ((104f - lum) / 56f).coerceIn(0f, 1f) * hairEffect
                            val sc = 1f - (1f - hairShadowScale) * hairWeight
                            r *= sc
                            g *= sc
                            b *= sc
                        }
                        srcPx[idx] = (0xFF shl 24) or
                            (r.toInt().coerceIn(0, 255) shl 16) or
                            (g.toInt().coerceIn(0, 255) shl 8) or
                            b.toInt().coerceIn(0, 255)
                    }
                }
                outAlpha[idx] = a
                outSkinWeight[idx] = sw
            }
        }

        // 3. Complete FULL natural hair crown & side hair curves wherever Source photo frame was cropped
        //    AND seamlessly merge left/right shoulder hair into Target's draped black hair strands!
        val crownCenterY = eyeMidY - eyeDist * 0.25f
        val baseRx = eyeDist * 1.41f
        val crownRy = eyeDist * 1.74f
        val skirtBottomY = (eyeMidY + eyeDist * 2.45f).toInt().coerceIn(10, canvasSize - 4)
        val domeTopY = (crownCenterY - crownRy - 4f).toInt().coerceIn(4, canvasSize - 10)
        val domeMinX = (eyeMidX - baseRx * 1.46f).toInt().coerceIn(4, canvasSize - 10)
        val domeMaxX = (eyeMidX + baseRx * 1.46f).toInt().coerceIn(domeMinX + 2, canvasSize - 10)

        var croppedCount = 0
        for (y in domeTopY until (eyeMidY - eyeDist * 0.35f).toInt().coerceIn(domeTopY + 1, canvasSize - 4) step 2) {
            val row = y * canvasSize
            for (x in domeMinX..domeMaxX step 2) {
                val idx = row + x
                if (srcBoundsMask[idx] < 0.88f || outAlpha[idx] < 0.55f) croppedCount++
            }
        }

        if (croppedCount > 12) {
            val origSrcPx = srcPx.copyOf()
            val origOutAlpha = outAlpha.copyOf()

            for (y in domeTopY..skirtBottomY) {
                val row = y * canvasSize
                val nyAbove = (y - crownCenterY) / crownRy
                val lowerProgress = ((y - eyeMidY) / (eyeDist * 1.85f)).coerceIn(0f, 1f)
                val localRx = baseRx * (1.0f + 0.16f * lowerProgress)
                val shoulderFlareT = ((y - mouthMidY) / (eyeDist * 1.05f)).coerceIn(0f, 1f)
                val smoothShoulderFlare = shoulderFlareT * shoulderFlareT * (3f - 2f * shoulderFlareT)

                for (x in domeMinX..domeMaxX) {
                    val idx = row + x
                    if (clothProtectMask[idx] > 0.88f) continue
                    // Never synthesize hair over the central bare neck zone below chin
                    if (y > chinApproxY + eyeDist * 0.08f && abs(x - eyeMidX) <= eyeDist * 0.64f) continue

                    // Check if Target has genuine dark hair strands at this crown/shoulder pixel
                    val tc = tgtPx[idx]
                    val tR = ((tc ushr 16) and 0xFF).toFloat()
                    val tG = ((tc ushr 8) and 0xFF).toFloat()
                    val tB = (tc and 0xFF).toFloat()
                    val tLum = 0.299f * tR + 0.587f * tG + 0.114f * tB
                    val hasTgtDarkHair = tgtInBoundsMask[idx] > 0.85f &&
                        decorProtectMask[idx] < 0.12f &&
                        tgtAlpha[idx] > 0.45f &&
                        tgtSkinWeight[idx] < 0.30f &&
                        tLum < 78f &&
                        (tR - tB) < 28f

                    // Gradually flare hair radius along Target's draped shoulder hair with zero corner stair-step
                    val effRx = if (hasTgtDarkHair) {
                        localRx + (baseRx * 1.44f - localRx) * smoothShoulderFlare
                    } else {
                        localRx
                    }
                    val nx = (x - eyeMidX) / effRx
                    val effRadial = if (y < crownCenterY) {
                        sqrt(nx * nx + nyAbove * nyAbove)
                    } else {
                        abs(nx) * (1.0f - 0.12f * lowerProgress)
                    }
                    if (effRadial >= 1.0f) {
                        // Trim any stray choppy outer crown hair spikes beyond the clean silhouette ellipse
                        if (y < eyeMidY && outSkinWeight[idx] < 0.16f && origOutAlpha[idx] > 0.01f) {
                            val trimT = ((effRadial - 1.0f) / 0.10f).coerceIn(0f, 1f)
                            outAlpha[idx] = origOutAlpha[idx] * (1f - trimT)
                        }
                        continue
                    }

                    val curReady = min(srcBoundsMask[idx], origOutAlpha[idx])
                    if (curReady >= 0.94f) continue

                    var domeAlpha = when {
                        effRadial <= 0.88f -> 1.0f
                        effRadial >= 0.985f -> 0.0f
                        else -> {
                            val t = (effRadial - 0.88f) / 0.105f
                            (0.5f * (1f + kotlin.math.cos(Math.PI * t))).toFloat()
                        }
                    } * tgtInBoundsMask[idx] * borderEnvelope[idx]
                    // On lower hair (left shoulder drape), keep full soft continuity onto Target hair while strictly protecting garlands
                    if (decorProtectMask[idx] > 0.08f && !hasTgtDarkHair) {
                        domeAlpha *= (1f - decorProtectMask[idx]).coerceIn(0f, 1f)
                    }
                    if (domeAlpha <= 0.02f) continue

                    val rayDx = (eyeMidX - x).toFloat()
                    val rayDy = ((eyeMidY - eyeDist * 0.15f) - y).toFloat()
                    val rayLen = hypot(rayDx.toDouble(), rayDy.toDouble()).toFloat().coerceAtLeast(1f)
                    val ux = rayDx / rayLen
                    val uy = rayDy / rayLen

                    var foundX = -1
                    var foundY = -1
                    var hitStep = 0f
                    var stepD = 2f
                    val maxScan = (eyeDist * 1.45f)
                    while (stepD <= maxScan) {
                        val sx = (x + ux * stepD).toInt()
                        val sy = (y + uy * stepD).toInt()
                        if (sx !in 2 until canvasSize - 2 || sy !in 2 until canvasSize - 2) break
                        val sIdx = sy * canvasSize + sx
                        if (srcBoundsMask[sIdx] >= 0.93f && origOutAlpha[sIdx] >= 0.80f && outSkinWeight[sIdx] < 0.22f) {
                            val sc = origSrcPx[sIdx]
                            val sr = (sc ushr 16) and 0xFF
                            val sg = (sc ushr 8) and 0xFF
                            val sb = sc and 0xFF
                            val sLum = 0.299f * sr + 0.587f * sg + 0.114f * sb
                            if (sLum < 96f && sb <= sr + 10) {
                                foundX = sx
                                foundY = sy
                                hitStep = stepD
                                break
                            }
                        }
                        stepD += 2f
                    }

                    var synR = hairR
                    var synG = hairG
                    var synB = hairB
                    if (foundX >= 0) {
                        val mirrorD = (hitStep + min(hitStep * 0.35f, eyeDist * 0.32f)).coerceAtMost(maxScan)
                        val mx = (x + ux * mirrorD).toInt().coerceIn(1, canvasSize - 2)
                        val my = (y + uy * mirrorD).toInt().coerceIn(1, canvasSize - 2)
                        val mIdx = my * canvasSize + mx
                        val mc = if (srcBoundsMask[mIdx] >= 0.90f && outSkinWeight[mIdx] < 0.22f) {
                            origSrcPx[mIdx]
                        } else {
                            origSrcPx[foundY * canvasSize + foundX]
                        }
                        val mr = ((mc ushr 16) and 0xFF).toFloat()
                        val mg = ((mc ushr 8) and 0xFF).toFloat()
                        val mb = (mc and 0xFF).toFloat()
                        val mLum = 0.299f * mr + 0.587f * mg + 0.114f * mb
                        if (mLum in 6f..96f && mb <= mr + 10f) {
                            val shade = (1.0f - (hitStep / (eyeDist * 2.5f))).coerceIn(0.84f, 1.0f)
                            synR = mr * shade
                            synG = mg * shade
                            synB = mb * shade
                        }
                    }

                    if (hasTgtDarkHair) {
                        val targetWeight = if (y > eyeMidY + eyeDist * 0.55f) 0.85f else 0.50f
                        val lumScale = (hairLum.coerceAtLeast(14f) / tLum.coerceAtLeast(14f)).coerceIn(0.82f, 1.18f)
                        synR = synR * (1f - targetWeight) + (tR * lumScale).coerceIn(0f, 82f) * targetWeight
                        synG = synG * (1f - targetWeight) + (tG * lumScale).coerceIn(0f, 82f) * targetWeight
                        synB = synB * (1f - targetWeight) + (tB * lumScale).coerceIn(0f, 82f) * targetWeight
                    }

                    val mix = (1.0f - curReady).coerceIn(0f, 1f)
                    val curC = origSrcPx[idx]
                    val fR = (((curC ushr 16) and 0xFF) * (1f - mix) + synR * mix).toInt().coerceIn(0, 255)
                    val fG = (((curC ushr 8) and 0xFF) * (1f - mix) + synG * mix).toInt().coerceIn(0, 255)
                    val fB = ((curC and 0xFF) * (1f - mix) + synB * mix).toInt().coerceIn(0, 255)
                    srcPx[idx] = (0xFF shl 24) or (fR shl 16) or (fG shl 8) or fB
                    outAlpha[idx] = max(origOutAlpha[idx], domeAlpha).coerceIn(0f, 1f)
                }
            }
        }

        // Light 2-pass perimeter feather on outer hair alpha only (strictly leaves inner face & chin opaque/crisp)
        repeat(2) {
            val prevA = outAlpha.copyOf()
            for (y in 2 until canvasSize - 2) {
                val row = y * canvasSize
                val sdy = (y - (eyeMidY + eyeDist * 0.28f)) / skinFaceRy
                for (x in 2 until canvasSize - 2) {
                    val idx = row + x
                    val cur = prevA[idx]
                    if (cur !in 0.01f..0.94f || outSkinWeight[idx] > 0.24f) continue
                    val sdx = (x - eyeMidX) / skinFaceRx
                    if (sdx * sdx + sdy * sdy < 1.10f) continue
                    val avg = (
                        prevA[idx - 1] + prevA[idx + 1] +
                            prevA[idx - canvasSize] + prevA[idx + canvasSize] +
                            cur * 2f
                        ) / 6f
                    outAlpha[idx] = avg.coerceIn(0f, 1f)
                }
            }
        }

        warpedSourceHead.setPixels(srcPx, 0, canvasSize, 0, 0, canvasSize, canvasSize)
        return outAlpha to outSkinWeight
    }

    private fun dilateAlpha(alpha: FloatArray, size: Int, radiusPx: Int): FloatArray {
        val out = FloatArray(size * size)
        for (y in 0 until size) {
            val y0 = max(0, y - radiusPx)
            val y1 = min(size - 1, y + radiusPx)
            val row = y * size
            for (x in 0 until size) {
                val x0 = max(0, x - radiusPx)
                val x1 = min(size - 1, x + radiusPx)
                var maxA = alpha[row + x]
                for (ny in y0..y1 step 2) {
                    val nRow = ny * size
                    for (nx in x0..x1 step 2) {
                        val a = alpha[nRow + nx]
                        if (a > maxA) maxA = a
                    }
                }
                out[row + x] = maxA
            }
        }
        return out
    }

    private fun eraseTargetHeadCleanly(
        ortEnv: OrtEnvironment?,
        targetCanvas: Bitmap,
        eraseMask: FloatArray,
        clothProtectMask: FloatArray,
        decorProtectMask: FloatArray,
        tgtInBoundsMask: FloatArray,
        tgtCanvasMatrix: FloatArray,
        targetDepthMap: DepthMapResult?,
        targetWidth: Int,
        targetHeight: Int,
        lamaModelFile: File?,
        preferHardwareAccel: Boolean,
        isCancelled: () -> Boolean = { false }
    ): Pair<Bitmap, Boolean> {
        if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
        val size = targetCanvas.width
        val total = size * size
        var erasePixels = 0
        for (i in 0 until total) {
            if (eraseMask[i] > 0.12f) erasePixels++
        }
        if (erasePixels < 16) {
            return targetCanvas.copy(Bitmap.Config.ARGB_8888, true) to false
        }

        if (ortEnv != null && lamaModelFile != null && lamaModelFile.exists() && lamaModelFile.length() > 1024L) {
            val lamaAttempt = runCatching {
                HeadSegmentationAndInpainting.runLamaInpaintOnnx(
                    ortEnv = ortEnv,
                    lamaFile = lamaModelFile,
                    cropBitmap = targetCanvas,
                    gapMask = eraseMask,
                    preferHardwareAccel = preferHardwareAccel,
                    isCancelled = isCancelled
                )
            }
            val lamaErr = lamaAttempt.exceptionOrNull()
            if (lamaErr is java.util.concurrent.CancellationException || isCancelled()) {
                throw (lamaErr as? java.util.concurrent.CancellationException
                    ?: java.util.concurrent.CancellationException("Head replacement cancelled by user."))
            }
            val lamaOut = lamaAttempt.getOrNull()
            if (lamaOut != null) {
                return lamaOut to true
            }
        }
        if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

        // Strict Non-Flower/Non-Earring Eroded Clean Background Anchor Fill + Harmonic Laplace Relaxation
        val pixels = IntArray(total)
        targetCanvas.getPixels(pixels, 0, size, 0, 0, size, size)

        val rawCleanBg = BooleanArray(total) { i ->
            if (tgtInBoundsMask[i] < 0.88f || eraseMask[i] >= 0.05f || clothProtectMask[i] >= 0.22f || decorProtectMask[i] >= 0.04f) {
                false
            } else {
                val col = pixels[i]
                val r = (col ushr 16) and 0xFF
                val g = (col ushr 8) and 0xFF
                val b = col and 0xFF
                val chroma = max(r, max(g, b)) - min(r, min(g, b))
                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                // Accept BOTH bright skies/walls AND rich warm brown wood doors, rejecting bright flowers AND dark crimson petals
                val isCrimsonPetal = (r >= 95 && r > g * 2.15f && r > b * 2.15f && chroma >= 55)
                lum in 12f..248f && !(chroma >= 82 && lum >= 115f) && !isCrimsonPetal
            }
        }

        // Erode rawCleanBg around erase/bounds and ~16px around flower garlands so garland shadows never darken wood fill
        val erosionR = max(5, size / 115)
        val decorMarginR = max(14, size / 44)
        val isCleanAnchor = BooleanArray(total) { i ->
            if (!rawCleanBg[i]) {
                false
            } else {
                val y = i / size
                val x = i % size
                var safe = true
                for (dy in -erosionR..erosionR step erosionR) {
                    val ny = (y + dy).coerceIn(0, size - 1)
                    val nRow = ny * size
                    for (dx in -erosionR..erosionR step erosionR) {
                        val nx = (x + dx).coerceIn(0, size - 1)
                        val nIdx = nRow + nx
                        if (eraseMask[nIdx] > 0.06f || decorProtectMask[nIdx] > 0.08f || tgtInBoundsMask[nIdx] < 0.78f) {
                            safe = false
                            break
                        }
                    }
                    if (!safe) break
                }
                if (safe) {
                    for (dy in -decorMarginR..decorMarginR step max(3, decorMarginR / 2)) {
                        val ny = (y + dy).coerceIn(0, size - 1)
                        val nRow = ny * size
                        for (dx in -decorMarginR..decorMarginR step max(3, decorMarginR / 2)) {
                            val nx = (x + dx).coerceIn(0, size - 1)
                            if (decorProtectMask[nRow + nx] > 0.10f) {
                                safe = false
                                break
                            }
                        }
                        if (!safe) break
                    }
                }
                safe
            }
        }

        val rBuf = FloatArray(total) { i -> ((pixels[i] ushr 16) and 0xFF).toFloat() }
        val gBuf = FloatArray(total) { i -> ((pixels[i] ushr 8) and 0xFF).toFloat() }
        val bBuf = FloatArray(total) { i -> (pixels[i] and 0xFF).toFloat() }
        val activeMask = BooleanArray(total)

        fun sampleWindowMedianBg(cx: Int, cy: Int, verticalSpan: Boolean): FloatArray {
            val rs = FloatArray(11)
            val gs = FloatArray(11)
            val bs = FloatArray(11)
            var cnt = 0
            for (d in -15..15 step 3) {
                val xx = if (verticalSpan) cx else (cx + d).coerceIn(1, size - 2)
                val yy = if (verticalSpan) (cy + d).coerceIn(1, size - 2) else cy
                val idx = yy * size + xx
                if (isCleanAnchor[idx] || (cnt == 0 && rawCleanBg[idx])) {
                    val col = pixels[idx]
                    rs[cnt] = ((col ushr 16) and 0xFF).toFloat()
                    gs[cnt] = ((col ushr 8) and 0xFF).toFloat()
                    bs[cnt] = (col and 0xFF).toFloat()
                    cnt++
                }
            }
            if (cnt == 0) {
                val col = pixels[cy.coerceIn(0, size - 1) * size + cx.coerceIn(0, size - 1)]
                return floatArrayOf(
                    ((col ushr 16) and 0xFF).toFloat(),
                    ((col ushr 8) and 0xFF).toFloat(),
                    (col and 0xFF).toFloat()
                )
            }
            rs.sort(0, cnt)
            gs.sort(0, cnt)
            bs.sort(0, cnt)
            val mid = cnt / 2
            return floatArrayOf(rs[mid], gs[mid], bs[mid])
        }

        var minEY = size - 1
        var maxEY = 0
        var minEX = size - 1
        var maxEX = 0

        for (y in 2 until size - 2) {
            val rowOff = y * size
            for (x in 2 until size - 2) {
                val idx = rowOff + x
                if (eraseMask[idx] > 0.04f && clothProtectMask[idx] < 0.45f) {
                    activeMask[idx] = true
                    if (y < minEY) minEY = y
                    if (y > maxEY) maxEY = y
                    if (x < minEX) minEX = x
                    if (x > maxEX) maxEX = x
                }
            }
        }

        // Fallback global clean background mean computed directly from non-erased anchor pixels (never hardcoded)
        var gBgR = 0.0
        var gBgG = 0.0
        var gBgB = 0.0
        var gBgCnt = 0
        for (i in 0 until total step 6) {
            if (isCleanAnchor[i] || rawCleanBg[i]) {
                val col = pixels[i]
                gBgR += (col ushr 16) and 0xFF
                gBgG += (col ushr 8) and 0xFF
                gBgB += col and 0xFF
                gBgCnt++
            }
        }
        val fbBgR = if (gBgCnt > 0) (gBgR / gBgCnt).toFloat() else 128f
        val fbBgG = if (gBgCnt > 0) (gBgG / gBgCnt).toFloat() else 128f
        val fbBgB = if (gBgCnt > 0) (gBgB / gBgCnt).toFloat() else 128f

        val rowLeftX = IntArray(size) { -1 }
        val rowRightX = IntArray(size) { -1 }
        for (y in minEY..maxEY) {
            val row = y * size
            var lx = -1
            for (x in minEX downTo 2) {
                if (isCleanAnchor[row + x]) {
                    lx = x
                    break
                }
            }
            if (lx < 0) {
                for (x in 2 until minEX) {
                    if (isCleanAnchor[row + x]) {
                        lx = x
                        break
                    }
                }
            }
            var rx = -1
            for (x in maxEX until size - 2) {
                if (isCleanAnchor[row + x]) {
                    rx = x
                    break
                }
            }
            if (rx < 0) {
                for (x in size - 3 downTo maxEX) {
                    if (isCleanAnchor[row + x]) {
                        rx = x
                        break
                    }
                }
            }
            rowLeftX[y] = lx
            rowRightX[y] = rx
        }

        for (y in minEY..maxEY) {
            val row = y * size
            for (x in minEX..maxEX) {
                val idx = row + x
                if (!activeMask[idx]) continue

                var lx = -1
                for (tx in (x - 1) downTo 2) {
                    if (isCleanAnchor[row + tx]) {
                        lx = tx
                        break
                    }
                }
                var rx = -1
                for (tx in (x + 1) until size - 2) {
                    if (isCleanAnchor[row + tx]) {
                        rx = tx
                        break
                    }
                }
                var ty = -1
                for (yy in (y - 1) downTo 2) {
                    if (isCleanAnchor[yy * size + x]) {
                        ty = yy
                        break
                    }
                }

                var wSum = 0f
                var rs = 0f
                var gs = 0f
                var bs = 0f

                if (lx >= 0) {
                    val d = (x - lx).toFloat()
                    val w = 1f / ((d + 16f) * (d + 16f))
                    val c = sampleWindowMedianBg(lx, y, verticalSpan = true)
                    rs += c[0] * w
                    gs += c[1] * w
                    bs += c[2] * w
                    wSum += w
                }
                if (rx >= 0) {
                    val d = (rx - x).toFloat()
                    val w = 1f / ((d + 16f) * (d + 16f))
                    val c = sampleWindowMedianBg(rx, y, verticalSpan = true)
                    rs += c[0] * w
                    gs += c[1] * w
                    bs += c[2] * w
                    wSum += w
                }
                if (ty >= 0) {
                    val d = (y - ty).toFloat()
                    val w = 0.20f / ((d + 22f) * (d + 22f))
                    val c = sampleWindowMedianBg(x, ty, verticalSpan = false)
                    rs += c[0] * w
                    gs += c[1] * w
                    bs += c[2] * w
                    wSum += w
                }

                if (wSum > 1e-6f) {
                    rBuf[idx] = rs / wSum
                    gBuf[idx] = gs / wSum
                    bBuf[idx] = bs / wSum
                } else {
                    rBuf[idx] = fbBgR
                    gBuf[idx] = fbBgG
                    bBuf[idx] = fbBgB
                }
            }
        }

        // 28-pass Harmonic Laplace Diffusion strictly inside active erase mask so seams vanish completely
        val rWork = rBuf.copyOf()
        val gWork = gBuf.copyOf()
        val bWork = bBuf.copyOf()
        repeat(28) {
            for (y in max(2, minEY)..min(size - 3, maxEY)) {
                val row = y * size
                for (x in max(2, minEX)..min(size - 3, maxEX)) {
                    val idx = row + x
                    if (!activeMask[idx]) continue
                    val nr = (rWork[idx - 1] + rWork[idx + 1] + rWork[idx - size] + rWork[idx + size]) * 0.25f
                    val ng = (gWork[idx - 1] + gWork[idx + 1] + gWork[idx - size] + gWork[idx + size]) * 0.25f
                    val nb = (bWork[idx - 1] + bWork[idx + 1] + bWork[idx - size] + bWork[idx + size]) * 0.25f
                    rWork[idx] = rWork[idx] * 0.16f + nr * 0.84f
                    gWork[idx] = gWork[idx] * 0.16f + ng * 0.84f
                    bWork[idx] = bWork[idx] * 0.16f + nb * 0.84f
                }
            }
        }

        val inpainted = pixels.copyOf()
        for (y in max(2, minEY)..min(size - 3, maxEY)) {
            val row = y * size
            for (x in max(2, minEX)..min(size - 3, maxEX)) {
                val idx = row + x
                if (!activeMask[idx]) continue

                // Subtle 2D-smoothed vertical wood texture continuation (no sharp 1D row stripes)
                var grainDelta = 0f
                val ax = if (rowLeftX[y] >= 2 && (rowRightX[y] < 0 || abs(x - rowLeftX[y]) <= abs(rowRightX[y] - x))) {
                    (rowLeftX[y] - 6).coerceIn(4, size - 5)
                } else {
                    (rowRightX[y] + 6).coerceIn(4, size - 5)
                }
                val gy = (y + ((x - size / 2) / 3)).coerceIn(4, size - 5)
                if (isCleanAnchor[gy * size + ax]) {
                    val cC = pixels[gy * size + ax]
                    val cL = pixels[gy * size + (ax - 3)]
                    val cR = pixels[gy * size + (ax + 3)]
                    val lumC = 0.299f * (cC ushr 16 and 0xFF) + 0.587f * (cC ushr 8 and 0xFF) + 0.114f * (cC and 0xFF)
                    val lumAvg = 0.5f * (
                        (0.299f * (cL ushr 16 and 0xFF) + 0.587f * (cL ushr 8 and 0xFF) + 0.114f * (cL and 0xFF)) +
                            (0.299f * (cR ushr 16 and 0xFF) + 0.587f * (cR ushr 8 and 0xFF) + 0.114f * (cR and 0xFF))
                        )
                    grainDelta = (lumC - lumAvg).coerceIn(-6f, 6f) * 0.40f
                }

                val eAlpha = eraseMask[idx]
                val blend = ((eAlpha - 0.04f) / 0.42f).coerceIn(0f, 1f)
                val orig = pixels[idx]
                val oR = (orig ushr 16) and 0xFF
                val oG = (orig ushr 8) and 0xFF
                val oB = orig and 0xFF

                val tR = (rWork[idx] + grainDelta).coerceIn(0f, 255f)
                val tG = (gWork[idx] + grainDelta).coerceIn(0f, 255f)
                val tB = (bWork[idx] + grainDelta).coerceIn(0f, 255f)

                val fR = (oR * (1f - blend) + tR * blend).toInt().coerceIn(0, 255)
                val fG = (oG * (1f - blend) + tG * blend).toInt().coerceIn(0, 255)
                val fB = (oB * (1f - blend) + tB * blend).toInt().coerceIn(0, 255)
                inpainted[idx] = (0xFF shl 24) or (fR shl 16) or (fG shl 8) or fB
            }
        }

        val outBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(inpainted, 0, size, 0, 0, size, size)
        return outBmp to false
    }

    private fun compositeSourceOverHeadlessCanvas(
        sourceHead: Bitmap,
        headlessTarget: Bitmap,
        sourceAlpha: FloatArray,
        clothProtectMask: FloatArray
    ): Bitmap {
        val size = sourceHead.width
        val total = size * size
        val srcPx = IntArray(total)
        val bgPx = IntArray(total)
        sourceHead.getPixels(srcPx, 0, size, 0, 0, size, size)
        headlessTarget.getPixels(bgPx, 0, size, 0, 0, size, size)

        val outPx = IntArray(total)
        for (i in 0 until total) {
            val a = (sourceAlpha[i] * (1f - clothProtectMask[i] * 0.90f)).coerceIn(0f, 1f)
            val invA = 1f - a
            val sc = srcPx[i]
            val bc = bgPx[i]
            val r = (((sc ushr 16) and 0xFF) * a + ((bc ushr 16) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val g = (((sc ushr 8) and 0xFF) * a + ((bc ushr 8) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val b = ((sc and 0xFF) * a + (bc and 0xFF) * invA).toInt().coerceIn(0, 255)
            outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val outBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(outPx, 0, size, 0, 0, size, size)
        return outBmp
    }

    private fun runDiffusionSeamBridge(
        context: Context,
        ortEnv: OrtEnvironment,
        pack: InstalledSdInpaintPack,
        stage4Canvas: Bitmap,
        sourceAlpha: FloatArray,
        targetEraseMask: FloatArray,
        innerFaceProtectMask: FloatArray,
        clothProtectMask: FloatArray,
        preferHardwareAccel: Boolean,
        maxPreferredResolutionPx: Int = 512,
        isCancelled: () -> Boolean = { false },
        onDetailProgress: (String) -> Unit = {}
    ): Bitmap? {
        val size = stage4Canvas.width
        val total = size * size
        val maskPx = IntArray(total)
        var activeMaskCount = 0

        for (i in 0 until total) {
            val a = sourceAlpha[i]
            val isHairPerimeter = (a in 0.08f..0.82f)
            val isDisocclusionRing = (targetEraseMask[i] > 0.15f && a < 0.72f)
            val rawBridge = when {
                isHairPerimeter -> 1.0f
                isDisocclusionRing -> 0.85f
                else -> 0.0f
            }
            val protectedBridge = rawBridge * (1f - innerFaceProtectMask[i]) * (1f - clothProtectMask[i] * 0.85f)
            if (protectedBridge > 0.25f) {
                maskPx[i] = (0xFF shl 24) or (0xFF shl 16) or (0xFF shl 8) or 0xFF
                activeMaskCount++
            } else {
                maskPx[i] = 0xFF000000.toInt()
            }
        }

        if (activeMaskCount < 32) return null

        val maskBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        maskBmp.setPixels(maskPx, 0, size, 0, 0, size, size)

        val sdRes = try {
            SdInpaintOnnxEngine.inpaint(
                ortEnv = ortEnv,
                pack = pack,
                sourceBitmap = stage4Canvas,
                maskBitmap = maskBmp,
                prompt = "natural realistic hair strands, seamless skin neck collar transition, photorealistic lighting",
                negativePrompt = "double face, extra head, blurry hair, artifacts, deformed neck, plastic skin",
                fastMode = true,
                preferHardwareAccel = preferHardwareAccel,
                strength = 0.25f,
                guidanceScale = 6.0f,
                releaseEachStageAfterUse = true,
                context = context,
                maxPreferredResolutionPx = maxPreferredResolutionPx,
                isCancelled = isCancelled,
                onProgress = onDetailProgress
            )
        } finally {
            maskBmp.recycle()
        }

        val diffOut = if (sdRes.outputBitmap.width == size && sdRes.outputBitmap.height == size) {
            sdRes.outputBitmap
        } else {
            Bitmap.createScaledBitmap(sdRes.outputBitmap, size, size, true).also {
                sdRes.outputBitmap.recycle()
            }
        }

        // Clamp inner face identity pixels back from stage4Canvas so Diffusion never shifts Source identity!
        val stage4Px = IntArray(total)
        val diffPx = IntArray(total)
        stage4Canvas.getPixels(stage4Px, 0, size, 0, 0, size, size)
        diffOut.getPixels(diffPx, 0, size, 0, 0, size, size)

        for (i in 0 until total) {
            val lock = max(innerFaceProtectMask[i], clothProtectMask[i]).coerceIn(0f, 1f)
            if (lock > 0.01f) {
                val invL = 1f - lock
                val s = stage4Px[i]
                val d = diffPx[i]
                val r = (((s ushr 16) and 0xFF) * lock + ((d ushr 16) and 0xFF) * invL).toInt().coerceIn(0, 255)
                val g = (((s ushr 8) and 0xFF) * lock + ((d ushr 8) and 0xFF) * invL).toInt().coerceIn(0, 255)
                val b = ((s and 0xFF) * lock + (d and 0xFF) * invL).toInt().coerceIn(0, 255)
                diffPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        diffOut.setPixels(diffPx, 0, size, 0, 0, size, size)
        return diffOut
    }

    private fun applyFastSeamFeatherBridge(
        composited: Bitmap,
        sourceAlpha: FloatArray,
        innerFaceProtectMask: FloatArray,
        clothProtectMask: FloatArray? = null
    ): Bitmap {
        val size = composited.width
        val total = size * size
        var current = IntArray(total)
        composited.getPixels(current, 0, size, 0, 0, size, size)

        // 2-pass multi-band boundary smoothing along outer hair/neck transition ring only
        // (strictly preserves innerFaceProtectMask & collar/jewelry clothProtectMask)
        val radii = intArrayOf(2, 1)
        val weights = floatArrayOf(0.42f, 0.32f)
        for (passIdx in radii.indices) {
            val rad = radii[passIdx]
            val baseW = weights[passIdx]
            val next = current.copyOf()
            for (y in rad until size - rad) {
                val row = y * size
                for (x in rad until size - rad) {
                    val idx = row + x
                    val a = sourceAlpha[idx]
                    val clothLocked = (clothProtectMask?.getOrNull(idx) ?: 0f) > 0.45f
                    if (a in 0.04f..0.82f && innerFaceProtectMask[idx] < 0.25f && !clothLocked) {
                        val n = current[(y - rad) * size + x]
                        val s = current[(y + rad) * size + x]
                        val l = current[row + x - rad]
                        val r = current[row + x + rad]
                        val nw = current[(y - rad) * size + (x - rad)]
                        val ne = current[(y - rad) * size + (x + rad)]
                        val sw = current[(y + rad) * size + (x - rad)]
                        val se = current[(y + rad) * size + (x + rad)]
                        val c = current[idx]

                        val avgR = (
                            ((n ushr 16 and 0xFF) + (s ushr 16 and 0xFF) + (l ushr 16 and 0xFF) + (r ushr 16 and 0xFF)) * 2f +
                            (nw ushr 16 and 0xFF) + (ne ushr 16 and 0xFF) + (sw ushr 16 and 0xFF) + (se ushr 16 and 0xFF)
                        ) / 12f
                        val avgG = (
                            ((n ushr 8 and 0xFF) + (s ushr 8 and 0xFF) + (l ushr 8 and 0xFF) + (r ushr 8 and 0xFF)) * 2f +
                            (nw ushr 8 and 0xFF) + (ne ushr 8 and 0xFF) + (sw ushr 8 and 0xFF) + (se ushr 8 and 0xFF)
                        ) / 12f
                        val avgB = (
                            ((n and 0xFF) + (s and 0xFF) + (l and 0xFF) + (r and 0xFF)) * 2f +
                            (nw and 0xFF) + (ne and 0xFF) + (sw and 0xFF) + (se and 0xFF)
                        ) / 12f

                        val edgePeak = 1f - abs(a - 0.43f) / 0.43f
                        val w = (baseW * (0.65f + 0.35f * edgePeak.coerceIn(0f, 1f)) * (1f - innerFaceProtectMask[idx])).coerceIn(0f, 0.55f)
                        val fr = (((c ushr 16 and 0xFF) * (1f - w)) + avgR * w).toInt().coerceIn(0, 255)
                        val fg = (((c ushr 8 and 0xFF) * (1f - w)) + avgG * w).toInt().coerceIn(0, 255)
                        val fb = (((c and 0xFF) * (1f - w)) + avgB * w).toInt().coerceIn(0, 255)
                        next[idx] = (0xFF shl 24) or (fr shl 16) or (fg shl 8) or fb
                    }
                }
            }
            current = next
        }

        val outBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(current, 0, size, 0, 0, size, size)
        return outBmp
    }

    internal fun resolveStage5BridgePlan(
        lowMemoryMode: Boolean,
        ramStatus: SdInpaintOnnxEngine.DeviceRamStatus?,
        isRuntimeCrashLatched: Boolean
    ): Pair<Boolean, Int> {
        if (lowMemoryMode || isRuntimeCrashLatched) {
            return false to 384
        }
        if (ramStatus != null) {
            if (ramStatus.isLowMemory || ramStatus.availMb < 1500L) {
                return false to 384
            }
            val is4GbOrTight = ramStatus.totalMb in 1L..4600L || ramStatus.availMb < 2400L
            return true to if (is4GbOrTight) 384 else 512
        }
        return true to 512
    }
}
