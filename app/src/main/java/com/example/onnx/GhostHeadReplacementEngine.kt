package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Android-Compatible GHOST 2.0 Full Head Replacement Pipeline (`MODE 2 — FULL HEAD REPLACEMENT`).
 *
 * Official GHOST 2.0 Reference Stages Mapped to On-Device Android Execution:
 *  Stage 1 — Detection:       `det_10g.onnx` SCRFD-10G_KPS face, 5-point landmark & cranial/hair/neck volume detection.
 *  Stage 2 — Alignment:       5-Point Cranial-Neck Pose Similarity Alignment (scale, roll, yaw offset) into canonical head canvas.
 *  Stage 3 — Head Generation: Hybrid Source Cranial/Hair/Neck Warp + Optional `inswapper_128.onnx` inner facial expression/gaze reenactment.
 *  Stage 4 — Segmentation:    `segformer_B5_ce.onnx` + `modnet.onnx` (or Chromatic-Geodesic + Joint Bilateral Guided Hair/Skull/Neck Matte).
 *  Stage 5 — Blending:        `lama_fp32.onnx` / Boundary-Marching Background-Gap Inpainting + Selective Skin Illumination Adaptation + Organic Edge Feathering.
 *  Stage 6 — Enhancement:     `gfpgan_1.4.onnx` (or Tiled Unsharp Luminance Micro-Contrast & Detail Enhancement).
 */
object GhostHeadReplacementEngine {

    fun executeFullHeadReplacement(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        sourceFace: DetectedFace,
        targetBitmap: Bitmap,
        targetFacesToReplace: List<DetectedFace>,
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
        headPipelineMode: HeadReplacementPipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_FAST,
        enableNextGenSwapStack: Boolean = false,
        isCancelled: () -> Boolean = { false },
        onProgress: (SwapStageProgress) -> Unit
    ): FaceSwapExecutionResult {
        require(targetFacesToReplace.isNotEmpty()) {
            "Please select at least one detected target head to replace."
        }
        if (headPipelineMode != HeadReplacementPipelineMode.LEGACY_COPY_PASTE) {
            return CleanEraseHeadReplacementEngine.execute(
                context = context,
                ortEnv = ortEnv,
                sourceBitmap = sourceBitmap,
                sourceFace = sourceFace,
                targetBitmap = targetBitmap,
                targetFacesToReplace = targetFacesToReplace,
                pipelineMode = headPipelineMode,
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
                enableNextGenSwapStack = enableNextGenSwapStack,
                isCancelled = isCancelled,
                onProgress = onProgress
            )
        }
        val tStart = System.currentTimeMillis()

        val headCropSize = if (lowMemoryMode) 256 else 320
        val recFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.RECOGNIZER)
        val swapFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SWAPPER)
        val segFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SEGMENTATION)
        val matFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.MATTING)
        val lamaFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.INPAINTING)
        val gfpFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.ENHANCEMENT)
        val reagingFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.FACE_REAGING)

        // Pre-resolve target body age plans with MiVOLO v2 before heavy SegFormer/GFPGAN sessions occupy RAM
        val ageConfig = AgeHarmonyEngine.AgeHarmonyConfig(
            autoBodyAgeMatch = autoBodyAgeMatch,
            manualAgingFactor = faceAgingFactor
        )
        val targetAgePlans = if (autoBodyAgeMatch || kotlin.math.abs(faceAgingFactor) >= 0.05f) {
            targetFacesToReplace.map { tgtFace ->
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

        // STAGE 1: Detection & 3D-Aware Cranial Pose Analysis
        val tDet0 = System.currentTimeMillis()
        onProgress(
            SwapStageProgress(
                stepIndex = 1,
                totalSteps = 6,
                stageTitle = "Stage 1/6: Detection",
                detailMessage = "Analyzing source & target head pose, roll angle, yaw ratio, and cranial/hair/neck bounds...",
                progressFraction = 0.12f
            )
        )
        val srcHeadPose = HeadSegmentationAndInpainting.analyzeHeadPoseAndBounds(
            sourceFace,
            sourceBitmap.width,
            sourceBitmap.height
        )
        val detMs = (System.currentTimeMillis() - tDet0).coerceAtLeast(1L)

        // STAGE 2: Head Alignment & Source Identity Extraction
        val tAlign0 = System.currentTimeMillis()
        onProgress(
            SwapStageProgress(
                stepIndex = 2,
                totalSteps = 6,
                stageTitle = "Stage 2/6: Alignment",
                detailMessage = "Aligning source head, hair crown, skull & neck to canonical ${headCropSize}x${headCropSize} space...",
                progressFraction = 0.28f
            )
        )

        val srcHeadMatrix = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(
            srcLandmarks5 = sourceFace.landmarks5,
            cropSize = headCropSize,
            targetYawRatio = srcHeadPose.yawRatio
        )
        val alignedSourceHead = FaceAlignment.warpAffineCrop(
            srcBitmap = sourceBitmap,
            forwardMatrix2x3 = srcHeadMatrix,
            dstSize = headCropSize
        )

        // If inswapper_128.onnx is installed/preloaded, also extract the 512-D source identity vector for facial gaze/expression fusion
        val hasInswapper = ortEnv != null && (preloadedSwapSession != null || (swapFile.exists() && swapFile.length() > 1024L))
        val sourceEmbedding: SourceEmbeddingResult? = if (ortEnv != null && hasInswapper &&
            (preloadedArcFaceSession != null || (recFile.exists() && recFile.length() > 1024L) || allowTwoModelFallbackForTesting)
        ) {
            val emap = preloadedEmap512x512 ?: OnnxProtobufInspector.loadOrExtractInswapperEmap(context, swapFile)
            ArcFaceRecognizer.extractSourceLatentEmbedding(
                ortEnv = ortEnv,
                sourceBitmap = sourceBitmap,
                sourceLandmarks5 = sourceFace.landmarks5,
                arcFaceModelFile = if (recFile.exists()) recFile else null,
                emap512x512 = emap,
                allowTwoModelFallbackForTesting = allowTwoModelFallbackForTesting,
                preloadedArcFaceSession = preloadedArcFaceSession
            )
        } else {
            null
        }
        val alignAndEmbMs = (System.currentTimeMillis() - tAlign0).coerceAtLeast(1L)

        // STAGE 3/4 Prep (Source Pass): Segment Source Head / Hair / Skull / Neck Matte
        onProgress(
            SwapStageProgress(
                stepIndex = 3,
                totalSteps = 6,
                stageTitle = "Stage 3/6: Source Segmentation",
                detailMessage = "Segmenting source head, hair strands, ears, and upper neck silhouette...",
                progressFraction = 0.42f
            )
        )
        val rawSrcMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
            ortEnv = ortEnv,
            alignedHeadCrop = alignedSourceHead,
            segModelFile = if (segFile.exists()) segFile else null,
            mattingModelFile = if (matFile.exists()) matFile else null,
            preferHardwareAccel = preferHardwareAccel,
            preloadedSegformerSession = preloadedSegformerSession
        )
        val canonicalSrcTemplate = HeadSegmentationAndInpainting.getCanonicalHeadTemplate(
            cropSize = headCropSize,
            yawRatio = srcHeadPose.yawRatio
        )
        val canonicalCenterNx = (canonicalSrcTemplate[2].x / headCropSize.toFloat()).coerceIn(0.38f, 0.62f)
        val srcBoundsMask = computeCopyPasteInBoundsMask(
            forwardHeadMatrix2x3 = srcHeadMatrix,
            srcWidth = sourceBitmap.width,
            srcHeight = sourceBitmap.height,
            cropSize = headCropSize,
            featherPx = 14f
        )
        val cleanedSrcAlpha = sanitizeCopyPasteHeadAlpha(
            rawAlpha = rawSrcMasks.fullHeadAlpha,
            cropBitmap = alignedSourceHead,
            cropSize = headCropSize,
            centerNx = canonicalCenterNx,
            inBoundsMask = srcBoundsMask,
            isSourceMask = true
        )
        val srcMasks = rawSrcMasks.copy(fullHeadAlpha = cleanedSrcAlpha)

        val targetW = targetBitmap.width
        val targetH = targetBitmap.height
        val compositePixels = IntArray(targetW * targetH)
        targetBitmap.getPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)

        var firstTargetHeadPreview: Bitmap? = null
        var firstGeneratedHeadPreview: Bitmap? = null
        var totalGenMs = 0L
        var totalBlendMs = 0L
        var usedGfpganPass = false

        // Process each target head sequentially, releasing intermediate Bitmaps & sessions to conserve RAM
        targetFacesToReplace.forEachIndexed { idx, targetFace ->
            val tgtHeadMatrix = HeadSegmentationAndInpainting.estimateHeadAlignmentMatrix(
                srcLandmarks5 = targetFace.landmarks5,
                cropSize = headCropSize,
                targetYawRatio = srcHeadPose.yawRatio
            )
            val alignedTargetHead = FaceAlignment.warpAffineCrop(
                srcBitmap = targetBitmap,
                forwardMatrix2x3 = tgtHeadMatrix,
                dstSize = headCropSize
            )

            // STAGE 4: Head Generation (Combine Source Cranial/Hair/Neck with Target Expression/Gaze Core)
            val tGen0 = System.currentTimeMillis()
            onProgress(
                SwapStageProgress(
                    stepIndex = 4,
                    totalSteps = 6,
                    stageTitle = "Stage 4/6: Head generation (${idx + 1}/${targetFacesToReplace.size})",
                    detailMessage = "Synthesizing aligned head structure, hair volume, and facial identity...",
                    progressFraction = 0.58f
                )
            )

            val synthesizedHeadCrop = alignedSourceHead.copy(Bitmap.Config.ARGB_8888, true)
            if (faceReactionMode == FaceReactionSourceMode.TARGET_REACTION &&
                ortEnv != null && hasInswapper && sourceEmbedding != null
            ) {
                // Fuse InSwapper-128 reenacted Target facial expression (smile/teeth/tongue) into the source head
                runCatching {
                    fuseInswapperInnerExpressionIntoSourceHead(
                        ortEnv = ortEnv,
                        swapFile = swapFile,
                        preloadedSwapSession = preloadedSwapSession,
                        sourceEmbedding = sourceEmbedding,
                        targetBitmap = targetBitmap,
                        targetFace = targetFace,
                        synthesizedHeadCrop = synthesizedHeadCrop,
                        headCropSize = headCropSize,
                        srcYawRatio = srcHeadPose.yawRatio,
                        preferHardwareAccel = preferHardwareAccel
                    )
                }
            }
            totalGenMs += (System.currentTimeMillis() - tGen0).coerceAtLeast(1L)

            // STAGE 5: Background-Gap Disocclusion Inpainting + Selective Skin Color Adaptation + Organic Blending
            val tBlend0 = System.currentTimeMillis()
            onProgress(
                SwapStageProgress(
                    stepIndex = 5,
                    totalSteps = 6,
                    stageTitle = "Stage 5/6: Blending",
                    detailMessage = "Inpainting target hair disocclusion gaps and applying ${skinToneMode.title}...",
                    progressFraction = 0.78f
                )
            )

            val rawTgtMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
                ortEnv = ortEnv,
                alignedHeadCrop = alignedTargetHead,
                segModelFile = if (segFile.exists()) segFile else null,
                mattingModelFile = if (matFile.exists()) matFile else null,
                preferHardwareAccel = preferHardwareAccel,
                preloadedSegformerSession = preloadedSegformerSession
            )
            val tgtBoundsMask = computeCopyPasteInBoundsMask(
                forwardHeadMatrix2x3 = tgtHeadMatrix,
                srcWidth = targetW,
                srcHeight = targetH,
                cropSize = headCropSize,
                featherPx = 14f
            )
            val cleanedTgtAlpha = sanitizeCopyPasteHeadAlpha(
                rawAlpha = rawTgtMasks.fullHeadAlpha,
                cropBitmap = alignedTargetHead,
                cropSize = headCropSize,
                centerNx = canonicalCenterNx,
                inBoundsMask = tgtBoundsMask,
                isSourceMask = false
            )
            val tgtMasks = rawTgtMasks.copy(fullHeadAlpha = cleanedTgtAlpha)

            // 5a. Only run neural LaMa disocclusion inpainting when lama_fp32.onnx is installed;
            // never paint horizontal scanline stripes over Target's natural outer hair.
            val hasNeuralLama = ortEnv != null && lamaFile.exists() && lamaFile.length() > 1024L
            val inpaintedTargetCrop = if (hasNeuralLama) {
                HeadSegmentationAndInpainting.inpaintTargetBackgroundGap(
                    ortEnv = ortEnv,
                    targetAlignedCrop = alignedTargetHead,
                    targetHeadAlpha = tgtMasks.fullHeadAlpha,
                    sourceHeadAlpha = srcMasks.fullHeadAlpha,
                    lamaModelFile = lamaFile,
                    preferHardwareAccel = preferHardwareAccel
                )
            } else {
                alignedTargetHead.copy(Bitmap.Config.ARGB_8888, true)
            }

            // 5b. Adapt skin & neck lighting/warmth according to SkinToneSourceMode without recoloring source hair
            val illuminationAdaptedHead = if (enableColorTransfer) {
                adaptHeadSkinAndLightingSelective(
                    sourceHeadCrop = synthesizedHeadCrop,
                    targetHeadCrop = alignedTargetHead,
                    skinAndNeckWeight = srcMasks.skinAndNeckWeight,
                    skinToneMode = skinToneMode
                )
            } else {
                synthesizedHeadCrop
            }

            // 5c. Composite the inpainted background + new source head/hair/neck in canonical head space
            val compositedHeadCrop = compositeHeadOverInpaintedCrop(
                sourceHeadCrop = illuminationAdaptedHead,
                inpaintedTargetCrop = inpaintedTargetCrop,
                sourceHeadAlpha = srcMasks.fullHeadAlpha,
                targetHeadAlpha = tgtMasks.fullHeadAlpha
            )

            // STAGE 6: Enhancement (GFPGAN 1.4 512x512 ONNX Restoration + Tiled Detail & Micro-Contrast)
            onProgress(
                SwapStageProgress(
                    stepIndex = 6,
                    totalSteps = 6,
                    stageTitle = "Stage 6/6: Enhancement",
                    detailMessage = "Enhancing facial & hair strand detail and re-projecting onto target body...",
                    progressFraction = 0.92f
                )
            )
            val headAgePlan = targetAgePlans.getOrNull(idx)
            val clampedEnhance = enhancementStrength.coerceIn(0f, 1f)
            val gfpRestoredHead512 = if (clampedEnhance > 0.05f && ortEnv != null && (preloadedGfpganSession != null || (gfpFile.exists() && gfpFile.length() > 1024L))) {
                val restored512 = FaceBlender.runOptionalGfpganEnhancement512(
                    ortEnv = ortEnv,
                    gfpganFile = if (gfpFile.exists()) gfpFile else null,
                    crop128Or512 = compositedHeadCrop,
                    preferHardwareAccel = preferHardwareAccel,
                    preloadedGfpganSession = preloadedGfpganSession
                )
                if (restored512 != null) {
                    usedGfpganPass = true
                    val ageScale = (headAgePlan?.gfpganWeightScale ?: 1.0f).coerceIn(0.35f, 1.0f)
                    if (ageScale < 0.99f) {
                        val base512 = Bitmap.createScaledBitmap(compositedHeadCrop, 512, 512, true)
                        val rPx = IntArray(512 * 512)
                        val bPx = IntArray(512 * 512)
                        restored512.getPixels(rPx, 0, 512, 0, 0, 512, 512)
                        base512.getPixels(bPx, 0, 512, 0, 0, 512, 512)
                        base512.recycle()
                        val invS = 1f - ageScale
                        for (i in rPx.indices) {
                            val rc = rPx[i]
                            val bc = bPx[i]
                            val r = (((rc ushr 16) and 0xFF) * ageScale + ((bc ushr 16) and 0xFF) * invS).toInt().coerceIn(0, 255)
                            val g = (((rc ushr 8) and 0xFF) * ageScale + ((bc ushr 8) and 0xFF) * invS).toInt().coerceIn(0, 255)
                            val b = ((rc and 0xFF) * ageScale + (bc and 0xFF) * invS).toInt().coerceIn(0, 255)
                            rPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                        }
                        restored512.setPixels(rPx, 0, 512, 0, 0, 512, 512)
                    }
                    restored512
                } else {
                    null
                }
            } else {
                null
            }
            // Keep full 512x512 HD resolution if GFPGAN restored at 512x512 (NEVER downscale back to 128x128 or headCropSize!)
            val rawEnhancedHeadCrop = applyTiledDetailEnhancement(gfpRestoredHead512 ?: compositedHeadCrop)
            gfpRestoredHead512?.recycle()

            // Stage 6b: Apply Body-Age Re-Aging onto Face & Neck Skin strictly (protecting Hair/Crown)
            val enhancedHeadCrop = if (headAgePlan != null && headAgePlan.shouldApplyReAging) {
                val work512 = if (rawEnhancedHeadCrop.width == 512 && rawEnhancedHeadCrop.height == 512) {
                    if (rawEnhancedHeadCrop.isMutable) rawEnhancedHeadCrop else rawEnhancedHeadCrop.copy(Bitmap.Config.ARGB_8888, true).also { rawEnhancedHeadCrop.recycle() }
                } else {
                    Bitmap.createScaledBitmap(rawEnhancedHeadCrop, 512, 512, true).also { rawEnhancedHeadCrop.recycle() }
                }
                val headPx512 = IntArray(512 * 512)
                work512.getPixels(headPx512, 0, 512, 0, 0, 512, 512)
                val skinMask512 = FloatArray(512 * 512) { idx512 ->
                    val y512 = idx512 / 512
                    val x512 = idx512 % 512
                    val sy = (y512 * headCropSize / 512).coerceIn(0, headCropSize - 1)
                    val sx = (x512 * headCropSize / 512).coerceIn(0, headCropSize - 1)
                    srcMasks.skinAndNeckWeight[sy * headCropSize + sx].coerceIn(0f, 1f)
                }
                val headGeom512 = FaceBlender.computeWarpedGeometry(
                    forwardMatrix128 = FloatArray(6) { i -> tgtHeadMatrix[i] * (512f / headCropSize.toFloat()) },
                    targetLandmarks5 = targetFace.landmarks5,
                    scale = 1.0f
                )
                AgeHarmonyEngine.reageCrop512(
                    ortEnv = ortEnv,
                    reagingFile = if (reagingFile.exists() && reagingFile.length() > 1024L) reagingFile else null,
                    pixels512 = headPx512,
                    geom512 = headGeom512,
                    agePlan = headAgePlan,
                    maskWeights512 = skinMask512,
                    preferHardwareAccel = preferHardwareAccel
                )
                work512.setPixels(headPx512, 0, 512, 0, 0, 512, 512)
                work512
            } else {
                rawEnhancedHeadCrop
            }

            // Paste the full head (plus neural LaMa disocclusion envelope if installed) back onto target
            val baseEnvelopeAlpha = buildCleanCopyPasteEnvelopeAlpha(
                sourceHeadAlpha = srcMasks.fullHeadAlpha,
                targetHeadAlpha = tgtMasks.fullHeadAlpha,
                cropSize = headCropSize,
                centerNx = canonicalCenterNx,
                hasNeuralInpaint = hasNeuralLama
            )
            val warpSize = enhancedHeadCrop.width
            val warpScale = warpSize.toFloat() / headCropSize.toFloat()
            val warpMatrix2x3 = if (warpSize == headCropSize) {
                tgtHeadMatrix
            } else {
                FloatArray(6) { i -> tgtHeadMatrix[i] * warpScale }
            }
            val warpEnvelopeAlpha = if (warpSize == headCropSize) {
                baseEnvelopeAlpha
            } else {
                upscaleAlphaBilinear(baseEnvelopeAlpha, headCropSize, warpSize)
            }

            warpHeadCropBackToTarget(
                targetPixels = compositePixels,
                targetWidth = targetW,
                targetHeight = targetH,
                headCrop = enhancedHeadCrop,
                forwardHeadMatrix2x3 = warpMatrix2x3,
                envelopeAlpha = warpEnvelopeAlpha,
                blendStrength = blendStrength,
                offsetX = offsetX,
                offsetY = offsetY,
                scaleAdjust = scaleAdjust
            )

            // Final post-swap retouch: match output face skin colour directly to Target's actual neck colour
            if (enableColorTransfer) {
                applyFinalNeckToFaceColorRetouch(
                    compositePixels = compositePixels,
                    targetBitmap = targetBitmap,
                    targetWidth = targetW,
                    targetHeight = targetH,
                    forwardHeadMatrix2x3 = tgtHeadMatrix,
                    headCropSize = headCropSize,
                    envelopeAlpha = baseEnvelopeAlpha,
                    skinAndNeckWeight = srcMasks.skinAndNeckWeight,
                    centerNx = canonicalCenterNx,
                    skinToneMode = skinToneMode,
                    offsetX = offsetX,
                    offsetY = offsetY,
                    scaleAdjust = scaleAdjust
                )
            }

            if (illuminationAdaptedHead !== synthesizedHeadCrop) illuminationAdaptedHead.recycle()
            inpaintedTargetCrop.recycle()
            compositedHeadCrop.recycle()

            if (firstTargetHeadPreview == null) {
                firstTargetHeadPreview = alignedTargetHead.copy(Bitmap.Config.ARGB_8888, false)
                firstGeneratedHeadPreview = enhancedHeadCrop.copy(Bitmap.Config.ARGB_8888, false)
            }
            alignedTargetHead.recycle()
            synthesizedHeadCrop.recycle()
            enhancedHeadCrop.recycle()

            totalBlendMs += (System.currentTimeMillis() - tBlend0).coerceAtLeast(1L)
        }

        val srcThumb112 = sourceEmbedding?.aligned112Crop
            ?: Bitmap.createScaledBitmap(alignedSourceHead, 112, 112, true)
        alignedSourceHead.recycle()

        if (lowMemoryMode) {
            System.gc()
        }

        val finalBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        finalBitmap.setPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)

        val targetDepthMap = if (portraitBlurStrength > 0.03f &&
            DepthEstimationEngine.hasInstalledDepthModel(context)
        ) {
            runCatching {
                DepthEstimationEngine.estimateDepth(
                    context = context,
                    ortEnv = ortEnv,
                    sourceBitmap = targetBitmap,
                    preferHardwareAccel = preferHardwareAccel
                )
            }.getOrNull()
        } else null

        if (portraitBlurStrength > 0.03f) {
            HeadSegmentationAndInpainting.applyPortraitModeBackgroundBokeh(
                bitmap = finalBitmap,
                faces = targetFacesToReplace,
                blurStrength = portraitBlurStrength,
                depthMap = targetDepthMap
            )
        }

        if (enableProvenanceWatermark) {
            FaceBlender.applyEthicalProvenanceWatermark(finalBitmap)
        }

        val engineUsed = if (hasInswapper && sourceEmbedding?.usedArcFaceModel == true) {
            "NEURAL"
        } else {
            "SYNTHETIC"
        }
        val totalMs = (System.currentTimeMillis() - tStart).coerceAtLeast(1L)
        val summary = buildString {
            append("GHOST-Android Head Replace: ")
            append(srcMasks.segmentationSourceLabel)
            if (hasInswapper && sourceEmbedding != null) {
                append(" + inswapper/HyperSwap Expression Core")
            }
            if (usedGfpganPass) {
                append(" + gfpgan_1.4.onnx (512x512 HD)")
            }
            if (portraitBlurStrength > 0.03f) {
                val prefix = if (targetDepthMap != null) " + 3D Depth Bokeh" else " + DSLR Bokeh"
                append("$prefix ${(portraitBlurStrength * 100).toInt()}%")
            }
            append(" | engineUsed=$engineUsed")
        }

        return FaceSwapExecutionResult(
            outputBitmap = finalBitmap,
            alignedSource112 = srcThumb112,
            alignedTarget128 = firstTargetHeadPreview
                ?: Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888),
            rawSwapped128 = firstGeneratedHeadPreview
                ?: Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888),
            swappedFacesCount = targetFacesToReplace.size,
            detectionMs = detMs,
            embeddingMs = alignAndEmbMs,
            inswapperMs = totalGenMs.coerceAtLeast(1L),
            blendingMs = totalBlendMs.coerceAtLeast(1L),
            totalMs = totalMs,
            pipelineSummary = summary,
            engineUsed = engineUsed,
            ageHarmonySummary = firstAgeHarmonySummary
        )
    }

    /**
     * Runs the active swap model (`hyperswap_1b_256.onnx` 256x256 or `inswapper_128.onnx` 128x128)
     * on the target face and blends the reenacted inner facial expression cleanly into the aligned source head.
     */
    internal fun fuseInswapperInnerExpressionIntoSourceHead(
        ortEnv: OrtEnvironment,
        swapFile: java.io.File,
        preloadedSwapSession: OrtSession? = null,
        sourceEmbedding: SourceEmbeddingResult,
        targetBitmap: Bitmap,
        targetFace: DetectedFace,
        synthesizedHeadCrop: Bitmap,
        headCropSize: Int,
        srcYawRatio: Float,
        preferHardwareAccel: Boolean,
        overrideHeadLandmarks5: List<PointF>? = null
    ) {
        val m128 = FaceAlignment.estimateNorm(targetFace.landmarks5, 128)
        val targetCrop128 = FaceAlignment.warpAffineCrop(targetBitmap, m128, 128)

        var rawSwappedNative: Bitmap? = null
        var nativeFusionSize = 128
        val runWithSession: (OrtSession) -> Unit = { session ->
            var targetName = "target"
            var sourceName = "source"
            var cropSize = 128
            for ((name, nodeInfo) in session.inputInfo) {
                val tInfo = nodeInfo.info as? ai.onnxruntime.TensorInfo ?: continue
                if (tInfo.shape.size == 4) {
                    targetName = name
                    val hDim = tInfo.shape[2].toInt()
                    if (hDim == 256 || hDim == 128) cropSize = hDim
                }
                if (tInfo.shape.size == 2) sourceName = name
            }
            val isHyperSwap256 = (cropSize == 256)
            val mCrop = if (cropSize == 128) m128 else FaceAlignment.estimateNorm(targetFace.landmarks5, cropSize)
            val alignedCrop = if (cropSize == 128) targetCrop128 else FaceAlignment.warpAffineCrop(targetBitmap, mCrop, cropSize)
            val hw = cropSize * cropSize
            val pixels = IntArray(hw)
            alignedCrop.getPixels(pixels, 0, cropSize, 0, 0, cropSize, cropSize)

            val targetBuf = java.nio.FloatBuffer.allocate(3 * hw)
            if (isHyperSwap256) {
                for (i in 0 until hw) {
                    val c = pixels[i]
                    targetBuf.put(i, (((c ushr 16) and 0xFF) - 127.5f) / 127.5f)
                    targetBuf.put(hw + i, (((c ushr 8) and 0xFF) - 127.5f) / 127.5f)
                    targetBuf.put(2 * hw + i, ((c and 0xFF) - 127.5f) / 127.5f)
                }
            } else {
                for (i in 0 until hw) {
                    val c = pixels[i]
                    targetBuf.put(i, ((c ushr 16) and 0xFF) / 255.0f)
                    targetBuf.put(hw + i, ((c ushr 8) and 0xFF) / 255.0f)
                    targetBuf.put(2 * hw + i, (c and 0xFF) / 255.0f)
                }
            }
            targetBuf.rewind()

            val srcVec = if (isHyperSwap256 && !sourceEmbedding.usedEmbeddedEmap) {
                sourceEmbedding.rawNormedEmbedding512
            } else {
                sourceEmbedding.latentSourceVector512
            }
            val srcBuf = java.nio.FloatBuffer.wrap(srcVec)
            val outPixels: IntArray
            var decodedPixels = IntArray(hw)
            OnnxTensor.createTensor(ortEnv, targetBuf, longArrayOf(1L, 3L, cropSize.toLong(), cropSize.toLong())).use { tTensor ->
                OnnxTensor.createTensor(ortEnv, srcBuf, longArrayOf(1L, 512L)).use { sTensor ->
                    session.run(mapOf(targetName to tTensor, sourceName to sTensor)).use { res ->
                        val outT = res[0] as OnnxTensor
                        val outFloats = FloatArray(3 * hw)
                        outT.floatBuffer.get(outFloats)
                        val rangeHint = if (isHyperSwap256) {
                            OnnxProtobufInspector.SwapperOutputRangeHint.AUTO
                        } else {
                            OnnxProtobufInspector.SwapperOutputRangeHint.UNSIGNED_0_TO_1
                        }
                        val isSigned = InSwapperEngine.inferSignedOutputMode(outFloats, rangeHint)
                        decodedPixels = InSwapperEngine.decodeRawOutputToPixels(
                            outFloats = outFloats,
                            hw = hw,
                            channelOrder = InSwapperEngine.SwapChannelOrder.RGB,
                            isSignedOutput = isSigned
                        )
                    }
                }
            }
            outPixels = decodedPixels
            if (alignedCrop !== targetCrop128) alignedCrop.recycle()
            val outBmp = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
            outBmp.setPixels(outPixels, 0, cropSize, 0, 0, cropSize, cropSize)
            nativeFusionSize = cropSize
            rawSwappedNative = outBmp
        }
        if (preloadedSwapSession != null) {
            runWithSession(preloadedSwapSession)
        } else {
            val cached = OnnxProtobufInspector.getOrCreateCachedSession(ortEnv, swapFile, preferHardwareAccel)
            if (cached != null) runWithSession(cached)
        }

        val activeSwapBmp = rawSwappedNative ?: run {
            targetCrop128.recycle()
            return
        }
        val hasTrueArcFaceLatent = sourceEmbedding.usedArcFaceModel
        val fusionSize = nativeFusionSize
        val swappedPixels = IntArray(fusionSize * fusionSize)
        if (fusionSize == 128) {
            val eyeRestoredBmp = FaceBlender.restoreEyesAndEliminateNegativeArtifacts128(
                swapped128 = activeSwapBmp,
                alignedTarget128 = targetCrop128,
                alignedSource112 = sourceEmbedding.aligned112Crop,
                hasTrueArcFaceLatent = hasTrueArcFaceLatent,
                forwardMatrix128 = m128,
                targetLandmarks5 = targetFace.landmarks5
            )
            eyeRestoredBmp.getPixels(swappedPixels, 0, 128, 0, 0, 128, 128)
            eyeRestoredBmp.recycle()
        } else {
            // Keep full 256px HyperSwap native resolution (Option B+C HD) without 256 -> 128 downscale
            activeSwapBmp.getPixels(swappedPixels, 0, fusionSize, 0, 0, fusionSize, fusionSize)
        }
        activeSwapBmp.recycle()
        targetCrop128.recycle()

        // Map canonical head keypoints (or override canvas landmarks) to fusionSize coordinates
        val headPts = if (overrideHeadLandmarks5 != null && overrideHeadLandmarks5.size >= 5) {
            overrideHeadLandmarks5.take(5)
        } else {
            HeadSegmentationAndInpainting.getCanonicalHeadTemplate(headCropSize, srcYawRatio).toList()
        }
        val headToFusion = FaceAlignment.estimateNorm(headPts, fusionSize)
        val featheredMask128 = FaceBlender.createFeatheredFaceMask128()
        val headPixels = IntArray(headCropSize * headCropSize)
        synthesizedHeadCrop.getPixels(headPixels, 0, headCropSize, 0, 0, headCropSize, headCropSize)
        val expressionBlendScale = if (hasTrueArcFaceLatent) 0.88f else 0.25f
        val validMin = 6f * (fusionSize / 128f)
        val validMax = (fusionSize - 7).toFloat()

        for (y in 0 until headCropSize) {
            for (x in 0 until headCropSize) {
                val u = headToFusion[0] * x + headToFusion[1] * y + headToFusion[2]
                val v = headToFusion[3] * x + headToFusion[4] * y + headToFusion[5]
                if (u in validMin..validMax && v in validMin..validMax) {
                    val uMask = (u * 128f / fusionSize).toInt().coerceIn(0, 127)
                    val vMask = (v * 128f / fusionSize).toInt().coerceIn(0, 127)
                    val w = featheredMask128[vMask * 128 + uMask] * expressionBlendScale
                    if (w > 0.005f) {
                        val swapC = FaceAlignment.sampleBicubicClamped(swappedPixels, fusionSize, fusionSize, u, v)
                        val srcC = headPixels[y * headCropSize + x]
                        val r = ((srcC ushr 16 and 0xFF) * (1f - w) + (swapC ushr 16 and 0xFF) * w).toInt().coerceIn(0, 255)
                        val g = ((srcC ushr 8 and 0xFF) * (1f - w) + (swapC ushr 8 and 0xFF) * w).toInt().coerceIn(0, 255)
                        val b = ((srcC and 0xFF) * (1f - w) + (swapC and 0xFF) * w).toInt().coerceIn(0, 255)
                        headPixels[y * headCropSize + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                }
            }
        }
        synthesizedHeadCrop.setPixels(headPixels, 0, headCropSize, 0, 0, headCropSize, headCropSize)
    }

    /**
     * Adapts skin and neck illumination/color statistics to match the target scene while preserving
     * the source hairstyle's natural color and protecting the eye sockets from pupil/sclera washout.
     */
    internal fun adaptHeadSkinAndLightingSelective(
        sourceHeadCrop: Bitmap,
        targetHeadCrop: Bitmap,
        skinAndNeckWeight: FloatArray,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        overrideLandmarks5: List<PointF>? = null
    ): Bitmap {
        val size = sourceHeadCrop.width
        val total = size * size
        val srcPx = IntArray(total)
        val tgtPx = IntArray(total)
        sourceHeadCrop.getPixels(srcPx, 0, size, 0, 0, size, size)
        targetHeadCrop.getPixels(tgtPx, 0, size, 0, 0, size, size)

        // Compute mean & std strictly on mid-face skin (cheeks & nose bridge, excluding eye row)
        var sRMean = 0.0
        var sGMean = 0.0
        var sBMean = 0.0
        var tRMean = 0.0
        var tGMean = 0.0
        var tBMean = 0.0
        var count = 0

        val hasOverride = overrideLandmarks5 != null && overrideLandmarks5.size >= 5
        val ovEyeDist = if (hasOverride) {
            hypot(
                (overrideLandmarks5!![1].x - overrideLandmarks5[0].x).toDouble(),
                (overrideLandmarks5[1].y - overrideLandmarks5[0].y).toDouble()
            ).toFloat().coerceAtLeast(10f)
        } else size * (60f / 256f)
        val ovEyeMidX = if (hasOverride) (overrideLandmarks5!![0].x + overrideLandmarks5[1].x) * 0.5f else size * 0.5f
        val ovEyeMidY = if (hasOverride) (overrideLandmarks5!![0].y + overrideLandmarks5[1].y) * 0.5f else size * (114f / 256f)
        val ovMouthY = if (hasOverride) (overrideLandmarks5!![3].y + overrideLandmarks5[4].y) * 0.5f else size * (178f / 256f)

        val x0 = if (hasOverride) (ovEyeMidX - ovEyeDist * 0.52f).toInt().coerceIn(0, size - 2) else (size * 0.38f).toInt()
        val x1 = if (hasOverride) (ovEyeMidX + ovEyeDist * 0.52f).toInt().coerceIn(x0 + 1, size - 1) else (size * 0.62f).toInt()
        val y0 = if (hasOverride) (ovEyeMidY + ovEyeDist * 0.25f).toInt().coerceIn(0, size - 2) else (size * 0.50f).toInt()
        val y1 = if (hasOverride) (ovMouthY - ovEyeDist * 0.12f).toInt().coerceIn(y0 + 1, size - 1) else (size * 0.66f).toInt()

        for (y in y0..y1) {
            for (x in x0..x1) {
                val idx = y * size + x
                val sc = srcPx[idx]
                val tc = tgtPx[idx]
                val sr = (sc ushr 16) and 0xFF
                val sg = (sc ushr 8) and 0xFF
                val sb = sc and 0xFF
                val tr = (tc ushr 16) and 0xFF
                val tg = (tc ushr 8) and 0xFF
                val tb = tc and 0xFF
                val sLum = 0.299f * sr + 0.587f * sg + 0.114f * sb
                val tLum = 0.299f * tr + 0.587f * tg + 0.114f * tb
                if (sLum !in 38f..244f || tLum !in 38f..244f) continue
                sRMean += sr
                sGMean += sg
                sBMean += sb
                tRMean += tr
                tGMean += tg
                tBMean += tb
                count++
            }
        }
        if (count < 16) return sourceHeadCrop.copy(Bitmap.Config.ARGB_8888, true)

        sRMean /= count
        sGMean /= count
        sBMean /= count
        tRMean /= count
        tGMean /= count
        tBMean /= count

        var sRVar = 0.0
        var sGVar = 0.0
        var sBVar = 0.0
        var tRVar = 0.0
        var tGVar = 0.0
        var tBVar = 0.0

        for (y in y0..y1) {
            for (x in x0..x1) {
                val idx = y * size + x
                val sc = srcPx[idx]
                val tc = tgtPx[idx]
                val srRaw = (sc ushr 16) and 0xFF
                val sgRaw = (sc ushr 8) and 0xFF
                val sbRaw = sc and 0xFF
                val trRaw = (tc ushr 16) and 0xFF
                val tgRaw = (tc ushr 8) and 0xFF
                val tbRaw = tc and 0xFF
                val sLum = 0.299f * srRaw + 0.587f * sgRaw + 0.114f * sbRaw
                val tLum = 0.299f * trRaw + 0.587f * tgRaw + 0.114f * tbRaw
                if (sLum !in 38f..244f || tLum !in 38f..244f) continue
                val sr = srRaw - sRMean
                val sg = sgRaw - sGMean
                val sb = sbRaw - sBMean
                val tr = trRaw - tRMean
                val tg = tgRaw - tGMean
                val tb = tbRaw - tBMean
                sRVar += sr * sr
                sGVar += sg * sg
                sBVar += sb * sb
                tRVar += tr * tr
                tGVar += tg * tg
                tBVar += tb * tb
            }
        }

        val scaleR = (sqrt(tRVar / count).coerceAtLeast(8.0) / sqrt(sRVar / count).coerceAtLeast(8.0)).coerceIn(0.88, 1.15)
        val scaleG = (sqrt(tGVar / count).coerceAtLeast(8.0) / sqrt(sGVar / count).coerceAtLeast(8.0)).coerceIn(0.88, 1.15)
        val scaleB = (sqrt(tBVar / count).coerceAtLeast(8.0) / sqrt(sBVar / count).coerceAtLeast(8.0)).coerceIn(0.88, 1.15)

        // Ambient luminance ratio for gentle hair exposure adaptation
        val srcLum = 0.299 * sRMean + 0.587 * sGMean + 0.114 * sBMean
        val tgtLum = 0.299 * tRMean + 0.587 * tGMean + 0.114 * tBMean
        val hairAmbientScale = (tgtLum / srcLum.coerceAtLeast(15.0)).coerceIn(0.85, 1.12).toFloat()

        val leftEyeX = if (hasOverride) overrideLandmarks5!![0].x else size * (98.0f / 256.0f)
        val rightEyeX = if (hasOverride) overrideLandmarks5!![1].x else size * (158.0f / 256.0f)
        val eyeY = if (hasOverride) ovEyeMidY else size * (114.0f / 256.0f)
        val eyeRx = ovEyeDist * 0.43f
        val eyeRy = ovEyeDist * 0.27f

        val skinStrength = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> 0.84f
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.16f
            SkinToneSourceMode.BALANCED_BLEND -> 0.52f
        }

        val outPx = IntArray(total)
        for (y in 0 until size) {
            val row = y * size
            val dy = (y - eyeY) / eyeRy
            for (x in 0 until size) {
                val i = row + x
                val sc = srcPx[i]
                val r = (sc ushr 16) and 0xFF
                val g = (sc ushr 8) and 0xFF
                val b = sc and 0xFF

                val ldx = (x - leftEyeX) / eyeRx
                val rdx = (x - rightEyeX) / eyeRx
                val eyeDistSq = min(ldx * ldx + dy * dy, rdx * rdx + dy * dy)
                val eyeProt = (1.0f - eyeDistSq).coerceIn(0f, 1f)

                val skinAdaptR = ((r - sRMean) * scaleR + tRMean).toFloat()
                val skinAdaptG = ((g - sGMean) * scaleG + tGMean).toFloat()
                val skinAdaptB = ((b - sBMean) * scaleB + tBMean).toFloat()

                val hairAdaptR = r * hairAmbientScale
                val hairAdaptG = g * hairAmbientScale
                val hairAdaptB = b * hairAmbientScale

                val sw = (skinAndNeckWeight[i] * skinStrength * (1.0f - 0.90f * eyeProt)).coerceIn(0f, 0.80f)
                val hw = (1f - skinAndNeckWeight[i]).coerceIn(0f, 1f) * 0.22f
                val origW = (1f - sw - hw).coerceIn(0f, 1f)

                val finalR = (r * origW + skinAdaptR * sw + hairAdaptR * hw).toInt().coerceIn(0, 255)
                val finalG = (g * origW + skinAdaptG * sw + hairAdaptG * hw).toInt().coerceIn(0, 255)
                val finalB = (b * origW + skinAdaptB * sw + hairAdaptB * hw).toInt().coerceIn(0, 255)
                outPx[i] = (0xFF shl 24) or (finalR shl 16) or (finalG shl 8) or finalB
            }
        }

        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, size, 0, 0, size, size)
        return out
    }

    private fun compositeHeadOverInpaintedCrop(
        sourceHeadCrop: Bitmap,
        inpaintedTargetCrop: Bitmap,
        sourceHeadAlpha: FloatArray,
        targetHeadAlpha: FloatArray
    ): Bitmap {
        val size = sourceHeadCrop.width
        val total = size * size
        val srcPx = IntArray(total)
        val bgPx = IntArray(total)
        sourceHeadCrop.getPixels(srcPx, 0, size, 0, 0, size, size)
        inpaintedTargetCrop.getPixels(bgPx, 0, size, 0, 0, size, size)

        val outPx = IntArray(total)
        for (i in 0 until total) {
            val a = sourceHeadAlpha[i].coerceIn(0f, 1f)
            val invA = 1.0f - a
            val sc = srcPx[i]
            val bc = bgPx[i]
            val r = (((sc ushr 16) and 0xFF) * a + ((bc ushr 16) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val g = (((sc ushr 8) and 0xFF) * a + ((bc ushr 8) and 0xFF) * invA).toInt().coerceIn(0, 255)
            val b = ((sc and 0xFF) * a + (bc and 0xFF) * invA).toInt().coerceIn(0, 255)
            outPx[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, size, 0, 0, size, size)
        return out
    }

    /**
     * Tiled 64x64 unsharp luminance micro-contrast enhancement pass that sharpens fine hair strands,
     * eyes, and skin texture without increasing memory pressure.
     */
    fun applyTiledDetailEnhancement(bitmap: Bitmap, amount: Float = 0.28f): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = pixels.copyOf()
        val tileSize = 64

        var ty = 1
        while (ty < h - 1) {
            val yEnd = min(ty + tileSize, h - 1)
            var tx = 1
            while (tx < w - 1) {
                val xEnd = min(tx + tileSize, w - 1)
                for (y in ty until yEnd) {
                    val row = y * w
                    for (x in tx until xEnd) {
                        val idx = row + x
                        val c = pixels[idx]
                        val n = pixels[(y - 1) * w + x]
                        val s = pixels[(y + 1) * w + x]
                        val l = pixels[row + x - 1]
                        val r = pixels[row + x + 1]

                        val cr = (c ushr 16) and 0xFF
                        val cg = (c ushr 8) and 0xFF
                        val cb = c and 0xFF

                        val avgR = (((n ushr 16 and 0xFF) + (s ushr 16 and 0xFF) +
                            (l ushr 16 and 0xFF) + (r ushr 16 and 0xFF)) * 0.25f)
                        val avgG = (((n ushr 8 and 0xFF) + (s ushr 8 and 0xFF) +
                            (l ushr 8 and 0xFF) + (r ushr 8 and 0xFF)) * 0.25f)
                        val avgB = (((n and 0xFF) + (s and 0xFF) +
                            (l and 0xFF) + (r and 0xFF)) * 0.25f)

                        val nr = (cr + (cr - avgR) * amount).toInt().coerceIn(0, 255)
                        val ng = (cg + (cg - avgG) * amount).toInt().coerceIn(0, 255)
                        val nb = (cb + (cb - avgB) * amount).toInt().coerceIn(0, 255)
                        out[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
                    }
                }
                tx += tileSize
            }
            ty += tileSize
        }

        val enhanced = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        enhanced.setPixels(out, 0, w, 0, 0, w, h)
        return enhanced
    }

    internal fun warpHeadCropBackToTarget(
        targetPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        headCrop: Bitmap,
        forwardHeadMatrix2x3: FloatArray,
        envelopeAlpha: FloatArray,
        blendStrength: Float = 1.0f,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f
    ) {
        val cropSize = headCrop.width
        val cropPixels = IntArray(cropSize * cropSize)
        headCrop.getPixels(cropPixels, 0, cropSize, 0, 0, cropSize, cropSize)

        val inv = FaceAlignment.invertAffine2x3(forwardHeadMatrix2x3)
        val corners = arrayOf(
            0f to 0f,
            cropSize.toFloat() to 0f,
            0f to cropSize.toFloat(),
            cropSize.toFloat() to cropSize.toFloat()
        )
        var minX = targetWidth - 1
        var maxX = 0
        var minY = targetHeight - 1
        var maxY = 0
        for ((cx, cy) in corners) {
            val tx = (inv[0] * cx + inv[1] * cy + inv[2]).toInt()
            val ty = (inv[3] * cx + inv[4] * cy + inv[5]).toInt()
            minX = min(minX, tx)
            maxX = max(maxX, tx)
            minY = min(minY, ty)
            maxY = max(maxY, ty)
        }
        minX = (minX - 16).coerceIn(0, targetWidth - 1)
        maxX = (maxX + 16).coerceIn(0, targetWidth - 1)
        minY = (minY - 16).coerceIn(0, targetHeight - 1)
        maxY = (maxY + 16).coerceIn(0, targetHeight - 1)

        val m00 = forwardHeadMatrix2x3[0]
        val m01 = forwardHeadMatrix2x3[1]
        val m02 = forwardHeadMatrix2x3[2]
        val m10 = forwardHeadMatrix2x3[3]
        val m11 = forwardHeadMatrix2x3[4]
        val m12 = forwardHeadMatrix2x3[5]

        val center = cropSize * 0.5f
        val clampedBlend = blendStrength.coerceIn(0.20f, 1.0f)
        val safeScale = scaleAdjust.coerceIn(0.85f, 1.15f)
        val shiftU = offsetX.coerceIn(-36f, 36f) * (cropSize / 320f)
        val shiftV = offsetY.coerceIn(-36f, 36f) * (cropSize / 320f)

        for (y in minY..maxY) {
            val rowOffset = y * targetWidth
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            for (x in minX..maxX) {
                val rawU = m00 * x + baseU
                val rawV = m10 * x + baseV
                val u = (rawU - center - shiftU) / safeScale + center
                val v = (rawV - center - shiftV) / safeScale + center
                if (u >= 1f && u < cropSize - 2f && v >= 1f && v < cropSize - 2f) {
                    val x0 = u.toInt().coerceIn(0, cropSize - 2)
                    val y0 = v.toInt().coerceIn(0, cropSize - 2)
                    val fx = u - x0
                    val fy = v - y0
                    val alpha = (
                        envelopeAlpha[y0 * cropSize + x0] * (1f - fx) * (1f - fy) +
                            envelopeAlpha[y0 * cropSize + x0 + 1] * fx * (1f - fy) +
                            envelopeAlpha[(y0 + 1) * cropSize + x0] * (1f - fx) * fy +
                            envelopeAlpha[(y0 + 1) * cropSize + x0 + 1] * fx * fy
                        ) * clampedBlend

                    if (alpha > 0.003f) {
                        val headColor = FaceAlignment.sampleBicubicClamped(cropPixels, cropSize, cropSize, u, v)
                        val dstColor = targetPixels[rowOffset + x]

                        val sr = (headColor ushr 16) and 0xFF
                        val sg = (headColor ushr 8) and 0xFF
                        val sb = headColor and 0xFF

                        val dr = (dstColor ushr 16) and 0xFF
                        val dg = (dstColor ushr 8) and 0xFF
                        val db = dstColor and 0xFF

                        val invA = 1.0f - alpha
                        val outR = (sr * alpha + dr * invA).toInt().coerceIn(0, 255)
                        val outG = (sg * alpha + dg * invA).toInt().coerceIn(0, 255)
                        val outB = (sb * alpha + db * invA).toInt().coerceIn(0, 255)

                        targetPixels[rowOffset + x] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
                    }
                }
            }
        }
    }

    /**
     * Computes an inverse-warped in-bounds visibility mask in `[0, 1]` for `cropSize x cropSize`,
     * fading clamped border extrapolations (such as top hair square horns on tight crops) to `0.0f`.
     */
    internal fun computeCopyPasteInBoundsMask(
        forwardHeadMatrix2x3: FloatArray,
        srcWidth: Int,
        srcHeight: Int,
        cropSize: Int,
        featherPx: Float = 14f
    ): FloatArray {
        val inv = FaceAlignment.invertAffine2x3(forwardHeadMatrix2x3)
        val i00 = inv[0]
        val i01 = inv[1]
        val i02 = inv[2]
        val i10 = inv[3]
        val i11 = inv[4]
        val i12 = inv[5]
        val maxX = (srcWidth - 1).toFloat()
        val maxY = (srcHeight - 1).toFloat()
        val safeFeather = featherPx.coerceAtLeast(1f)
        val out = FloatArray(cropSize * cropSize)
        for (dy in 0 until cropSize) {
            val row = dy * cropSize
            val baseX = i01 * dy + i02
            val baseY = i11 * dy + i12
            for (dx in 0 until cropSize) {
                val sx = i00 * dx + baseX
                val sy = i10 * dx + baseY
                val dist = min(min(sx, maxX - sx), min(sy, maxY - sy))
                out[row + dx] = when {
                    dist <= 0f -> 0f
                    dist >= safeFeather -> 1f
                    else -> {
                        val t = dist / safeFeather
                        (0.5f * (1.0 - cos(Math.PI * t))).toFloat()
                    }
                }
            }
        }
        return out
    }

    /**
     * LEGACY_COPY_PASTE-only mask cleanup:
     * 1. Enforces out-of-bounds (`inBoundsMask`) fade so clamped square top/side corners vanish.
     * 2. Guarantees a solid `1.0f` interior facial core so bangs/shadows never ghost over eyes/cheeks.
     * 3. Softly feathers outer crown & temple hair into Target hair (no helmet/headband seams).
     * 4. Protects the full smiling chin down to `ny = 0.855` while deleting lower shoulder wings and
     *    fading lower neck smoothly across `ny = 0.855..0.945` + suppressing dangling earrings smoothly.
     */
    internal fun sanitizeCopyPasteHeadAlpha(
        rawAlpha: FloatArray,
        cropBitmap: Bitmap,
        cropSize: Int,
        centerNx: Float,
        inBoundsMask: FloatArray? = null,
        isSourceMask: Boolean
    ): FloatArray {
        val pixels = IntArray(cropSize * cropSize)
        cropBitmap.getPixels(pixels, 0, cropSize, 0, 0, cropSize, cropSize)
        return sanitizeCopyPasteHeadAlphaPixels(
            rawAlpha = rawAlpha,
            pixels = pixels,
            cropSize = cropSize,
            centerNx = centerNx,
            inBoundsMask = inBoundsMask,
            isSourceMask = isSourceMask
        )
    }

    internal fun sanitizeCopyPasteHeadAlphaPixels(
        rawAlpha: FloatArray,
        pixels: IntArray,
        cropSize: Int,
        centerNx: Float,
        inBoundsMask: FloatArray? = null,
        isSourceMask: Boolean
    ): FloatArray {
        val total = cropSize * cropSize
        val out = FloatArray(total)

        // Sample interior hair brightness near crown to detect metallic/bright dangling earrings
        var hairLumSum = 0f
        var hairCount = 0
        for (y in (cropSize * 0.18f).toInt()..(cropSize * 0.36f).toInt() step 2) {
            for (x in (cropSize * 0.34f).toInt()..(cropSize * 0.66f).toInt() step 2) {
                val idx = y * cropSize + x
                if (rawAlpha[idx] > 0.70f && (inBoundsMask == null || inBoundsMask[idx] > 0.75f)) {
                    val c = pixels[idx]
                    val lum = 0.299f * ((c ushr 16) and 0xFF) + 0.587f * ((c ushr 8) and 0xFF) + 0.114f * (c and 0xFF)
                    if (lum < 110f) {
                        hairLumSum += lum
                        hairCount++
                    }
                }
            }
        }
        val darkHairRef = if (hairCount >= 16) hairLumSum / hairCount else 55f

        for (y in 0 until cropSize) {
            val ny = y.toFloat() / cropSize.toFloat()
            val row = y * cropSize

            // Vertical lower-neck feathering starting well below smiling chin (~0.855) and finishing cleanly at 0.945
            val verticalNeckFactor = when {
                ny <= 0.855f -> 1.0f
                ny >= 0.945f -> 0.0f
                else -> {
                    val t = (ny - 0.855f) / 0.09f
                    (0.5f * (1.0 + cos(Math.PI * t))).toFloat()
                }
            }

            // Anatomical lateral half-width containment: full cranial dome above 0.64, smooth jaw curve
            // preserving smiling cheeks/chin through 0.855, then strict neck width below to delete shoulder wings.
            val allowedHalfW = when {
                ny <= 0.64f -> 0.43f
                ny <= 0.77f -> {
                    val t = (ny - 0.64f) / 0.13f
                    val s = (0.5f * (1.0 - cos(Math.PI * t))).toFloat()
                    0.43f * (1f - s) + 0.215f * s
                }
                ny <= 0.855f -> {
                    val t = (ny - 0.77f) / 0.085f
                    val s = (0.5f * (1.0 - cos(Math.PI * t))).toFloat()
                    0.215f * (1f - s) + 0.165f * s
                }
                else -> {
                    val t = ((ny - 0.855f) / 0.09f).coerceIn(0f, 1f)
                    0.165f * (1f - t) + 0.130f * t
                }
            }
            val featherBand = if (ny > 0.72f) 0.048f else 0.055f

            for (x in 0 until cropSize) {
                val idx = row + x
                val boundsF = inBoundsMask?.get(idx)?.coerceIn(0f, 1f) ?: 1f
                if (boundsF <= 0.002f || verticalNeckFactor <= 0.002f) {
                    out[idx] = 0f
                    continue
                }

                val nx = x.toFloat() / cropSize.toFloat()
                val dx = abs(nx - centerNx)

                val lateralFactor = when {
                    dx >= allowedHalfW -> 0.0f
                    dx <= (allowedHalfW - featherBand) -> 1.0f
                    else -> {
                        val t = (dx - (allowedHalfW - featherBand)) / featherBand
                        (0.5f * (1.0 + cos(Math.PI * t))).toFloat()
                    }
                }
                if (lateralFactor <= 0.002f) {
                    out[idx] = 0f
                    continue
                }

                // Interior facial core floor (forehead, eyes, nose, cheeks, mouth, chin) so MODNet hair/shadow
                // dips never make the swapped face translucent
                val coreDist = hypot(((nx - centerNx) / 0.162f).toDouble(), ((ny - 0.555f) / 0.260f).toDouble()).toFloat()
                val faceCoreFloor = when {
                    coreDist <= 0.85f -> 1.0f
                    coreDist >= 1.05f -> 0.0f
                    else -> {
                        val t = (coreDist - 0.85f) / 0.20f
                        (0.5f * (1.0 + cos(Math.PI * t))).toFloat()
                    }
                }

                var a = max(rawAlpha[idx].coerceIn(0f, 1f), faceCoreFloor)
                if (a <= 0.002f) {
                    out[idx] = 0f
                    continue
                }

                // Smooth outer hair/crown transition on Source so cropped top/temple hair blends organically into Target hair
                var hairCrownFeather = 1.0f
                if (isSourceMask && ny < 0.76f && coreDist > 0.92f) {
                    val crownR = hypot(((nx - centerNx) / 0.305f).toDouble(), ((ny - 0.455f) / 0.335f).toDouble()).toFloat()
                    if (crownR > 0.78f) {
                        val t = ((crownR - 0.78f) / 0.28f).coerceIn(0f, 1f)
                        val domeFade = (0.5f * (1.0 + cos(Math.PI * t))).toFloat()
                        val outsideFaceT = ((coreDist - 0.92f) / 0.22f).coerceIn(0f, 1f)
                        hairCrownFeather = 1.0f * (1f - outsideFaceT) + domeFade * outsideFaceT
                    }
                }

                // Softly erode semi-transparent outer ear/hair fringe to prevent halo & earring ghosting
                if (dx > 0.185f && a < 0.86f && x > 0 && x < cropSize - 1 && y > 0 && y < cropSize - 1) {
                    var minNb = a
                    for (dy in -1..1) {
                        val nRow = (y + dy) * cropSize
                        for (ox in -1..1) {
                            val na = rawAlpha[nRow + x + ox]
                            if (na < minNb) minNb = na
                        }
                    }
                    a = a * 0.45f + minNb * 0.55f
                }

                // Smoothly suppress dangling earrings / earlobe jewelry outliers in the lateral ear-drop zone (no step edge!)
                var earJewelryTrim = 1.0f
                if (ny in 0.48f..0.80f && dx > 0.185f) {
                    val nyRamp = ((ny - 0.48f) / 0.10f).coerceIn(0f, 1f) * ((0.80f - ny) / 0.08f).coerceIn(0f, 1f)
                    val c = pixels[idx]
                    val r = ((c ushr 16) and 0xFF).toFloat()
                    val g = ((c ushr 8) and 0xFF).toFloat()
                    val b = (c and 0xFF).toFloat()
                    val lum = 0.299f * r + 0.587f * g + 0.114f * b
                    val chromaSpan = max(r, max(g, b)) - min(r, min(g, b))
                    val isMetallicOrBrightJewelry = (lum > darkHairRef + 46f && chromaSpan < 30f) ||
                        (lum > 180f && dx > 0.205f) ||
                        (r > 130f && g > 95f && r > b * 1.42f && dx > 0.20f)

                    if (isSourceMask && ny > 0.50f && dx > 0.195f) {
                        val dangleFade = ((dx - 0.195f) / 0.055f).coerceIn(0f, 1f) * nyRamp
                        earJewelryTrim = (1.0f - dangleFade * 0.92f).coerceIn(0.08f, 1f)
                    }
                    if (isMetallicOrBrightJewelry) {
                        val jewelWeight = if (isSourceMask) 0.88f else 0.70f
                        earJewelryTrim *= (1.0f - jewelWeight * nyRamp).coerceIn(0.12f, 1f)
                    }
                }

                out[idx] = (a * boundsF * lateralFactor * verticalNeckFactor * hairCrownFeather * earJewelryTrim)
                    .coerceIn(0f, 1f)
            }
        }
        return out
    }

    internal fun buildCleanCopyPasteEnvelopeAlpha(
        sourceHeadAlpha: FloatArray,
        targetHeadAlpha: FloatArray,
        cropSize: Int,
        centerNx: Float,
        hasNeuralInpaint: Boolean = false
    ): FloatArray {
        val total = cropSize * cropSize
        if (!hasNeuralInpaint) {
            return FloatArray(total) { i -> sourceHeadAlpha[i].coerceIn(0f, 1f) }
        }
        return FloatArray(total) { i ->
            val y = i / cropSize
            val x = i % cropSize
            val ny = y.toFloat() / cropSize.toFloat()
            val nx = x.toFloat() / cropSize.toFloat()
            val dx = abs(nx - centerNx)
            val sA = sourceHeadAlpha[i].coerceIn(0f, 1f)
            var tA = targetHeadAlpha[i].coerceIn(0f, 1f) * 0.86f
            if (ny > 0.50f && dx > 0.185f && sA < 0.25f) {
                val suppress = (((dx - 0.185f) / 0.055f).coerceIn(0f, 1f) *
                    ((ny - 0.50f) / 0.12f).coerceIn(0f, 1f)).coerceIn(0f, 1f)
                tA *= (1.0f - suppress * 0.95f)
            }
            max(sA, tA).coerceIn(0f, 1f)
        }
    }

    internal fun upscaleAlphaBilinear(srcAlpha: FloatArray, srcSize: Int, dstSize: Int): FloatArray {
        if (srcSize == dstSize) return srcAlpha.copyOf()
        val out = FloatArray(dstSize * dstSize)
        val scale = (srcSize - 1).toFloat() / (dstSize - 1).coerceAtLeast(1).toFloat()
        for (y in 0 until dstSize) {
            val sy = (y * scale).coerceIn(0f, (srcSize - 1).toFloat())
            val y0 = sy.toInt().coerceIn(0, (srcSize - 2).coerceAtLeast(0))
            val y1 = min(y0 + 1, srcSize - 1)
            val fy = sy - y0
            val row = y * dstSize
            val r0 = y0 * srcSize
            val r1 = y1 * srcSize
            for (x in 0 until dstSize) {
                val sx = (x * scale).coerceIn(0f, (srcSize - 1).toFloat())
                val x0 = sx.toInt().coerceIn(0, (srcSize - 2).coerceAtLeast(0))
                val x1 = min(x0 + 1, srcSize - 1)
                val fx = sx - x0
                val a00 = srcAlpha[r0 + x0]
                val a10 = srcAlpha[r0 + x1]
                val a01 = srcAlpha[r1 + x0]
                val a11 = srcAlpha[r1 + x1]
                out[row + x] = ((a00 * (1f - fx) + a10 * fx) * (1f - fy) +
                    (a01 * (1f - fx) + a11 * fx) * fy).coerceIn(0f, 1f)
            }
        }
        return out
    }

    /**
     * Final post-swap retouch pass (runs AFTER GFPGAN & blending onto `compositePixels`):
     * Samples the Target's actual exposed neck skin colour (below the faded seam and above clothing/jewelry),
     * measures the output face's CIE LAB skin colour, and retouches ONLY facial & upper-neck skin pixels
     * to match the Target neck's exact colour & warmth while protecting eyes, eyebrows, lips/teeth, and hair.
     */
    internal fun applyFinalNeckToFaceColorRetouch(
        compositePixels: IntArray,
        targetBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        forwardHeadMatrix2x3: FloatArray,
        headCropSize: Int,
        envelopeAlpha: FloatArray,
        skinAndNeckWeight: FloatArray,
        centerNx: Float,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f
    ) {
        val inv = FaceAlignment.invertAffine2x3(forwardHeadMatrix2x3)
        val center = headCropSize * 0.5f
        val safeScale = scaleAdjust.coerceIn(0.85f, 1.15f)
        val shiftU = offsetX.coerceIn(-36f, 36f) * (headCropSize / 320f)
        val shiftV = offsetY.coerceIn(-36f, 36f) * (headCropSize / 320f)

        // Compute target bounding box covering head + lower neck sample band (ny up to 1.12)
        val corners = arrayOf(
            0f to 0f,
            headCropSize.toFloat() to 0f,
            0f to (headCropSize * 1.15f),
            headCropSize.toFloat() to (headCropSize * 1.15f)
        )
        var minX = targetWidth - 1
        var maxX = 0
        var minY = targetHeight - 1
        var maxY = 0
        for ((cx, cy) in corners) {
            val ux = (cx - center) * safeScale + center + shiftU
            val vy = (cy - center) * safeScale + center + shiftV
            val tx = (inv[0] * ux + inv[1] * vy + inv[2]).toInt()
            val ty = (inv[3] * ux + inv[4] * vy + inv[5]).toInt()
            minX = min(minX, tx)
            maxX = max(maxX, tx)
            minY = min(minY, ty)
            maxY = max(maxY, ty)
        }
        minX = (minX - 8).coerceIn(0, targetWidth - 1)
        maxX = (maxX + 8).coerceIn(0, targetWidth - 1)
        minY = (minY - 8).coerceIn(0, targetHeight - 1)
        maxY = (maxY + 8).coerceIn(0, targetHeight - 1)

        val roiW = maxX - minX + 1
        val roiH = maxY - minY + 1
        if (roiW <= 4 || roiH <= 4) return
        val origTargetRoiPx = IntArray(roiW * roiH)
        targetBitmap.getPixels(origTargetRoiPx, 0, roiW, minX, minY, roiW, roiH)

        applyFinalNeckToFaceColorRetouchRoi(
            compositePixels = compositePixels,
            origTargetRoiPx = origTargetRoiPx,
            minX = minX,
            maxX = maxX,
            minY = minY,
            maxY = maxY,
            targetWidth = targetWidth,
            forwardHeadMatrix2x3 = forwardHeadMatrix2x3,
            headCropSize = headCropSize,
            envelopeAlpha = envelopeAlpha,
            skinAndNeckWeight = skinAndNeckWeight,
            centerNx = centerNx,
            skinToneMode = skinToneMode,
            offsetX = offsetX,
            offsetY = offsetY,
            scaleAdjust = scaleAdjust
        )
    }

    internal fun applyFinalNeckToFaceColorRetouchRoi(
        compositePixels: IntArray,
        origTargetRoiPx: IntArray,
        minX: Int,
        maxX: Int,
        minY: Int,
        maxY: Int,
        targetWidth: Int,
        forwardHeadMatrix2x3: FloatArray,
        headCropSize: Int,
        envelopeAlpha: FloatArray,
        skinAndNeckWeight: FloatArray,
        centerNx: Float,
        skinToneMode: SkinToneSourceMode = SkinToneSourceMode.TARGET_SCENE,
        offsetX: Float = 0f,
        offsetY: Float = 0f,
        scaleAdjust: Float = 1.0f
    ) {
        val m00 = forwardHeadMatrix2x3[0]
        val m01 = forwardHeadMatrix2x3[1]
        val m02 = forwardHeadMatrix2x3[2]
        val m10 = forwardHeadMatrix2x3[3]
        val m11 = forwardHeadMatrix2x3[4]
        val m12 = forwardHeadMatrix2x3[5]
        val center = headCropSize * 0.5f
        val safeScale = scaleAdjust.coerceIn(0.85f, 1.15f)
        val shiftU = offsetX.coerceIn(-36f, 36f) * (headCropSize / 320f)
        val shiftV = offsetY.coerceIn(-36f, 36f) * (headCropSize / 320f)
        val roiW = maxX - minX + 1

        val neckLabSamples = ArrayList<FloatArray>(512)
        val tgtFallbackSkinSamples = ArrayList<FloatArray>(512)
        val faceLabSamples = ArrayList<FloatArray>(512)
        val labScratch = FloatArray(3)

        val eyeY = 114f / 256f
        val leftEyeNx = centerNx - (30f / 256f)
        val rightEyeNx = centerNx + (30f / 256f)
        val mouthY = 178f / 256f
        val step = if (roiW > 420) 2 else 1

        for (y in minY..maxY step step) {
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            val roiRow = (y - minY) * roiW
            val outRow = y * targetWidth
            for (x in minX..maxX step step) {
                val u = ((m00 * x + baseU) - center - shiftU) / safeScale + center
                val v = ((m10 * x + baseV) - center - shiftV) / safeScale + center
                val nx = u / headCropSize.toFloat()
                val ny = v / headCropSize.toFloat()
                val dx = abs(nx - centerNx)

                // 1. Sample Target's actual exposed neck skin (ny 0.88..1.14, |nx-cx| <= 0.125)
                if (ny in 0.88f..1.14f && dx <= 0.125f) {
                    val tc = origTargetRoiPx[roiRow + (x - minX)]
                    if (isCleanSkinPixelForLab(tc)) {
                        rgbToLab((tc ushr 16) and 0xFF, (tc ushr 8) and 0xFF, tc and 0xFF, labScratch)
                        neckLabSamples.add(labScratch.copyOf())
                    }
                } else if (ny in 0.56f..0.78f && dx <= 0.135f) {
                    val tc = origTargetRoiPx[roiRow + (x - minX)]
                    if (isCleanSkinPixelForLab(tc)) {
                        rgbToLab((tc ushr 16) and 0xFF, (tc ushr 8) and 0xFF, tc and 0xFF, labScratch)
                        tgtFallbackSkinSamples.add(labScratch.copyOf())
                    }
                }

                // 2. Sample output face skin (cheeks, forehead, chin, excluding eyes & mouth)
                if (ny in 0.34f..0.80f && dx <= 0.15f) {
                    val lEyeD = hypot(((nx - leftEyeNx) / 0.085f).toDouble(), ((ny - eyeY) / 0.055f).toDouble())
                    val rEyeD = hypot(((nx - rightEyeNx) / 0.085f).toDouble(), ((ny - eyeY) / 0.055f).toDouble())
                    val mouthD = hypot(((nx - centerNx) / 0.105f).toDouble(), ((ny - mouthY) / 0.055f).toDouble())
                    if (lEyeD > 1.0 && rEyeD > 1.0 && mouthD > 1.0) {
                        val fc = compositePixels[outRow + x]
                        if (isCleanSkinPixelForLab(fc)) {
                            rgbToLab((fc ushr 16) and 0xFF, (fc ushr 8) and 0xFF, fc and 0xFF, labScratch)
                            faceLabSamples.add(labScratch.copyOf())
                        }
                    }
                }
            }
        }

        val activeNeckSamples = if (neckLabSamples.size >= 24) neckLabSamples else tgtFallbackSkinSamples
        if (activeNeckSamples.size < 12 || faceLabSamples.size < 16) return

        val neckStats = computeTrimmedLabStats(activeNeckSamples)
        val faceStats = computeTrimmedLabStats(faceLabSamples)

        val modeStrength = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> 0.94f
            SkinToneSourceMode.BALANCED_BLEND -> 0.62f
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.35f
        }
        val dL = (neckStats[0] - faceStats[0]).coerceIn(-30f, 30f)
        val da = (neckStats[1] - faceStats[1]).coerceIn(-14f, 14f)
        val db = (neckStats[2] - faceStats[2]).coerceIn(-18f, 18f)
        val lScale = (neckStats[3].coerceAtLeast(3.5f) / faceStats[3].coerceAtLeast(3.5f)).coerceIn(0.84f, 1.18f)
        val aScale = (neckStats[4].coerceAtLeast(2.0f) / faceStats[4].coerceAtLeast(2.0f)).coerceIn(0.85f, 1.15f)
        val bScale = (neckStats[5].coerceAtLeast(2.5f) / faceStats[5].coerceAtLeast(2.5f)).coerceIn(0.85f, 1.18f)

        val rgbOut = IntArray(3)
        for (y in minY..maxY) {
            val baseU = m01 * y + m02
            val baseV = m11 * y + m12
            val outRow = y * targetWidth
            for (x in minX..maxX) {
                val u = ((m00 * x + baseU) - center - shiftU) / safeScale + center
                val v = ((m10 * x + baseV) - center - shiftV) / safeScale + center
                if (u < 1f || u >= headCropSize - 2f || v < 1f || v >= headCropSize - 2f) continue

                val ny = v / headCropSize.toFloat()
                if (ny < 0.16f || ny > 0.935f) continue

                val x0 = u.toInt().coerceIn(0, headCropSize - 2)
                val y0 = v.toInt().coerceIn(0, headCropSize - 2)
                val fx = u - x0
                val fy = v - y0
                val envA = (
                    envelopeAlpha[y0 * headCropSize + x0] * (1f - fx) * (1f - fy) +
                        envelopeAlpha[y0 * headCropSize + x0 + 1] * fx * (1f - fy) +
                        envelopeAlpha[(y0 + 1) * headCropSize + x0] * (1f - fx) * fy +
                        envelopeAlpha[(y0 + 1) * headCropSize + x0 + 1] * fx * fy
                    ).coerceIn(0f, 1f)
                if (envA <= 0.05f) continue

                val c = compositePixels[outRow + x]
                val r = (c ushr 16) and 0xFF
                val g = (c ushr 8) and 0xFF
                val b = c and 0xFF
                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                // Never tint black/dark hair or clipped highlights
                if (lum < 34f || lum > 250f) continue

                val maskSkinW = (
                    skinAndNeckWeight[y0 * headCropSize + x0] * (1f - fx) * (1f - fy) +
                        skinAndNeckWeight[y0 * headCropSize + x0 + 1] * fx * (1f - fy) +
                        skinAndNeckWeight[(y0 + 1) * headCropSize + x0] * (1f - fx) * fy +
                        skinAndNeckWeight[(y0 + 1) * headCropSize + x0 + 1] * fx * fy
                    ).coerceIn(0f, 1f)

                val nx = u / headCropSize.toFloat()
                val fDx = (nx - centerNx) / 0.215f
                val fDy = (ny - 0.57f) / 0.31f
                val faceOval = (1.0f - (fDx * fDx + fDy * fDy)).coerceIn(0f, 1f)
                val chromaSkinGate = computeSkinChromaLikelihood(r.toFloat(), g.toFloat(), b.toFloat())
                val skinW = max(maskSkinW, faceOval) * chromaSkinGate
                if (skinW <= 0.04f) continue

                // Protect Eyes + Brows + Mouth/Lips + Teeth
                val lEyeDx = (nx - leftEyeNx) / 0.082f
                val rEyeDx = (nx - rightEyeNx) / 0.082f
                val eyeDy = (ny - (eyeY - 0.012f)) / 0.058f
                val eyeProt = (1.0f - min(lEyeDx * lEyeDx + eyeDy * eyeDy, rEyeDx * rEyeDx + eyeDy * eyeDy)).coerceIn(0f, 1f)

                val mDx = (nx - centerNx) / 0.098f
                val mDy = (ny - mouthY) / 0.050f
                val mouthProt = (1.0f - (mDx * mDx + mDy * mDy)).coerceIn(0f, 1f)

                val retouchW = (envA * skinW * modeStrength * (1.0f - 0.94f * eyeProt) * (1.0f - 0.88f * mouthProt))
                    .coerceIn(0f, 0.94f)
                if (retouchW <= 0.02f) continue

                rgbToLab(r, g, b, labScratch)
                val targetL = (faceStats[0] + (labScratch[0] - faceStats[0]) * (0.55f + 0.45f * lScale) + dL * 0.96f)
                    .coerceIn(2f, 98f)
                val targetA = (faceStats[1] + (labScratch[1] - faceStats[1]) * (0.60f + 0.40f * aScale) + da * 0.95f)
                    .coerceIn(-80f, 80f)
                val targetB = (faceStats[2] + (labScratch[2] - faceStats[2]) * (0.60f + 0.40f * bScale) + db * 0.95f)
                    .coerceIn(-80f, 80f)

                val blendL = labScratch[0] * (1f - retouchW) + targetL * retouchW
                val blendA = labScratch[1] * (1f - retouchW) + targetA * retouchW
                val blendB = labScratch[2] * (1f - retouchW) + targetB * retouchW
                labToRgb(blendL, blendA, blendB, rgbOut)
                compositePixels[outRow + x] = (0xFF shl 24) or (rgbOut[0] shl 16) or (rgbOut[1] shl 8) or rgbOut[2]
            }
        }
    }

    private fun isCleanSkinPixelForLab(argb: Int): Boolean {
        val r = (argb ushr 16) and 0xFF
        val g = (argb ushr 8) and 0xFF
        val b = argb and 0xFF
        val lum = 0.299f * r + 0.587f * g + 0.114f * b
        if (lum !in 38f..242f) return false
        // Exclude green/blue clothing, bright gold jewelry, and near-achromatic dark hair
        if (r < g - 6 || r < b + 6) return false
        if (r > 195 && g > 160 && (r - b) > 95) return false
        return (r - b) in 7..118 && (g - b) >= -8
    }

    private fun computeSkinChromaLikelihood(r: Float, g: Float, b: Float): Float {
        val lum = 0.299f * r + 0.587f * g + 0.114f * b
        if (lum < 34f || lum > 250f) return 0f
        if (r < b - 2f || r < g - 10f) return 0f
        val warmth = (r - b).coerceAtLeast(0f)
        val lumOk = ((lum - 34f) / 28f).coerceIn(0f, 1f)
        val warmOk = (warmth / 16f).coerceIn(0.15f, 1f)
        val notTooRed = if (r - g > 68f) 0.25f else 1.0f
        return (lumOk * warmOk * notTooRed).coerceIn(0f, 1f)
    }

    private fun computeTrimmedLabStats(samples: List<FloatArray>): FloatArray {
        val sortedByL = samples.sortedBy { it[0] }
        val trim = (sortedByL.size * 0.15f).toInt().coerceAtMost((sortedByL.size - 4).coerceAtLeast(0) / 2)
        val sub = sortedByL.subList(trim, (sortedByL.size - trim).coerceAtLeast(trim + 1))
        val n = sub.size.coerceAtLeast(1)
        var sL = 0.0
        var sA = 0.0
        var sB = 0.0
        for (p in sub) {
            sL += p[0]
            sA += p[1]
            sB += p[2]
        }
        val mL = (sL / n).toFloat()
        val mA = (sA / n).toFloat()
        val mB = (sB / n).toFloat()
        var vL = 0.0
        var vA = 0.0
        var vB = 0.0
        for (p in sub) {
            vL += (p[0] - mL) * (p[0] - mL)
            vA += (p[1] - mA) * (p[1] - mA)
            vB += (p[2] - mB) * (p[2] - mB)
        }
        return floatArrayOf(
            mL,
            mA,
            mB,
            sqrt(vL / n).toFloat(),
            sqrt(vA / n).toFloat(),
            sqrt(vB / n).toFloat()
        )
    }

    internal fun rgbToLab(r8: Int, g8: Int, b8: Int, outLab: FloatArray) {
        fun toLinear(v: Int): Double {
            val c = v / 255.0
            return if (c <= 0.04045) c / 12.92 else ((c + 0.055) / 1.055).pow(2.4)
        }
        val r = toLinear(r8)
        val g = toLinear(g8)
        val b = toLinear(b8)

        val x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
        val y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b)
        val z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

        fun f(t: Double): Double = if (t > 0.008856) cbrt(t) else (7.787 * t + 16.0 / 116.0)
        val fx = f(x)
        val fy = f(y)
        val fz = f(z)
        outLab[0] = (116.0 * fy - 16.0).toFloat()
        outLab[1] = (500.0 * (fx - fy)).toFloat()
        outLab[2] = (200.0 * (fy - fz)).toFloat()
    }

    internal fun labToRgb(l: Float, a: Float, b: Float, outRgb: IntArray) {
        val fy = (l + 16.0) / 116.0
        val fx = fy + a / 500.0
        val fz = fy - b / 200.0

        fun fInv(t: Double): Double = if (t > 0.206893) t * t * t else (t - 16.0 / 116.0) / 7.787
        val x = 0.95047 * fInv(fx)
        val y = fInv(fy)
        val z = 1.08883 * fInv(fz)

        val rLin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        val gLin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        val bLin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

        fun toSrgb(lin: Double): Int {
            val clamped = lin.coerceIn(0.0, 1.0)
            val srgb = if (clamped <= 0.0031308) {
                12.92 * clamped
            } else {
                1.055 * clamped.pow(1.0 / 2.4) - 0.055
            }
            return (srgb * 255.0 + 0.5).toInt().coerceIn(0, 255)
        }
        outRgb[0] = toSrgb(rLin)
        outRgb[1] = toSrgb(gLin)
        outRgb[2] = toSrgb(bLin)
    }
}
