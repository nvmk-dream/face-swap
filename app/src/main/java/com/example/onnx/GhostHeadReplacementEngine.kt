package com.example.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
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
        headPipelineMode: HeadReplacementPipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_DIFFUSION,
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
        val srcMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
            ortEnv = ortEnv,
            alignedHeadCrop = alignedSourceHead,
            segModelFile = if (segFile.exists()) segFile else null,
            mattingModelFile = if (matFile.exists()) matFile else null,
            preferHardwareAccel = preferHardwareAccel,
            preloadedSegformerSession = preloadedSegformerSession
        )

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

            val tgtMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
                ortEnv = ortEnv,
                alignedHeadCrop = alignedTargetHead,
                segModelFile = if (segFile.exists()) segFile else null,
                mattingModelFile = if (matFile.exists()) matFile else null,
                preferHardwareAccel = preferHardwareAccel,
                preloadedSegformerSession = preloadedSegformerSession
            )

            // 5a. Inpaint disoccluded background where target's old hair/ears extended beyond the new source head
            val inpaintedTargetCrop = HeadSegmentationAndInpainting.inpaintTargetBackgroundGap(
                ortEnv = ortEnv,
                targetAlignedCrop = alignedTargetHead,
                targetHeadAlpha = tgtMasks.fullHeadAlpha,
                sourceHeadAlpha = srcMasks.fullHeadAlpha,
                lamaModelFile = if (lamaFile.exists()) lamaFile else null,
                preferHardwareAccel = preferHardwareAccel
            )

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

            // Paste the full head + inpainted disocclusion envelope back onto the full-resolution target photo
            val baseEnvelopeAlpha = FloatArray(headCropSize * headCropSize) { i ->
                max(srcMasks.fullHeadAlpha[i], tgtMasks.fullHeadAlpha[i] * 0.90f).coerceIn(0f, 1f)
            }
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
                FloatArray(warpSize * warpSize) { idx512 ->
                    val y512 = idx512 / warpSize
                    val x512 = idx512 % warpSize
                    val sy = (y512 * headCropSize / warpSize).coerceIn(0, headCropSize - 1)
                    val sx = (x512 * headCropSize / warpSize).coerceIn(0, headCropSize - 1)
                    baseEnvelopeAlpha[sy * headCropSize + sx]
                }
            }

            warpHeadCropBackToTarget(
                targetPixels = compositePixels,
                targetWidth = targetW,
                targetHeight = targetH,
                headCrop = enhancedHeadCrop,
                forwardHeadMatrix2x3 = warpMatrix2x3,
                envelopeAlpha = warpEnvelopeAlpha,
                blendStrength = blendStrength,
                offsetX = offsetX * warpScale,
                offsetY = offsetY * warpScale,
                scaleAdjust = scaleAdjust
            )

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

        var rawSwapped128: Bitmap? = null
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
            val outPixels = IntArray(hw)
            OnnxTensor.createTensor(ortEnv, targetBuf, longArrayOf(1L, 3L, cropSize.toLong(), cropSize.toLong())).use { tTensor ->
                OnnxTensor.createTensor(ortEnv, srcBuf, longArrayOf(1L, 512L)).use { sTensor ->
                    session.run(mapOf(targetName to tTensor, sourceName to sTensor)).use { res ->
                        val outT = res[0] as OnnxTensor
                        val outFloats = FloatArray(3 * hw)
                        outT.floatBuffer.get(outFloats)
                        var minVal = Float.MAX_VALUE
                        var maxVal = -Float.MAX_VALUE
                        for (v in outFloats) {
                            if (v.isFinite()) {
                                if (v < minVal) minVal = v
                                if (v > maxVal) maxVal = v
                            }
                        }
                        val isSigned = if (isHyperSwap256) {
                            minVal < -0.01f || maxVal > 1.05f || minVal == Float.MAX_VALUE
                        } else {
                            minVal < -0.05f
                        }
                        if (isSigned) {
                            for (i in 0 until hw) {
                                val rRaw = if (outFloats[i].isFinite()) outFloats[i] else 0f
                                val gRaw = if (outFloats[hw + i].isFinite()) outFloats[hw + i] else 0f
                                val bRaw = if (outFloats[2 * hw + i].isFinite()) outFloats[2 * hw + i] else 0f
                                val r = ((rRaw * 0.5f + 0.5f) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                val g = ((gRaw * 0.5f + 0.5f) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                val b = ((bRaw * 0.5f + 0.5f) * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                            }
                        } else {
                            for (i in 0 until hw) {
                                val rRaw = if (outFloats[i].isFinite()) outFloats[i] else 0f
                                val gRaw = if (outFloats[hw + i].isFinite()) outFloats[hw + i] else 0f
                                val bRaw = if (outFloats[2 * hw + i].isFinite()) outFloats[2 * hw + i] else 0f
                                val r = (rRaw * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                val g = (gRaw * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                val b = (bRaw * 255.0f + 0.5f).toInt().coerceIn(0, 255)
                                outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                            }
                        }
                    }
                }
            }
            if (alignedCrop !== targetCrop128) alignedCrop.recycle()
            val outBmp = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
            outBmp.setPixels(outPixels, 0, cropSize, 0, 0, cropSize, cropSize)
            rawSwapped128 = if (cropSize == 128) {
                outBmp
            } else {
                val scaled128 = Bitmap.createScaledBitmap(outBmp, 128, 128, true)
                outBmp.recycle()
                scaled128
            }
        }
        if (preloadedSwapSession != null) {
            runWithSession(preloadedSwapSession)
        } else {
            val cached = OnnxProtobufInspector.getOrCreateCachedSession(ortEnv, swapFile, preferHardwareAccel)
            if (cached != null) runWithSession(cached)
        }

        val activeSwapBmp = rawSwapped128 ?: run {
            targetCrop128.recycle()
            return
        }
        val hasTrueArcFaceLatent = sourceEmbedding.usedArcFaceModel
        val eyeRestoredBmp = FaceBlender.restoreEyesAndEliminateNegativeArtifacts128(
            swapped128 = activeSwapBmp,
            alignedTarget128 = targetCrop128,
            alignedSource112 = sourceEmbedding.aligned112Crop,
            hasTrueArcFaceLatent = hasTrueArcFaceLatent,
            forwardMatrix128 = m128,
            targetLandmarks5 = targetFace.landmarks5
        )
        activeSwapBmp.recycle()
        targetCrop128.recycle()

        val swappedPixels128 = IntArray(128 * 128)
        eyeRestoredBmp.getPixels(swappedPixels128, 0, 128, 0, 0, 128, 128)
        eyeRestoredBmp.recycle()

        // Map canonical head keypoints (or override canvas landmarks) to 128x128 coordinates
        val headPts = if (overrideHeadLandmarks5 != null && overrideHeadLandmarks5.size >= 5) {
            overrideHeadLandmarks5.take(5)
        } else {
            HeadSegmentationAndInpainting.getCanonicalHeadTemplate(headCropSize, srcYawRatio).toList()
        }
        val headTo128 = FaceAlignment.estimateNorm(headPts, 128)
        val featheredMask128 = FaceBlender.createFeatheredFaceMask128()
        val headPixels = IntArray(headCropSize * headCropSize)
        synthesizedHeadCrop.getPixels(headPixels, 0, headCropSize, 0, 0, headCropSize, headCropSize)
        // Use high core weight (0.88f) when neural ArcFace is present so inner mouth/eyes NEVER double-ghost
        val expressionBlendScale = if (hasTrueArcFaceLatent) 0.88f else 0.25f

        for (y in 0 until headCropSize) {
            for (x in 0 until headCropSize) {
                val u = headTo128[0] * x + headTo128[1] * y + headTo128[2]
                val v = headTo128[3] * x + headTo128[4] * y + headTo128[5]
                if (u in 6f..121f && v in 6f..121f) {
                    val u0 = u.toInt().coerceIn(0, 126)
                    val v0 = v.toInt().coerceIn(0, 126)
                    val w = featheredMask128[v0 * 128 + u0] * expressionBlendScale
                    if (w > 0.005f) {
                        val swapC = FaceAlignment.sampleBicubicClamped(swappedPixels128, 128, 128, u, v)
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
                sRMean += (sc ushr 16) and 0xFF
                sGMean += (sc ushr 8) and 0xFF
                sBMean += sc and 0xFF
                tRMean += (tc ushr 16) and 0xFF
                tGMean += (tc ushr 8) and 0xFF
                tBMean += tc and 0xFF
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
                val sr = ((sc ushr 16) and 0xFF) - sRMean
                val sg = ((sc ushr 8) and 0xFF) - sGMean
                val sb = (sc and 0xFF) - sBMean
                val tr = ((tc ushr 16) and 0xFF) - tRMean
                val tg = ((tc ushr 8) and 0xFF) - tGMean
                val tb = (tc and 0xFF) - tBMean
                sRVar += sr * sr
                sGVar += sg * sg
                sBVar += sb * sb
                tRVar += tr * tr
                tGVar += tg * tg
                tBVar += tb * tb
            }
        }

        val scaleR = (sqrt(tRVar / count).coerceAtLeast(6.0) / sqrt(sRVar / count).coerceAtLeast(6.0)).coerceIn(0.80, 1.25)
        val scaleG = (sqrt(tGVar / count).coerceAtLeast(6.0) / sqrt(sGVar / count).coerceAtLeast(6.0)).coerceIn(0.80, 1.25)
        val scaleB = (sqrt(tBVar / count).coerceAtLeast(6.0) / sqrt(sBVar / count).coerceAtLeast(6.0)).coerceIn(0.80, 1.25)

        // Ambient luminance ratio for gentle hair exposure adaptation
        val srcLum = 0.299 * sRMean + 0.587 * sGMean + 0.114 * sBMean
        val tgtLum = 0.299 * tRMean + 0.587 * tGMean + 0.114 * tBMean
        val hairAmbientScale = (tgtLum / srcLum.coerceAtLeast(15.0)).coerceIn(0.85, 1.18).toFloat()

        val leftEyeX = if (hasOverride) overrideLandmarks5!![0].x else size * (98.0f / 256.0f)
        val rightEyeX = if (hasOverride) overrideLandmarks5!![1].x else size * (158.0f / 256.0f)
        val eyeY = if (hasOverride) ovEyeMidY else size * (114.0f / 256.0f)
        val eyeRx = ovEyeDist * 0.43f
        val eyeRy = ovEyeDist * 0.27f

        val skinStrength = when (skinToneMode) {
            SkinToneSourceMode.TARGET_SCENE -> 0.76f
            SkinToneSourceMode.SOURCE_IDENTITY -> 0.14f
            SkinToneSourceMode.BALANCED_BLEND -> 0.45f
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
                        val headColor = FaceAlignment.sampleBilinearClamped(cropPixels, cropSize, cropSize, u, v)
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
}
