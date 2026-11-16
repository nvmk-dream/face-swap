package com.example.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import java.io.File
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Clean-Erase 3D Head Replacement + Diffusion Seam Bridge Pipeline (`MODE 2 — FULL HEAD REPLACEMENT`).
 *
 * Unlike 2D copy-paste over the target head, this engine executes a strict 6-stage architecture:
 *  Stage 1 — 3D FaceMesh & Age Plan: Extracts 468-pt 3D landmarks (`facemesh-v2-v1.onnx` or PnP-106)
 *            for Source & Target + MiVOLO v2 body-age matching plan.
 *  Stage 2 — Depth & Target Parsing: Depth Anything V2 (`depth-anything-v2-small-fp16-v1.onnx`) +
 *            SegFormer/MODNet segmentation to separate target head/hair from background & clothing.
 *  Stage 3 — Complete Target Head Erasure: Removes 100% of Target's old face, hair & ears using
 *            `lama_fp32.onnx` or Depth-Aware Boundary Marching to build a clean headless canvas
 *            (protecting Target clothing/shoulders).
 *  Stage 4 — 3D Pose Source Head Reconstruction: Warps Source face, hair & neck via 31-pt 3D triangle
 *            mesh (`FaceMesh3DEngine`), fuses optional target expression (`inswapper_128`/`hyperswap`),
 *            adapts skin lighting, and composites onto the headless canvas.
 *  Stage 5 — Diffusion Seam & Neck Bridge: In `CLEAN_ERASE_3D_DIFFUSION`, runs `SdInpaintOnnxEngine`
 *            strictly on the outer hair/neck-to-collar seam while keeping Source inner-face identity 100%
 *            locked via `buildInnerFaceIdentityMask`. (Uses fast multi-scale seam bridge if SD pack is
 *            not installed or `CLEAN_ERASE_3D_FAST` is selected.)
 *  Stage 6 — 512x512 HD Enhancement & Composite: Runs `gfpgan-v1.4-w16-v1.onnx` / `gfpgan_1.4.onnx`
 *            + Age Harmony + Tiled Micro-Contrast + seamless composite onto target body & optional bokeh.
 */
object CleanEraseHeadReplacementEngine {

    fun execute(
        context: Context,
        ortEnv: OrtEnvironment?,
        sourceBitmap: Bitmap,
        sourceFace: DetectedFace,
        targetBitmap: Bitmap,
        targetFacesToReplace: List<DetectedFace>,
        pipelineMode: HeadReplacementPipelineMode = HeadReplacementPipelineMode.CLEAN_ERASE_3D_DIFFUSION,
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
        onProgress: (SwapStageProgress) -> Unit
    ): FaceSwapExecutionResult {
        val tStart = System.currentTimeMillis()
        require(targetFacesToReplace.isNotEmpty()) {
            "Please select at least one detected target head to replace."
        }

        val canvasSize = if (lowMemoryMode) 320 else 512
        val recFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.RECOGNIZER)
        val swapFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SWAPPER)
        val segFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SEGMENTATION)
        val matFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.MATTING)
        val lamaFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.INPAINTING)
        val gfpFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.ENHANCEMENT)
        val reagingFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.FACE_REAGING)
        val faceMeshFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.FACE_MESH_3D)

        // =========================================================================
        // STAGE 1/6: 3D FaceMesh V2 Extraction + Body Age Harmony Plan
        // =========================================================================
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
        // STAGE 2/6: Depth Anything V2 + Identity Embedding
        // =========================================================================
        val tEmb0 = System.currentTimeMillis()
        onProgress(
            SwapStageProgress(
                stepIndex = 2,
                totalSteps = 6,
                stageTitle = "Stage 2/6: Depth & Identity",
                detailMessage = "Computing Depth Anything V2 foreground/background map & ArcFace identity...",
                progressFraction = 0.28f
            )
        )

        val targetDepthMap = if (DepthEstimationEngine.hasInstalledDepthModel(context)) {
            runCatching {
                DepthEstimationEngine.estimateDepth(
                    context = context,
                    ortEnv = ortEnv,
                    sourceBitmap = targetBitmap,
                    preferHardwareAccel = preferHardwareAccel
                )
            }.getOrNull()
        } else {
            null
        }

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

        val sdPack = if (pipelineMode == HeadReplacementPipelineMode.CLEAN_ERASE_3D_DIFFUSION) {
            SdInpaintPackManager.resolvePackForOption(context, InpaintModelOption.AUTO)
        } else {
            null
        }
        val embMs = (System.currentTimeMillis() - tEmb0).coerceAtLeast(1L)

        val targetW = targetBitmap.width
        val targetH = targetBitmap.height
        val compositePixels = IntArray(targetW * targetH)
        targetBitmap.getPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)

        var firstTargetHeadPreview: Bitmap? = null
        var firstGeneratedHeadPreview: Bitmap? = null
        var totalGenMs = 0L
        var totalBlendMs = 0L
        var usedGfpganPass = false
        var usedDiffusionBridge = false
        var usedLamaErasure = false
        var segLabelSummary = "3D-Clean-Erase"

        targetFacesToReplace.forEachIndexed { idx, targetFace ->
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

            val alignedTargetHead = FaceAlignment.warpAffineCrop(
                srcBitmap = targetBitmap,
                forwardMatrix2x3 = tgtCanvasMatrix,
                dstSize = canvasSize
            )

            // =========================================================================
            // STAGE 3/6: Complete Target Head Removal (True Headless Background Canvas)
            // =========================================================================
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
                preloadedSegformerSession = preloadedSegformerSession
            )
            val tgtMasks = if (rawTgtMasks.usedSegmentationOnnx) {
                rawTgtMasks
            } else {
                boostMatteForWideHeadCanvas(
                    alignedHeadCrop = alignedTargetHead,
                    baseMasks = rawTgtMasks,
                    canvasPts5 = tgtCanvasPts5,
                    suppressBgHalo = false
                )
            }
            segLabelSummary = tgtMasks.segmentationSourceLabel

            val clothProtectMask = buildClothingCollarProtectMask(
                canvasSize = canvasSize,
                tgtCanvasPts5 = tgtCanvasPts5,
                tgtMasks = tgtMasks,
                tgtCanvasMatrix = tgtCanvasMatrix,
                targetDepthMap = targetDepthMap,
                targetWidth = targetW,
                targetHeight = targetH
            )

            // Dilate target head mask slightly (radius ~5 px) so zero target hair halo survives
            val dilatedTargetEraseMask = dilateAlpha(tgtMasks.fullHeadAlpha, canvasSize, radiusPx = 5)
            val cleanEraseMask = FloatArray(canvasSize * canvasSize) { i ->
                (dilatedTargetEraseMask[i] * (1f - clothProtectMask[i])).coerceIn(0f, 1f)
            }

            val (cleanHeadlessTargetCanvas, lamaSuccess) = eraseTargetHeadCleanly(
                ortEnv = ortEnv,
                targetCanvas = alignedTargetHead,
                eraseMask = cleanEraseMask,
                clothProtectMask = clothProtectMask,
                tgtCanvasMatrix = tgtCanvasMatrix,
                targetDepthMap = targetDepthMap,
                targetWidth = targetW,
                targetHeight = targetH,
                lamaModelFile = if (lamaFile.exists()) lamaFile else null,
                preferHardwareAccel = preferHardwareAccel
            )
            if (lamaSuccess) usedLamaErasure = true

            // =========================================================================
            // STAGE 4/6: 3D Pose Source Head Reconstruction & Neck-to-Collar Composite
            // =========================================================================
            val tGen0 = System.currentTimeMillis()
            onProgress(
                SwapStageProgress(
                    stepIndex = 4,
                    totalSteps = 6,
                    stageTitle = "Stage 4/6: 3D Head Reconstruction",
                    detailMessage = "Warping source head onto target 3D pose (${tgtMesh.sourceLabel})...",
                    progressFraction = 0.60f
                )
            )

            val warpedSourceHead = FaceMesh3DEngine.warpSourceHeadToTarget3DPose(
                sourceBitmap = sourceBitmap,
                sourceMesh = srcMesh3D,
                targetMesh = tgtMesh,
                targetCanvasMatrix2x3 = tgtCanvasMatrix,
                canvasSize = canvasSize,
                offsetX = offsetX,
                offsetY = offsetY,
                scaleAdjust = scaleAdjust
            )

            val rawWarpedSrcMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
                ortEnv = ortEnv,
                alignedHeadCrop = warpedSourceHead,
                segModelFile = if (segFile.exists()) segFile else null,
                mattingModelFile = if (matFile.exists()) matFile else null,
                preferHardwareAccel = preferHardwareAccel,
                preloadedSegformerSession = preloadedSegformerSession
            )
            val adjPts5ForMask = adjustLandmarks5(tgtCanvasPts5, canvasSize, offsetX, offsetY, scaleAdjust)
            val warpedSrcMasks = if (rawWarpedSrcMasks.usedSegmentationOnnx) {
                rawWarpedSrcMasks
            } else {
                boostMatteForWideHeadCanvas(
                    alignedHeadCrop = warpedSourceHead,
                    baseMasks = rawWarpedSrcMasks,
                    canvasPts5 = adjPts5ForMask,
                    suppressBgHalo = true
                )
            }

            if (faceReactionMode == FaceReactionSourceMode.TARGET_REACTION &&
                ortEnv != null && hasInswapper && sourceEmbedding != null
            ) {
                // Adjust canvas 5-pt coordinates with offset/scale for exact inner expression alignment
                val adjPts5 = adjustLandmarks5(tgtCanvasPts5, canvasSize, offsetX, offsetY, scaleAdjust)
                runCatching {
                    GhostHeadReplacementEngine.fuseInswapperInnerExpressionIntoSourceHead(
                        ortEnv = ortEnv,
                        swapFile = swapFile,
                        preloadedSwapSession = preloadedSwapSession,
                        sourceEmbedding = sourceEmbedding,
                        targetBitmap = targetBitmap,
                        targetFace = targetFace,
                        synthesizedHeadCrop = warpedSourceHead,
                        headCropSize = canvasSize,
                        srcYawRatio = tgtYawRatio,
                        preferHardwareAccel = preferHardwareAccel,
                        overrideHeadLandmarks5 = adjPts5
                    )
                }
            }
            totalGenMs += (System.currentTimeMillis() - tGen0).coerceAtLeast(1L)

            val tBlend0 = System.currentTimeMillis()
            val adjPts5 = adjustLandmarks5(tgtCanvasPts5, canvasSize, offsetX, offsetY, scaleAdjust)
            val litSourceHead = if (enableColorTransfer) {
                GhostHeadReplacementEngine.adaptHeadSkinAndLightingSelective(
                    sourceHeadCrop = warpedSourceHead,
                    targetHeadCrop = alignedTargetHead,
                    skinAndNeckWeight = warpedSrcMasks.skinAndNeckWeight,
                    skinToneMode = skinToneMode,
                    overrideLandmarks5 = adjPts5
                )
            } else {
                warpedSourceHead
            }

            // Composite warped Source head onto the completely headless Target canvas (protecting collar)
            val stage4Composited = compositeSourceOverHeadlessCanvas(
                sourceHead = litSourceHead,
                headlessTarget = cleanHeadlessTargetCanvas,
                sourceAlpha = warpedSrcMasks.fullHeadAlpha,
                clothProtectMask = clothProtectMask
            )

            // =========================================================================
            // STAGE 5/6: Diffusion Seam & Hair/Neck Bridge (Strict Identity Protection)
            // =========================================================================
            onProgress(
                SwapStageProgress(
                    stepIndex = 5,
                    totalSteps = 6,
                    stageTitle = "Stage 5/6: Diffusion Seam Bridge",
                    detailMessage = if (sdPack != null) {
                        "Blending hair & neck-collar seam with SD 1.5 Diffusion (inner face locked)..."
                    } else {
                        "Applying multi-band neck-collar & hair perimeter seam bridge..."
                    },
                    progressFraction = 0.76f
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

            val seamResolvedCanvas = if (sdPack != null && ortEnv != null) {
                val sdOut = runCatching {
                    runDiffusionSeamBridge(
                        ortEnv = ortEnv,
                        pack = sdPack,
                        stage4Canvas = stage4Composited,
                        sourceAlpha = warpedSrcMasks.fullHeadAlpha,
                        targetEraseMask = cleanEraseMask,
                        innerFaceProtectMask = innerFaceProtectMask,
                        clothProtectMask = clothProtectMask,
                        preferHardwareAccel = preferHardwareAccel
                    )
                }.getOrNull()
                if (sdOut != null) {
                    usedDiffusionBridge = true
                    sdOut
                } else {
                    applyFastSeamFeatherBridge(
                        composited = stage4Composited,
                        sourceAlpha = warpedSrcMasks.fullHeadAlpha,
                        innerFaceProtectMask = innerFaceProtectMask
                    )
                }
            } else {
                applyFastSeamFeatherBridge(
                    composited = stage4Composited,
                    sourceAlpha = warpedSrcMasks.fullHeadAlpha,
                    innerFaceProtectMask = innerFaceProtectMask
                )
            }

            // =========================================================================
            // STAGE 6/6: 512x512 HD Enhancement (GFPGAN w16/FP32) + Age Harmony + Composite
            // =========================================================================
            onProgress(
                SwapStageProgress(
                    stepIndex = 6,
                    totalSteps = 6,
                    stageTitle = "Stage 6/6: HD Enhancement & Composite",
                    detailMessage = "Enhancing 512px facial & hair detail (GFPGAN) and compositing onto target...",
                    progressFraction = 0.92f
                )
            )

            val headAgePlan = targetAgePlans.getOrNull(idx)
            val clampedEnhance = enhancementStrength.coerceIn(0f, 1f)
            val gfpRestored512 = if (clampedEnhance > 0.05f && ortEnv != null &&
                (preloadedGfpganSession != null || (gfpFile.exists() && gfpFile.length() > 1024L))
            ) {
                val restored512 = FaceBlender.runOptionalGfpganEnhancement512(
                    ortEnv = ortEnv,
                    gfpganFile = if (gfpFile.exists()) gfpFile else null,
                    crop128Or512 = seamResolvedCanvas,
                    preferHardwareAccel = preferHardwareAccel,
                    preloadedGfpganSession = preloadedGfpganSession
                )
                if (restored512 != null) {
                    usedGfpganPass = true
                    val ageScale = (headAgePlan?.gfpganWeightScale ?: 1.0f).coerceIn(0.35f, 1.0f)
                    if (ageScale < 0.99f) {
                        val base512 = Bitmap.createScaledBitmap(seamResolvedCanvas, 512, 512, true)
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

            val rawEnhancedHead = GhostHeadReplacementEngine.applyTiledDetailEnhancement(
                gfpRestored512 ?: seamResolvedCanvas
            )
            gfpRestored512?.recycle()

            val finalHeadCanvas = if (headAgePlan != null && headAgePlan.shouldApplyReAging) {
                val work512 = if (rawEnhancedHead.width == 512 && rawEnhancedHead.height == 512) {
                    if (rawEnhancedHead.isMutable) rawEnhancedHead else rawEnhancedHead.copy(Bitmap.Config.ARGB_8888, true).also { rawEnhancedHead.recycle() }
                } else {
                    Bitmap.createScaledBitmap(rawEnhancedHead, 512, 512, true).also { rawEnhancedHead.recycle() }
                }
                val headPx512 = IntArray(512 * 512)
                work512.getPixels(headPx512, 0, 512, 0, 0, 512, 512)
                val skinMask512 = FloatArray(512 * 512) { idx512 ->
                    val y512 = idx512 / 512
                    val x512 = idx512 % 512
                    val sy = (y512 * canvasSize / 512).coerceIn(0, canvasSize - 1)
                    val sx = (x512 * canvasSize / 512).coerceIn(0, canvasSize - 1)
                    warpedSrcMasks.skinAndNeckWeight[sy * canvasSize + sx].coerceIn(0f, 1f)
                }
                val headGeom512 = FaceBlender.computeWarpedGeometry(
                    forwardMatrix128 = FloatArray(6) { i -> tgtCanvasMatrix[i] * (512f / canvasSize.toFloat()) },
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
                rawEnhancedHead
            }

            // Union envelope covers both the erased old target head & the new 3D source head while preserving exterior background & torso
            val baseEnvelopeAlpha = FloatArray(canvasSize * canvasSize) { i ->
                val rawUnion = max(warpedSrcMasks.fullHeadAlpha[i], cleanEraseMask[i] * 0.96f)
                val protected = rawUnion * (1f - clothProtectMask[i] * 0.88f)
                protected.coerceIn(0f, 1f)
            }
            val warpSize = finalHeadCanvas.width
            val warpScale = warpSize.toFloat() / canvasSize.toFloat()
            val warpMatrix2x3 = if (warpSize == canvasSize) {
                tgtCanvasMatrix
            } else {
                FloatArray(6) { i -> tgtCanvasMatrix[i] * warpScale }
            }
            val warpEnvelopeAlpha = if (warpSize == canvasSize) {
                baseEnvelopeAlpha
            } else {
                FloatArray(warpSize * warpSize) { idxW ->
                    val yW = idxW / warpSize
                    val xW = idxW % warpSize
                    val sy = (yW * canvasSize / warpSize).coerceIn(0, canvasSize - 1)
                    val sx = (xW * canvasSize / warpSize).coerceIn(0, canvasSize - 1)
                    baseEnvelopeAlpha[sy * canvasSize + sx]
                }
            }

            // Note: because Source head was already shifted inside stage 4 mesh warp, keep composite warp offset=0, scale=1
            // so the clean erased background & warped head stay locked to target coordinates.
            GhostHeadReplacementEngine.warpHeadCropBackToTarget(
                targetPixels = compositePixels,
                targetWidth = targetW,
                targetHeight = targetH,
                headCrop = finalHeadCanvas,
                forwardHeadMatrix2x3 = warpMatrix2x3,
                envelopeAlpha = warpEnvelopeAlpha,
                blendStrength = blendStrength,
                offsetX = 0f,
                offsetY = 0f,
                scaleAdjust = 1.0f
            )

            if (firstTargetHeadPreview == null) {
                firstTargetHeadPreview = alignedTargetHead.copy(Bitmap.Config.ARGB_8888, false)
                firstGeneratedHeadPreview = finalHeadCanvas.copy(Bitmap.Config.ARGB_8888, false)
            }

            if (litSourceHead !== warpedSourceHead) litSourceHead.recycle()
            warpedSourceHead.recycle()
            cleanHeadlessTargetCanvas.recycle()
            if (stage4Composited !== seamResolvedCanvas) stage4Composited.recycle()
            seamResolvedCanvas.recycle()
            alignedTargetHead.recycle()
            finalHeadCanvas.recycle()

            totalBlendMs += (System.currentTimeMillis() - tBlend0).coerceAtLeast(1L)
        }

        val srcThumb112 = sourceEmbedding?.aligned112Crop ?: run {
            val m112 = FaceAlignment.estimateNorm(sourceFace.landmarks5, 112)
            FaceAlignment.warpAffineCrop(sourceBitmap, m112, 112)
        }

        if (lowMemoryMode) {
            System.gc()
        }

        val finalBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        finalBitmap.setPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)

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
            append("Clean-Erase 3D Head Replace (${pipelineMode.title}): ")
            append(srcMesh3D.sourceLabel)
            append(" + $segLabelSummary")
            if (targetDepthMap != null) append(" + ${targetDepthMap.modelUsedLabel}")
            if (usedLamaErasure) append(" + LaMa Clean Erase")
            if (usedDiffusionBridge) append(" + SD 1.5 Diffusion Seam Bridge")
            if (usedGfpganPass) append(" + GFPGAN 512px HD")
            if (portraitBlurStrength > 0.03f) append(" + Depth Bokeh ${(portraitBlurStrength * 100).toInt()}%")
            append(" | engineUsed=$engineUsed")
        }

        return FaceSwapExecutionResult(
            outputBitmap = finalBitmap,
            alignedSource112 = srcThumb112,
            alignedTarget128 = firstTargetHeadPreview ?: Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888),
            rawSwapped128 = firstGeneratedHeadPreview ?: Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888),
            swappedFacesCount = targetFacesToReplace.size,
            detectionMs = detMs,
            embeddingMs = embMs,
            inswapperMs = totalGenMs.coerceAtLeast(1L),
            blendingMs = totalBlendMs.coerceAtLeast(1L),
            totalMs = totalMs,
            pipelineSummary = summary,
            engineUsed = engineUsed,
            ageHarmonySummary = firstAgeHarmonySummary
        )
    }

    private fun estimateWideHeadCanvasMatrix(
        srcLandmarks5: List<PointF>,
        cropSize: Int,
        yawRatio: Float = 0f
    ): FloatArray {
        val s = cropSize / 256.0f
        val yawShiftX = (-yawRatio * 7.0f * s).coerceIn(-12f * s, 12f * s)
        // Map inter-ocular distance to 38/256 (~0.148 * cropSize) centered at y=132/256
        // leaving ~3.47x eyeDist headroom above eyes for tall/voluminous hair and ~3.26x below for neck/collar.
        val dstPoints = arrayOf(
            PointF(109.0f * s + yawShiftX, 132.0f * s),
            PointF(147.0f * s + yawShiftX, 132.0f * s),
            PointF(128.0f * s + yawShiftX, 153.0f * s),
            PointF(112.0f * s + yawShiftX, 172.0f * s),
            PointF(144.0f * s + yawShiftX, 172.0f * s)
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
     * Enhances the Chromatic-Geodesic fallback matte in wide-headroom canvases so large target hairstyles
     * (up to ~2.65x eyeDist above eyes and ~2.45x horizontally) are cleanly detected via exterior
     * background chromatic divergence while preserving untouched borders.
     */
    private fun boostMatteForWideHeadCanvas(
        alignedHeadCrop: Bitmap,
        baseMasks: HeadSegmentationMasks,
        canvasPts5: List<PointF>,
        suppressBgHalo: Boolean = false
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
        val eyeDist = hypot((rEye.x - lEye.x).toDouble(), (rEye.y - lEye.y).toDouble()).toFloat().coerceAtLeast(10f)

        // Sample exterior background colors ONLY from top-left/top-right corners and left/right borders
        // (never center top row where tall hair crowns may reach)
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
        for (y in 4 until (size * 0.55f).toInt() step step) {
            for (x in listOf(3, size - 4)) {
                val c = pixels[y * size + x]
                bgR.add(((c ushr 16) and 0xFF).toFloat())
                bgG.add(((c ushr 8) and 0xFF).toFloat())
                bgB.add((c and 0xFF).toFloat())
            }
        }

        val boostedAlpha = baseMasks.fullHeadAlpha.copyOf()
        val hairCenterY = eyeMidY - eyeDist * 0.55f
        val hairRx = eyeDist * 2.45f
        val hairRy = eyeDist * 2.45f
        val faceCoreRx = eyeDist * 1.15f
        val faceCoreRy = eyeDist * 1.35f
        val borderMargin = (size * 0.025f).toInt().coerceAtLeast(3)

        for (y in borderMargin until size - borderMargin) {
            val rowOffset = y * size
            val dy = (y - hairCenterY) / hairRy
            val fDy = (y - (eyeMidY + eyeDist * 0.35f)) / faceCoreRy
            for (x in borderMargin until size - borderMargin) {
                val dx = (x - eyeMidX) / hairRx
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

    private fun buildClothingCollarProtectMask(
        canvasSize: Int,
        tgtCanvasPts5: List<PointF>,
        tgtMasks: HeadSegmentationMasks,
        tgtCanvasMatrix: FloatArray,
        targetDepthMap: DepthMapResult?,
        targetWidth: Int,
        targetHeight: Int
    ): FloatArray {
        val total = canvasSize * canvasSize
        val protect = FloatArray(total)
        val chinEstY = if (tgtCanvasPts5.size >= 5) {
            val eyeY = (tgtCanvasPts5[0].y + tgtCanvasPts5[1].y) * 0.5f
            val mouthY = (tgtCanvasPts5[3].y + tgtCanvasPts5[4].y) * 0.5f
            mouthY + (mouthY - eyeY) * 0.62f
        } else {
            canvasSize * 0.78f
        }
        val collarStartY = (chinEstY + canvasSize * 0.08f).coerceIn(canvasSize * 0.68f, canvasSize * 0.90f)
        val inv = FaceAlignment.invertAffine2x3(tgtCanvasMatrix)
        val torsoCenterDepth = targetDepthMap?.let { dm ->
            val cx = ((inv[0] * (canvasSize * 0.5f) + inv[1] * (canvasSize * 0.92f) + inv[2]) / targetWidth).coerceIn(0f, 1f)
            val cy = ((inv[3] * (canvasSize * 0.5f) + inv[4] * (canvasSize * 0.92f) + inv[5]) / targetHeight).coerceIn(0f, 1f)
            dm.sampleBilinear(cx, cy)
        } ?: 0.5f

        for (y in 0 until canvasSize) {
            if (y < collarStartY) continue
            val rowOffset = y * canvasSize
            val vWeight = ((y - collarStartY) / (canvasSize * 0.14f)).coerceIn(0f, 1f)
            for (x in 0 until canvasSize) {
                val idx = rowOffset + x
                // Regions below collarStartY that are NOT skin/neck (e.g. clothing/shoulders) get protected
                val isNotSkin = (1f - tgtMasks.skinAndNeckWeight[idx]).coerceIn(0f, 1f)
                val depthFgBoost = if (targetDepthMap != null) {
                    val tx = inv[0] * x + inv[1] * y + inv[2]
                    val ty = inv[3] * x + inv[4] * y + inv[5]
                    val d = targetDepthMap.sampleBilinear(tx / targetWidth, ty / targetHeight)
                    if (abs(d - torsoCenterDepth) < 0.20f) 1.0f else 0.5f
                } else {
                    1.0f
                }
                protect[idx] = (vWeight * isNotSkin * depthFgBoost).coerceIn(0f, 1f)
            }
        }
        return protect
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
        tgtCanvasMatrix: FloatArray,
        targetDepthMap: DepthMapResult?,
        targetWidth: Int,
        targetHeight: Int,
        lamaModelFile: File?,
        preferHardwareAccel: Boolean
    ): Pair<Bitmap, Boolean> {
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
            val lamaOut = runCatching {
                HeadSegmentationAndInpainting.runLamaInpaintOnnx(
                    ortEnv = ortEnv,
                    lamaFile = lamaModelFile,
                    cropBitmap = targetCanvas,
                    gapMask = eraseMask,
                    preferHardwareAccel = preferHardwareAccel
                )
            }.getOrNull()
            if (lamaOut != null) {
                return lamaOut to true
            }
        }

        // Depth-aware multi-directional exterior boundary marching across the entire erased head & upper neck
        val pixels = IntArray(total)
        targetCanvas.getPixels(pixels, 0, size, 0, 0, size, size)
        val inpainted = pixels.copyOf()

        for (y in 0 until size) {
            val rowOffset = y * size
            var leftAnchorX = 2
            for (x in 2 until size / 2) {
                val idx = rowOffset + x
                if (eraseMask[idx] < 0.10f && clothProtectMask[idx] < 0.35f) {
                    leftAnchorX = x
                } else if (eraseMask[idx] >= 0.10f) {
                    break
                }
            }
            val leftColor = pixels[rowOffset + leftAnchorX]

            var rightAnchorX = size - 3
            for (x in (size - 3) downTo size / 2) {
                val idx = rowOffset + x
                if (eraseMask[idx] < 0.10f && clothProtectMask[idx] < 0.35f) {
                    rightAnchorX = x
                } else if (eraseMask[idx] >= 0.10f) {
                    break
                }
            }
            val rightColor = pixels[rowOffset + rightAnchorX]
            val span = (rightAnchorX - leftAnchorX).coerceAtLeast(1).toFloat()

            for (x in leftAnchorX..rightAnchorX) {
                val idx = rowOffset + x
                val eAlpha = eraseMask[idx]
                if (eAlpha > 0.04f) {
                    val tHoriz = ((x - leftAnchorX) / span).coerceIn(0f, 1f)
                    val topIdx = min(y, 4) * size + x
                    val hasCleanTop = eraseMask[topIdx] < 0.10f && clothProtectMask[topIdx] < 0.35f

                    val hR = (leftColor ushr 16 and 0xFF) * (1f - tHoriz) + (rightColor ushr 16 and 0xFF) * tHoriz
                    val hG = (leftColor ushr 8 and 0xFF) * (1f - tHoriz) + (rightColor ushr 8 and 0xFF) * tHoriz
                    val hB = (leftColor and 0xFF) * (1f - tHoriz) + (rightColor and 0xFF) * tHoriz

                    val bgR: Int
                    val bgG: Int
                    val bgB: Int
                    if (hasCleanTop) {
                        val topColor = pixels[topIdx]
                        bgR = (0.76f * hR + 0.24f * (topColor ushr 16 and 0xFF)).toInt().coerceIn(0, 255)
                        bgG = (0.76f * hG + 0.24f * (topColor ushr 8 and 0xFF)).toInt().coerceIn(0, 255)
                        bgB = (0.76f * hB + 0.24f * (topColor and 0xFF)).toInt().coerceIn(0, 255)
                    } else {
                        bgR = hR.toInt().coerceIn(0, 255)
                        bgG = hG.toInt().coerceIn(0, 255)
                        bgB = hB.toInt().coerceIn(0, 255)
                    }

                    val orig = pixels[idx]
                    val oR = (orig ushr 16) and 0xFF
                    val oG = (orig ushr 8) and 0xFF
                    val oB = orig and 0xFF

                    val blend = ((eAlpha - 0.04f) / 0.38f).coerceIn(0f, 1f)
                    val fR = (oR * (1f - blend) + bgR * blend).toInt().coerceIn(0, 255)
                    val fG = (oG * (1f - blend) + bgG * blend).toInt().coerceIn(0, 255)
                    val fB = (oB * (1f - blend) + bgB * blend).toInt().coerceIn(0, 255)
                    inpainted[idx] = (0xFF shl 24) or (fR shl 16) or (fG shl 8) or fB
                }
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
        ortEnv: OrtEnvironment,
        pack: InstalledSdInpaintPack,
        stage4Canvas: Bitmap,
        sourceAlpha: FloatArray,
        targetEraseMask: FloatArray,
        innerFaceProtectMask: FloatArray,
        clothProtectMask: FloatArray,
        preferHardwareAccel: Boolean
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

        val sdRes = SdInpaintOnnxEngine.inpaint(
            ortEnv = ortEnv,
            pack = pack,
            sourceBitmap = stage4Canvas,
            maskBitmap = maskBmp,
            prompt = "natural realistic hair strands, seamless skin neck collar transition, photorealistic lighting",
            negativePrompt = "double face, extra head, blurry hair, artifacts, deformed neck, plastic skin",
            fastMode = true,
            preferHardwareAccel = preferHardwareAccel,
            strength = 0.45f,
            guidanceScale = 6.5f
        )
        maskBmp.recycle()

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
        innerFaceProtectMask: FloatArray
    ): Bitmap {
        val size = composited.width
        val total = size * size
        val px = IntArray(total)
        composited.getPixels(px, 0, size, 0, 0, size, size)
        val out = px.copyOf()

        for (y in 1 until size - 1) {
            val row = y * size
            for (x in 1 until size - 1) {
                val idx = row + x
                val a = sourceAlpha[idx]
                // Smooth only along the outer hair/neck transition ring (never inside inner face)
                if (a in 0.06f..0.75f && innerFaceProtectMask[idx] < 0.25f) {
                    val n = px[(y - 1) * size + x]
                    val s = px[(y + 1) * size + x]
                    val l = px[row + x - 1]
                    val r = px[row + x + 1]
                    val c = px[idx]

                    val avgR = ((n ushr 16 and 0xFF) + (s ushr 16 and 0xFF) + (l ushr 16 and 0xFF) + (r ushr 16 and 0xFF)) * 0.25f
                    val avgG = ((n ushr 8 and 0xFF) + (s ushr 8 and 0xFF) + (l ushr 8 and 0xFF) + (r ushr 8 and 0xFF)) * 0.25f
                    val avgB = ((n and 0xFF) + (s and 0xFF) + (l and 0xFF) + (r and 0xFF)) * 0.25f

                    val w = 0.35f
                    val fr = (((c ushr 16 and 0xFF) * (1f - w)) + avgR * w).toInt().coerceIn(0, 255)
                    val fg = (((c ushr 8 and 0xFF) * (1f - w)) + avgG * w).toInt().coerceIn(0, 255)
                    val fb = (((c and 0xFF) * (1f - w)) + avgB * w).toInt().coerceIn(0, 255)
                    out[idx] = (0xFF shl 24) or (fr shl 16) or (fg shl 8) or fb
                }
            }
        }
        val outBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(out, 0, size, 0, 0, size, size)
        return outBmp
    }
}
