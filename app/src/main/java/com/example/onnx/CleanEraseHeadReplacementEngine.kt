package com.example.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import java.io.File
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
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
        isCancelled: () -> Boolean = { false },
        onProgress: (SwapStageProgress) -> Unit
    ): FaceSwapExecutionResult {
        val tStart = System.currentTimeMillis()
        require(targetFacesToReplace.isNotEmpty()) {
            "Please select at least one detected target head to replace."
        }

        val canvasSize = if (lowMemoryMode) 320 else 512
        val detFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.DETECTOR)
        val segFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.SEGMENTATION)
        val matFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.MATTING)
        val lamaFile = OnnxProtobufInspector.resolveModelFile(context, ModelSlot.INPAINTING)
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
        // STAGE 2/6: Depth Anything V2 + Seam Pack Plan
        // =========================================================================
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
        val borderEnvelope = computeCanvasBorderEnvelope(canvasSize)

        var firstTargetHeadPreview: Bitmap? = null
        var firstHeadlessPreview512: Bitmap? = null
        var firstIntermediatePortraitPreview512: Bitmap? = null
        var firstTgtCanvasMatrix: FloatArray? = null
        val placedHeadPoses = mutableListOf<PlacedHeadPose>()
        var totalGenMs = 0L
        var totalBlendMs = 0L
        var usedDiffusionBridge = false
        var usedLamaErasure = false
        var segLabelSummary = "3D-Clean-Erase"

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

            // Dilate target head mask slightly (radius ~5 px) so zero target hair halo survives,
            // and taper cleanly near canvas borders so no rectangular crop boundary ever appears.
            val dilatedTargetEraseMask = dilateAlpha(tgtMasks.fullHeadAlpha, canvasSize, radiusPx = 5)
            val cleanEraseMask = FloatArray(canvasSize * canvasSize) { i ->
                (dilatedTargetEraseMask[i] * (1f - clothProtectMask[i]) * borderEnvelope[i]).coerceIn(0f, 1f)
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
            val warpedSourceHead = warpedSrcResult.bitmap
            val srcBoundsMask = warpedSrcResult.inBoundsMask

            val rawWarpedSrcMasks = HeadSegmentationAndInpainting.segmentHeadHairNeck(
                ortEnv = ortEnv,
                alignedHeadCrop = warpedSourceHead,
                segModelFile = if (segFile.exists()) segFile else null,
                mattingModelFile = if (matFile.exists()) matFile else null,
                preferHardwareAccel = preferHardwareAccel,
                preloadedSegformerSession = preloadedSegformerSession
            )
            val adjPts5 = adjustLandmarks5(tgtCanvasPts5, canvasSize, offsetX, offsetY, scaleAdjust)
            val rawOrBoostedSrcMasks = if (rawWarpedSrcMasks.usedSegmentationOnnx) {
                rawWarpedSrcMasks
            } else {
                boostMatteForWideHeadCanvas(
                    alignedHeadCrop = warpedSourceHead,
                    baseMasks = rawWarpedSrcMasks,
                    canvasPts5 = adjPts5,
                    suppressBgHalo = true
                )
            }
            // Clamp out-of-bounds top/side clamped source pixels and outer canvas borders cleanly to 0
            val filteredFullHeadAlpha = FloatArray(canvasSize * canvasSize) { i ->
                (rawOrBoostedSrcMasks.fullHeadAlpha[i] * srcBoundsMask[i] * borderEnvelope[i]).coerceIn(0f, 1f)
            }
            val filteredSkinAndNeckWeight = FloatArray(canvasSize * canvasSize) { i ->
                (rawOrBoostedSrcMasks.skinAndNeckWeight[i] * srcBoundsMask[i]).coerceIn(0f, 1f)
            }
            totalGenMs += (System.currentTimeMillis() - tGen0).coerceAtLeast(1L)

            val tBlend0 = System.currentTimeMillis()
            // Harmonize hair, neck & ambient scene lighting on the intermediate portrait cleanly;
            // final exact facial skin tone & ArcFace identity come from Step 3 at high resolution.
            val step2ToneMode = if (skinToneMode == SkinToneSourceMode.SOURCE_IDENTITY) {
                SkinToneSourceMode.BALANCED_BLEND
            } else {
                skinToneMode
            }
            val litSourceHead = if (enableColorTransfer) {
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

            // Composite warped Source head onto the completely headless Target canvas (protecting collar)
            val stage4Composited = compositeSourceOverHeadlessCanvas(
                sourceHead = litSourceHead,
                headlessTarget = cleanHeadlessTargetCanvas,
                sourceAlpha = filteredFullHeadAlpha,
                clothProtectMask = clothProtectMask
            )

            // =========================================================================
            // STAGE 5/6: Diffusion Seam & Hair/Neck Bridge (Strict Identity Protection)
            // =========================================================================
            if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")
            onProgress(
                SwapStageProgress(
                    stepIndex = 5,
                    totalSteps = 6,
                    stageTitle = "Stage 5/6: Diffusion Seam Bridge",
                    detailMessage = if (sdPack != null) {
                        "Freeing Stage 1-4 sessions & preparing SD 1.5 (${sdPack.packType.displayLabel})..."
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

            val seamResolvedCanvas = if (sdPack != null && ortEnv != null) {
                // Free all Stage 1-4 ONNX sessions (det_10g, facemesh, arcface, hyperswap, segformer, mivolo)
                // so SD 1.5 UNet has maximum native RAM and cannot OOM/swap-thrash on mobile devices.
                OnnxProtobufInspector.clearCachedSessions()
                SdInpaintOnnxEngine.clearSessionCache()
                System.gc()

                val sdAttempt = runCatching {
                    runDiffusionSeamBridge(
                        ortEnv = ortEnv,
                        pack = sdPack,
                        stage4Canvas = stage4Composited,
                        sourceAlpha = filteredFullHeadAlpha,
                        targetEraseMask = cleanEraseMask,
                        innerFaceProtectMask = innerFaceProtectMask,
                        clothProtectMask = clothProtectMask,
                        preferHardwareAccel = preferHardwareAccel,
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
                        composited = stage4Composited,
                        sourceAlpha = filteredFullHeadAlpha,
                        innerFaceProtectMask = innerFaceProtectMask,
                        clothProtectMask = clothProtectMask
                    )
                }
            } else {
                applyFastSeamFeatherBridge(
                    composited = stage4Composited,
                    sourceAlpha = filteredFullHeadAlpha,
                    innerFaceProtectMask = innerFaceProtectMask,
                    clothProtectMask = clothProtectMask
                )
            }

            if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

            // Mild hair/neck crispness before warping into Intermediate Full Portrait (NO zoomed-out GFPGAN!)
            val step2PortraitHeadCanvas = GhostHeadReplacementEngine.applyTiledDetailEnhancement(
                bitmap = seamResolvedCanvas,
                amount = 0.16f
            )

            // Union envelope covers both the erased old target head & the new 3D source head while
            // strictly feathering canvas borders and protecting target collar/clothing.
            val baseEnvelopeAlpha = FloatArray(canvasSize * canvasSize) { i ->
                val rawUnion = max(filteredFullHeadAlpha[i], cleanEraseMask[i] * 0.96f)
                val protected = rawUnion * (1f - clothProtectMask[i] * 0.90f) * borderEnvelope[i]
                protected.coerceIn(0f, 1f)
            }

            // Warp Step 2 intermediate portrait onto compositePixels
            GhostHeadReplacementEngine.warpHeadCropBackToTarget(
                targetPixels = compositePixels,
                targetWidth = targetW,
                targetHeight = targetH,
                headCrop = step2PortraitHeadCanvas,
                forwardHeadMatrix2x3 = tgtCanvasMatrix,
                envelopeAlpha = baseEnvelopeAlpha,
                blendStrength = blendStrength,
                offsetX = 0f,
                offsetY = 0f,
                scaleAdjust = 1.0f
            )

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

            if (firstTargetHeadPreview == null) {
                firstTargetHeadPreview = alignedTargetHead.copy(Bitmap.Config.ARGB_8888, false)
                firstTgtCanvasMatrix = tgtCanvasMatrix.copyOf()
                firstHeadlessPreview512 = if (cleanHeadlessTargetCanvas.width == 512) {
                    cleanHeadlessTargetCanvas.copy(Bitmap.Config.ARGB_8888, false)
                } else {
                    Bitmap.createScaledBitmap(cleanHeadlessTargetCanvas, 512, 512, true)
                }
                firstIntermediatePortraitPreview512 = if (step2PortraitHeadCanvas.width == 512) {
                    step2PortraitHeadCanvas.copy(Bitmap.Config.ARGB_8888, false)
                } else {
                    Bitmap.createScaledBitmap(step2PortraitHeadCanvas, 512, 512, true)
                }
            }

            if (litSourceHead !== warpedSourceHead) litSourceHead.recycle()
            warpedSourceHead.recycle()
            cleanHeadlessTargetCanvas.recycle()
            if (stage4Composited !== seamResolvedCanvas) stage4Composited.recycle()
            seamResolvedCanvas.recycle()
            if (step2PortraitHeadCanvas !== seamResolvedCanvas) step2PortraitHeadCanvas.recycle()
            alignedTargetHead.recycle()

            totalBlendMs += (System.currentTimeMillis() - tBlend0).coerceAtLeast(1L)
        }

        if (isCancelled()) throw java.util.concurrent.CancellationException("Head replacement cancelled by user.")

        // =========================================================================
        // STAGE 6/6 (Step 3): High-Res Face Swap (`ArcFace` -> `HyperSwap 256px` -> `GFPGAN 512`)
        // on the Step 2 Intermediate Full Portrait
        // =========================================================================
        val intermediatePortraitBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        intermediatePortraitBitmap.setPixels(compositePixels, 0, targetW, 0, 0, targetW, targetH)

        // Release heavy Step 1/2 temporary ONNX sessions (LaMa/Depth/FaceMesh/SD) so Step 3 has full RAM
        if (ortEnv != null && preloadedArcFaceSession == null && preloadedSwapSession == null && preloadedGfpganSession == null) {
            SdInpaintOnnxEngine.clearSessionCache()
            OnnxProtobufInspector.clearCachedSessions()
            if (lowMemoryMode) System.gc()
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
            ageHarmonySummary = step3Res.ageHarmonySummary ?: firstAgeHarmonySummary
        )
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

        // 2D Multi-directional exterior anchor interpolation + gentle local smoothing (eliminates 1D bands)
        val pixels = IntArray(total)
        targetCanvas.getPixels(pixels, 0, size, 0, 0, size, size)
        val inpainted = pixels.copyOf()
        val isCleanAnchor = BooleanArray(total) { i ->
            eraseMask[i] < 0.08f && clothProtectMask[i] < 0.30f
        }
        val invTgt = if (targetDepthMap != null) FaceAlignment.invertAffine2x3(tgtCanvasMatrix) else null
        val dirDx = intArrayOf(-1, 1, 0, -1, 1, -1, 1)
        val dirDy = intArrayOf(0, 0, -1, -1, -1, 1, 1)

        for (y in 0 until size) {
            val rowOffset = y * size
            for (x in 0 until size) {
                val idx = rowOffset + x
                val eAlpha = eraseMask[idx]
                if (eAlpha <= 0.04f) continue

                val centerDepth = if (targetDepthMap != null && invTgt != null) {
                    val tx = invTgt[0] * x + invTgt[1] * y + invTgt[2]
                    val ty = invTgt[3] * x + invTgt[4] * y + invTgt[5]
                    targetDepthMap.sampleBilinear(tx / targetWidth, ty / targetHeight)
                } else {
                    0.5f
                }

                var wSum = 0f
                var rSum = 0f
                var gSum = 0f
                var bSum = 0f

                for (d in dirDx.indices) {
                    val stepX = dirDx[d]
                    val stepY = dirDy[d]
                    var cx = x + stepX
                    var cy = y + stepY
                    var steps = 1
                    var foundIdx = -1
                    while (cx in 2 until (size - 2) && cy in 2 until (size - 2)) {
                        val cIdx = cy * size + cx
                        if (isCleanAnchor[cIdx]) {
                            foundIdx = cIdx
                            break
                        }
                        cx += stepX
                        cy += stepY
                        steps++
                    }
                    if (foundIdx >= 0) {
                        val dist = steps.toFloat() * if (stepX != 0 && stepY != 0) 1.414f else 1.0f
                        var w = 1f / ((dist + 4f) * sqrt(dist + 4f))
                        if (targetDepthMap != null && invTgt != null) {
                            val ax = foundIdx % size
                            val ay = foundIdx / size
                            val tx = invTgt[0] * ax + invTgt[1] * ay + invTgt[2]
                            val ty = invTgt[3] * ax + invTgt[4] * ay + invTgt[5]
                            val aDepth = targetDepthMap.sampleBilinear(tx / targetWidth, ty / targetHeight)
                            if (abs(aDepth - centerDepth) > 0.25f) w *= 0.45f
                        }
                        val col = pixels[foundIdx]
                        rSum += ((col ushr 16) and 0xFF) * w
                        gSum += ((col ushr 8) and 0xFF) * w
                        bSum += (col and 0xFF) * w
                        wSum += w
                    }
                }

                if (wSum <= 1e-6f) {
                    val fallbackPts = arrayOf(
                        pixels[min(y, 4) * size + 4],
                        pixels[min(y, 4) * size + (size - 5)]
                    )
                    for (col in fallbackPts) {
                        rSum += (col ushr 16) and 0xFF
                        gSum += (col ushr 8) and 0xFF
                        bSum += col and 0xFF
                        wSum += 1f
                    }
                }

                val bgR = (rSum / wSum).coerceIn(0f, 255f)
                val bgG = (gSum / wSum).coerceIn(0f, 255f)
                val bgB = (bSum / wSum).coerceIn(0f, 255f)

                val orig = pixels[idx]
                val oR = (orig ushr 16) and 0xFF
                val oG = (orig ushr 8) and 0xFF
                val oB = orig and 0xFF

                val blend = ((eAlpha - 0.04f) / 0.34f).coerceIn(0f, 1f)
                val fR = (oR * (1f - blend) + bgR * blend).toInt().coerceIn(0, 255)
                val fG = (oG * (1f - blend) + bgG * blend).toInt().coerceIn(0, 255)
                val fB = (oB * (1f - blend) + bgB * blend).toInt().coerceIn(0, 255)
                inpainted[idx] = (0xFF shl 24) or (fR shl 16) or (fG shl 8) or fB
            }
        }

        // Gentle 2-pass 3x3 smoothing strictly inside erased zone to eliminate any ray transitions
        repeat(2) {
            val prev = inpainted.copyOf()
            for (y in 2 until size - 2) {
                val row = y * size
                for (x in 2 until size - 2) {
                    val idx = row + x
                    if (eraseMask[idx] > 0.12f && clothProtectMask[idx] < 0.35f) {
                        val c = prev[idx]
                        val n = prev[(y - 1) * size + x]
                        val s = prev[(y + 1) * size + x]
                        val l = prev[row + x - 1]
                        val r = prev[row + x + 1]
                        val sr = (((c ushr 16 and 0xFF) * 2 + (n ushr 16 and 0xFF) + (s ushr 16 and 0xFF) + (l ushr 16 and 0xFF) + (r ushr 16 and 0xFF)) / 6)
                        val sg = (((c ushr 8 and 0xFF) * 2 + (n ushr 8 and 0xFF) + (s ushr 8 and 0xFF) + (l ushr 8 and 0xFF) + (r ushr 8 and 0xFF)) / 6)
                        val sb = (((c and 0xFF) * 2 + (n and 0xFF) + (s and 0xFF) + (l and 0xFF) + (r and 0xFF)) / 6)
                        inpainted[idx] = (0xFF shl 24) or (sr.coerceIn(0, 255) shl 16) or (sg.coerceIn(0, 255) shl 8) or sb.coerceIn(0, 255)
                    }
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
        preferHardwareAccel: Boolean,
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
        val px = IntArray(total)
        composited.getPixels(px, 0, size, 0, 0, size, size)
        val out = px.copyOf()

        for (y in 1 until size - 1) {
            val row = y * size
            for (x in 1 until size - 1) {
                val idx = row + x
                val a = sourceAlpha[idx]
                val clothLocked = (clothProtectMask?.getOrNull(idx) ?: 0f) > 0.45f
                // Smooth only along the outer hair/neck transition ring (never inside inner face or protected collar/jewelry)
                if (a in 0.06f..0.75f && innerFaceProtectMask[idx] < 0.25f && !clothLocked) {
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
