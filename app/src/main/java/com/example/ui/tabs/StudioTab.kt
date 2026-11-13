package com.example.ui.tabs

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.material3.HorizontalDivider
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.FaceRetouchingNatural
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Preview
import androidx.compose.material.icons.filled.Radar
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.VerifiedUser
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.R
import com.example.onnx.DetectedFace
import com.example.onnx.FaceReactionSourceMode
import com.example.onnx.AccessoryTarget
import com.example.onnx.InpaintModelOption
import com.example.onnx.ModelSlot
import com.example.onnx.SkinToneSourceMode
import com.example.ui.FaceSwapUiState
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.material.icons.filled.Crop
import androidx.compose.material.icons.filled.RotateRight
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.Slider
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.activity.compose.BackHandler
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import com.example.ui.OutputResolutionOption
import com.example.ui.ProcessingQualityLevel
import com.example.ui.StudioMode
import com.example.ui.StudioSubPage
import com.example.ui.components.FaceDetectionCanvas
import com.example.ui.theme.AmberWarning
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.NeonEmerald
import com.example.ui.theme.ObsidianBg
import com.example.ui.theme.RoyalViolet

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun StudioTab(
    uiState: FaceSwapUiState,
    onSelectStudioMode: (StudioMode) -> Unit,
    onPickSourcePhoto: () -> Unit,
    onPickTargetPhoto: () -> Unit,
    onDetectHead: () -> Unit,
    onPreviewHead: () -> Unit,
    onSelectSourceFace: (Int) -> Unit,
    onSelectTargetFace: (Int) -> Unit,
    onToggleReplaceAllFaces: (Boolean) -> Unit,
    onConsentChanged: (Boolean) -> Unit,
    onColorTransferChanged: (Boolean) -> Unit,
    onWatermarkChanged: (Boolean) -> Unit,
    onTwoModelFallbackChanged: (Boolean) -> Unit,
    onHardwareAccelChanged: (Boolean) -> Unit,
    onLowMemoryChanged: (Boolean) -> Unit,
    onRunSwap: () -> Unit,
    onSaveToGallery: () -> Unit,
    onSaveHdToGallery: () -> Unit = onSaveToGallery,
    onShareResult: () -> Unit = {},
    onRotateResult90: () -> Unit = {},
    onCropResultAspect: (Int, Int) -> Unit = { _, _ -> },
    onQualityLevelChanged: (ProcessingQualityLevel) -> Unit = {},
    onOutputResolutionChanged: (OutputResolutionOption) -> Unit = {},
    onBlendStrengthChanged: (Float) -> Unit = {},
    onEnhancementStrengthChanged: (Float) -> Unit = {},
    onOcclusionProtectionChanged: (Boolean) -> Unit = {},
    onPortraitBlurStrengthChanged: (Float) -> Unit = {},
    onFaceOffsetXChanged: (Float) -> Unit = {},
    onFaceOffsetYChanged: (Float) -> Unit = {},
    onFaceScaleChanged: (Float) -> Unit = {},
    onResetAdjustments: () -> Unit = {},
    onToggleCompareOriginal: (Boolean) -> Unit,
    onOpenModelsTab: () -> Unit,
    onImportAllFromFolder: () -> Unit = {},
    onSkinToneModeChanged: (SkinToneSourceMode) -> Unit = {},
    onFaceReactionModeChanged: (FaceReactionSourceMode) -> Unit = {},
    onNegativeArtifactSuppressionChanged: (Boolean) -> Unit = {},
    onLcmInpaintingChanged: (Boolean) -> Unit = {},
    onFaceAgingFactorChanged: (Float) -> Unit = {},
    onBrowseSourceFile: () -> Unit = onPickSourcePhoto,
    onBrowseTargetFile: () -> Unit = onPickTargetPhoto,
    onRunVisualValidation: () -> Unit = {},
    onSaveValidationSheet: (Boolean) -> Unit = {},
    onSelectStudioSubPage: (StudioSubPage) -> Unit = {},
    onPickInpaintSourceFromFolder: () -> Unit = {},
    onSelectInpaintModelOption: (InpaintModelOption) -> Unit = {},
    onInpaintMaskUpdated: (android.graphics.Bitmap?) -> Unit = {},
    onInpaintPromptChanged: (String) -> Unit = {},
    onInpaintNegativePromptChanged: (String) -> Unit = {},
    onResetInpaintNegativePrompt: () -> Unit = {},
    onInpaintFastModeChanged: (Boolean) -> Unit = {},
    onInpaintStrengthChanged: (Float) -> Unit = {},
    onInpaintCfgScaleChanged: (Float) -> Unit = {},
    onInpaintDepthGuardChanged: (Boolean) -> Unit = {},
    onComputeInpaintDepth: () -> Unit = {},
    onToggleQuickAccessoryRemoval: (AccessoryTarget) -> Unit = {},
    onComputeAccessoryLayers: () -> Unit = {},
    onApplyAccessorySelectionAndExpand: (Set<AccessoryTarget>, Float) -> Unit = { _, _ -> },
    onInpaintMaskAndPromptUpdated: (android.graphics.Bitmap?, String?) -> Unit = { bmp, _ -> onInpaintMaskUpdated(bmp) },
    onRunInpaint: () -> Unit = {},
    onReuseInpaintResult: () -> Unit = {},
    onSaveBitmapToFolder: (android.graphics.Bitmap) -> Unit = {},
    onDownloadBitmapToGallery: (android.graphics.Bitmap) -> Unit = {},
    onOpenInpaintGallery: (android.graphics.Bitmap) -> Unit = {},
    onViewImageFullScreen: (android.graphics.Bitmap, String) -> Unit = { _, _ -> },
    modifier: Modifier = Modifier
) {
    // Front Page: Clean, focused UI matching the user's reference design
    if (uiState.studioSubPage == StudioSubPage.FRONT_HOME) {
        StudioFrontPage(
            uiState = uiState,
            onSelectStudioMode = onSelectStudioMode,
            onPickSourcePhoto = onPickSourcePhoto,
            onPickTargetPhoto = onPickTargetPhoto,
            onDetectFaces = onDetectHead,
            onPreviewAlignmentPage = {
                if (uiState.sourceBitmap != null && uiState.targetBitmap != null) {
                    onPreviewHead()
                }
                onSelectStudioSubPage(StudioSubPage.ALIGNMENT_PREVIEW)
            },
            onOpenSettingsPage = {
                onOpenModelsTab()
            },
            onRunSwap = onRunSwap,
            onDownloadHd = onSaveHdToGallery,
            onSaveToGallery = onSaveToGallery,
            onInspectResultDetails = {
                onSelectStudioSubPage(StudioSubPage.ALIGNMENT_PREVIEW)
            },
            onImportAllFromFolder = onImportAllFromFolder,
            onSkinToneModeChanged = onSkinToneModeChanged,
            onFaceReactionModeChanged = onFaceReactionModeChanged,
            onOpenInpaintPage = {
                onSelectStudioSubPage(StudioSubPage.INPAINT_PAGE)
            },
            onViewResultFullScreen = { bmp ->
                onViewImageFullScreen(bmp, "Swap Result")
            },
            modifier = modifier
        )
        return
    }

    if (uiState.studioSubPage == StudioSubPage.INPAINT_PAGE) {
        InpaintPage(
            uiState = uiState,
            onBack = { onSelectStudioSubPage(StudioSubPage.FRONT_HOME) },
            onPickSourceFromFolder = onPickInpaintSourceFromFolder,
            onSelectModelOption = onSelectInpaintModelOption,
            onImportModelFolder = onImportAllFromFolder,
            onMaskUpdated = onInpaintMaskUpdated,
            onPromptChanged = onInpaintPromptChanged,
            onNegativePromptChanged = onInpaintNegativePromptChanged,
            onResetNegativePrompt = onResetInpaintNegativePrompt,
            onFastModeChanged = onInpaintFastModeChanged,
            onStrengthChanged = onInpaintStrengthChanged,
            onCfgScaleChanged = onInpaintCfgScaleChanged,
            onDepthGuardChanged = onInpaintDepthGuardChanged,
            onComputeInpaintDepth = onComputeInpaintDepth,
            onToggleQuickAccessoryRemoval = onToggleQuickAccessoryRemoval,
            onComputeAccessoryLayers = onComputeAccessoryLayers,
            onApplyAccessorySelectionAndExpand = onApplyAccessorySelectionAndExpand,
            onMaskAndPromptUpdated = onInpaintMaskAndPromptUpdated,
            onCreateInpaint = onRunInpaint,
            onSaveToFolder = onSaveBitmapToFolder,
            onDownloadToGallery = onDownloadBitmapToGallery,
            onReuseResult = onReuseInpaintResult,
            onOpenGalleryView = onOpenInpaintGallery,
            onViewResultFullScreen = { bmp ->
                onViewImageFullScreen(bmp, "Inpaint Result")
            },
            modifier = modifier
        )
        return
    }

    // Secondary Pages ("மற்றவற்றை வேறு page இல் வை"): Skin Tone & Face Reaction Selector / Alignment Preview
    BackHandler {
        onSelectStudioSubPage(StudioSubPage.FRONT_HOME)
    }

    val isHeadMode = uiState.studioMode == StudioMode.HEAD_REPLACEMENT
    val isSettingsPage = uiState.studioSubPage == StudioSubPage.SETTINGS_AND_DIAGNOSTICS

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 640.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Top Secondary Page Navigation Bar: Back to Front Page + Sub-Page Switcher
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFF101932)
                )
            ) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        OutlinedButton(
                            onClick = { onSelectStudioSubPage(StudioSubPage.FRONT_HOME) },
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier.testTag("btn_back_to_front_page")
                        ) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back to Front Page",
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Back to Swap Page")
                        }

                        Text(
                            text = if (isSettingsPage) "Skin Tone & Reaction" else "Alignment & Diagnostics",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = ElectricCyan
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        FilterChip(
                            selected = isSettingsPage,
                            onClick = { onSelectStudioSubPage(StudioSubPage.SETTINGS_AND_DIAGNOSTICS) },
                            label = { Text("Skin Tone & Reaction") },
                            modifier = Modifier
                                .weight(1f)
                                .testTag("subpage_tab_settings")
                        )
                        FilterChip(
                            selected = !isSettingsPage,
                            onClick = { onSelectStudioSubPage(StudioSubPage.ALIGNMENT_PREVIEW) },
                            label = { Text("Alignment & Inspector") },
                            modifier = Modifier
                                .weight(1f)
                                .testTag("subpage_tab_alignment")
                        )
                    }
                }
            }

            if (isSettingsPage) {
                // User selects Skin Tone, Face Reaction, and Negative Artifact / LCM / StyleGAN engines
                SkinToneAndReactionSelectorCard(
                    skinToneMode = uiState.skinToneMode,
                    faceReactionMode = uiState.faceReactionMode,
                    onSkinToneModeChanged = onSkinToneModeChanged,
                    onFaceReactionModeChanged = onFaceReactionModeChanged
                )
                AdvancedNegativeAndStyleSettingsCard(
                    enableNegativeArtifactSuppression = uiState.enableNegativeArtifactSuppression,
                    enableLcmInpainting = uiState.enableLcmInpainting,
                    faceAgingFactor = uiState.faceAgingFactor,
                    onNegativeArtifactSuppressionChanged = onNegativeArtifactSuppressionChanged,
                    onLcmInpaintingChanged = onLcmInpaintingChanged,
                    onFaceAgingFactorChanged = onFaceAgingFactorChanged
                )
            } else {
                // Page B: Alignment Preview, Multi-Face Selection, Interactive Before/After & Visual Validation
                uiState.swapResult?.let {
                    SwapResultCard(
                        uiState = uiState,
                        onToggleCompareOriginal = onToggleCompareOriginal,
                        onSaveToGallery = onSaveToGallery,
                        onSaveHdToGallery = onSaveHdToGallery,
                        onShareResult = onShareResult,
                        onRotateResult90 = onRotateResult90,
                        onCropResultAspect = onCropResultAspect
                    )
                }

                // 1. Source Photo Selection & 5-Point Landmark Overlay Card
                PhotoSelectionCard(
                    title = if (isHeadMode) "SOURCE HEAD" else "1. Source Identity Face & Landmarks",
                    subtitle = if (isHeadMode) {
                        "Head, hair, skull structure & upper neck to transfer"
                    } else {
                        "Select photo containing the donor face identity (112x112 -> 512-D embedding)"
                    },
                    buttonLabel = stringResource(R.string.btn_select_source),
                    buttonTestTag = "select_source_photo_button",
                    browseTestTag = "browse_source_file_button",
                    bitmap = uiState.sourceBitmap,
                    faces = uiState.sourceFaces,
                    selectedFaceIndex = uiState.selectedSourceFaceIndex,
                    replaceAllFaces = false,
                    showCranialHeadBounds = isHeadMode,
                    isDetecting = uiState.isDetectingSource,
                    onPickPhoto = onPickSourcePhoto,
                    onBrowseFile = onBrowseSourceFile,
                    onSelectFace = onSelectSourceFace,
                    showMultiFaceToggle = false,
                    onToggleMultiFace = {}
                )

                // 2. Target Photo Selection & Multi-Face Selector Card
                PhotoSelectionCard(
                    title = if (isHeadMode) "TARGET PHOTO" else "2. Target Scene Photo & Landmarks",
                    subtitle = if (isHeadMode) {
                        "Body, pose, clothing & background to preserve (tap head to replace)"
                    } else {
                        "Select target photo and tap any detected face to replace"
                    },
                    buttonLabel = stringResource(R.string.btn_select_target),
                    buttonTestTag = "select_target_photo_button",
                    browseTestTag = "browse_target_file_button",
                    bitmap = uiState.targetBitmap,
                    faces = uiState.targetFaces,
                    selectedFaceIndex = uiState.selectedTargetFaceIndex,
                    replaceAllFaces = uiState.replaceAllTargetFaces,
                    showCranialHeadBounds = isHeadMode,
                    isDetecting = uiState.isDetectingTarget,
                    onPickPhoto = onPickTargetPhoto,
                    onBrowseFile = onBrowseTargetFile,
                    onSelectFace = onSelectTargetFace,
                    showMultiFaceToggle = uiState.targetFaces.size > 1,
                    onToggleMultiFace = onToggleReplaceAllFaces
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedButton(
                        onClick = onDetectHead,
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .testTag("detect_head_button"),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Radar,
                            contentDescription = stringResource(R.string.btn_detect_head),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(if (isHeadMode) stringResource(R.string.btn_detect_head) else "Detect Faces")
                    }

                    OutlinedButton(
                        onClick = onPreviewHead,
                        enabled = !uiState.isGeneratingHeadPreview,
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .testTag("preview_head_button"),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Preview,
                            contentDescription = stringResource(R.string.btn_preview_head),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            if (uiState.isGeneratingHeadPreview) {
                                "Segmenting..."
                            } else {
                                "Generate Alignment Preview"
                            }
                        )
                    }
                }

                // Live Head / Hair / Neck Segmentation & Cranial Alignment Preview
                uiState.headPreviewState?.let { preview ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surface
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(14.dp),
                            verticalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Text(
                                text = "Head / Hair / Neck Segmentation & Pose Preview",
                                style = MaterialTheme.typography.titleMedium,
                                color = ElectricCyan
                            )
                            Text(
                                text = "Parser: ${preview.segmentationLabel}",
                                style = MaterialTheme.typography.labelMedium,
                                color = NeonEmerald
                            )
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceEvenly
                            ) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Image(
                                        bitmap = preview.segmentedSourceHeadBitmap.asImageBitmap(),
                                        contentDescription = "Segmented Source Head & Hair Mask",
                                        modifier = Modifier
                                            .size(128.dp)
                                            .clip(RoundedCornerShape(12.dp))
                                            .border(
                                                1.5.dp,
                                                ElectricCyan,
                                                RoundedCornerShape(12.dp)
                                            )
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = "Source Matte (${"%.1f".format(preview.sourceRollDeg)}°)",
                                        style = MaterialTheme.typography.labelMedium
                                    )
                                }
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Image(
                                        bitmap = preview.alignedTargetHeadBitmap.asImageBitmap(),
                                        contentDescription = "Aligned Target Head Region",
                                        modifier = Modifier
                                            .size(128.dp)
                                            .clip(RoundedCornerShape(12.dp))
                                            .border(
                                                1.5.dp,
                                                RoyalViolet,
                                                RoundedCornerShape(12.dp)
                                            )
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = "Target Pose (${"%.1f".format(preview.targetRollDeg)}°)",
                                        style = MaterialTheme.typography.labelMedium
                                    )
                                }
                            }
                        }
                    }
                }

                // Real Visual Validation: 7-Panel A/B/C/D Comparison & 3-Stage Upper-Lip Moustache Diagnostic
                VisualValidationAndStageInspectorCard(
                    uiState = uiState,
                    onRunVisualValidation = onRunVisualValidation,
                    onSaveValidationSheet = onSaveValidationSheet
                )
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun StudioModeSwitcherCard(
    currentMode: StudioMode,
    onSelectMode: (StudioMode) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                StudioMode.entries.forEach { mode ->
                    val selected = currentMode == mode
                    val containerColor = if (selected) {
                        when (mode) {
                            StudioMode.HEAD_REPLACEMENT -> NeonEmerald
                            StudioMode.IDENTITY_PRESERVED -> AmberWarning
                            else -> ElectricCyan
                        }
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    }
                    val contentColor = if (selected) {
                        ObsidianBg
                    } else {
                        MaterialTheme.colorScheme.onSurface
                    }

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(containerColor)
                            .clickable { onSelectMode(mode) }
                            .testTag(
                                when (mode) {
                                    StudioMode.FACE_SWAP -> "mode_selector_face_swap"
                                    StudioMode.IDENTITY_PRESERVED -> "mode_selector_identity_preserved"
                                    StudioMode.HEAD_REPLACEMENT -> "mode_selector_head_replacement"
                                }
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = mode.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = contentColor
                        )
                    }
                }
            }
            Text(
                text = currentMode.badge,
                style = MaterialTheme.typography.labelMedium,
                color = when (currentMode) {
                    StudioMode.HEAD_REPLACEMENT -> NeonEmerald
                    StudioMode.IDENTITY_PRESERVED -> AmberWarning
                    else -> ElectricCyan
                }
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun HeroPipelineStatusCard(
    uiState: FaceSwapUiState,
    onOpenModelsTab: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(116.dp)
            ) {
                Image(
                    painter = painterResource(id = R.drawable.img_hero_pipeline),
                    contentDescription = "Biometric face and head alignment banner",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            Brush.verticalGradient(
                                colors = listOf(
                                    ObsidianBg.copy(alpha = 0.25f),
                                    ObsidianBg.copy(alpha = 0.92f)
                                )
                            )
                        )
                )
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(14.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.VerifiedUser,
                            contentDescription = "Offline Air-Gapped Badge",
                            tint = NeonEmerald,
                            modifier = Modifier.size(18.dp)
                        )
                        Text(
                            text = "AIRPLANE-MODE READY • 100% LOCAL ON-DEVICE INFERENCE",
                            style = MaterialTheme.typography.labelMedium,
                            color = NeonEmerald
                        )
                    }
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = if (uiState.studioMode == StudioMode.HEAD_REPLACEMENT) {
                            "Mode 2: Full Head, Hair, Skull & Neck Replacement"
                        } else {
                            "Mode 1: SCRFD-10G + ArcFace + HyperSwap 1b (256px)"
                        },
                        style = MaterialTheme.typography.titleLarge,
                        color = Color.White
                    )
                }
            }

            Column(
                modifier = Modifier.padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    ModelStatusPill(
                        label = "det_10g.onnx",
                        isReady = uiState.isDetectorReady,
                        onClick = onOpenModelsTab
                    )
                    ModelStatusPill(
                        label = "hyperswap_1b_256.onnx",
                        isReady = uiState.isSwapperReady,
                        onClick = onOpenModelsTab
                    )
                    ModelStatusPill(
                        label = "w600k_r50.onnx",
                        isReady = uiState.isRecognizerReady,
                        onClick = onOpenModelsTab
                    )
                    ModelStatusPill(
                        label = "gfpgan_1.4.onnx",
                        isReady = uiState.memoryServiceState.gfpganLoaded ||
                            uiState.modelInspections.any { it.slot == ModelSlot.ENHANCEMENT && it.isValidOnnx },
                        onClick = onOpenModelsTab
                    )
                    if (uiState.studioMode == StudioMode.HEAD_REPLACEMENT) {
                        ModelStatusPill(
                            label = if (uiState.isSegmentationOnnxReady) {
                                "segformer_B5_ce.onnx"
                            } else {
                                "Head/Hair Matte: Built-In CV"
                            },
                            isReady = true,
                            onClick = onOpenModelsTab
                        )
                    }
                }

                // Interactive Tamil In-Memory Model Status & Purpose Panel
                var showTamilModelGuide by remember { mutableStateOf(true) }
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(ObsidianBg)
                        .border(1.dp, ElectricCyan.copy(alpha = 0.35f), RoundedCornerShape(12.dp))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { showTamilModelGuide = !showTamilModelGuide }
                            .testTag("toggle_tamil_model_guide"),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Memory,
                                contentDescription = "RAM Models",
                                tint = NeonEmerald,
                                modifier = Modifier.size(18.dp)
                            )
                            Text(
                                text = "மெமரியில் (RAM) உள்ள மாடல்கள் (${uiState.memoryServiceState.activeSessionCount}/5) & தமிழ் விளக்கம்",
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.Bold,
                                color = NeonEmerald
                            )
                        }
                        Text(
                            text = if (showTamilModelGuide) "மறை ▲" else "விளக்கம் ▼",
                            style = MaterialTheme.typography.labelSmall,
                            color = ElectricCyan,
                            fontWeight = FontWeight.Bold
                        )
                    }

                    AnimatedVisibility(visible = showTamilModelGuide) {
                        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            StudioTamilModelRow(
                                fileName = "det_10g.onnx",
                                isInRam = uiState.memoryServiceState.detectorLoaded,
                                tamilRole = "முகத்தைக் கண்டறிய: படத்தில் முகம் எங்கே உள்ளது மற்றும் கண், மூக்கு, வாய் (5 புள்ளிகள்) ஆகியவற்றைக் துல்லியமாகக் கண்டறிய."
                            )
                            StudioTamilModelRow(
                                fileName = "w600k_r50.onnx",
                                isInRam = uiState.memoryServiceState.w600kLoaded,
                                tamilRole = "முக அடையாளம் (512-D Identity): மாற்றுவதற்கான (Source) முகத்தின் தனித்துவமான 512-பரிமாண அடையாளத்தைப் பிரித்தெடுக்க."
                            )
                            StudioTamilModelRow(
                                fileName = "hyperswap_1b_256.onnx",
                                isInRam = uiState.memoryServiceState.inswapperLoaded,
                                tamilRole = "முகம் மாற்றம் (Primary 256×256 Face Swap): Target முகத்தின் பாவனை மற்றும் ஒளியை மாற்றாமல் Source முகத்தைப் பொருத்த."
                            )
                            StudioTamilModelRow(
                                fileName = "gfpgan_1.4.onnx",
                                isInRam = uiState.memoryServiceState.gfpganLoaded,
                                tamilRole = "512×512 HD மெருகூட்டல் (Face Restoration): முகம் மாற்றிய பிறகு மங்கலாக இல்லாமல் கண்கள், புருவம் மற்றும் தோலைத் தெளிவாக்க."
                            )
                            StudioTamilModelRow(
                                fileName = "segformer_B5_ce.onnx",
                                isInRam = uiState.memoryServiceState.segformerLoaded,
                                tamilRole = "தலை & முடி பிரிப்பு (Head/Hair Parser): Mode 2-ல் தலை, முடி, காது மற்றும் கழுத்தைத் துல்லியமாகப் பிரித்து முழு தலையையும் மாற்ற."
                            )
                        }
                    }
                }

                if (!uiState.isDetectorReady || !uiState.isSwapperReady) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(MaterialTheme.colorScheme.surfaceVariant)
                            .clickable(onClick = onOpenModelsTab)
                            .padding(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Memory,
                            contentDescription = "Model Status",
                            tint = AmberWarning,
                            modifier = Modifier.size(20.dp)
                        )
                        Text(
                            text = "Tap to open ONNX Models tab to import .onnx files or view full Tamil guide.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StudioTamilModelRow(
    fileName: String,
    isInRam: Boolean,
    tamilRole: String
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.6f))
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = fileName,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = ElectricCyan
            )
            Text(
                text = if (isInRam) "RAM-ல் உள்ளது ●" else "டிஸ்க்கில் / இல்லை ○",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Bold,
                color = if (isInRam) NeonEmerald else AmberWarning
            )
        }
        Text(
            text = tamilRole,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
}

@Composable
private fun ModelStatusPill(
    label: String,
    isReady: Boolean,
    onClick: () -> Unit
) {
    val tint = if (isReady) NeonEmerald else AmberWarning
    Surface(
        modifier = Modifier.clickable(onClick = onClick),
        shape = RoundedCornerShape(50),
        color = tint.copy(alpha = 0.14f)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                imageVector = if (isReady) Icons.Default.CheckCircle else Icons.Default.WarningAmber,
                contentDescription = label,
                tint = tint,
                modifier = Modifier.size(15.dp)
            )
            Text(
                text = if (isReady) "✓ $label" else "✗ $label",
                style = MaterialTheme.typography.labelMedium,
                color = tint
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PhotoSelectionCard(
    title: String,
    subtitle: String,
    buttonLabel: String,
    buttonTestTag: String,
    browseTestTag: String,
    bitmap: android.graphics.Bitmap?,
    faces: List<DetectedFace>,
    selectedFaceIndex: Int,
    replaceAllFaces: Boolean,
    showCranialHeadBounds: Boolean,
    isDetecting: Boolean,
    onPickPhoto: () -> Unit,
    onBrowseFile: () -> Unit,
    onSelectFace: (Int) -> Unit,
    showMultiFaceToggle: Boolean,
    onToggleMultiFace: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    OutlinedButton(
                        onClick = onPickPhoto,
                        modifier = Modifier
                            .weight(1f)
                            .testTag(buttonTestTag),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AddPhotoAlternate,
                            contentDescription = buttonLabel,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = buttonLabel)
                    }
                    OutlinedButton(
                        onClick = onBrowseFile,
                        modifier = Modifier
                            .weight(1f)
                            .testTag(browseTestTag),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AddPhotoAlternate,
                            contentDescription = "Browse Files",
                            modifier = Modifier.size(18.dp),
                            tint = NeonEmerald
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = "Browse Files", color = NeonEmerald)
                    }
                }
            }

            if (isDetecting) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(ObsidianBg),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = ElectricCyan)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Detecting Face, 106-Point 3D Landmarks & 3D Head-Pose...",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            } else if (bitmap != null) {
                FaceDetectionCanvas(
                    bitmap = bitmap,
                    faces = faces,
                    selectedFaceIndex = selectedFaceIndex,
                    highlightAllFaces = replaceAllFaces,
                    showCranialHeadBounds = showCranialHeadBounds,
                    onFaceTapped = onSelectFace,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(250.dp)
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Detected: ${faces.size} (${bitmap.width}x${bitmap.height}px)",
                        style = MaterialTheme.typography.labelMedium,
                        color = if (faces.isNotEmpty()) NeonEmerald else AmberWarning
                    )
                    faces.firstOrNull()?.let { first ->
                        Text(
                            text = "${first.landmarks106.size}-Pt 3DMM • ${first.detectorSource}",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                val activeFace = faces.getOrNull(selectedFaceIndex) ?: faces.firstOrNull()
                if (activeFace != null) {
                    val pose = activeFace.headPose3D
                    val anatCount = activeFace.reconstructed3D.anatomicalPoints34.size
                    Text(
                        text = "3D Head-Pose: ${pose.poseSummaryLabel} • ${anatCount}-Pt 3D + 106-Pt Mesh • VAE+GAN Fusion (Vis L:${(pose.leftSideVisibility * 100).toInt()}% R:${(pose.rightSideVisibility * 100).toInt()}%)",
                        style = MaterialTheme.typography.labelSmall,
                        color = ElectricCyan
                    )
                }

                if (faces.size > 1) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(ObsidianBg)
                            .border(1.dp, ElectricCyan.copy(alpha = 0.4f), RoundedCornerShape(12.dp))
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Text(
                            text = if (showMultiFaceToggle) {
                                "பல நபர்கள் கண்டறியப்பட்டனர் (${faces.size} பேர்) — யாரை மாற்ற வேண்டும்?"
                            } else {
                                "Source படத்தில் ${faces.size} முகங்கள் உள்ளன — எந்த முகத்தை எடுக்க வேண்டும்?"
                            },
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.Bold,
                            color = NeonEmerald
                        )

                        if (showMultiFaceToggle) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                val singleSelected = !replaceAllFaces
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(
                                            if (singleSelected) NeonEmerald else MaterialTheme.colorScheme.surfaceVariant
                                        )
                                        .clickable { onToggleMultiFace(false) }
                                        .padding(vertical = 10.dp, horizontal = 8.dp)
                                        .testTag("single_person_target_mode_button"),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text(
                                        text = "✓ குறிப்பிட்ட 1 நபர் மட்டும் (#${selectedFaceIndex + 1})",
                                        style = MaterialTheme.typography.labelMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = if (singleSelected) ObsidianBg else MaterialTheme.colorScheme.onSurface
                                    )
                                }

                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(
                                            if (replaceAllFaces) ElectricCyan else MaterialTheme.colorScheme.surfaceVariant
                                        )
                                        .clickable { onToggleMultiFace(true) }
                                        .padding(vertical = 10.dp, horizontal = 8.dp)
                                        .testTag("all_persons_target_mode_button"),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text(
                                        text = "அனைவர் முகமும் (${faces.size} பேர்)",
                                        style = MaterialTheme.typography.labelMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = if (replaceAllFaces) ObsidianBg else MaterialTheme.colorScheme.onSurface
                                    )
                                }
                            }

                            Text(
                                text = if (!replaceAllFaces) {
                                    "கீழே உள்ள படங்களில் எந்த நபரின் முகத்தை மாற்ற வேண்டுமோ அவரைத் தொட்டுத் தேர்ந்தெடுக்கவும் (தற்போது: நபர் #${selectedFaceIndex + 1} மட்டும் மாற்றப்படும்):"
                                } else {
                                    "தற்போது படத்தில் உள்ள ${faces.size} நபர்களின் முகங்களும் வரிசையாக மாற்றப்படும் (குறிப்பிட்ட ஒருவரை மட்டும் மாற்ற அவருடைய படத்தைத் தொடவும்):"
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        FlowRow(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            faces.forEach { face ->
                                val isPersonActive = replaceAllFaces || (face.index == selectedFaceIndex)
                                DetectedPersonThumbCard(
                                    fullBitmap = bitmap,
                                    face = face,
                                    isSelected = isPersonActive,
                                    isTargetCard = showMultiFaceToggle,
                                    onClick = {
                                        if (showMultiFaceToggle) {
                                            onToggleMultiFace(false)
                                        }
                                        onSelectFace(face.index)
                                    }
                                )
                            }
                        }
                    }
                }
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(140.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(ObsidianBg)
                        .border(
                            1.dp,
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(12.dp)
                        )
                        .clickable(onClick = onPickPhoto),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.FaceRetouchingNatural,
                            contentDescription = buttonLabel,
                            tint = ElectricCyan,
                            modifier = Modifier.size(34.dp)
                        )
                        Text(
                            text = "Tap to select photo from Android Gallery",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun SafeguardsAndBlendingCard(
    uiState: FaceSwapUiState,
    onConsentChanged: (Boolean) -> Unit,
    onColorTransferChanged: (Boolean) -> Unit,
    onWatermarkChanged: (Boolean) -> Unit,
    onTwoModelFallbackChanged: (Boolean) -> Unit,
    onHardwareAccelChanged: (Boolean) -> Unit,
    onLowMemoryChanged: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = "Hardware Acceleration, Memory & Ethical Safeguards",
                style = MaterialTheme.typography.titleMedium
            )

            if (!uiState.isRecognizerReady && uiState.studioMode == StudioMode.FACE_SWAP) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(AmberWarning.copy(alpha = 0.14f))
                        .border(1.dp, AmberWarning.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "Missing w600k_r50.onnx (512-D ArcFace Recognizer)",
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Bold,
                        color = AmberWarning
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Enable 2-Model Testing Mode (when w600k_r50.onnx is absent)",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.weight(1f)
                        )
                        Switch(
                            checked = uiState.allowTwoModelFallbackForTesting,
                            onCheckedChange = onTwoModelFallbackChanged,
                            modifier = Modifier.testTag("two_model_fallback_switch")
                        )
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Android NNAPI / GPU Acceleration (with CPU Fallback)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Uses hardware delegate where supported and falls back to multi-thread CPU",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = uiState.preferHardwareAcceleration,
                    onCheckedChange = onHardwareAccelChanged
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Low-Memory Tiled Optimization Mode",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Sequential model unloading + 256x256 head tiles for low-RAM devices",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = uiState.lowMemoryMode,
                    onCheckedChange = onLowMemoryChanged
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = uiState.enableColorTransfer,
                    onCheckedChange = onColorTransferChanged
                )
                Text(
                    text = stringResource(R.string.color_transfer_checkbox_label),
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = uiState.enableProvenanceWatermark,
                    onCheckedChange = onWatermarkChanged
                )
                Text(
                    text = stringResource(R.string.watermark_checkbox_label),
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .padding(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Checkbox(
                    checked = uiState.consentConfirmed,
                    onCheckedChange = onConsentChanged,
                    modifier = Modifier.testTag("consent_checkbox")
                )
                Text(
                    text = stringResource(R.string.consent_checkbox_label),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SwapResultCard(
    uiState: FaceSwapUiState,
    onToggleCompareOriginal: (Boolean) -> Unit,
    onSaveToGallery: () -> Unit,
    onSaveHdToGallery: () -> Unit = onSaveToGallery,
    onShareResult: () -> Unit = {},
    onRotateResult90: () -> Unit = {},
    onCropResultAspect: (Int, Int) -> Unit = { _, _ -> }
) {
    val result = uiState.swapResult ?: return
    val displayBitmap = if (uiState.showOriginalInComparison && uiState.targetBitmap != null) {
        uiState.targetBitmap
    } else {
        result.outputBitmap
    }

    var zoomScale by remember { mutableFloatStateOf(1.0f) }
    var panOffset by remember { mutableStateOf(Offset.Zero) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (uiState.showOriginalInComparison) {
                            "BEFORE (Original Target)"
                        } else {
                            "AFTER (${uiState.studioMode.title} Result)"
                        },
                        style = MaterialTheme.typography.titleLarge,
                        color = ElectricCyan
                    )
                    Text(
                        text = "Latency: ${result.totalMs} ms • ${result.swappedFacesCount} Replaced • ${displayBitmap.width}×${displayBitmap.height}",
                        style = MaterialTheme.typography.labelMedium,
                        color = NeonEmerald
                    )
                }

                OutlinedButton(
                    onClick = { onToggleCompareOriginal(!uiState.showOriginalInComparison) },
                    modifier = Modifier.testTag("compare_before_after_button")
                ) {
                    Icon(
                        imageVector = Icons.Default.Compare,
                        contentDescription = stringResource(R.string.btn_before_after),
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("BEFORE | AFTER")
                }
            }

            // Interactive Before | After & Multi-Stage Split-Screen Comparison
            var splitPosition by remember { mutableFloatStateOf(0.50f) }
            var selectedSplitModeIndex by remember { mutableStateOf(0) }
            val splitModeOptions = listOf(
                "Final vs Target",
                "Stage 2 (Restore) vs Stage 1 (Swap-Only)",
                "Stage 3 (Final) vs Stage 2 (Restore)",
                "Stage 3 (512px) vs Target Crop"
            )

            if (result.stage1SwapOnly512 != null && result.stage2SwapRestore512 != null && result.stage3FinalBlend512 != null) {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    splitModeOptions.forEachIndexed { idx, label ->
                        FilterChip(
                            selected = selectedSplitModeIndex == idx,
                            onClick = {
                                selectedSplitModeIndex = idx
                                if (splitPosition > 0.95f || splitPosition < 0.05f) {
                                    splitPosition = 0.50f
                                }
                            },
                            label = {
                                Text(
                                    text = label,
                                    style = MaterialTheme.typography.labelSmall
                                )
                            },
                            modifier = Modifier.testTag("split_stage_mode_$idx")
                        )
                    }
                }
            }

            val splitComparisonPreview = remember(
                result.outputBitmap,
                result.stage1SwapOnly512,
                result.stage2SwapRestore512,
                result.stage3FinalBlend512,
                result.alignedTarget128,
                uiState.targetBitmap,
                splitPosition,
                selectedSplitModeIndex,
                uiState.showOriginalInComparison
            ) {
                if (uiState.showOriginalInComparison && uiState.targetBitmap != null) {
                    uiState.targetBitmap
                } else {
                    val leftBmp: android.graphics.Bitmap
                    val rightBmp: android.graphics.Bitmap?
                    when (selectedSplitModeIndex) {
                        1 -> {
                            leftBmp = result.stage2SwapRestore512 ?: result.outputBitmap
                            rightBmp = result.stage1SwapOnly512
                        }
                        2 -> {
                            leftBmp = result.stage3FinalBlend512 ?: result.outputBitmap
                            rightBmp = result.stage2SwapRestore512
                        }
                        3 -> {
                            leftBmp = result.stage3FinalBlend512 ?: result.outputBitmap
                            rightBmp = result.alignedTarget128.let { crop ->
                                if (crop.width != leftBmp.width || crop.height != leftBmp.height) {
                                    android.graphics.Bitmap.createScaledBitmap(crop, leftBmp.width, leftBmp.height, true)
                                } else crop
                            }
                        }
                        else -> {
                            leftBmp = result.outputBitmap
                            rightBmp = uiState.targetBitmap
                        }
                    }

                    if (splitPosition in 0.02f..0.98f && rightBmp != null &&
                        rightBmp.width == leftBmp.width && rightBmp.height == leftBmp.height
                    ) {
                        val w = leftBmp.width
                        val h = leftBmp.height
                        val splitX = (w * splitPosition).toInt().coerceIn(1, w - 1)
                        val combined = leftBmp.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
                        val origSlice = IntArray((w - splitX) * h)
                        rightBmp.getPixels(origSlice, 0, w - splitX, splitX, 0, w - splitX, h)
                        combined.setPixels(origSlice, 0, w - splitX, splitX, 0, w - splitX, h)
                        val canvas = android.graphics.Canvas(combined)
                        val linePaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
                            color = android.graphics.Color.rgb(0, 229, 255)
                            strokeWidth = (w / 240f).coerceAtLeast(3f)
                        }
                        canvas.drawLine(splitX.toFloat(), 0f, splitX.toFloat(), h.toFloat(), linePaint)
                        combined
                    } else {
                        leftBmp
                    }
                }
            }

            // Zoomable & Pannable Result Canvas
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(340.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(ObsidianBg)
                    .pointerInput(Unit) {
                        detectTransformGestures { _, pan, zoom, _ ->
                            val newScale = (zoomScale * zoom).coerceIn(1.0f, 4.0f)
                            zoomScale = newScale
                            panOffset = if (newScale > 1.02f) {
                                Offset(
                                    x = (panOffset.x + pan.x).coerceIn(-320f, 320f),
                                    y = (panOffset.y + pan.y).coerceIn(-320f, 320f)
                                )
                            } else {
                                Offset.Zero
                            }
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                Image(
                    bitmap = splitComparisonPreview.asImageBitmap(),
                    contentDescription = "Zoomable Output Preview",
                    modifier = Modifier
                        .fillMaxSize()
                        .graphicsLayer(
                            scaleX = zoomScale,
                            scaleY = zoomScale,
                            translationX = panOffset.x,
                            translationY = panOffset.y
                        ),
                    contentScale = ContentScale.Fit
                )

                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(8.dp),
                    shape = RoundedCornerShape(8.dp),
                    color = ObsidianBg.copy(alpha = 0.78f)
                ) {
                    Text(
                        text = "Zoom: ${"%.1f".format(zoomScale)}x (Pinch or tap below)",
                        style = MaterialTheme.typography.labelSmall,
                        color = ElectricCyan,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            // Interactive Before | After Split-Screen Wipe Slider
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "AFTER | BEFORE Split Slider (${splitModeOptions.getOrElse(selectedSplitModeIndex) { "Final vs Target" }})",
                        style = MaterialTheme.typography.labelSmall,
                        color = NeonEmerald,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "${(splitPosition * 100).toInt()}% Left",
                        style = MaterialTheme.typography.labelSmall,
                        color = ElectricCyan
                    )
                }
                Slider(
                    value = splitPosition,
                    onValueChange = { splitPosition = it },
                    valueRange = 0.0f..1.0f,
                    modifier = Modifier.testTag("before_after_split_slider")
                )
            }

            // Zoom, Rotate & Crop Interactive Toolbar
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FilterChip(
                    selected = zoomScale > 1.05f,
                    onClick = {
                        if (zoomScale < 1.9f) {
                            zoomScale = 2.0f
                        } else if (zoomScale < 2.9f) {
                            zoomScale = 3.0f
                        } else {
                            zoomScale = 1.0f
                            panOffset = Offset.Zero
                        }
                    },
                    label = { Text("Zoom (${"%.0f".format(zoomScale)}x)") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.ZoomIn,
                            contentDescription = "Zoom Result",
                            modifier = Modifier.size(16.dp)
                        )
                    },
                    modifier = Modifier.testTag("result_zoom_button")
                )

                FilterChip(
                    selected = false,
                    onClick = {
                        zoomScale = 1.0f
                        panOffset = Offset.Zero
                        onRotateResult90()
                    },
                    label = { Text("Rotate 90°") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.RotateRight,
                            contentDescription = "Rotate 90 degrees",
                            modifier = Modifier.size(16.dp)
                        )
                    },
                    modifier = Modifier.testTag("result_rotate_button")
                )

                FilterChip(
                    selected = false,
                    onClick = { onCropResultAspect(1, 1) },
                    label = { Text("Crop 1:1") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.Crop,
                            contentDescription = "Crop 1:1 Square",
                            modifier = Modifier.size(16.dp)
                        )
                    },
                    modifier = Modifier.testTag("result_crop_1_1_button")
                )

                FilterChip(
                    selected = false,
                    onClick = { onCropResultAspect(4, 5) },
                    label = { Text("Crop 4:5") },
                    modifier = Modifier.testTag("result_crop_4_5_button")
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                AlignedCropThumb(bitmap = result.alignedSource112, label = "Source Crop")
                AlignedCropThumb(bitmap = result.alignedTarget128, label = "Target Crop")
                result.stage1SwapOnly512?.let {
                    AlignedCropThumb(bitmap = it, label = "1. Swap-Only")
                }
                result.stage2SwapRestore512?.let {
                    AlignedCropThumb(bitmap = it, label = "2. Restore")
                }
                AlignedCropThumb(bitmap = result.rawSwapped128, label = "3. Final 512")
            }

            // 10-Stage Production Timing Breakdown Panel
            val st = result.stageTimings
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(ObsidianBg)
                    .padding(10.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "Production 10-Stage Latency Breakdown (Total: ${result.totalMs} ms)",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = NeonEmerald
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("1. Model Load: ${st.modelLoadingMs}ms", style = MaterialTheme.typography.labelSmall, color = ElectricCyan)
                    Text("2. Face Det: ${st.faceDetectionMs}ms", style = MaterialTheme.typography.labelSmall, color = ElectricCyan)
                    Text("3. Landmarks: ${st.landmarkDetectionMs}ms", style = MaterialTheme.typography.labelSmall, color = ElectricCyan)
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("4. ArcFace: ${st.arcFaceEmbeddingMs}ms", style = MaterialTheme.typography.labelSmall, color = RoyalViolet)
                    Text("5. HyperSwap 1b: ${st.hyperSwapInferenceMs}ms", style = MaterialTheme.typography.labelSmall, color = RoyalViolet)
                    Text("6. 512 Restore: ${st.restoration512Ms}ms", style = MaterialTheme.typography.labelSmall, color = NeonEmerald)
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("7. Lip Guard: ${st.upperLipGuardMs}ms", style = MaterialTheme.typography.labelSmall, color = NeonEmerald)
                    Text("8. Mask Gen: ${st.maskGenerationMs}ms", style = MaterialTheme.typography.labelSmall, color = NeonEmerald)
                    Text("9. Blend: ${st.finalBlendingMs}ms", style = MaterialTheme.typography.labelSmall, color = NeonEmerald)
                    Text("10. Export: ${st.imageEncodingExportMs}ms", style = MaterialTheme.typography.labelSmall, color = ElectricCyan)
                }
            }

            // [ Save HD ] | [ Save ] | [ Share ] Action Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = onSaveHdToGallery,
                    modifier = Modifier
                        .weight(1.15f)
                        .height(50.dp)
                        .testTag("save_hd_button"),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = NeonEmerald,
                        contentColor = ObsidianBg
                    )
                ) {
                    Icon(
                        imageVector = Icons.Default.SaveAlt,
                        contentDescription = "Save HD",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Save HD",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold
                    )
                }

                OutlinedButton(
                    onClick = onSaveToGallery,
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp)
                        .testTag("save_to_gallery_button"),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = "Save",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = ElectricCyan
                    )
                }

                OutlinedButton(
                    onClick = onShareResult,
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp)
                        .testTag("share_result_button"),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Share,
                        contentDescription = "Share Result",
                        modifier = Modifier.size(16.dp),
                        tint = ElectricCyan
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Share",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = ElectricCyan
                    )
                }
            }
        }
    }
}

@Composable
private fun AlignedCropThumb(
    bitmap: android.graphics.Bitmap,
    label: String
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = label,
            modifier = Modifier
                .size(76.dp)
                .clip(RoundedCornerShape(10.dp))
                .border(1.dp, ElectricCyan.copy(alpha = 0.5f), RoundedCornerShape(10.dp))
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium
        )
    }
}

@Composable
private fun SwapRequirementsCard(
    uiState: FaceSwapUiState,
    onOpenModelsTab: () -> Unit
) {
    val isHeadMode = uiState.studioMode == StudioMode.HEAD_REPLACEMENT

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.65f)
        )
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.WarningAmber,
                    contentDescription = null,
                    tint = AmberWarning,
                    modifier = Modifier.size(18.dp)
                )
                Text(
                    text = if (isHeadMode) {
                        "Requirements to enable Head Replacement:"
                    } else {
                        "Requirements to enable Face Swap button:"
                    },
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = ElectricCyan
                )
            }

            // 1. Source Face requirement
            val hasSourceFace = uiState.sourceBitmap != null && uiState.sourceFaces.isNotEmpty()
            RequirementRowItem(
                isSatisfied = hasSourceFace,
                text = if (uiState.sourceBitmap == null) {
                    "1. Select Source Photo (Donor Face)"
                } else if (uiState.sourceFaces.isEmpty()) {
                    "1. No face detected in Source Photo (choose a clearer photo)"
                } else {
                    "1. Source face detected (${uiState.sourceFaces.size} face found)"
                }
            )

            // 2. Target Face requirement
            val hasTargetFace = uiState.targetBitmap != null && uiState.targetFaces.isNotEmpty()
            RequirementRowItem(
                isSatisfied = hasTargetFace,
                text = if (uiState.targetBitmap == null) {
                    "2. Select Target Photo (Scene Photo)"
                } else if (uiState.targetFaces.isEmpty()) {
                    "2. No face detected in Target Photo (choose a clearer photo)"
                } else {
                    "2. Target face detected (${uiState.targetFaces.size} face found)"
                }
            )

            // 3. Models requirement (Face Swap mode requires det_10g, hyperswap_1b_256, and w600k_r50/fallback)
            if (!isHeadMode) {
                RequirementRowItem(
                    isSatisfied = uiState.isDetectorReady,
                    text = "det_10g.onnx installed",
                    actionLabel = if (!uiState.isDetectorReady) "Install" else null,
                    onAction = onOpenModelsTab
                )
                RequirementRowItem(
                    isSatisfied = uiState.isSwapperReady,
                    text = "hyperswap_1b_256.onnx installed",
                    actionLabel = if (!uiState.isSwapperReady) "Install" else null,
                    onAction = onOpenModelsTab
                )
                val recognizerReady = uiState.isRecognizerReady || uiState.allowTwoModelFallbackForTesting
                RequirementRowItem(
                    isSatisfied = recognizerReady,
                    text = if (uiState.isRecognizerReady) {
                        "w600k_r50.onnx installed"
                    } else if (uiState.allowTwoModelFallbackForTesting) {
                        "2-Model Testing Mode enabled (w600k_r50 bypassed)"
                    } else {
                        "w600k_r50.onnx installed (or enable 2-Model Testing switch above)"
                    },
                    actionLabel = if (!recognizerReady) "Install" else null,
                    onAction = onOpenModelsTab
                )
            }

            // 4. Consent requirement
            RequirementRowItem(
                isSatisfied = uiState.consentConfirmed,
                text = "Explicit consent confirmed (tick checkbox above)"
            )
        }
    }
}

@Composable
private fun RequirementRowItem(
    isSatisfied: Boolean,
    text: String,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(
            modifier = Modifier.weight(1f),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Icon(
                imageVector = if (isSatisfied) Icons.Default.CheckCircle else Icons.Default.WarningAmber,
                contentDescription = null,
                tint = if (isSatisfied) NeonEmerald else AmberWarning,
                modifier = Modifier.size(16.dp)
            )
            Text(
                text = text,
                style = MaterialTheme.typography.bodyMedium,
                color = if (isSatisfied) MaterialTheme.colorScheme.onSurface else AmberWarning
            )
        }
        if (actionLabel != null && onAction != null) {
            Surface(
                modifier = Modifier.clickable(onClick = onAction),
                shape = RoundedCornerShape(6.dp),
                color = ElectricCyan.copy(alpha = 0.16f)
            ) {
                Text(
                    text = actionLabel,
                    style = MaterialTheme.typography.labelSmall,
                    color = ElectricCyan,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
private fun DetectedPersonThumbCard(
    fullBitmap: android.graphics.Bitmap,
    face: DetectedFace,
    isSelected: Boolean,
    isTargetCard: Boolean,
    onClick: () -> Unit
) {
    val faceThumb = remember(fullBitmap, face.index, face.boundingBox) {
        val padW = (face.boundingBox.width() * 0.25f).toInt()
        val padH = (face.boundingBox.height() * 0.25f).toInt()
        val x = (face.boundingBox.left.toInt() - padW).coerceIn(0, (fullBitmap.width - 1).coerceAtLeast(0))
        val y = (face.boundingBox.top.toInt() - padH).coerceIn(0, (fullBitmap.height - 1).coerceAtLeast(0))
        val w = (face.boundingBox.width().toInt() + padW * 2).coerceIn(1, (fullBitmap.width - x).coerceAtLeast(1))
        val h = (face.boundingBox.height().toInt() + padH * 2).coerceIn(1, (fullBitmap.height - y).coerceAtLeast(1))
        runCatching {
            android.graphics.Bitmap.createBitmap(fullBitmap, x, y, w, h)
        }.getOrElse { fullBitmap }
    }

    val borderColor = if (isSelected) NeonEmerald else RoyalViolet.copy(alpha = 0.55f)
    val bgColor = if (isSelected) NeonEmerald.copy(alpha = 0.16f) else MaterialTheme.colorScheme.surface

    Column(
        modifier = Modifier
            .width(96.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(bgColor)
            .border(if (isSelected) 2.dp else 1.dp, borderColor, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .padding(8.dp)
            .testTag("select_person_${face.index}_card"),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Image(
            bitmap = faceThumb.asImageBitmap(),
            contentDescription = "Person #${face.index + 1}",
            modifier = Modifier
                .size(64.dp)
                .clip(RoundedCornerShape(10.dp))
                .border(1.dp, borderColor, RoundedCornerShape(10.dp)),
            contentScale = ContentScale.Crop
        )
        Text(
            text = "நபர் #${face.index + 1}",
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            color = if (isSelected) NeonEmerald else MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = if (isTargetCard) {
                if (isSelected) "✓ மாற்றப்படும்" else "மாற்றப்படாது"
            } else {
                if (isSelected) "✓ தேர்ந்தெடுக்கப்பட்டது" else "தேர்ந்தெடு"
            },
            style = MaterialTheme.typography.labelSmall,
            color = if (isSelected) NeonEmerald else MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
        )
    }
}

@Composable
private fun SkinToneAndReactionSelectorCard(
    skinToneMode: SkinToneSourceMode,
    faceReactionMode: FaceReactionSourceMode,
    onSkinToneModeChanged: (SkinToneSourceMode) -> Unit,
    onFaceReactionModeChanged: (FaceReactionSourceMode) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // Section 1: Skin Tone Selection (Source vs Target vs 50/50)
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = "1. தோல் நிறம் தேர்வு (Skin Tone Source Selection)",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = ElectricCyan
                )
                Text(
                    text = "முகம் மாற்றும்போது யாருடைய தோல் நிறம் (Skin Tone) வர வேண்டும் என்பதைத் தேர்ந்தெடுக்கவும்:",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                SkinToneSourceMode.entries.forEach { mode ->
                    val selected = skinToneMode == mode
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(
                                if (selected) ElectricCyan.copy(alpha = 0.14f) else ObsidianBg
                            )
                            .border(
                                width = if (selected) 1.8.dp else 1.dp,
                                color = if (selected) ElectricCyan else MaterialTheme.colorScheme.surfaceVariant,
                                shape = RoundedCornerShape(12.dp)
                            )
                            .clickable { onSkinToneModeChanged(mode) }
                            .padding(12.dp)
                            .testTag("skin_tone_mode_${mode.name.lowercase()}"),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = mode.tamilTitle,
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = FontWeight.Bold,
                                color = if (selected) NeonEmerald else MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = if (selected) "✓ SELECTED" else "SELECT",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = if (selected) NeonEmerald else ElectricCyan
                            )
                        }
                        Text(
                            text = mode.tamilSubtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // Section 2: Face Reaction Selection (Smile, Visible Teeth, Visible Tongue)
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = "2. முகபாவனை தேர்வு (Face Reaction: Smile, பற்கள், நாக்கு)",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = NeonEmerald
                )
                Text(
                    text = "சிரிப்பு (Smile), தெரியும் பற்கள் (Teeth), நாக்கு (Tongue Visible) மற்றும் கண் பாவனை யாருடைய படத்திலிருந்து வர வேண்டும்?",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                FaceReactionSourceMode.entries.forEach { mode ->
                    val selected = faceReactionMode == mode
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(
                                if (selected) NeonEmerald.copy(alpha = 0.14f) else ObsidianBg
                            )
                            .border(
                                width = if (selected) 1.8.dp else 1.dp,
                                color = if (selected) NeonEmerald else MaterialTheme.colorScheme.surfaceVariant,
                                shape = RoundedCornerShape(12.dp)
                            )
                            .clickable { onFaceReactionModeChanged(mode) }
                            .padding(12.dp)
                            .testTag("face_reaction_mode_${mode.name.lowercase()}"),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = mode.tamilTitle,
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = FontWeight.Bold,
                                color = if (selected) NeonEmerald else MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = if (selected) "✓ SELECTED" else "SELECT",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = if (selected) NeonEmerald else ElectricCyan
                            )
                        }
                        Text(
                            text = mode.tamilSubtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun QualityAndFineTuneControlsCard(
    uiState: FaceSwapUiState,
    onQualityLevelChanged: (ProcessingQualityLevel) -> Unit,
    onOutputResolutionChanged: (OutputResolutionOption) -> Unit,
    onBlendStrengthChanged: (Float) -> Unit,
    onEnhancementStrengthChanged: (Float) -> Unit,
    onOcclusionProtectionChanged: (Boolean) -> Unit,
    onPortraitBlurStrengthChanged: (Float) -> Unit,
    onFaceOffsetXChanged: (Float) -> Unit,
    onFaceOffsetYChanged: (Float) -> Unit,
    onFaceScaleChanged: (Float) -> Unit,
    onResetAdjustments: () -> Unit
) {
    var expandPositionControls by remember { mutableStateOf(false) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // 1. Quality Level Selector (FAST / BALANCED / HIGH QUALITY) + RAM Telemetry
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Quality Level & Auto-RAM Resolution",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = ElectricCyan
                    )
                    Text(
                        text = "RAM: ${"%.1f".format(uiState.deviceTotalRamGb)}GB (${uiState.deviceAvailRamMb}MB Free)",
                        style = MaterialTheme.typography.labelSmall,
                        color = NeonEmerald,
                        fontWeight = FontWeight.Bold
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    ProcessingQualityLevel.entries.forEach { level ->
                        val selected = uiState.qualityLevel == level
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(42.dp)
                                .clip(RoundedCornerShape(10.dp))
                                .background(
                                    if (selected) ElectricCyan else MaterialTheme.colorScheme.surfaceVariant
                                )
                                .clickable { onQualityLevelChanged(level) }
                                .testTag("quality_level_${level.name.lowercase()}"),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = level.title,
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.Bold,
                                color = if (selected) ObsidianBg else MaterialTheme.colorScheme.onSurface
                            )
                        }
                    }
                }

                Text(
                    text = uiState.qualityLevel.subtitle,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // 2. Blend Strength, GFPGAN Enhancement & DSLR Portrait Bokeh Blur Sliders
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Blend Strength (Multi-Band Boundary Fusion)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "${(uiState.blendStrength * 100).toInt()}%",
                        style = MaterialTheme.typography.labelMedium,
                        color = ElectricCyan,
                        fontWeight = FontWeight.Bold
                    )
                }
                Slider(
                    value = uiState.blendStrength,
                    onValueChange = onBlendStrengthChanged,
                    valueRange = 0.20f..1.0f,
                    modifier = Modifier.testTag("blend_strength_slider")
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "512×512 HD Enhancement (GFPGAN / Pore Detail)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "${(uiState.enhancementStrength * 100).toInt()}%",
                        style = MaterialTheme.typography.labelMedium,
                        color = NeonEmerald,
                        fontWeight = FontWeight.Bold
                    )
                }
                Slider(
                    value = uiState.enhancementStrength,
                    onValueChange = onEnhancementStrengthChanged,
                    valueRange = 0.0f..1.0f,
                    modifier = Modifier.testTag("enhancement_strength_slider")
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "DSLR Portrait Background Blur (பின்னணி Blur)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = if (uiState.portraitBlurStrength <= 0.03f) "OFF (0%)" else "${(uiState.portraitBlurStrength * 100).toInt()}%",
                        style = MaterialTheme.typography.labelMedium,
                        color = RoyalViolet,
                        fontWeight = FontWeight.Bold
                    )
                }
                Slider(
                    value = uiState.portraitBlurStrength,
                    onValueChange = onPortraitBlurStrengthChanged,
                    valueRange = 0.0f..1.0f,
                    modifier = Modifier.testTag("portrait_blur_slider")
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Occlusion Protection (கண்ணாடி, முடி & ஆபரணப் பாதுகாப்பு)",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "Uses segformer_B5_ce.onnx + luminance guard to protect glasses, hair bangs & earrings",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Switch(
                        checked = uiState.enableOcclusionProtection,
                        onCheckedChange = onOcclusionProtectionChanged,
                        modifier = Modifier.testTag("occlusion_protection_switch")
                    )
                }
            }

            // 3. Output Resolution Selector
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(
                    text = "Output Export Resolution",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = ElectricCyan
                )
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    OutputResolutionOption.entries.forEach { resOpt ->
                        FilterChip(
                            selected = uiState.outputResolution == resOpt,
                            onClick = { onOutputResolutionChanged(resOpt) },
                            label = { Text(resOpt.title) },
                            modifier = Modifier.testTag("output_res_${resOpt.name.lowercase()}")
                        )
                    }
                }
            }

            // 4. Collapsible Face/Head Position & Scale Adjustment
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(ObsidianBg)
                    .clickable { expandPositionControls = !expandPositionControls }
                    .padding(horizontal = 12.dp, vertical = 10.dp)
                    .testTag("toggle_position_adjustments_button"),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Face / Head Position & Scale Adjustment (X: ${uiState.faceOffsetX.toInt()}px, Y: ${uiState.faceOffsetY.toInt()}px, ${"%.2f".format(uiState.faceScaleAdjust)}x)",
                    style = MaterialTheme.typography.labelMedium,
                    color = ElectricCyan,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = if (expandPositionControls) "▲" else "▼",
                    style = MaterialTheme.typography.labelMedium,
                    color = NeonEmerald
                )
            }

            AnimatedVisibility(visible = expandPositionControls) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(ObsidianBg)
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Horizontal Shift (X): ${uiState.faceOffsetX.toInt()} px",
                            style = MaterialTheme.typography.labelMedium
                        )
                        OutlinedButton(
                            onClick = onResetAdjustments,
                            modifier = Modifier.height(32.dp)
                        ) {
                            Text("Reset All", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                    Slider(
                        value = uiState.faceOffsetX,
                        onValueChange = onFaceOffsetXChanged,
                        valueRange = -24f..24f
                    )

                    Text(
                        text = "Vertical Shift (Y): ${uiState.faceOffsetY.toInt()} px",
                        style = MaterialTheme.typography.labelMedium
                    )
                    Slider(
                        value = uiState.faceOffsetY,
                        onValueChange = onFaceOffsetYChanged,
                        valueRange = -24f..24f
                    )

                    Text(
                        text = "Face / Head Scale: ${"%.2f".format(uiState.faceScaleAdjust)}x",
                        style = MaterialTheme.typography.labelMedium
                    )
                    Slider(
                        value = uiState.faceScaleAdjust,
                        onValueChange = onFaceScaleChanged,
                        valueRange = 0.88f..1.12f
                    )
                }
            }
        }
    }
}

@Composable
private fun VisualValidationAndStageInspectorCard(
    uiState: FaceSwapUiState,
    onRunVisualValidation: () -> Unit,
    onSaveValidationSheet: (Boolean) -> Unit
) {
    val suite = uiState.visualValidationSuite

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Real Visual Validation: A/B/C/D Models & 3-Stage Upper-Lip Inspector",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = ElectricCyan
            )
            Text(
                text = "Generates actual output images for 1. Source, 2. Target (NO Moustache), 3. Current Output (Before Fix), 4. A (inswapper_128), 5. B (hyperswap_1a_256), 6. C (hyperswap_1b_256), 7. D (hyperswap_1c_256), plus the 3-Stage Upper-Lip Diagnostic.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Button(
                onClick = onRunVisualValidation,
                enabled = !uiState.isRunningVisualValidation,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp)
                    .testTag("run_visual_validation_button"),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = NeonEmerald,
                    contentColor = ObsidianBg
                )
            ) {
                if (uiState.isRunningVisualValidation) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = ObsidianBg,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Generating A/B/C/D & 3-Stage Visual Comparison...",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.Compare,
                        contentDescription = "Run Visual Validation",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Run A/B/C/D & 3-Stage Upper-Lip Visual Validation",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            if (suite != null) {
                // 1. 7-Panel Visual Comparison Sheet Preview
                Text(
                    text = "1. 7-Panel Visual Comparison (Source • Target • Before Fix • A • B • C • D)",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = NeonEmerald
                )
                Image(
                    bitmap = suite.comparison7PanelSheet.asImageBitmap(),
                    contentDescription = "7-Panel Visual Comparison Sheet",
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .border(1.5.dp, ElectricCyan, RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.FillWidth
                )

                // 2. 3-Stage Upper-Lip Moustache Diagnostic Sheet Preview
                Text(
                    text = "2. 3-Stage Upper-Lip Moustache Diagnostic (Winning Model: ${suite.winningCandidate.code} — ${suite.winningCandidate.canonicalFileName})",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = ElectricCyan
                )
                Image(
                    bitmap = suite.stage3PanelSheet.asImageBitmap(),
                    contentDescription = "3-Stage Upper-Lip Diagnostic Sheet",
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .border(1.5.dp, NeonEmerald, RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.FillWidth
                )

                // Exact Stage Root-Cause Summary Box
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(ObsidianBg)
                        .border(1.dp, NeonEmerald.copy(alpha = 0.45f), RoundedCornerShape(10.dp))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "Exact Stage Root-Cause Analysis (Upper-Lip Moustache):",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        color = NeonEmerald
                    )
                    Text(
                        text = suite.rootCauseStageSummary,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }

                // Export Buttons for the 2 Visual Comparison Sheets
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedButton(
                        onClick = { onSaveValidationSheet(false) },
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .testTag("save_7panel_sheet_button"),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text(
                            text = "Save 7-Panel PNG",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = ElectricCyan
                        )
                    }
                    OutlinedButton(
                        onClick = { onSaveValidationSheet(true) },
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp)
                            .testTag("save_3stage_sheet_button"),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text(
                            text = "Save 3-Stage PNG",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = NeonEmerald
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun AdvancedNegativeAndStyleSettingsCard(
    enableNegativeArtifactSuppression: Boolean,
    enableLcmInpainting: Boolean,
    faceAgingFactor: Float,
    onNegativeArtifactSuppressionChanged: (Boolean) -> Unit,
    onLcmInpaintingChanged: (Boolean) -> Unit,
    onFaceAgingFactorChanged: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Negative Artifact & Style Controls",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = ElectricCyan
                    )
                    Text(
                        text = "கண்கள், வாய் & நாக்கு பிழைத் தடுப்பு மற்றும் StyleGAN கட்டுப்பாடுகள்",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = NeonEmerald.copy(alpha = 0.15f),
                    border = BorderStroke(1.dp, NeonEmerald.copy(alpha = 0.4f))
                ) {
                    Text(
                        text = "100% OFFLINE",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = NeonEmerald,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f))

            // 1. Negative Artifact Suppression (Tongue, Mouth, Eye, Makeup)
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "1. Negative Artifact Suppression",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = "நாக்கு வெளியே நீட்டியிருந்தாலோ வாய் திறந்திருந்தாலோ பிளவுபடாமல் (Anti-Split Tongue) பாதுகாக்கும்; கண் இமை மேக்கப் & கருவிழி சிதைவைத் தடுக்கும்.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = enableNegativeArtifactSuppression,
                    onCheckedChange = onNegativeArtifactSuppressionChanged,
                    modifier = Modifier.testTag("negative_artifact_suppression_switch")
                )
            }

            // 2. LCM Inpainting + Custom LoRA
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "2. LCM Inpainting + Custom LoRA",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = "2-4 Step Latent Consistency Model அதிவேக விளிம்பு சீரமைப்பு & நுண்ணிய சருமத் தோற்ற ஒருங்கிணைப்பு.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = enableLcmInpainting,
                    onCheckedChange = onLcmInpaintingChanged,
                    modifier = Modifier.testTag("lcm_inpainting_switch")
                )
            }

            // 3. Face-Aging & Expression StyleGAN
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "3. Face-Aging & Expression (StyleGAN)",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = when {
                            faceAgingFactor < -0.15f -> "Youthful / இளமை (${String.format("%.1f", faceAgingFactor)})"
                            faceAgingFactor > 0.15f -> "Mature / முதிர்ச்சி (+${String.format("%.1f", faceAgingFactor)})"
                            else -> "Natural / இயல்பு (0.0)"
                        },
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = if (faceAgingFactor != 0f) AmberWarning else NeonEmerald
                    )
                }
                Text(
                    text = "இயற்கையான முகச் சுருக்கங்கள் அல்லது இளமையான மென்மைத் தோற்றத்தைக் கட்டுப்படுத்துகிறது.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Slider(
                    value = faceAgingFactor,
                    onValueChange = onFaceAgingFactorChanged,
                    valueRange = -1.0f..1.0f,
                    modifier = Modifier.testTag("face_aging_slider")
                )
            }
        }
    }
}



