package com.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
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
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import android.content.Intent
import android.graphics.Bitmap
import android.provider.MediaStore
import com.example.onnx.ModelSlot
import com.example.ui.AppTab
import com.example.ui.FaceSwapViewModel
import com.example.ui.StudioSubPage
import com.example.ui.components.ZoomableImageViewerDialog
import com.example.ui.tabs.HistoryAuditTab
import com.example.ui.tabs.ModelsInspectorTab
import com.example.ui.tabs.SafetyLicensesTab
import com.example.ui.tabs.SplitFaceMaskLogo
import com.example.ui.tabs.StudioTab
import com.example.ui.theme.CoralError
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.MyApplicationTheme
import com.example.ui.theme.NeonEmerald
import com.example.ui.theme.RoyalViolet

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                FaceSwapStudioApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FaceSwapStudioApp(
    viewModel: FaceSwapViewModel = viewModel()
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val auditLogs by viewModel.auditLogs.collectAsStateWithLifecycle()

    var pendingImportSlot by remember { mutableStateOf(ModelSlot.DETECTOR) }
    var pendingSaveBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var zoomViewerState by remember { mutableStateOf<Pair<Bitmap, String>?>(null) }

    val sourcePhotoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onSourceImageSelected(uri)
        }
    }

    val targetPhotoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onTargetImageSelected(uri)
        }
    }

    val sourceFilePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onSourceImageSelected(uri)
        }
    }

    val targetFilePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onTargetImageSelected(uri)
        }
    }

    val onnxDocumentPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.importOnnxModel(uri, pendingImportSlot)
        }
    }

    val onnxFolderTreePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { treeUri ->
        if (treeUri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    treeUri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.importAllModelsFromFolder(treeUri)
        }
    }

    // Mobile Memory -> Folder photo picker exclusively for Inpaint Source Image
    val inpaintSourceFolderPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            viewModel.onInpaintSourceImageSelected(uri)
        }
    }

    // Mobile Memory -> Folder saver for Inpaint & History results
    val folderSaveLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("image/png")
    ) { uri ->
        val bmp = pendingSaveBitmap
        pendingSaveBitmap = null
        if (uri != null && bmp != null) {
            viewModel.saveBitmapToFolderUri(uri, bmp)
        }
    }

    val requestSaveToFolder: (Bitmap) -> Unit = { bmp ->
        pendingSaveBitmap = bmp
        folderSaveLauncher.launch("FaceSwapStudio_${System.currentTimeMillis()}.png")
    }

    if (uiState.currentTab != AppTab.STUDIO) {
        BackHandler {
            viewModel.selectTab(AppTab.STUDIO)
        }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        SplitFaceMaskLogo(
                            modifier = Modifier.size(38.dp)
                        )
                        Column {
                            Text(
                                text = buildAnnotatedString {
                                    withStyle(
                                        SpanStyle(
                                            color = Color.White,
                                            fontWeight = FontWeight.ExtraBold
                                        )
                                    ) {
                                        append("FaceSwap ")
                                    }
                                    withStyle(
                                        SpanStyle(
                                            brush = Brush.horizontalGradient(
                                                colors = listOf(
                                                    Color(0xFF00D4FF),
                                                    Color(0xFF8B5CF6)
                                                )
                                            ),
                                            fontWeight = FontWeight.ExtraBold
                                        )
                                    ) {
                                        append("Studio")
                                    }
                                },
                                fontSize = 21.sp,
                                lineHeight = 24.sp
                            )
                            Text(
                                text = stringResource(R.string.subtitle_offline_engine),
                                style = MaterialTheme.typography.labelMedium,
                                color = Color(0xFFA6B4D0),
                                fontSize = 11.5.sp
                            )
                        }
                    }
                },
                actions = {
                    // [ ● Offline Ready ] pill
                    Row(
                        modifier = Modifier
                            .clip(RoundedCornerShape(50))
                            .background(Color(0xFF0A2E26))
                            .border(1.dp, Color(0xFF15694F), RoundedCornerShape(50))
                            .clickable {
                                viewModel.selectTab(AppTab.MODELS)
                            }
                            .padding(horizontal = 11.dp, vertical = 6.dp)
                            .testTag("top_offline_ready_badge"),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(NeonEmerald)
                        )
                        Text(
                            text = "Offline Ready",
                            color = Color.White,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 11.5.sp
                        )
                    }

                    Spacer(modifier = Modifier.width(4.dp))

                    // Settings gear icon button (⚙️) -> highly visible luminous frosted button
                    val isInsideSettings = uiState.currentTab != AppTab.STUDIO ||
                        uiState.studioSubPage == StudioSubPage.SETTINGS_AND_DIAGNOSTICS
                    IconButton(
                        onClick = {
                            if (isInsideSettings) {
                                viewModel.selectStudioSubPage(StudioSubPage.FRONT_HOME)
                                viewModel.selectTab(AppTab.STUDIO)
                            } else {
                                viewModel.selectTab(AppTab.MODELS)
                            }
                        },
                        modifier = Modifier
                            .size(38.dp)
                            .clip(CircleShape)
                            .background(
                                if (isInsideSettings) Color(0xFFE0F2FE) else Color(0xF2FFFFFF)
                            )
                            .border(
                                width = 1.2.dp,
                                color = if (isInsideSettings) Color(0xFF0284C7) else Color(0x9938BDF8),
                                shape = CircleShape
                            )
                            .testTag("top_settings_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Settings,
                            contentDescription = "Settings (Models, History, Safety)",
                            tint = if (isInsideSettings) Color(0xFF0284C7) else Color(0xFF0F172A),
                            modifier = Modifier.size(22.dp)
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xB3FFFFFF),
                    titleContentColor = Color(0xFF0F172A)
                )
            )
        }
    ) { innerPadding ->
        val isInsideSettings = uiState.currentTab != AppTab.STUDIO ||
            uiState.studioSubPage == StudioSubPage.SETTINGS_AND_DIAGNOSTICS

        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            // Dreamy pastel bubbles background wallpaper - 100% CLEAN & FULLY VISIBLE (matching Image 1)
            androidx.compose.foundation.Image(
                painter = androidx.compose.ui.res.painterResource(id = R.drawable.bg_dreamy_bubbles),
                contentDescription = null,
                contentScale = androidx.compose.ui.layout.ContentScale.Crop,
                modifier = Modifier.fillMaxSize()
            )

            Column(
                modifier = Modifier.fillMaxSize()
            ) {
                // Settings Hub Header & Switcher (Models, History, Safety) shown ONLY when ⚙️ Settings is opened
                AnimatedVisibility(visible = isInsideSettings) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xD9FFFFFF))
                            .border(1.dp, Color(0x6638BDF8))
                            .padding(horizontal = 14.dp, vertical = 10.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(
                            modifier = Modifier
                                .clip(RoundedCornerShape(50))
                                .background(Color(0xFF132042))
                                .border(1.dp, Color(0xFF284078), RoundedCornerShape(50))
                                .clickable {
                                    viewModel.selectStudioSubPage(StudioSubPage.FRONT_HOME)
                                    viewModel.selectTab(AppTab.STUDIO)
                                }
                                .padding(horizontal = 12.dp, vertical = 6.dp)
                                .testTag("nav_tab_studio"),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.AutoFixHigh,
                                contentDescription = stringResource(R.string.tab_studio),
                                tint = ElectricCyan,
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = "← Back to Swap",
                                color = Color.White,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 12.5.sp
                            )
                        }

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Settings,
                                contentDescription = null,
                                tint = ElectricCyan,
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = "Settings",
                                color = ElectricCyan,
                                fontWeight = FontWeight.Bold,
                                fontSize = 14.sp
                            )
                        }
                    }

                    NavigationBar(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(16.dp))
                            .border(1.dp, Color(0xFF1E2D52), RoundedCornerShape(16.dp)),
                        containerColor = Color(0xFF090F20),
                        tonalElevation = 0.dp
                    ) {
                        val navItemColors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Color(0xFFA855F7),
                            selectedTextColor = Color(0xFFA855F7),
                            indicatorColor = Color(0xFF1C163B),
                            unselectedIconColor = Color(0xFF7D90B8),
                            unselectedTextColor = Color(0xFF7D90B8)
                        )
                        NavigationBarItem(
                            selected = uiState.currentTab == AppTab.MODELS ||
                                uiState.currentTab == AppTab.STUDIO,
                            onClick = { viewModel.selectTab(AppTab.MODELS) },
                            icon = {
                                Icon(
                                    imageVector = Icons.Default.Memory,
                                    contentDescription = stringResource(R.string.tab_models)
                                )
                            },
                            label = { Text(stringResource(R.string.tab_models)) },
                            colors = navItemColors,
                            modifier = Modifier.testTag("nav_tab_models")
                        )
                        NavigationBarItem(
                            selected = uiState.currentTab == AppTab.HISTORY,
                            onClick = { viewModel.selectTab(AppTab.HISTORY) },
                            icon = {
                                Icon(
                                    imageVector = Icons.Default.History,
                                    contentDescription = stringResource(R.string.tab_history)
                                )
                            },
                            label = { Text(stringResource(R.string.tab_history)) },
                            colors = navItemColors,
                            modifier = Modifier.testTag("nav_tab_history")
                        )
                        NavigationBarItem(
                            selected = uiState.currentTab == AppTab.ETHICS,
                            onClick = { viewModel.selectTab(AppTab.ETHICS) },
                            icon = {
                                Icon(
                                    imageVector = Icons.Default.Shield,
                                    contentDescription = stringResource(R.string.tab_ethics)
                                )
                            },
                            label = { Text(stringResource(R.string.tab_ethics)) },
                            colors = navItemColors,
                            modifier = Modifier.testTag("nav_tab_ethics")
                        )
                    }
                }
            }

            AnimatedVisibility(
                visible = uiState.errorBannerMessage != null || uiState.statusBannerMessage != null
            ) {
                val isError = uiState.errorBannerMessage != null
                val bannerBg = Color(0xF7FFFFFF) // Solid frosted white surface
                val borderColor = if (isError) Color(0xFFEF4444) else Color(0xFF10B981)
                val textAccentColor = if (isError) Color(0xFF991B1B) else Color(0xFF065F46) // Deep, high-contrast readable text
                val message = uiState.errorBannerMessage ?: uiState.statusBannerMessage.orEmpty()

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 14.dp, vertical = 6.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(bannerBg)
                        .border(1.5.dp, borderColor, RoundedCornerShape(14.dp))
                        .clickable { viewModel.clearMessages() }
                        .padding(horizontal = 14.dp, vertical = 11.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = textAccentColor,
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.5.sp,
                        lineHeight = 17.sp,
                        modifier = Modifier.weight(1f)
                    )
                    IconButton(
                        onClick = { viewModel.clearMessages() },
                        modifier = Modifier.size(28.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Dismiss notification",
                            tint = textAccentColor,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
            }

            when (uiState.currentTab) {
                AppTab.STUDIO -> StudioTab(
                    uiState = uiState,
                    onSelectStudioMode = viewModel::selectStudioMode,
                    onPickSourcePhoto = {
                        sourcePhotoPicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    },
                    onPickTargetPhoto = {
                        targetPhotoPicker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    },
                    onBrowseSourceFile = {
                        sourceFilePicker.launch(arrayOf("image/*"))
                    },
                    onBrowseTargetFile = {
                        targetFilePicker.launch(arrayOf("image/*"))
                    },
                    onDetectHead = viewModel::detectHeadsOnLoadedPhotos,
                    onPreviewHead = viewModel::generateHeadReplacementPreview,
                    onSelectSourceFace = viewModel::selectSourceFace,
                    onSelectTargetFace = viewModel::selectTargetFace,
                    onToggleReplaceAllFaces = viewModel::setReplaceAllTargetFaces,
                    onSkinToneModeChanged = viewModel::setSkinToneMode,
                    onFaceReactionModeChanged = viewModel::setFaceReactionMode,
                    onConsentChanged = viewModel::setConsentConfirmed,
                    onColorTransferChanged = viewModel::setEnableColorTransfer,
                    onWatermarkChanged = viewModel::setEnableProvenanceWatermark,
                    onTwoModelFallbackChanged = viewModel::setAllowTwoModelFallback,
                    onHardwareAccelChanged = viewModel::setPreferHardwareAcceleration,
                    onLowMemoryChanged = viewModel::setLowMemoryMode,
                    onRunSwap = viewModel::runActiveModePipeline,
                    onSaveToGallery = { viewModel.saveResultToGallery(isHd = false) },
                    onSaveHdToGallery = { viewModel.saveResultToGallery(isHd = true) },
                    onShareResult = viewModel::shareResultImage,
                    onRotateResult90 = viewModel::rotateResultBitmap90,
                    onCropResultAspect = viewModel::cropResultBitmapAspect,
                    onQualityLevelChanged = viewModel::setQualityLevel,
                    onOutputResolutionChanged = viewModel::setOutputResolution,
                    onBlendStrengthChanged = viewModel::setBlendStrength,
                    onEnhancementStrengthChanged = viewModel::setEnhancementStrength,
                    onOcclusionProtectionChanged = viewModel::setEnableOcclusionProtection,
                    onPortraitBlurStrengthChanged = viewModel::setPortraitBlurStrength,
                    onFaceOffsetXChanged = viewModel::setFaceOffsetX,
                    onFaceOffsetYChanged = viewModel::setFaceOffsetY,
                    onFaceScaleChanged = viewModel::setFaceScaleAdjust,
                    onResetAdjustments = viewModel::resetPositionAdjustments,
                    onToggleCompareOriginal = viewModel::toggleComparisonMode,
                    onNegativeArtifactSuppressionChanged = viewModel::setNegativeArtifactSuppression,
                    onLcmInpaintingChanged = viewModel::setLcmInpainting,
                    onFaceAgingFactorChanged = viewModel::setFaceAgingFactor,
                    onOpenModelsTab = { viewModel.selectTab(AppTab.MODELS) },
                    onImportAllFromFolder = {
                        onnxFolderTreePicker.launch(null)
                    },
                    onRunVisualValidation = viewModel::runVisualValidationSuite,
                    onSaveValidationSheet = viewModel::saveVisualValidationSheetToGallery,
                    onSelectStudioSubPage = viewModel::selectStudioSubPage,
                    onPickInpaintSourceFromFolder = {
                        inpaintSourceFolderPicker.launch(arrayOf("image/*"))
                    },
                    onSelectInpaintModelOption = viewModel::selectInpaintModelOption,
                    onInpaintMaskUpdated = viewModel::setInpaintMaskBitmap,
                    onInpaintPromptChanged = viewModel::setInpaintPrompt,
                    onInpaintFastModeChanged = viewModel::setInpaintFastMode,
                    onRunInpaint = viewModel::runInpaint,
                    onReuseInpaintResult = viewModel::reuseInpaintResultAsSource,
                    onSaveBitmapToFolder = requestSaveToFolder,
                    onDownloadBitmapToGallery = { bmp ->
                        viewModel.saveAnyBitmapToGallery(bmp, isHd = true, label = "Inpaint")
                    },
                    onOpenInpaintGallery = { bmp ->
                        viewModel.saveAnyBitmapToGallery(bmp, isHd = true, label = "Inpaint")
                        val launched = runCatching {
                            val galleryIntent = Intent(
                                Intent.ACTION_VIEW,
                                MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                            )
                            context.startActivity(galleryIntent)
                        }.isSuccess
                        if (!launched) {
                            zoomViewerState = bmp to "Inpaint Result"
                        }
                    },
                    onViewImageFullScreen = { bmp, title ->
                        zoomViewerState = bmp to title
                    }
                )

                AppTab.MODELS -> ModelsInspectorTab(
                    inspections = uiState.modelInspections,
                    isInspecting = uiState.isInspectingModels,
                    memoryServiceState = uiState.memoryServiceState,
                    onImportModelForSlot = { slot ->
                        pendingImportSlot = slot
                        onnxDocumentPicker.launch(arrayOf("*/*"))
                    },
                    onRefreshModels = viewModel::refreshModelInspections,
                    onImportAllFromFolder = {
                        onnxFolderTreePicker.launch(null)
                    },
                    onPreloadMemoryModels = viewModel::preloadModelsIntoMemory,
                    onReleaseMemoryModels = viewModel::releaseModelsFromMemory
                )

                AppTab.HISTORY -> HistoryAuditTab(
                    auditLogs = auditLogs,
                    onDeleteLog = viewModel::deleteAuditLog,
                    onClearAll = viewModel::clearAllAuditLogs,
                    onViewImageFullScreen = { bmp, title ->
                        zoomViewerState = bmp to title
                    },
                    onSaveImageToFolder = requestSaveToFolder
                )

                AppTab.ETHICS -> SafetyLicensesTab()
            }

            zoomViewerState?.let { (bmp, title) ->
                ZoomableImageViewerDialog(
                    bitmap = bmp,
                    title = title,
                    onDismiss = { zoomViewerState = null },
                    onSaveToFolder = { requestSaveToFolder(bmp) },
                    onSaveToGallery = {
                        viewModel.saveAnyBitmapToGallery(bmp, isHd = true, label = title)
                    }
                )
            }
        }
    }
    }
}
