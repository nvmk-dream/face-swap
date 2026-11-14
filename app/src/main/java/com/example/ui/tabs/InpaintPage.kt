package com.example.ui.tabs

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Diamond
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
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
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.R
import com.example.onnx.AccessoryTarget
import com.example.onnx.InpaintModelOption
import com.example.onnx.SdInpaintOnnxEngine
import com.example.onnx.SdInpaintPackType
import com.example.ui.FaceSwapUiState
import com.example.ui.components.DrawMaskEditorDialog
import java.util.Locale
import kotlin.math.roundToInt

private val SAFE_FASHION_PRESETS = listOf(
    "hand on hip-arched back",
    "over shoulder-direct gaze",
    "lips parted-hair tuck",
    "one leg extended-lean forward",
    "spreading both legs"
)

private val POSITIVE_BOOSTER_TAGS = listOf(
    "photorealistic",
    "sharp focus",
    "natural skin texture",
    "high detail",
    "clean lighting",
    "naked",
    "nude",
    "nudity",
    "exposed breasts",
    "exposed buttocks",
    "visible genitals",
    "see through fabric",
    "sexy",
    "cleavage",
    "nipples",
    "visible nipples",
    "sitting straddled",
    "show perky boobs"
    "huge saggy natural boobs",
    "boobs",
    "curvy",
    "big ass hole",
    "show ass hole",
    "gigantic ass",
    "gaping anus",
    "gegantic boobs",
    "skinny voluptuous body",
    "wide hip",
    "wide waist",
    "large giant breasts",

)

private val NEGATIVE_QUALITY_TAGS = listOf(
    "blurry",
    "low quality",
    "deformed hands",
    "extra fingers",
    "bad anatomy",
    "distorted face",
    "mutated limbs",
    "watermark",
    "text",
    "oversaturated"
    "fully clothed",
    "cloths",
    "dress",
    "covered breasts",
    "full body coverage"
)

/**
 * Offline Mage.space-style Inpaint Screen with FP32/INT8 CPU Priority + FP16 Secondary,
 * Dynamic 512x768/768x512 aspect-locked resolution, and Dedicated Positive/Negative Prompt Screens.
 */
@Composable
fun InpaintPage(
    uiState: FaceSwapUiState,
    onBack: () -> Unit,
    onPickSourceFromFolder: () -> Unit,
    onSelectModelOption: (InpaintModelOption) -> Unit,
    onImportModelFolder: () -> Unit,
    onMaskUpdated: (Bitmap?) -> Unit,
    onPromptChanged: (String) -> Unit,
    onNegativePromptChanged: (String) -> Unit = {},
    onResetNegativePrompt: () -> Unit = {},
    onFastModeChanged: (Boolean) -> Unit,
    onStrengthChanged: (Float) -> Unit = {},
    onCfgScaleChanged: (Float) -> Unit = {},
    onDepthGuardChanged: (Boolean) -> Unit = {},
    onComputeInpaintDepth: () -> Unit = {},
    onToggleQuickAccessoryRemoval: (AccessoryTarget) -> Unit = {},
    onComputeAccessoryLayers: () -> Unit = {},
    onApplyAccessorySelectionAndExpand: (Set<AccessoryTarget>, Float) -> Unit = { _, _ -> },
    onMaskAndPromptUpdated: (Bitmap?, String?) -> Unit = { bmp, _ -> onMaskUpdated(bmp) },
    onCreateInpaint: () -> Unit,
    onSaveToFolder: (Bitmap) -> Unit,
    onDownloadToGallery: (Bitmap) -> Unit,
    onReuseResult: () -> Unit,
    onOpenGalleryView: (Bitmap) -> Unit,
    onViewResultFullScreen: (Bitmap) -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    var showModelDropdown by remember { mutableStateOf(false) }
    var showDrawMaskDialog by remember { mutableStateOf(false) }
    var showOriginalCompare by remember { mutableStateOf(false) }
    var showPositivePromptScreen by remember { mutableStateOf(false) }
    var showNegativePromptScreen by remember { mutableStateOf(false) }

    val frostedCardBg = Color(0xD9FFFFFF)
    val frostedBorderColor = Color(0x99FFFFFF)
    val darkSlate = Color(0xFF0F172A)
    val mutedSlate = Color(0xFF475569)

    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter
    ) {
        Image(
            painter = painterResource(id = R.drawable.bg_dreamy_bubbles),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 600.dp)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Top Bar: Back Button + "Inpaint" Title
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Surface(
                    shape = CircleShape,
                    color = frostedCardBg,
                    border = BorderStroke(1.dp, frostedBorderColor),
                    modifier = Modifier
                        .size(44.dp)
                        .clickable(onClick = onBack)
                        .testTag("inpaint_back_btn")
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back to Studio Home",
                            tint = darkSlate,
                            modifier = Modifier.size(22.dp)
                        )
                    }
                }

                Text(
                    text = "Inpaint",
                    color = darkSlate,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 30.sp,
                    letterSpacing = (-0.5).sp
                )
            }

            // Top Row: Left Source Image Card + Right (Model Dropdown, Draw Mask, Create)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.Top
            ) {
                val sourceBmp = uiState.inpaintSourceBitmap
                val maskBmp = uiState.inpaintMaskBitmap
                val cardShape = RoundedCornerShape(18.dp)

                Box(
                    modifier = Modifier
                        .weight(1.05f)
                        .height(176.dp)
                        .clip(cardShape)
                        .background(frostedCardBg)
                        .border(1.4.dp, frostedBorderColor, cardShape)
                        .clickable(onClick = onPickSourceFromFolder)
                        .testTag("inpaint_source_image_card"),
                    contentAlignment = Alignment.Center
                ) {
                    if (sourceBmp != null) {
                        Image(
                            bitmap = sourceBmp.asImageBitmap(),
                            contentDescription = "Inpaint Source Image",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize()
                        )
                        if (maskBmp != null) {
                            Image(
                                bitmap = maskBmp.asImageBitmap(),
                                contentDescription = "Painted Inpaint Mask Overlay",
                                contentScale = ContentScale.Crop,
                                alpha = 0.55f,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                        Row(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .fillMaxWidth()
                                .background(Color(0x990F172A))
                                .padding(horizontal = 8.dp, vertical = 5.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "${sourceBmp.width} x ${sourceBmp.height}",
                                color = Color.White,
                                fontSize = 11.5.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                text = if (maskBmp != null) "Mask Ready ✓" else "Change",
                                color = if (maskBmp != null) Color(0xFF67E8F9) else Color(0xFFE2E8F0),
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    } else {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier.padding(10.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Add,
                                contentDescription = "Select Source Image from Folder",
                                tint = darkSlate,
                                modifier = Modifier.size(34.dp)
                            )
                            Text(
                                text = "Source Image",
                                color = darkSlate,
                                fontWeight = FontWeight.Bold,
                                fontSize = 16.sp
                            )
                            Text(
                                text = "Device Folder",
                                color = mutedSlate,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }

                // Right Column: Model Dropdown (Priority 1: FP16, Priority 2: INT8), Draw Mask, Create
                Column(
                    modifier = Modifier.weight(0.95f),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "Model",
                        color = darkSlate,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )

                    Box {
                        val dropShape = RoundedCornerShape(12.dp)
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(46.dp)
                                .clip(dropShape)
                                .background(frostedCardBg)
                                .border(1.2.dp, frostedBorderColor, dropShape)
                                .clickable { showModelDropdown = true }
                                .padding(horizontal = 10.dp)
                                .testTag("inpaint_model_dropdown"),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = uiState.inpaintModelLabel,
                                color = darkSlate,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 13.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f)
                            )
                            Icon(
                                imageVector = Icons.Default.KeyboardArrowDown,
                                contentDescription = "Model options",
                                tint = darkSlate,
                                modifier = Modifier.size(20.dp)
                            )
                        }

                        val hasFp32 = SdInpaintPackType.SD15_FP32 in uiState.installedSdInpaintPacks
                        val hasInt8 = SdInpaintPackType.REALVIS_INT8 in uiState.installedSdInpaintPacks
                        val hasFp16 = SdInpaintPackType.SD15_FP16 in uiState.installedSdInpaintPacks
                        val hasLama = uiState.isLamaInpaintReady

                        DropdownMenu(
                            expanded = showModelDropdown,
                            onDismissRequest = { showModelDropdown = false }
                        ) {
                            ModelDropdownEntry(
                                title = "Auto (FP32 / INT8 > FP16)",
                                statusBadge = if (hasFp32 || hasInt8 || hasFp16) "Priority" else "Coming Soon",
                                selected = uiState.inpaintModelOption == InpaintModelOption.AUTO,
                                enabled = true,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.AUTO)
                                    showModelDropdown = false
                                }
                            )
                            ModelDropdownEntry(
                                title = "SD 1.5 Inpaint (FP32 CPU)",
                                statusBadge = if (hasFp32) "1st • Ready" else "Missing",
                                selected = uiState.inpaintModelOption == InpaintModelOption.SD15_FP32,
                                enabled = hasFp32,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.SD15_FP32)
                                    showModelDropdown = false
                                }
                            )
                            ModelDropdownEntry(
                                title = "RealVis 5.1 (INT8 CPU)",
                                statusBadge = if (hasInt8) "Fast • Ready" else "Missing",
                                selected = uiState.inpaintModelOption == InpaintModelOption.REALVIS_INT8,
                                enabled = hasInt8,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.REALVIS_INT8)
                                    showModelDropdown = false
                                }
                            )
                            ModelDropdownEntry(
                                title = "SD 1.5 Inpaint (FP16)",
                                statusBadge = if (hasFp16) "Secondary" else "Missing",
                                selected = uiState.inpaintModelOption == InpaintModelOption.SD15_FP16,
                                enabled = hasFp16,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.SD15_FP16)
                                    showModelDropdown = false
                                }
                            )
                            ModelDropdownEntry(
                                title = "LaMa (Object Remove)",
                                statusBadge = if (hasLama) "Ready" else "Missing",
                                selected = uiState.inpaintModelOption == InpaintModelOption.LAMA_ONNX,
                                enabled = hasLama,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.LAMA_ONNX)
                                    showModelDropdown = false
                                }
                            )
                            ModelDropdownEntry(
                                title = "Fast Offline CV",
                                statusBadge = "Built-in",
                                selected = uiState.inpaintModelOption == InpaintModelOption.CV_FALLBACK,
                                enabled = true,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.CV_FALLBACK)
                                    showModelDropdown = false
                                }
                            )
                            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                            DropdownMenuItem(
                                leadingIcon = {
                                    Icon(
                                        imageVector = Icons.Default.FolderOpen,
                                        contentDescription = "Import Model Folder",
                                        tint = Color(0xFF0284C7)
                                    )
                                },
                                text = {
                                    Text(
                                        text = "Import Model Folder (FP16 / INT8)",
                                        fontWeight = FontWeight.Bold,
                                        color = Color(0xFF0284C7),
                                        fontSize = 13.sp
                                    )
                                },
                                onClick = {
                                    showModelDropdown = false
                                    onImportModelFolder()
                                }
                            )
                        }
                    }

                    // Draw Mask Button
                    val actionBtnShape = RoundedCornerShape(12.dp)
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(46.dp)
                            .clip(actionBtnShape)
                            .background(
                                if (uiState.inpaintMaskBitmap != null) Color(0xFFE0F2FE) else frostedCardBg
                            )
                            .border(
                                width = 1.2.dp,
                                color = if (uiState.inpaintMaskBitmap != null) Color(0xFF0284C7) else frostedBorderColor,
                                shape = actionBtnShape
                            )
                            .clickable {
                                if (uiState.inpaintSourceBitmap != null) {
                                    showDrawMaskDialog = true
                                } else {
                                    onPickSourceFromFolder()
                                }
                            }
                            .padding(horizontal = 10.dp)
                            .testTag("inpaint_draw_mask_btn"),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Edit,
                            contentDescription = "Draw Mask",
                            tint = if (uiState.inpaintMaskBitmap != null) Color(0xFF0284C7) else darkSlate,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (uiState.inpaintMaskBitmap != null) "Edit Mask" else "Draw Mask",
                            color = if (uiState.inpaintMaskBitmap != null) Color(0xFF0284C7) else darkSlate,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 15.sp
                        )
                    }

                    // Create Button
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(46.dp)
                            .clip(actionBtnShape)
                            .background(
                                if (uiState.isInpainting) {
                                    Brush.horizontalGradient(
                                        listOf(Color(0xFF94A3B8), Color(0xFF64748B))
                                    )
                                } else {
                                    Brush.horizontalGradient(
                                        listOf(Color(0xF2FFFFFF), Color(0xE6E0F2FE))
                                    )
                                }
                            )
                            .border(1.4.dp, frostedBorderColor, actionBtnShape)
                            .clickable(enabled = !uiState.isInpainting, onClick = onCreateInpaint)
                            .testTag("inpaint_create_btn"),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = if (uiState.isInpainting) "Working..." else "Create",
                            color = if (uiState.isInpainting) Color.White else darkSlate,
                            fontWeight = FontWeight.Bold,
                            fontSize = 15.5.sp
                        )
                    }
                }
            }

            // Diffusion Tuning Card (Strength 1..10 & CFG 1..15) directly above Prompt
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(frostedCardBg)
                    .border(1.2.dp, frostedBorderColor, RoundedCornerShape(16.dp))
                    .padding(horizontal = 14.dp, vertical = 8.dp)
                    .testTag("inpaint_tuning_card"),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                val strengthDisplay = (uiState.inpaintStrength * 10f).coerceIn(1.0f, 10.0f)
                CleanTuningSliderRow(
                    label = "Strength",
                    value = strengthDisplay,
                    valueRange = 1.0f..10.0f,
                    steps = 17,
                    onValueChange = { sliderVal ->
                        val snapped = (sliderVal * 2f).roundToInt() / 2f
                        onStrengthChanged((snapped / 10f).coerceIn(0.10f, 1.00f))
                    },
                    onResetDefault = { onStrengthChanged(0.70f) },
                    sliderTag = "inpaint_strength_slider",
                    valueTag = "inpaint_strength_value",
                    labelColor = darkSlate
                )
                CleanTuningSliderRow(
                    label = "CFG",
                    value = uiState.inpaintCfgScale.coerceIn(1.0f, 15.0f),
                    valueRange = 1.0f..15.0f,
                    steps = 27,
                    onValueChange = { sliderVal ->
                        val snapped = (sliderVal * 2f).roundToInt() / 2f
                        onCfgScaleChanged(snapped.coerceIn(1.0f, 15.0f))
                    },
                    onResetDefault = { onCfgScaleChanged(7.0f) },
                    sliderTag = "inpaint_cfg_slider",
                    valueTag = "inpaint_cfg_value",
                    labelColor = darkSlate
                )
            }

            // Prompt Header Row: "Prompt" left | "Fast Mode" + Switch right
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Prompt",
                        color = darkSlate,
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "Fast Mode",
                            color = darkSlate,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 15.5.sp
                        )
                        Switch(
                            checked = uiState.inpaintFastMode,
                            onCheckedChange = onFastModeChanged,
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = Color.White,
                                checkedTrackColor = Color(0xFF0284C7),
                                uncheckedThumbColor = Color.White,
                                uncheckedTrackColor = Color(0xFF94A3B8)
                            ),
                            modifier = Modifier.testTag("inpaint_fast_mode_switch")
                        )
                    }
                }

                // Dedicated Positive Prompt Card (tap to open full-screen Positive Prompt editor)
                val promptShape = RoundedCornerShape(14.dp)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(promptShape)
                        .background(frostedCardBg)
                        .border(1.2.dp, frostedBorderColor, promptShape)
                        .clickable { showPositivePromptScreen = true }
                        .padding(horizontal = 14.dp, vertical = 12.dp)
                        .testTag("inpaint_prompt_input"),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Text(
                            text = "Positive Prompt",
                            color = Color(0xFF0284C7),
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        )
                        Text(
                            text = uiState.inpaintPrompt.ifBlank { "Tap to open Positive Prompt screen (e.g. Modern dress)..." },
                            color = if (uiState.inpaintPrompt.isBlank()) mutedSlate.copy(alpha = 0.75f) else darkSlate,
                            fontSize = 14.5.sp,
                            fontWeight = FontWeight.Medium,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Icon(
                        imageVector = Icons.Default.Edit,
                        contentDescription = "Edit Positive Prompt",
                        tint = Color(0xFF0284C7),
                        modifier = Modifier.size(18.dp)
                    )
                }

                // Dedicated Negative Prompt Card (tap to open full-screen Negative Prompt editor)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(promptShape)
                        .background(Color(0xD9FFF7F7))
                        .border(1.2.dp, Color(0xB3FECACA), promptShape)
                        .clickable { showNegativePromptScreen = true }
                        .padding(horizontal = 14.dp, vertical = 10.dp)
                        .testTag("inpaint_negative_prompt_card"),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp)
                    ) {
                        Text(
                            text = "Negative Prompt",
                            color = Color(0xFFDC2626),
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        )
                        Text(
                            text = uiState.inpaintNegativePrompt.ifBlank { SdInpaintOnnxEngine.DEFAULT_NEGATIVE_PROMPT },
                            color = darkSlate,
                            fontSize = 13.5.sp,
                            fontWeight = FontWeight.Medium,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    Icon(
                        imageVector = Icons.Default.Edit,
                        contentDescription = "Edit Negative Prompt",
                        tint = Color(0xFFDC2626),
                        modifier = Modifier.size(18.dp)
                    )
                }

                // Optional Depth Guard Row (protects foreground subject from background mask bleed)
                if (uiState.isDepthModelReady) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(Color(0xB3FFFFFF))
                            .border(1.dp, frostedBorderColor, RoundedCornerShape(12.dp))
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Depth Guard (Protect Foreground)",
                                color = darkSlate,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 13.5.sp
                            )
                            Text(
                                text = if (uiState.inpaintDepthMap != null) {
                                    "Depth cached • ${uiState.inpaintDepthMap.modelUsedLabel}"
                                } else {
                                    "Uses Depth Anything V2 / sentis-MiDaS"
                                },
                                color = mutedSlate,
                                fontSize = 11.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                        Switch(
                            checked = uiState.inpaintDepthGuardEnabled,
                            onCheckedChange = onDepthGuardChanged,
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = Color.White,
                                checkedTrackColor = Color(0xFF059669),
                                uncheckedThumbColor = Color.White,
                                uncheckedTrackColor = Color(0xFF94A3B8)
                            ),
                            modifier = Modifier.testTag("inpaint_depth_guard_switch")
                        )
                    }
                }

                // 1-Tap Accessory Removal Chips (Multi-select: Cap, Cooling Glass, Jewels -> Auto Mask + Auto Prompt)
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "1-Tap Auto Remove (Mask + Prompt)",
                            color = darkSlate,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                        if (uiState.isComputingAccessoryMask) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(5.dp)
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(12.dp),
                                    strokeWidth = 1.8.dp,
                                    color = Color(0xFF0284C7)
                                )
                                Text(
                                    text = "Scanning...",
                                    color = Color(0xFF0284C7),
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }
                        } else if (uiState.selectedAccessoryTargets.isNotEmpty() && uiState.inpaintMaskBitmap != null) {
                            Text(
                                text = "Mask Ready ✓",
                                color = Color(0xFF059669),
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf(
                            Triple(AccessoryTarget.CAP, Icons.Default.Face, "inpaint_quick_remove_cap"),
                            Triple(AccessoryTarget.GLASSES, Icons.Default.Visibility, "inpaint_quick_remove_glasses"),
                            Triple(AccessoryTarget.JEWELS, Icons.Default.Diamond, "inpaint_quick_remove_jewels")
                        ).forEach { (target, iconVec, tag) ->
                            val isSelected = target in uiState.selectedAccessoryTargets
                            Row(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(50))
                                    .background(if (isSelected) Color(0xFF0284C7) else Color(0xD9FFFFFF))
                                    .border(
                                        1.dp,
                                        if (isSelected) Color(0xFF38BDF8) else frostedBorderColor,
                                        RoundedCornerShape(50)
                                    )
                                    .clickable(enabled = !uiState.isComputingAccessoryMask && !uiState.isInpainting) {
                                        onToggleQuickAccessoryRemoval(target)
                                    }
                                    .padding(horizontal = 12.dp, vertical = 7.dp)
                                    .testTag(tag),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(5.dp)
                            ) {
                                Icon(
                                    imageVector = iconVec,
                                    contentDescription = target.removalButtonLabel,
                                    tint = if (isSelected) Color.White else Color(0xFF0284C7),
                                    modifier = Modifier.size(15.dp)
                                )
                                Text(
                                    text = target.removalButtonLabel,
                                    color = if (isSelected) Color.White else darkSlate,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }

                // Safe Fashion Design Quick Preset Chips
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    SAFE_FASHION_PRESETS.forEach { preset ->
                        val isSelected = uiState.inpaintPrompt.equals(preset, ignoreCase = true)
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(50))
                                .background(if (isSelected) Color(0xFF0284C7) else Color(0xCCFFFFFF))
                                .border(
                                    1.dp,
                                    if (isSelected) Color(0xFF38BDF8) else frostedBorderColor,
                                    RoundedCornerShape(50)
                                )
                                .clickable { onPromptChanged(preset) }
                                .padding(horizontal = 12.dp, vertical = 6.dp)
                        ) {
                            Text(
                                text = preset,
                                color = if (isSelected) Color.White else darkSlate,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                }
            }

            // Large Frosted Result Card
            val resultBmp = uiState.inpaintResultBitmap
            val displayedBmp = if (showOriginalCompare && uiState.inpaintSourceBitmap != null) {
                uiState.inpaintSourceBitmap
            } else {
                resultBmp
            }
            val resultShape = RoundedCornerShape(20.dp)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(275.dp)
                    .clip(resultShape)
                    .background(Color(0xCCFFFFFF))
                    .border(1.4.dp, frostedBorderColor, resultShape)
                    .clickable(enabled = resultBmp != null && !uiState.isInpainting) {
                        if (resultBmp != null) onViewResultFullScreen(resultBmp)
                    }
                    .padding(14.dp)
                    .testTag("inpaint_result_card")
            ) {
                Row(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (showOriginalCompare && resultBmp != null) "Original" else "Result",
                        color = darkSlate,
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp
                    )
                    if (resultBmp != null && !uiState.isInpainting) {
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            if (uiState.inpaintSourceBitmap != null) {
                                Row(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(50))
                                        .background(
                                            if (showOriginalCompare) Color(0xFF0284C7) else Color(0x990F172A)
                                        )
                                        .clickable { showOriginalCompare = !showOriginalCompare }
                                        .padding(horizontal = 9.dp, vertical = 4.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Compare,
                                        contentDescription = "Compare original and result",
                                        tint = Color.White,
                                        modifier = Modifier.size(14.dp)
                                    )
                                    Text(
                                        text = if (showOriginalCompare) "After" else "Before",
                                        color = Color.White,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                            }
                            Row(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(50))
                                    .background(Color(0xCC0F172A))
                                    .padding(horizontal = 9.dp, vertical = 4.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.ZoomIn,
                                    contentDescription = "Tap to zoom",
                                    tint = Color(0xFF38BDF8),
                                    modifier = Modifier.size(14.dp)
                                )
                                Text(
                                    text = "Zoom",
                                    color = Color.White,
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }
                        }
                    }
                }

                // Center Area: Loading OR Result Image OR Empty Hint
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(top = 34.dp, bottom = 24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    when {
                        uiState.isInpainting -> {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(10.dp),
                                modifier = Modifier.padding(horizontal = 12.dp)
                            ) {
                                CircularProgressIndicator(
                                    color = Color(0xFF0284C7),
                                    strokeWidth = 3.5.dp,
                                    modifier = Modifier.size(44.dp)
                                )
                                Text(
                                    text = uiState.inpaintProgressText ?: "Inpainting...",
                                    color = darkSlate,
                                    fontWeight = FontWeight.SemiBold,
                                    fontSize = 14.5.sp,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                        displayedBmp != null -> {
                            Image(
                                bitmap = displayedBmp.asImageBitmap(),
                                contentDescription = "Inpaint Result Image (Tap to Zoom)",
                                contentScale = ContentScale.Fit,
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clip(RoundedCornerShape(12.dp))
                            )
                        }
                        else -> {
                            Text(
                                text = "Select Source Image → Draw Mask → Tap Create",
                                color = mutedSlate,
                                fontSize = 13.5.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }

                // Bottom Row: Engine/Steps Badge (Left) + Dimensions & Time Info (Right)
                val dimsText = if (resultBmp != null) {
                    "${resultBmp.width} x ${resultBmp.height}"
                } else if (uiState.inpaintSourceBitmap != null) {
                    "${uiState.inpaintSourceBitmap.width} x ${uiState.inpaintSourceBitmap.height}"
                } else {
                    "0 x 0"
                }
                val secText = "%.2f".format(uiState.inpaintElapsedSec ?: 0.0f)
                Row(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (!uiState.inpaintEngineLabel.isNullOrBlank()) {
                        Box(
                            modifier = Modifier
                                .weight(1f, fill = false)
                                .padding(end = 8.dp)
                                .clip(RoundedCornerShape(50))
                                .background(Color(0xD90F172A))
                                .padding(horizontal = 9.dp, vertical = 3.dp)
                                .testTag("inpaint_result_engine_badge")
                        ) {
                            Text(
                                text = uiState.inpaintEngineLabel,
                                color = Color(0xFF67E8F9),
                                fontSize = 10.5.sp,
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    } else {
                        Spacer(modifier = Modifier.width(1.dp))
                    }
                    Text(
                        text = "$dimsText, ${secText}s",
                        color = darkSlate,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 14.sp
                    )
                }
            }

            // Bottom 4 Action Buttons: Save | Download | Reuse | Gallery
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                InpaintBottomActionButton(
                    icon = Icons.Default.Save,
                    label = "Save",
                    enabled = resultBmp != null,
                    onClick = { resultBmp?.let(onSaveToFolder) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("inpaint_btn_save")
                )
                InpaintBottomActionButton(
                    icon = Icons.Default.Download,
                    label = "Download",
                    enabled = resultBmp != null,
                    onClick = { resultBmp?.let(onDownloadToGallery) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("inpaint_btn_download")
                )
                InpaintBottomActionButton(
                    icon = Icons.Default.Refresh,
                    label = "Reuse",
                    enabled = resultBmp != null,
                    onClick = onReuseResult,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("inpaint_btn_reuse")
                )
                InpaintBottomActionButton(
                    icon = Icons.Default.PhotoLibrary,
                    label = "Gallery",
                    enabled = resultBmp != null,
                    onClick = { resultBmp?.let(onOpenGalleryView) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("inpaint_btn_gallery")
                )
            }

            Spacer(modifier = Modifier.height(10.dp))
        }
    }

    if (showDrawMaskDialog && uiState.inpaintSourceBitmap != null) {
        DrawMaskEditorDialog(
            sourceBitmap = uiState.inpaintSourceBitmap,
            existingMaskBitmap = uiState.inpaintMaskBitmap,
            depthMap = uiState.inpaintDepthMap,
            isDepthModelReady = uiState.isDepthModelReady,
            isComputingDepth = uiState.isComputingInpaintDepth,
            onRequestDepthMap = onComputeInpaintDepth,
            accessoryLayers = uiState.inpaintAccessoryLayers,
            initialAccessoryTargets = uiState.selectedAccessoryTargets,
            initialAccessoryExpandPx = uiState.inpaintAccessoryExpandPx,
            isComputingAccessoryMask = uiState.isComputingAccessoryMask,
            accessoryNotice = uiState.accessoryMaskNotice,
            onRequestAccessoryLayers = onComputeAccessoryLayers,
            onAccessorySelectionChanged = onApplyAccessorySelectionAndExpand,
            onDismiss = { showDrawMaskDialog = false },
            onApplyMask = { generatedMask ->
                onMaskUpdated(generatedMask)
                showDrawMaskDialog = false
            },
            onApplyMaskWithPrompt = { generatedMask, suggestedPrompt ->
                onMaskAndPromptUpdated(generatedMask, suggestedPrompt)
                showDrawMaskDialog = false
            }
        )
    }

    if (showPositivePromptScreen) {
        FullScreenPromptEditorDialog(
            title = "Positive Prompt",
            subtitle = "Describe clearly what you want generated inside the painted mask",
            accentColor = Color(0xFF0284C7),
            initialText = uiState.inpaintPrompt,
            placeholderText = "e.g. modern dress, natural skin texture, sharp focus...",
            quickTags = POSITIVE_BOOSTER_TAGS + SAFE_FASHION_PRESETS,
            showResetDefault = false,
            defaultText = "",
            dialogTag = "positive_prompt_fullscreen_dialog",
            inputTag = "positive_prompt_fullscreen_input",
            applyTag = "positive_prompt_apply_btn",
            onDismiss = { showPositivePromptScreen = false },
            onApply = { updated ->
                onPromptChanged(updated)
                showPositivePromptScreen = false
            }
        )
    }

    if (showNegativePromptScreen) {
        FullScreenPromptEditorDialog(
            title = "Negative Prompt",
            subtitle = "Things the AI should strictly avoid generating",
            accentColor = Color(0xFFDC2626),
            initialText = uiState.inpaintNegativePrompt,
            placeholderText = SdInpaintOnnxEngine.DEFAULT_NEGATIVE_PROMPT,
            quickTags = NEGATIVE_QUALITY_TAGS,
            showResetDefault = true,
            defaultText = SdInpaintOnnxEngine.DEFAULT_NEGATIVE_PROMPT,
            dialogTag = "negative_prompt_fullscreen_dialog",
            inputTag = "negative_prompt_fullscreen_input",
            applyTag = "negative_prompt_apply_btn",
            onDismiss = { showNegativePromptScreen = false },
            onApply = { updated ->
                if (updated.isBlank()) {
                    onResetNegativePrompt()
                } else {
                    onNegativePromptChanged(updated)
                }
                showNegativePromptScreen = false
            }
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun FullScreenPromptEditorDialog(
    title: String,
    subtitle: String,
    accentColor: Color,
    initialText: String,
    placeholderText: String,
    quickTags: List<String>,
    showResetDefault: Boolean,
    defaultText: String,
    dialogTag: String,
    inputTag: String,
    applyTag: String,
    onDismiss: () -> Unit,
    onApply: (String) -> Unit
) {
    var draftText by remember(initialText) { mutableStateOf(initialText) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            dismissOnBackPress = true,
            dismissOnClickOutside = false
        )
    ) {
        BackHandler { onDismiss() }

        Surface(
            modifier = Modifier
                .fillMaxSize()
                .testTag(dialogTag),
            color = Color(0xFFF8FAFC)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .systemBarsPadding()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(44.dp)
                                .clip(CircleShape)
                                .background(Color.White)
                                .border(1.dp, Color(0xFFE2E8F0), CircleShape)
                                .clickable(onClick = onDismiss),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back",
                                tint = Color(0xFF0F172A)
                            )
                        }
                        Column {
                            Text(
                                text = title,
                                color = Color(0xFF0F172A),
                                fontWeight = FontWeight.ExtraBold,
                                fontSize = 20.sp
                            )
                            Text(
                                text = subtitle,
                                color = Color(0xFF64748B),
                                fontSize = 12.sp
                            )
                        }
                    }

                    Button(
                        onClick = { onApply(draftText.trim()) },
                        colors = ButtonDefaults.buttonColors(containerColor = accentColor),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier
                            .height(44.dp)
                            .testTag(applyTag)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = "Apply",
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Apply", fontWeight = FontWeight.Bold)
                    }
                }

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(170.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color.White)
                        .border(1.5.dp, accentColor.copy(alpha = 0.45f), RoundedCornerShape(16.dp))
                        .padding(14.dp)
                ) {
                    if (draftText.isEmpty()) {
                        Text(
                            text = placeholderText,
                            color = Color(0xFF94A3B8),
                            fontSize = 15.sp
                        )
                    }
                    BasicTextField(
                        value = draftText,
                        onValueChange = { draftText = it },
                        textStyle = TextStyle(
                            color = Color(0xFF0F172A),
                            fontSize = 15.5.sp,
                            fontWeight = FontWeight.Medium,
                            lineHeight = 22.sp
                        ),
                        cursorBrush = SolidColor(accentColor),
                        modifier = Modifier
                            .fillMaxSize()
                            .testTag(inputTag)
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    val wordCount = draftText.trim().split(Regex("\\s+")).count { it.isNotBlank() }
                    Text(
                        text = "$wordCount words (CLIP max ~75 tokens)",
                        color = Color(0xFF64748B),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (showResetDefault) {
                            OutlinedButton(
                                onClick = { draftText = defaultText },
                                shape = RoundedCornerShape(10.dp),
                                border = BorderStroke(1.dp, accentColor.copy(alpha = 0.5f))
                            ) {
                                Text("Reset Default", color = accentColor, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                        OutlinedButton(
                            onClick = { draftText = "" },
                            shape = RoundedCornerShape(10.dp),
                            border = BorderStroke(1.dp, Color(0xFFCBD5E1))
                        ) {
                            Text("Clear", color = Color(0xFF475569), fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }
                }

                Text(
                    text = "Tap to Add Quick Tags",
                    color = Color(0xFF0F172A),
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp
                )

                FlowRow(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .verticalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    quickTags.forEach { tag ->
                        val isPresent = draftText.contains(tag, ignoreCase = true)
                        Row(
                            modifier = Modifier
                                .clip(RoundedCornerShape(50))
                                .background(if (isPresent) accentColor else Color.White)
                                .border(
                                    1.dp,
                                    if (isPresent) accentColor else Color(0xFFCBD5E1),
                                    RoundedCornerShape(50)
                                )
                                .clickable {
                                    draftText = if (isPresent) {
                                        draftText
                                            .split(",")
                                            .map { it.trim() }
                                            .filter { it.isNotEmpty() && !it.equals(tag, ignoreCase = true) }
                                            .joinToString(", ")
                                    } else {
                                        val clean = draftText.trim().trimEnd(',')
                                        if (clean.isEmpty()) tag else "$clean, $tag"
                                    }
                                }
                                .padding(horizontal = 12.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Icon(
                                imageVector = if (isPresent) Icons.Default.Check else Icons.Default.Add,
                                contentDescription = tag,
                                tint = if (isPresent) Color.White else accentColor,
                                modifier = Modifier.size(14.dp)
                            )
                            Text(
                                text = tag,
                                color = if (isPresent) Color.White else Color(0xFF0F172A),
                                fontSize = 12.5.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelDropdownEntry(
    title: String,
    statusBadge: String,
    selected: Boolean,
    enabled: Boolean,
    onClick: () -> Unit
) {
    DropdownMenuItem(
        enabled = enabled,
        trailingIcon = if (selected) {
            {
                Icon(
                    imageVector = Icons.Default.Check,
                    contentDescription = "Selected",
                    tint = Color(0xFF0284C7),
                    modifier = Modifier.size(18.dp)
                )
            }
        } else null,
        text = {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = title,
                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                    fontSize = 13.5.sp
                )
                Text(
                    text = statusBadge,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (statusBadge.contains("Ready") || statusBadge.contains("Priority")) {
                        Color(0xFF059669)
                    } else if (statusBadge == "Built-in") {
                        Color(0xFF0284C7)
                    } else {
                        Color(0xFF64748B)
                    }
                )
            }
        },
        onClick = onClick
    )
}

@Composable
private fun InpaintBottomActionButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(12.dp)
    Row(
        modifier = modifier
            .height(48.dp)
            .clip(shape)
            .background(if (enabled) Color(0xE6FFFFFF) else Color(0x88FFFFFF))
            .border(1.2.dp, Color(0xB3FFFFFF), shape)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = if (enabled) Color(0xFF0F172A) else Color(0xFF64748B),
            modifier = Modifier.size(16.dp)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = label,
            color = if (enabled) Color(0xFF0F172A) else Color(0xFF64748B),
            fontWeight = FontWeight.SemiBold,
            fontSize = 12.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CleanTuningSliderRow(
    label: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    onValueChange: (Float) -> Unit,
    onResetDefault: () -> Unit,
    sliderTag: String,
    valueTag: String,
    labelColor: Color
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = label,
            color = labelColor,
            fontWeight = FontWeight.SemiBold,
            fontSize = 13.sp,
            modifier = Modifier.width(60.dp)
        )
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = steps,
            modifier = Modifier
                .weight(1f)
                .height(34.dp)
                .testTag(sliderTag),
            thumb = {
                Box(
                    modifier = Modifier
                        .size(18.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF93C5FD))
                )
            },
            track = { sliderState ->
                val span = (sliderState.valueRange.endInclusive - sliderState.valueRange.start).coerceAtLeast(0.001f)
                val fraction = ((sliderState.value - sliderState.valueRange.start) / span).coerceIn(0f, 1f)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(4.dp)
                        .clip(CircleShape)
                        .background(Color(0xFFE2E8F0))
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(fraction)
                            .height(4.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF026AA7))
                    )
                }
            }
        )
        Text(
            text = "(${String.format(Locale.US, "%.1f", value)})",
            color = Color(0xFF475569),
            fontWeight = FontWeight.Medium,
            fontSize = 14.sp,
            textAlign = TextAlign.End,
            modifier = Modifier
                .width(48.dp)
                .clickable(onClick = onResetDefault)
                .testTag(valueTag)
        )
    }
}

