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
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
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
import com.example.R
import com.example.onnx.InpaintModelOption
import com.example.onnx.SdInpaintPackType
import com.example.ui.FaceSwapUiState
import com.example.ui.components.DrawMaskEditorDialog

private val SAFE_FASHION_PRESETS = listOf(
    "hand on hip-arched back",
    "over shoulder-direct gaze",
    "lips parted-hair tuck",
    "one leg extended-lean forward",
    "spreading both legs"
)

/**
 * Offline Mage.space-style Inpaint Screen with SD 1.5 FP16 (Priority 1) &
 * RealVis 5.1 INT8 (Priority 2) support, Dreamy Bubbles + Frosted Glassmorphism UI.
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
    onFastModeChanged: (Boolean) -> Unit,
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

                        val hasFp16 = SdInpaintPackType.SD15_FP16 in uiState.installedSdInpaintPacks
                        val hasInt8 = SdInpaintPackType.REALVIS_INT8 in uiState.installedSdInpaintPacks
                        val hasLama = uiState.isLamaInpaintReady

                        DropdownMenu(
                            expanded = showModelDropdown,
                            onDismissRequest = { showModelDropdown = false }
                        ) {
                            ModelDropdownEntry(
                                title = "Auto (FP16 > INT8)",
                                statusBadge = if (hasFp16 || hasInt8) "Priority" else "Coming Soon",
                                selected = uiState.inpaintModelOption == InpaintModelOption.AUTO,
                                enabled = true,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.AUTO)
                                    showModelDropdown = false
                                }
                            )
                            ModelDropdownEntry(
                                title = "SD 1.5 Inpaint (FP16)",
                                statusBadge = if (hasFp16) "1st • Ready" else "Missing",
                                selected = uiState.inpaintModelOption == InpaintModelOption.SD15_FP16,
                                enabled = hasFp16,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.SD15_FP16)
                                    showModelDropdown = false
                                }
                            )
                            ModelDropdownEntry(
                                title = "RealVis 5.1 (INT8)",
                                statusBadge = if (hasInt8) "2nd • Ready" else "Missing",
                                selected = uiState.inpaintModelOption == InpaintModelOption.REALVIS_INT8,
                                enabled = hasInt8,
                                onClick = {
                                    onSelectModelOption(InpaintModelOption.REALVIS_INT8)
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

                // Frosted Prompt Text Input
                val promptShape = RoundedCornerShape(14.dp)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp)
                        .clip(promptShape)
                        .background(frostedCardBg)
                        .border(1.2.dp, frostedBorderColor, promptShape)
                        .padding(horizontal = 14.dp),
                    contentAlignment = Alignment.CenterStart
                ) {
                    if (uiState.inpaintPrompt.isEmpty()) {
                        Text(
                            text = "Modern dress",
                            color = mutedSlate.copy(alpha = 0.75f),
                            fontSize = 16.sp
                        )
                    }
                    BasicTextField(
                        value = uiState.inpaintPrompt,
                        onValueChange = onPromptChanged,
                        singleLine = true,
                        textStyle = TextStyle(
                            color = darkSlate,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Medium
                        ),
                        cursorBrush = SolidColor(Color(0xFF0284C7)),
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("inpaint_prompt_input")
                    )
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

                // Bottom-Right Dimensions & Time Info
                val dimsText = if (resultBmp != null) {
                    "${resultBmp.width} x ${resultBmp.height}"
                } else if (uiState.inpaintSourceBitmap != null) {
                    "${uiState.inpaintSourceBitmap.width} x ${uiState.inpaintSourceBitmap.height}"
                } else {
                    "0 x 0"
                }
                val secText = "%.2f".format(uiState.inpaintElapsedSec ?: 0.0f)
                Text(
                    text = "$dimsText, ${secText}s",
                    color = darkSlate,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp,
                    modifier = Modifier.align(Alignment.BottomEnd)
                )
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
            onDismiss = { showDrawMaskDialog = false },
            onApplyMask = { generatedMask ->
                onMaskUpdated(generatedMask)
                showDrawMaskDialog = false
            }
        )
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
