package com.example.ui.tabs

import android.graphics.Bitmap
import android.graphics.Canvas as AndroidCanvas
import android.graphics.Paint as AndroidPaint
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Brush
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.R
import com.example.ui.FaceSwapUiState

private data class NormalizedMaskStroke(
    val points: List<Offset>, // Normalized 0f..1f coordinates inside canvas
    val brushFraction: Float  // Brush radius relative to min(width, height)
)

/**
 * Interactive Inpaint Screen matching the user's mockup layout and
 * Dreamy Bubbles + Frosted White Glassmorphism aesthetic.
 */
@Composable
fun InpaintPage(
    uiState: FaceSwapUiState,
    onBack: () -> Unit,
    onPickSourceFromFolder: () -> Unit,
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

    val frostedCardBg = Color(0xD9FFFFFF)
    val frostedBorderColor = Color(0x99FFFFFF)
    val darkSlate = Color(0xFF0F172A)
    val mutedSlate = Color(0xFF475569)

    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter
    ) {
        // Dreamy pastel bubbles wallpaper background matching Front Home
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
            verticalArrangement = Arrangement.spacedBy(14.dp)
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

            // Top Row: Left Source Image Card + Right (Model Dropdown [Coming Soon], Draw Mask, Create)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.Top
            ) {
                // Left: Source Image Card (tapping opens Mobile Memory -> Folder picker)
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
                        // Resolution + Folder badge overlay
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

                // Right Column: Model ("Coming Soon"), Draw Mask, Create (exact mockup top-right layout)
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

                    // Model Dropdown ("Coming Soon" only — no model names per user request)
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
                                .padding(horizontal = 12.dp)
                                .testTag("inpaint_model_dropdown"),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = uiState.inpaintModelLabel,
                                color = darkSlate,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 14.5.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Icon(
                                imageVector = Icons.Default.KeyboardArrowDown,
                                contentDescription = "Model options",
                                tint = darkSlate
                            )
                        }

                        DropdownMenu(
                            expanded = showModelDropdown,
                            onDismissRequest = { showModelDropdown = false }
                        ) {
                            DropdownMenuItem(
                                text = { Text("Coming Soon", fontWeight = FontWeight.SemiBold) },
                                onClick = { showModelDropdown = false }
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

                    // Create Button (Top-right under Draw Mask per mockup)
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
            }

            // Large Frosted Result Card
            val resultBmp = uiState.inpaintResultBitmap
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
                        text = "Result",
                        color = darkSlate,
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp
                    )
                    if (resultBmp != null && !uiState.isInpainting) {
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
                                text = "Tap to Zoom",
                                color = Color.White,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.SemiBold
                            )
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
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                CircularProgressIndicator(
                                    color = Color(0xFF0284C7),
                                    strokeWidth = 3.5.dp,
                                    modifier = Modifier.size(44.dp)
                                )
                                Text(
                                    text = "Inpainting...",
                                    color = darkSlate,
                                    fontWeight = FontWeight.SemiBold,
                                    fontSize = 17.sp
                                )
                            }
                        }
                        resultBmp != null -> {
                            Image(
                                bitmap = resultBmp.asImageBitmap(),
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

                // Bottom-Right Dimensions & Time Info (e.g. "1220 x 1600, 0.42s")
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

/**
 * Interactive Touch-Drawing Mask Overlay Dialog for painting areas to inpaint.
 */
@Composable
private fun DrawMaskEditorDialog(
    sourceBitmap: Bitmap,
    onDismiss: () -> Unit,
    onApplyMask: (Bitmap) -> Unit
) {
    val strokes = remember { mutableStateListOf<NormalizedMaskStroke>() }
    var activePoints by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var brushFraction by remember { mutableFloatStateOf(0.06f) } // 6% of min side
    var canvasSize by remember { mutableStateOf(IntSize(1, 1)) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            decorFitsSystemWindows = false
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF090F20))
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Top bar
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                IconButton(
                    onClick = onDismiss,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(Color(0xFF1E293B))
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Cancel Draw Mask",
                        tint = Color.White
                    )
                }

                Text(
                    text = "Paint Mask Area",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp
                )

                Row(
                    modifier = Modifier
                        .clip(RoundedCornerShape(50))
                        .background(Color(0xFF0284C7))
                        .clickable {
                            val maskBmp = renderMaskBitmap(
                                width = sourceBitmap.width,
                                height = sourceBitmap.height,
                                strokes = strokes.toList()
                            )
                            onApplyMask(maskBmp)
                        }
                        .padding(horizontal = 14.dp, vertical = 10.dp)
                        .testTag("draw_mask_apply_btn"),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = "Apply Mask",
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Done",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                }
            }

            // Drawing Canvas centered with Source Image aspect ratio
            BoxWithConstraints(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                val aspectRatio = sourceBitmap.width.toFloat() / sourceBitmap.height.coerceAtLeast(1).toFloat()
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(aspectRatio)
                        .clip(RoundedCornerShape(14.dp))
                        .border(1.5.dp, Color(0xFF38BDF8), RoundedCornerShape(14.dp))
                        .onSizeChanged { canvasSize = it }
                        .pointerInput(brushFraction) {
                            detectTapGestures { pos ->
                                val w = size.width.coerceAtLeast(1).toFloat()
                                val h = size.height.coerceAtLeast(1).toFloat()
                                val norm = Offset(
                                    (pos.x / w).coerceIn(0f, 1f),
                                    (pos.y / h).coerceIn(0f, 1f)
                                )
                                strokes.add(
                                    NormalizedMaskStroke(
                                        points = listOf(norm, norm),
                                        brushFraction = brushFraction
                                    )
                                )
                            }
                        }
                        .pointerInput(brushFraction) {
                            detectDragGestures(
                                onDragStart = { start ->
                                    val w = size.width.coerceAtLeast(1).toFloat()
                                    val h = size.height.coerceAtLeast(1).toFloat()
                                    activePoints = listOf(
                                        Offset(
                                            (start.x / w).coerceIn(0f, 1f),
                                            (start.y / h).coerceIn(0f, 1f)
                                        )
                                    )
                                },
                                onDrag = { change, _ ->
                                    change.consume()
                                    val w = size.width.coerceAtLeast(1).toFloat()
                                    val h = size.height.coerceAtLeast(1).toFloat()
                                    val p = Offset(
                                        (change.position.x / w).coerceIn(0f, 1f),
                                        (change.position.y / h).coerceIn(0f, 1f)
                                    )
                                    activePoints = activePoints + p
                                },
                                onDragEnd = {
                                    if (activePoints.isNotEmpty()) {
                                        strokes.add(
                                            NormalizedMaskStroke(
                                                points = activePoints,
                                                brushFraction = brushFraction
                                            )
                                        )
                                    }
                                    activePoints = emptyList()
                                },
                                onDragCancel = {
                                    activePoints = emptyList()
                                }
                            )
                        }
                ) {
                    Image(
                        bitmap = sourceBitmap.asImageBitmap(),
                        contentDescription = "Draw Mask Target",
                        contentScale = ContentScale.FillBounds,
                        modifier = Modifier.fillMaxSize()
                    )

                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val minSide = minOf(size.width, size.height)
                        val maskColor = Color(0xFF00E5FF).copy(alpha = 0.58f)

                        fun drawNormalizedStroke(stroke: NormalizedMaskStroke) {
                            val strokePx = (stroke.brushFraction * minSide).coerceAtLeast(6f)
                            if (stroke.points.size <= 1) {
                                stroke.points.firstOrNull()?.let { pt ->
                                    drawCircle(
                                        color = maskColor,
                                        radius = strokePx / 2f,
                                        center = Offset(pt.x * size.width, pt.y * size.height)
                                    )
                                }
                                return
                            }
                            val path = Path().apply {
                                val first = stroke.points.first()
                                moveTo(first.x * size.width, first.y * size.height)
                                for (i in 1 until stroke.points.size) {
                                    val pt = stroke.points[i]
                                    lineTo(pt.x * size.width, pt.y * size.height)
                                }
                            }
                            drawPath(
                                path = path,
                                color = maskColor,
                                style = Stroke(
                                    width = strokePx,
                                    cap = StrokeCap.Round,
                                    join = StrokeJoin.Round
                                )
                            )
                        }

                        strokes.forEach { drawNormalizedStroke(it) }
                        if (activePoints.isNotEmpty()) {
                            drawNormalizedStroke(
                                NormalizedMaskStroke(activePoints, brushFraction)
                            )
                        }
                    }
                }
            }

            // Brush Size + Undo / Clear controls
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color(0xFF15203B))
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Brush,
                        contentDescription = "Brush Size",
                        tint = Color(0xFF38BDF8),
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Brush Size",
                        color = Color.White,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Slider(
                        value = brushFraction,
                        onValueChange = { brushFraction = it },
                        valueRange = 0.02f..0.16f,
                        colors = SliderDefaults.colors(
                            thumbColor = Color(0xFF38BDF8),
                            activeTrackColor = Color(0xFF0284C7)
                        ),
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .weight(1f)
                            .height(44.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(Color(0xFF1E293B))
                            .clickable(enabled = strokes.isNotEmpty()) {
                                if (strokes.isNotEmpty()) strokes.removeAt(strokes.lastIndex)
                            },
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.Undo,
                            contentDescription = "Undo stroke",
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "Undo",
                            color = Color.White,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 13.5.sp
                        )
                    }

                    Row(
                        modifier = Modifier
                            .weight(1f)
                            .height(44.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(Color(0xFF3F1D2B))
                            .clickable(enabled = strokes.isNotEmpty()) {
                                strokes.clear()
                            },
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.DeleteOutline,
                            contentDescription = "Clear mask",
                            tint = Color(0xFFFCA5A5),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "Clear",
                            color = Color(0xFFFCA5A5),
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 13.5.sp
                        )
                    }
                }
            }
        }
    }
}

private fun renderMaskBitmap(
    width: Int,
    height: Int,
    strokes: List<NormalizedMaskStroke>
): Bitmap {
    val bmp = Bitmap.createBitmap(width.coerceAtLeast(1), height.coerceAtLeast(1), Bitmap.Config.ARGB_8888)
    val canvas = AndroidCanvas(bmp)
    val paint = AndroidPaint(AndroidPaint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.argb(255, 0, 229, 255)
        style = AndroidPaint.Style.STROKE
        strokeCap = AndroidPaint.Cap.ROUND
        strokeJoin = AndroidPaint.Join.ROUND
    }
    val minSide = minOf(width, height).toFloat()
    for (stroke in strokes) {
        paint.strokeWidth = (stroke.brushFraction * minSide).coerceAtLeast(8f)
        if (stroke.points.size <= 1) {
            stroke.points.firstOrNull()?.let { pt ->
                canvas.drawPoint(pt.x * width, pt.y * height, paint)
            }
        } else {
            val path = android.graphics.Path()
            val first = stroke.points.first()
            path.moveTo(first.x * width, first.y * height)
            for (i in 1 until stroke.points.size) {
                val pt = stroke.points[i]
                path.lineTo(pt.x * width, pt.y * height)
            }
            canvas.drawPath(path, paint)
        }
    }
    return bmp
}
