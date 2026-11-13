package com.example.ui.components

import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.Canvas as AndroidCanvas
import android.graphics.Paint as AndroidPaint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.AutoFixOff
import androidx.compose.material.icons.filled.Brush
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Flip
import androidx.compose.material.icons.filled.OpenWith
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Diamond
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.Wallpaper
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.onnx.AccessoryMaskEngine
import com.example.onnx.AccessoryMaskLayers
import com.example.onnx.AccessoryTarget
import com.example.onnx.DepthMapResult

private enum class MaskToolMode {
    BRUSH,
    ERASER,
    PAN
}

private enum class AutoDepthMaskMode {
    NONE,
    BACKGROUND,
    SUBJECT
}

private data class NormalizedMaskStroke(
    val points: List<Offset>, // Normalized 0f..1f coordinates inside source image
    val brushFraction: Float, // Brush radius relative to min(width, height)
    val isEraser: Boolean = false
)

/**
 * Interactive Touch-Drawing Mask Overlay Dialog supporting Brush, Eraser,
 * Invert Mask, 1x/2x/3x Zoom + Pan, Undo, and Clear.
 */
@Composable
fun DrawMaskEditorDialog(
    sourceBitmap: Bitmap,
    existingMaskBitmap: Bitmap? = null,
    depthMap: DepthMapResult? = null,
    isDepthModelReady: Boolean = false,
    isComputingDepth: Boolean = false,
    onRequestDepthMap: () -> Unit = {},
    accessoryLayers: AccessoryMaskLayers? = null,
    initialAccessoryTargets: Set<AccessoryTarget> = emptySet(),
    initialAccessoryExpandPx: Float = 6f,
    isComputingAccessoryMask: Boolean = false,
    accessoryNotice: String? = null,
    onRequestAccessoryLayers: () -> Unit = {},
    onAccessorySelectionChanged: (Set<AccessoryTarget>, Float) -> Unit = { _, _ -> },
    onDismiss: () -> Unit,
    onApplyMask: (Bitmap) -> Unit = {},
    onApplyMaskWithPrompt: ((Bitmap, String?) -> Unit)? = null
) {
    val strokes = remember { mutableStateListOf<NormalizedMaskStroke>() }
    var activePoints by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var brushFraction by remember { mutableFloatStateOf(0.06f) }
    var toolMode by remember { mutableStateOf(MaskToolMode.BRUSH) }
    var isInverted by remember { mutableStateOf(false) }
    var zoomLevel by remember { mutableFloatStateOf(1f) }
    var panOffset by remember { mutableStateOf(Offset.Zero) }
    var keepExistingMask by remember(existingMaskBitmap) { mutableStateOf(existingMaskBitmap != null) }

    var autoDepthMode by remember { mutableStateOf(AutoDepthMaskMode.NONE) }
    var pendingAutoMode by remember { mutableStateOf<AutoDepthMaskMode?>(null) }
    var depthCutoff by remember(depthMap) {
        mutableFloatStateOf(depthMap?.suggestForegroundCutoff() ?: 0.48f)
    }

    var selectedAccessories by remember(initialAccessoryTargets) {
        mutableStateOf(initialAccessoryTargets)
    }
    var pendingAccessoryTarget by remember { mutableStateOf<AccessoryTarget?>(null) }
    var accessoryExpandPx by remember(initialAccessoryExpandPx) {
        mutableFloatStateOf(initialAccessoryExpandPx)
    }
    var inlineHintMessage by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(depthMap) {
        if (depthMap != null && pendingAutoMode != null) {
            autoDepthMode = pendingAutoMode!!
            pendingAutoMode = null
            isInverted = false
            inlineHintMessage = null
        }
    }

    LaunchedEffect(accessoryLayers) {
        val pending = pendingAccessoryTarget
        if (accessoryLayers != null && pending != null) {
            val next = selectedAccessories + pending
            selectedAccessories = next
            pendingAccessoryTarget = null
            onAccessorySelectionChanged(next, accessoryExpandPx)
            inlineHintMessage = if (accessoryLayers.detectedFacesCount <= 0) {
                "No face detected in photo. Use Brush or Auto BG/Subject."
            } else null
        }
    }

    val previewAutoMaskBitmap = remember(
        depthMap,
        autoDepthMode,
        depthCutoff,
        isInverted,
        accessoryLayers,
        selectedAccessories,
        accessoryExpandPx
    ) {
        val hasDepth = depthMap != null && autoDepthMode != AutoDepthMaskMode.NONE
        val hasAcc = accessoryLayers != null &&
            accessoryLayers.detectedFacesCount > 0 &&
            selectedAccessories.isNotEmpty()
        if (!hasDepth && !hasAcc) {
            null
        } else {
            val maxDim = maxOf(sourceBitmap.width, sourceBitmap.height).coerceAtLeast(1)
            val scale = (384f / maxDim.toFloat()).coerceAtMost(1f)
            val pw = (sourceBitmap.width * scale).toInt().coerceAtLeast(32)
            val ph = (sourceBitmap.height * scale).toInt().coerceAtLeast(32)
            val combined = Bitmap.createBitmap(pw, ph, Bitmap.Config.ARGB_8888)
            val canvas = AndroidCanvas(combined)

            if (hasDepth) {
                val maskBg = if (isInverted) {
                    autoDepthMode != AutoDepthMaskMode.BACKGROUND
                } else {
                    autoDepthMode == AutoDepthMaskMode.BACKGROUND
                }
                val dBmp = depthMap!!.renderAutoMaskBitmap(
                    targetWidth = pw,
                    targetHeight = ph,
                    cutoff = depthCutoff,
                    maskBackground = maskBg
                )
                canvas.drawBitmap(dBmp, 0f, 0f, null)
                dBmp.recycle()
            }
            if (hasAcc) {
                val aBmp = accessoryLayers!!.renderMaskBitmap(
                    targetWidth = pw,
                    targetHeight = ph,
                    selectedTargets = selectedAccessories,
                    expandPx = accessoryExpandPx
                )
                canvas.drawBitmap(aBmp, 0f, 0f, null)
                aBmp.recycle()
            }
            combined
        }
    }

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
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Top Bar
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

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Paint Mask Area",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 17.sp
                    )
                    Text(
                        text = when {
                            isInverted -> "Inverted Mode • Editing outside strokes"
                            toolMode == MaskToolMode.ERASER -> "Eraser Active"
                            toolMode == MaskToolMode.PAN -> "Drag to Pan (${zoomLevel.toInt()}x)"
                            else -> "Paint area to change"
                        },
                        color = Color(0xFF94A3B8),
                        fontSize = 11.5.sp
                    )
                }

                Row(
                    modifier = Modifier
                        .clip(RoundedCornerShape(50))
                        .background(Color(0xFF0284C7))
                        .clickable {
                            val hasDepth = depthMap != null && autoDepthMode != AutoDepthMaskMode.NONE
                            val hasAcc = accessoryLayers != null &&
                                accessoryLayers.detectedFacesCount > 0 &&
                                selectedAccessories.isNotEmpty()
                            val preservedBase = if (keepExistingMask && existingMaskBitmap != null) existingMaskBitmap else null
                            val autoBase = if (hasDepth || hasAcc || preservedBase != null) {
                                val base = Bitmap.createBitmap(sourceBitmap.width, sourceBitmap.height, Bitmap.Config.ARGB_8888)
                                val baseCanvas = AndroidCanvas(base)
                                if (preservedBase != null && !hasAcc) {
                                    baseCanvas.drawBitmap(
                                        preservedBase,
                                        null,
                                        Rect(0, 0, sourceBitmap.width, sourceBitmap.height),
                                        null
                                    )
                                }
                                if (hasDepth) {
                                    val maskBg = if (isInverted) {
                                        autoDepthMode != AutoDepthMaskMode.BACKGROUND
                                    } else {
                                        autoDepthMode == AutoDepthMaskMode.BACKGROUND
                                    }
                                    val dBmp = depthMap!!.renderAutoMaskBitmap(
                                        targetWidth = sourceBitmap.width,
                                        targetHeight = sourceBitmap.height,
                                        cutoff = depthCutoff,
                                        maskBackground = maskBg
                                    )
                                    baseCanvas.drawBitmap(dBmp, 0f, 0f, null)
                                    dBmp.recycle()
                                }
                                if (hasAcc) {
                                    val aBmp = accessoryLayers!!.renderMaskBitmap(
                                        targetWidth = sourceBitmap.width,
                                        targetHeight = sourceBitmap.height,
                                        selectedTargets = selectedAccessories,
                                        expandPx = accessoryExpandPx
                                    )
                                    baseCanvas.drawBitmap(aBmp, 0f, 0f, null)
                                    aBmp.recycle()
                                }
                                base
                            } else null
                            val maskBmp = renderMaskBitmap(
                                width = sourceBitmap.width,
                                height = sourceBitmap.height,
                                strokes = strokes.toList(),
                                invert = isInverted && autoBase == null,
                                baseMask = autoBase
                            )
                            autoBase?.recycle()
                            onAccessorySelectionChanged(selectedAccessories, accessoryExpandPx)
                            val suggestedPrompt = if (selectedAccessories.isNotEmpty()) {
                                AccessoryMaskEngine.buildCombinedRemovalPrompt(selectedAccessories)
                            } else null
                            if (onApplyMaskWithPrompt != null) {
                                onApplyMaskWithPrompt(maskBmp, suggestedPrompt)
                            } else {
                                onApplyMask(maskBmp)
                            }
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

            // Tool Selector Bar: Brush | Eraser | Invert | Zoom | Pan
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                MaskModeChip(
                    icon = Icons.Default.Brush,
                    label = "Brush",
                    selected = toolMode == MaskToolMode.BRUSH,
                    onClick = { toolMode = MaskToolMode.BRUSH },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("mask_tool_brush")
                )
                MaskModeChip(
                    icon = Icons.Default.AutoFixOff,
                    label = "Eraser",
                    selected = toolMode == MaskToolMode.ERASER,
                    onClick = { toolMode = MaskToolMode.ERASER },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("mask_tool_eraser")
                )
                MaskModeChip(
                    icon = Icons.Default.Flip,
                    label = if (isInverted) "Inverted" else "Invert",
                    selected = isInverted,
                    onClick = {
                        when (autoDepthMode) {
                            AutoDepthMaskMode.BACKGROUND -> {
                                autoDepthMode = AutoDepthMaskMode.SUBJECT
                                isInverted = false
                            }
                            AutoDepthMaskMode.SUBJECT -> {
                                autoDepthMode = AutoDepthMaskMode.BACKGROUND
                                isInverted = false
                            }
                            AutoDepthMaskMode.NONE -> {
                                isInverted = !isInverted
                            }
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("mask_tool_invert")
                )
                MaskModeChip(
                    icon = Icons.Default.ZoomIn,
                    label = "${zoomLevel.toInt()}x",
                    selected = zoomLevel > 1f,
                    onClick = {
                        zoomLevel = when {
                            zoomLevel < 1.5f -> 2f
                            zoomLevel < 2.5f -> 3f
                            else -> 1f
                        }
                        if (zoomLevel == 1f) {
                            panOffset = Offset.Zero
                            if (toolMode == MaskToolMode.PAN) toolMode = MaskToolMode.BRUSH
                        }
                    },
                    modifier = Modifier
                        .weight(0.85f)
                        .testTag("mask_tool_zoom")
                )
                if (zoomLevel > 1f) {
                    MaskModeChip(
                        icon = Icons.Default.OpenWith,
                        label = "Pan",
                        selected = toolMode == MaskToolMode.PAN,
                        onClick = {
                            toolMode = if (toolMode == MaskToolMode.PAN) MaskToolMode.BRUSH else MaskToolMode.PAN
                        },
                        modifier = Modifier.weight(0.85f)
                    )
                }
            }

            // 1-Click Smart Auto-Mask Row: Auto BG | Auto Subject | Cap | Glasses | Jewels (Multi-Select)
            fun toggleAccessoryChip(target: AccessoryTarget) {
                if (target in selectedAccessories) {
                    val next = selectedAccessories - target
                    selectedAccessories = next
                    onAccessorySelectionChanged(next, accessoryExpandPx)
                    inlineHintMessage = null
                } else if (accessoryLayers != null) {
                    val next = selectedAccessories + target
                    selectedAccessories = next
                    onAccessorySelectionChanged(next, accessoryExpandPx)
                    inlineHintMessage = if (accessoryLayers.detectedFacesCount <= 0) {
                        "No face detected in photo. Use Brush or Auto BG."
                    } else null
                } else {
                    pendingAccessoryTarget = target
                    onAccessorySelectionChanged(selectedAccessories + target, accessoryExpandPx)
                    inlineHintMessage = null
                    onRequestAccessoryLayers()
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                MaskModeChip(
                    icon = Icons.Default.Wallpaper,
                    label = if (isComputingDepth && pendingAutoMode == AutoDepthMaskMode.BACKGROUND) {
                        "Scanning..."
                    } else {
                        "Auto BG"
                    },
                    selected = autoDepthMode == AutoDepthMaskMode.BACKGROUND,
                    onClick = {
                        if (autoDepthMode == AutoDepthMaskMode.BACKGROUND) {
                            autoDepthMode = AutoDepthMaskMode.NONE
                            isInverted = false
                        } else if (depthMap != null) {
                            autoDepthMode = AutoDepthMaskMode.BACKGROUND
                            isInverted = false
                            inlineHintMessage = null
                        } else if (isDepthModelReady) {
                            pendingAutoMode = AutoDepthMaskMode.BACKGROUND
                            inlineHintMessage = null
                            onRequestDepthMap()
                        } else {
                            inlineHintMessage = "Install Depth Anything V2 / sentis-MiDaS in Models tab first."
                            onRequestDepthMap()
                        }
                    },
                    modifier = Modifier
                        .padding(horizontal = 2.dp)
                        .testTag("mask_tool_auto_bg")
                )
                MaskModeChip(
                    icon = Icons.Default.Person,
                    label = if (isComputingDepth && pendingAutoMode == AutoDepthMaskMode.SUBJECT) {
                        "Scanning..."
                    } else {
                        "Auto Subject"
                    },
                    selected = autoDepthMode == AutoDepthMaskMode.SUBJECT,
                    onClick = {
                        if (autoDepthMode == AutoDepthMaskMode.SUBJECT) {
                            autoDepthMode = AutoDepthMaskMode.NONE
                            isInverted = false
                        } else if (depthMap != null) {
                            autoDepthMode = AutoDepthMaskMode.SUBJECT
                            isInverted = false
                            inlineHintMessage = null
                        } else if (isDepthModelReady) {
                            pendingAutoMode = AutoDepthMaskMode.SUBJECT
                            inlineHintMessage = null
                            onRequestDepthMap()
                        } else {
                            inlineHintMessage = "Install Depth Anything V2 / sentis-MiDaS in Models tab first."
                            onRequestDepthMap()
                        }
                    },
                    modifier = Modifier
                        .padding(horizontal = 2.dp)
                        .testTag("mask_tool_auto_subject")
                )
                MaskModeChip(
                    icon = Icons.Default.Face,
                    label = if (isComputingAccessoryMask && pendingAccessoryTarget == AccessoryTarget.CAP) {
                        "Scanning..."
                    } else {
                        "Cap"
                    },
                    selected = AccessoryTarget.CAP in selectedAccessories,
                    onClick = { toggleAccessoryChip(AccessoryTarget.CAP) },
                    modifier = Modifier.testTag("mask_tool_auto_cap")
                )
                MaskModeChip(
                    icon = Icons.Default.Visibility,
                    label = if (isComputingAccessoryMask && pendingAccessoryTarget == AccessoryTarget.GLASSES) {
                        "Scanning..."
                    } else {
                        "Glasses"
                    },
                    selected = AccessoryTarget.GLASSES in selectedAccessories,
                    onClick = { toggleAccessoryChip(AccessoryTarget.GLASSES) },
                    modifier = Modifier.testTag("mask_tool_auto_glasses")
                )
                MaskModeChip(
                    icon = Icons.Default.Diamond,
                    label = if (isComputingAccessoryMask && pendingAccessoryTarget == AccessoryTarget.JEWELS) {
                        "Scanning..."
                    } else {
                        "Jewels"
                    },
                    selected = AccessoryTarget.JEWELS in selectedAccessories,
                    onClick = { toggleAccessoryChip(AccessoryTarget.JEWELS) },
                    modifier = Modifier.testTag("mask_tool_auto_jewels")
                )
                if (isComputingDepth || isComputingAccessoryMask) {
                    CircularProgressIndicator(
                        color = Color(0xFF38BDF8),
                        strokeWidth = 2.dp,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            val shownHint = inlineHintMessage ?: if (selectedAccessories.isNotEmpty()) accessoryNotice else null
            if (!shownHint.isNullOrBlank()) {
                Text(
                    text = shownHint,
                    color = Color(0xFF7DD3FC),
                    fontSize = 11.5.sp,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color(0xFF15203B))
                        .padding(horizontal = 10.dp, vertical = 5.dp)
                )
            }

            // Drawing Canvas centered with Source Image aspect ratio
            BoxWithConstraints(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp)),
                contentAlignment = Alignment.Center
            ) {
                val aspectRatio = sourceBitmap.width.toFloat() / sourceBitmap.height.coerceAtLeast(1).toFloat()
                val availRatio = if (maxHeight.value > 0f) maxWidth.value / maxHeight.value else aspectRatio
                val fittedModifier = if (availRatio > aspectRatio) {
                    Modifier.fillMaxHeight().aspectRatio(aspectRatio)
                } else {
                    Modifier.fillMaxWidth().aspectRatio(aspectRatio)
                }
                Box(
                    modifier = fittedModifier
                        .graphicsLayer {
                            scaleX = zoomLevel
                            scaleY = zoomLevel
                            translationX = panOffset.x
                            translationY = panOffset.y
                        }
                        .clip(RoundedCornerShape(14.dp))
                        .border(1.5.dp, Color(0xFF38BDF8), RoundedCornerShape(14.dp))
                        .pointerInput(toolMode, brushFraction, zoomLevel) {
                            if (toolMode == MaskToolMode.PAN) return@pointerInput
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
                                        brushFraction = brushFraction,
                                        isEraser = toolMode == MaskToolMode.ERASER
                                    )
                                )
                            }
                        }
                        .pointerInput(toolMode, brushFraction, zoomLevel) {
                            detectDragGestures(
                                onDragStart = { start ->
                                    if (toolMode == MaskToolMode.PAN) return@detectDragGestures
                                    val w = size.width.coerceAtLeast(1).toFloat()
                                    val h = size.height.coerceAtLeast(1).toFloat()
                                    activePoints = listOf(
                                        Offset(
                                            (start.x / w).coerceIn(0f, 1f),
                                            (start.y / h).coerceIn(0f, 1f)
                                        )
                                    )
                                },
                                onDrag = { change, dragAmount ->
                                    change.consume()
                                    if (toolMode == MaskToolMode.PAN) {
                                        val maxPanX = size.width * (zoomLevel - 1f) * 0.55f
                                        val maxPanY = size.height * (zoomLevel - 1f) * 0.55f
                                        panOffset = Offset(
                                            (panOffset.x + dragAmount.x * zoomLevel).coerceIn(-maxPanX, maxPanX),
                                            (panOffset.y + dragAmount.y * zoomLevel).coerceIn(-maxPanY, maxPanY)
                                        )
                                        return@detectDragGestures
                                    }
                                    val w = size.width.coerceAtLeast(1).toFloat()
                                    val h = size.height.coerceAtLeast(1).toFloat()
                                    val p = Offset(
                                        (change.position.x / w).coerceIn(0f, 1f),
                                        (change.position.y / h).coerceIn(0f, 1f)
                                    )
                                    activePoints = activePoints + p
                                },
                                onDragEnd = {
                                    if (toolMode != MaskToolMode.PAN && activePoints.isNotEmpty()) {
                                        strokes.add(
                                            NormalizedMaskStroke(
                                                points = activePoints,
                                                brushFraction = brushFraction,
                                                isEraser = toolMode == MaskToolMode.ERASER
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

                    // Offscreen compositing layer so Eraser strokes clear ONLY the mask overlay!
                    Canvas(
                        modifier = Modifier
                            .fillMaxSize()
                            .graphicsLayer {
                                compositingStrategy = CompositingStrategy.Offscreen
                            }
                    ) {
                        val minSide = minOf(size.width, size.height)
                        val maskColor = Color(0xFF00E5FF).copy(alpha = 0.58f)

                        if (keepExistingMask && existingMaskBitmap != null && selectedAccessories.isEmpty()) {
                            drawImage(
                                image = existingMaskBitmap.asImageBitmap(),
                                dstSize = IntSize(size.width.toInt().coerceAtLeast(1), size.height.toInt().coerceAtLeast(1)),
                                alpha = 0.58f
                            )
                        }
                        if (previewAutoMaskBitmap != null) {
                            drawImage(
                                image = previewAutoMaskBitmap.asImageBitmap(),
                                dstSize = IntSize(size.width.toInt().coerceAtLeast(1), size.height.toInt().coerceAtLeast(1)),
                                alpha = 0.58f
                            )
                        } else if (isInverted) {
                            drawRect(color = maskColor)
                        }

                        fun drawNormalizedStroke(stroke: NormalizedMaskStroke) {
                            val strokePx = (stroke.brushFraction * minSide).coerceAtLeast(6f)
                            val effectiveClear = if (previewAutoMaskBitmap == null && isInverted) {
                                !stroke.isEraser
                            } else {
                                stroke.isEraser
                            }
                            val strokeColor = if (effectiveClear) Color.Transparent else maskColor
                            val blend = if (effectiveClear) BlendMode.Clear else BlendMode.SrcOver

                            if (stroke.points.size <= 1) {
                                stroke.points.firstOrNull()?.let { pt ->
                                    drawCircle(
                                        color = strokeColor,
                                        radius = strokePx / 2f,
                                        center = Offset(pt.x * size.width, pt.y * size.height),
                                        blendMode = blend
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
                                color = strokeColor,
                                style = Stroke(
                                    width = strokePx,
                                    cap = StrokeCap.Round,
                                    join = StrokeJoin.Round
                                ),
                                blendMode = blend
                            )
                        }

                        strokes.forEach { drawNormalizedStroke(it) }
                        if (activePoints.isNotEmpty()) {
                            drawNormalizedStroke(
                                NormalizedMaskStroke(
                                    points = activePoints,
                                    brushFraction = brushFraction,
                                    isEraser = toolMode == MaskToolMode.ERASER
                                )
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
                if (autoDepthMode != AutoDepthMaskMode.NONE && depthMap != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AutoAwesome,
                            contentDescription = "Depth Cutoff",
                            tint = Color(0xFF38BDF8),
                            modifier = Modifier.size(18.dp)
                        )
                        Text(
                            text = "Depth",
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                        Slider(
                            value = depthCutoff,
                            onValueChange = { depthCutoff = it },
                            valueRange = 0.10f..0.90f,
                            colors = SliderDefaults.colors(
                                thumbColor = Color(0xFF38BDF8),
                                activeTrackColor = Color(0xFF0284C7)
                            ),
                            modifier = Modifier
                                .weight(1f)
                                .testTag("mask_depth_cutoff_slider")
                        )
                    }
                }
                if (selectedAccessories.isNotEmpty() && accessoryLayers != null && accessoryLayers.detectedFacesCount > 0) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AutoAwesome,
                            contentDescription = "Mask Expand",
                            tint = Color(0xFF38BDF8),
                            modifier = Modifier.size(18.dp)
                        )
                        Text(
                            text = "Expand",
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                        Slider(
                            value = accessoryExpandPx,
                            onValueChange = { accessoryExpandPx = it },
                            valueRange = 0f..16f,
                            colors = SliderDefaults.colors(
                                thumbColor = Color(0xFF38BDF8),
                                activeTrackColor = Color(0xFF0284C7)
                            ),
                            modifier = Modifier
                                .weight(1f)
                                .testTag("mask_accessory_expand_slider")
                        )
                    }
                }
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = if (toolMode == MaskToolMode.ERASER) Icons.Default.AutoFixOff else Icons.Default.Brush,
                        contentDescription = "Brush Size",
                        tint = Color(0xFF38BDF8),
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Size",
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
                            .clickable(
                                enabled = strokes.isNotEmpty() ||
                                    keepExistingMask ||
                                    isInverted ||
                                    autoDepthMode != AutoDepthMaskMode.NONE ||
                                    selectedAccessories.isNotEmpty()
                            ) {
                                strokes.clear()
                                keepExistingMask = false
                                isInverted = false
                                autoDepthMode = AutoDepthMaskMode.NONE
                                pendingAutoMode = null
                                selectedAccessories = emptySet()
                                pendingAccessoryTarget = null
                                onAccessorySelectionChanged(emptySet(), accessoryExpandPx)
                                inlineHintMessage = null
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

@Composable
private fun MaskModeChip(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .height(38.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(if (selected) Color(0xFF0284C7) else Color(0xFF15203B))
            .border(
                width = 1.dp,
                color = if (selected) Color(0xFF38BDF8) else Color(0xFF283556),
                shape = RoundedCornerShape(10.dp)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = Color.White,
            modifier = Modifier.size(15.dp)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = label,
            color = Color.White,
            fontSize = 11.5.sp,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1
        )
    }
}

private fun renderMaskBitmap(
    width: Int,
    height: Int,
    strokes: List<NormalizedMaskStroke>,
    invert: Boolean,
    baseMask: Bitmap? = null
): Bitmap {
    val w = width.coerceAtLeast(1)
    val h = height.coerceAtLeast(1)
    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    val canvas = AndroidCanvas(bmp)
    if (baseMask != null) {
        canvas.drawBitmap(baseMask, null, Rect(0, 0, w, h), null)
    }
    val drawPaint = AndroidPaint(AndroidPaint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.argb(255, 0, 229, 255)
        style = AndroidPaint.Style.STROKE
        strokeCap = AndroidPaint.Cap.ROUND
        strokeJoin = AndroidPaint.Join.ROUND
    }
    val erasePaint = AndroidPaint(AndroidPaint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.TRANSPARENT
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
        style = AndroidPaint.Style.STROKE
        strokeCap = AndroidPaint.Cap.ROUND
        strokeJoin = AndroidPaint.Join.ROUND
    }
    val minSide = minOf(w, h).toFloat()
    for (stroke in strokes) {
        val paint = if (stroke.isEraser) erasePaint else drawPaint
        paint.strokeWidth = (stroke.brushFraction * minSide).coerceAtLeast(8f)
        if (stroke.points.size <= 1) {
            stroke.points.firstOrNull()?.let { pt ->
                canvas.drawPoint(pt.x * w, pt.y * h, paint)
            }
        } else {
            val path = android.graphics.Path()
            val first = stroke.points.first()
            path.moveTo(first.x * w, first.y * h)
            for (i in 1 until stroke.points.size) {
                val pt = stroke.points[i]
                path.lineTo(pt.x * w, pt.y * h)
            }
            canvas.drawPath(path, paint)
        }
    }
    if (invert) {
        val total = w * h
        val px = IntArray(total)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        val cyan = android.graphics.Color.argb(255, 0, 229, 255)
        for (i in 0 until total) {
            val a = (px[i] ushr 24) and 0xFF
            px[i] = if (a > 30) android.graphics.Color.TRANSPARENT else cyan
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)
    }
    return bmp
}
