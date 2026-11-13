package com.example.ui.components

import android.graphics.Bitmap
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
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
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
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
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

private enum class MaskToolMode {
    BRUSH,
    ERASER,
    PAN
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
    onDismiss: () -> Unit,
    onApplyMask: (Bitmap) -> Unit
) {
    val strokes = remember { mutableStateListOf<NormalizedMaskStroke>() }
    var activePoints by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var brushFraction by remember { mutableFloatStateOf(0.06f) }
    var toolMode by remember { mutableStateOf(MaskToolMode.BRUSH) }
    var isInverted by remember { mutableStateOf(false) }
    var zoomLevel by remember { mutableFloatStateOf(1f) }
    var panOffset by remember { mutableStateOf(Offset.Zero) }

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
                            val maskBmp = renderMaskBitmap(
                                width = sourceBitmap.width,
                                height = sourceBitmap.height,
                                strokes = strokes.toList(),
                                invert = isInverted
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
                    onClick = { isInverted = !isInverted },
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

            // Drawing Canvas centered with Source Image aspect ratio
            BoxWithConstraints(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp)),
                contentAlignment = Alignment.Center
            ) {
                val aspectRatio = sourceBitmap.width.toFloat() / sourceBitmap.height.coerceAtLeast(1).toFloat()
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(aspectRatio)
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

                        if (isInverted) {
                            drawRect(color = maskColor)
                        }

                        fun drawNormalizedStroke(stroke: NormalizedMaskStroke) {
                            val strokePx = (stroke.brushFraction * minSide).coerceAtLeast(6f)
                            val effectiveClear = if (isInverted) !stroke.isEraser else stroke.isEraser
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
                            .clickable(enabled = strokes.isNotEmpty() || isInverted) {
                                strokes.clear()
                                isInverted = false
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
            .padding(horizontal = 6.dp),
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
    invert: Boolean
): Bitmap {
    val w = width.coerceAtLeast(1)
    val h = height.coerceAtLeast(1)
    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    val canvas = AndroidCanvas(bmp)
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
