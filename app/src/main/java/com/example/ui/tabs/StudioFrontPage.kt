package com.example.ui.tabs

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CropFree
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.People
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.R
import com.example.ui.FaceSwapUiState
import com.example.ui.StudioMode
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.NeonEmerald
import com.example.ui.theme.RoyalViolet
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Dual-tone split face shield logo icon (Cyan left half, Neon Purple right half)
 * matching the top-left header identity in the reference design.
 */
@Composable
fun SplitFaceMaskLogo(
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val shieldPath = Path().apply {
            moveTo(w * 0.50f, h * 0.06f)
            cubicTo(w * 0.85f, h * 0.08f, w * 0.94f, h * 0.22f, w * 0.92f, h * 0.52f)
            cubicTo(w * 0.90f, h * 0.78f, w * 0.70f, h * 0.93f, w * 0.50f, h * 0.96f)
            cubicTo(w * 0.30f, h * 0.93f, w * 0.10f, h * 0.78f, w * 0.08f, h * 0.52f)
            cubicTo(w * 0.06f, h * 0.22f, w * 0.15f, h * 0.08f, w * 0.50f, h * 0.06f)
            close()
        }

        // Left half: Electric Cyan gradient
        clipRect(left = 0f, top = 0f, right = w * 0.50f, bottom = h) {
            drawPath(
                path = shieldPath,
                brush = Brush.verticalGradient(
                    colors = listOf(Color(0xFF00F0FF), Color(0xFF0099FF))
                )
            )
        }
        // Right half: Neon Violet/Purple gradient
        clipRect(left = w * 0.50f, top = 0f, right = w, bottom = h) {
            drawPath(
                path = shieldPath,
                brush = Brush.verticalGradient(
                    colors = listOf(Color(0xFFA855F7), Color(0xFF6D28D9))
                )
            )
        }

        // Center split line
        drawLine(
            color = Color(0xFF080E20),
            start = Offset(w * 0.50f, h * 0.05f),
            end = Offset(w * 0.50f, h * 0.97f),
            strokeWidth = w * 0.035f
        )

        // Left & Right eyes
        drawOval(
            color = Color(0xFF080E20),
            topLeft = Offset(w * 0.23f, h * 0.36f),
            size = Size(w * 0.18f, h * 0.11f)
        )
        drawOval(
            color = Color(0xFF080E20),
            topLeft = Offset(w * 0.59f, h * 0.36f),
            size = Size(w * 0.18f, h * 0.11f)
        )

        // Eyebrow arcs
        val leftBrow = Path().apply {
            moveTo(w * 0.20f, h * 0.31f)
            quadraticTo(w * 0.32f, h * 0.24f, w * 0.43f, h * 0.30f)
        }
        val rightBrow = Path().apply {
            moveTo(w * 0.57f, h * 0.30f)
            quadraticTo(w * 0.68f, h * 0.24f, w * 0.80f, h * 0.31f)
        }
        drawPath(leftBrow, color = Color(0xFF080E20), style = Stroke(width = w * 0.045f, cap = StrokeCap.Round))
        drawPath(rightBrow, color = Color(0xFF080E20), style = Stroke(width = w * 0.045f, cap = StrokeCap.Round))

        // Smile arc
        val smilePath = Path().apply {
            moveTo(w * 0.30f, h * 0.65f)
            quadraticTo(w * 0.50f, h * 0.81f, w * 0.70f, h * 0.65f)
        }
        drawPath(
            path = smilePath,
            color = Color(0xFF080E20),
            style = Stroke(width = w * 0.055f, cap = StrokeCap.Round)
        )
    }
}

/**
 * Clean, modern Front Page matching the user's reference screenshot.
 * Secondary tools, advanced sliders, and 7-panel / 3-stage diagnostics live on separate pages.
 */
@Composable
fun StudioFrontPage(
    uiState: FaceSwapUiState,
    onSelectStudioMode: (StudioMode) -> Unit,
    onPickSourcePhoto: () -> Unit,
    onPickTargetPhoto: () -> Unit,
    onDetectFaces: () -> Unit,
    onPreviewAlignmentPage: () -> Unit,
    onOpenSettingsPage: () -> Unit,
    onRunSwap: () -> Unit,
    onDownloadHd: () -> Unit,
    onSaveToGallery: () -> Unit,
    onInspectResultDetails: () -> Unit,
    onImportAllFromFolder: () -> Unit = {},
    onSkinToneModeChanged: (com.example.onnx.SkinToneSourceMode) -> Unit = {},
    onFaceReactionModeChanged: (com.example.onnx.FaceReactionSourceMode) -> Unit = {},
    onOpenInpaintPage: () -> Unit = {},
    onViewResultFullScreen: (Bitmap) -> Unit = {},
    modifier: Modifier = Modifier
) {
    val scrollState = rememberScrollState()

    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter
    ) {
        // Dreamy Pastel Soap Bubbles Wallpaper Background - 100% CLEAN & LUMINOUS
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
                .verticalScroll(scrollState)
                .padding(horizontal = 14.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // 1. Hero Story Banner ("Swap Faces / Create New Stories" + "Your Photos / Your Creativity / 100% Private")
            FrontHeroStoryBanner()

            // 2. Choose Mode Section ("Face Swap" vs "Full Head Replacement") + 1-Tap Auto-Import All .onnx from Folder pill
            FrontChooseModeSection(
                selectedMode = uiState.studioMode,
                onSelectMode = onSelectStudioMode,
                onImportAllFromFolder = onImportAllFromFolder,
                onOpenInpaintPage = onOpenInpaintPage
            )

            // 3. Main Workspace Card (1 Source Face, 2 Target Photo, Skin Tone & Face Reaction Selector, Swap Face CTA)
            FrontMainWorkspaceCard(
                uiState = uiState,
                onPickSourcePhoto = onPickSourcePhoto,
                onPickTargetPhoto = onPickTargetPhoto,
                onDetectFaces = onDetectFaces,
                onPreviewAlignmentPage = onPreviewAlignmentPage,
                onOpenSettingsPage = onOpenSettingsPage,
                onRunSwap = onRunSwap,
                onSkinToneModeChanged = onSkinToneModeChanged,
                onFaceReactionModeChanged = onFaceReactionModeChanged
            )

            // 4. Result Card (Source -> Target + Result Preview + Download + Save to Gallery + Completed Footer)
            FrontResultShowcaseCard(
                uiState = uiState,
                onDownloadHd = onDownloadHd,
                onSaveToGallery = onSaveToGallery,
                onInspectResultDetails = onInspectResultDetails,
                onViewResultFullScreen = onViewResultFullScreen
            )

            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

@Composable
private fun FrontHeroStoryBanner() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(106.dp)
            .clip(RoundedCornerShape(18.dp))
            .border(
                width = 1.dp,
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        Color(0xFF00D4FF).copy(alpha = 0.65f),
                        Color(0xFF3B82F6).copy(alpha = 0.45f),
                        Color(0xFF1E3A8A).copy(alpha = 0.65f)
                    )
                ),
                shape = RoundedCornerShape(18.dp)
            )
    ) {
        Image(
            painter = painterResource(id = R.drawable.img_hero_swap_banner),
            contentDescription = "Swap Faces Create New Stories Hero Banner",
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )

        // Subtle atmospheric gradient overlay for crisp text contrast
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.horizontalGradient(
                        0.0f to Color.Transparent,
                        0.28f to Color(0xFF0B1C48).copy(alpha = 0.55f),
                        0.68f to Color(0xFF0D1636).copy(alpha = 0.78f),
                        1.0f to Color(0xFF09132C).copy(alpha = 0.88f)
                    )
                )
        )

        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = 104.dp, end = 14.dp, top = 10.dp, bottom = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Center angled expressive headline
            Column(
                modifier = Modifier
                    .weight(1f)
                    .graphicsLayer { rotationZ = -5f }
            ) {
                Text(
                    text = "Swap Faces",
                    color = Color.White,
                    fontSize = 19.sp,
                    fontWeight = FontWeight.Bold,
                    fontStyle = FontStyle.Italic,
                    lineHeight = 22.sp
                )
                Box {
                    Text(
                        text = "Create New Stories",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.SemiBold,
                        fontStyle = FontStyle.Italic,
                        lineHeight = 22.sp,
                        modifier = Modifier.padding(bottom = 5.dp)
                    )
                    Canvas(
                        modifier = Modifier
                            .width(126.dp)
                            .height(6.dp)
                            .align(Alignment.BottomStart)
                            .padding(start = 34.dp)
                    ) {
                        val underlinePath = Path().apply {
                            moveTo(0f, size.height * 0.8f)
                            quadraticTo(
                                size.width * 0.5f,
                                0f,
                                size.width,
                                size.height * 0.45f
                            )
                        }
                        drawPath(
                            path = underlinePath,
                            brush = Brush.horizontalGradient(
                                colors = listOf(Color.White, ElectricCyan)
                            ),
                            style = Stroke(width = 2.2.dp.toPx(), cap = StrokeCap.Round)
                        )
                    }
                }
            }

            // Right angled privacy & creativity badge
            Row(
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Column(
                    modifier = Modifier.graphicsLayer { rotationZ = -6f },
                    horizontalAlignment = Alignment.Start
                ) {
                    Text(
                        text = "Your Photos",
                        color = Color(0xFFE2E8F0),
                        fontSize = 11.5.sp,
                        fontStyle = FontStyle.Italic,
                        fontWeight = FontWeight.Medium,
                        lineHeight = 14.sp
                    )
                    Text(
                        text = "Your Creativity",
                        color = Color(0xFFE2E8F0),
                        fontSize = 11.5.sp,
                        fontStyle = FontStyle.Italic,
                        fontWeight = FontWeight.Medium,
                        lineHeight = 14.sp
                    )
                    Text(
                        text = "100% Private",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontStyle = FontStyle.Italic,
                        fontWeight = FontWeight.Bold,
                        lineHeight = 15.sp
                    )
                }

                Icon(
                    imageVector = Icons.Default.Lock,
                    contentDescription = "100% Private On-Device Lock",
                    tint = ElectricCyan,
                    modifier = Modifier
                        .size(18.dp)
                        .padding(bottom = 2.dp)
                )
            }
        }
    }
}

@Composable
private fun FrontChooseModeSection(
    selectedMode: StudioMode,
    onSelectMode: (StudioMode) -> Unit,
    onImportAllFromFolder: () -> Unit,
    onOpenInpaintPage: () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "Choose Mode",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.ExtraBold,
                color = Color(0xFF0F172A),
                fontSize = 18.sp
            )

            // 1-Tap Auto-Import All .onnx from Folder pill button in Frosted Glass style
            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(50))
                    .background(Color(0xD9FFFFFF))
                    .border(
                        width = 1.2.dp,
                        brush = Brush.horizontalGradient(
                            colors = listOf(
                                Color(0xFF38BDF8),
                                Color(0xFFA855F7)
                            )
                        ),
                        shape = RoundedCornerShape(50)
                    )
                    .clickable(onClick = onImportAllFromFolder)
                    .padding(horizontal = 11.dp, vertical = 6.dp)
                    .testTag("front_auto_import_folder_btn"),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(7.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Folder,
                    contentDescription = "Auto-Import ONNX Folder",
                    tint = Color(0xFF18C8FF),
                    modifier = Modifier.size(17.dp)
                )
                Text(
                    text = "1-Tap Auto-Import All .onnx\nfrom Folder",
                    color = Color(0xFF18C8FF),
                    fontSize = 10.5.sp,
                    lineHeight = 12.5.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Icon(
                    imageVector = Icons.Default.Refresh,
                    contentDescription = "Sync ONNX Models from Folder",
                    tint = Color(0xFF64B5F6),
                    modifier = Modifier.size(16.dp)
                )
            }
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ModeSelectionCard(
                    title = "Face Swap",
                    subtitle = "Standard AI swap",
                    isSelected = selectedMode == StudioMode.FACE_SWAP,
                    iconType = 0,
                    onClick = { onSelectMode(StudioMode.FACE_SWAP) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("mode_tab_face_swap")
                )

                ModeSelectionCard(
                    title = "Identity Preserved",
                    subtitle = "Pure real eyes & lips",
                    isSelected = selectedMode == StudioMode.IDENTITY_PRESERVED,
                    iconType = 2,
                    onClick = { onSelectMode(StudioMode.IDENTITY_PRESERVED) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("mode_tab_identity_preserved")
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ModeSelectionCard(
                    title = "Full Head Replacement",
                    subtitle = "Replace entire head (face + hair + neck)",
                    isSelected = selectedMode == StudioMode.HEAD_REPLACEMENT,
                    iconType = 1,
                    onClick = { onSelectMode(StudioMode.HEAD_REPLACEMENT) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag("mode_tab_head_replacement")
                )

                ModeSelectionCard(
                    title = "Inpaint",
                    subtitle = "Edit specific parts",
                    isSelected = false,
                    iconType = 3,
                    onClick = onOpenInpaintPage,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("mode_tab_inpaint")
                )
            }
        }
    }
}

@Composable
private fun ModeSelectionCard(
    title: String,
    subtitle: String,
    isSelected: Boolean,
    iconType: Int,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(16.dp)
    val bgBrush = if (isSelected) {
        Brush.horizontalGradient(
            colors = listOf(
                Color(0xF2FFFFFF),
                Color(0xE6F0F9FF)
            )
        )
    } else {
        Brush.horizontalGradient(
            colors = listOf(
                Color(0xCCFFFFFF),
                Color(0xB3F8FAFC)
            )
        )
    }
    val borderColor = if (isSelected) {
        if (iconType == 2) Color(0xFFD97706) else Color(0xFF0284C7)
    } else Color(0x6638BDF8)

    Box(
        modifier = modifier
            .height(84.dp)
            .clip(shape)
            .background(bgBrush)
            .border(
                width = if (isSelected) 1.8.dp else 1.2.dp,
                color = borderColor,
                shape = shape
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Left circular icon
            Box(
                modifier = Modifier
                    .size(46.dp)
                    .clip(CircleShape)
                    .background(
                        if (isSelected) {
                            if (iconType == 2) Color(0xFFFEF3C7) else Color(0xFFE0F2FE)
                        } else Color(0xFFF1F5F9)
                    )
                    .border(
                        width = 1.4.dp,
                        color = if (isSelected) {
                            if (iconType == 2) Color(0xFFD97706) else Color(0xFF0284C7)
                        } else Color(0xFF94A3B8),
                        shape = CircleShape
                    ),
                contentAlignment = Alignment.Center
            ) {
                when (iconType) {
                    0 -> Icon(
                        imageVector = Icons.Default.People,
                        contentDescription = title,
                        tint = if (isSelected) Color(0xFF0284C7) else Color(0xFF475569),
                        modifier = Modifier.size(24.dp)
                    )
                    1 -> HeadWithHairSilhouetteIcon(
                        tint = if (isSelected) Color(0xFF0284C7) else Color(0xFF475569),
                        modifier = Modifier.size(25.dp)
                    )
                    3 -> Icon(
                        imageVector = Icons.Default.AutoFixHigh,
                        contentDescription = title,
                        tint = Color(0xFF475569),
                        modifier = Modifier.size(24.dp)
                    )
                    else -> Icon(
                        imageVector = Icons.Default.Face,
                        contentDescription = title,
                        tint = if (isSelected) Color(0xFFD97706) else Color(0xFF475569),
                        modifier = Modifier.size(25.dp)
                    )
                }
            }

            Column(
                modifier = Modifier
                    .weight(1f)
                    .padding(end = if (isSelected) 14.dp else 0.dp),
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = title,
                    color = Color(0xFF0F172A),
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 13.sp,
                    lineHeight = 15.5.sp,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = subtitle,
                    color = Color(0xFF475569),
                    fontSize = 10.sp,
                    lineHeight = 12.5.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }

        if (isSelected) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .size(20.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF1DA1F2)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Check,
                    contentDescription = "Selected Mode",
                    tint = Color.White,
                    modifier = Modifier.size(13.dp)
                )
            }
        }
    }
}

@Composable
private fun HeadWithHairSilhouetteIcon(
    tint: Color,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val hairPath = Path().apply {
            moveTo(w * 0.50f, h * 0.10f)
            cubicTo(w * 0.22f, h * 0.10f, w * 0.14f, h * 0.35f, w * 0.18f, h * 0.62f)
            cubicTo(w * 0.20f, h * 0.76f, w * 0.12f, h * 0.84f, w * 0.25f, h * 0.88f)
            cubicTo(w * 0.33f, h * 0.82f, w * 0.34f, h * 0.68f, w * 0.33f, h * 0.54f)
            cubicTo(w * 0.33f, h * 0.34f, w * 0.40f, h * 0.26f, w * 0.50f, h * 0.26f)
            cubicTo(w * 0.60f, h * 0.26f, w * 0.67f, h * 0.34f, w * 0.67f, h * 0.54f)
            cubicTo(w * 0.66f, h * 0.68f, w * 0.67f, h * 0.82f, w * 0.75f, h * 0.88f)
            cubicTo(w * 0.88f, h * 0.84f, w * 0.80f, h * 0.76f, w * 0.82f, h * 0.62f)
            cubicTo(w * 0.86f, h * 0.35f, w * 0.78f, h * 0.10f, w * 0.50f, h * 0.10f)
            close()
        }
        drawPath(hairPath, color = tint)
        drawOval(
            color = tint,
            topLeft = Offset(w * 0.34f, h * 0.30f),
            size = Size(w * 0.32f, h * 0.40f)
        )
    }
}

@Composable
private fun FrontMainWorkspaceCard(
    uiState: FaceSwapUiState,
    onPickSourcePhoto: () -> Unit,
    onPickTargetPhoto: () -> Unit,
    onDetectFaces: () -> Unit,
    onPreviewAlignmentPage: () -> Unit,
    onOpenSettingsPage: () -> Unit,
    onRunSwap: () -> Unit,
    onSkinToneModeChanged: (com.example.onnx.SkinToneSourceMode) -> Unit = {},
    onFaceReactionModeChanged: (com.example.onnx.FaceReactionSourceMode) -> Unit = {}
) {
    val cardShape = RoundedCornerShape(20.dp)
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = cardShape,
        color = Color(0xE6FFFFFF), // Luminous Frosted Glass (matching Image 1)
        border = BorderStroke(1.2.dp, Color(0x6638BDF8))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Row 1: 1 Source Face
            PhotoStepRow(
                stepNumber = "1",
                stepCircleBrush = Brush.verticalGradient(
                    colors = listOf(Color(0xFF0284C7), Color(0xFF0369A1))
                ),
                title = "Source Face",
                subtitle = if (uiState.sourceFaces.isNotEmpty()) {
                    "Face detected (${uiState.sourceFaces.size}) • Ready to swap"
                } else {
                    "Select or add the face you want to use"
                },
                bitmap = uiState.sourceBitmap,
                isLoading = uiState.isDetectingSource,
                thumbnailBorderColor = Color(0xFF0284C7),
                buttonFillBrush = Brush.horizontalGradient(
                    colors = listOf(Color(0xFFE0F2FE), Color(0xFFBAE6FD))
                ),
                buttonBorderColor = Color(0xFF0284C7),
                buttonTestTag = "btn_select_source",
                onPickPhoto = onPickSourcePhoto
            )

            HorizontalDivider(color = Color(0x330284C7), thickness = 1.dp)

            // Row 2: 2 Target Photo
            PhotoStepRow(
                stepNumber = "2",
                stepCircleBrush = Brush.verticalGradient(
                    colors = listOf(Color(0xFF9333EA), Color(0xFF7E22CE))
                ),
                title = "Target Photo",
                subtitle = if (uiState.targetFaces.isNotEmpty()) {
                    "Target face locked (${uiState.targetFaces.size}) • Ready"
                } else {
                    "Select the photo where you want to replace"
                },
                bitmap = uiState.targetBitmap,
                isLoading = uiState.isDetectingTarget,
                thumbnailBorderColor = Color(0xFF9333EA),
                buttonFillBrush = Brush.horizontalGradient(
                    colors = listOf(Color(0xFFF3E8FF), Color(0xFFE9D5FF))
                ),
                buttonBorderColor = Color(0xFF9333EA),
                buttonTestTag = "btn_select_target",
                onPickPhoto = onPickTargetPhoto
            )

            HorizontalDivider(color = Color(0x330284C7), thickness = 1.dp)

            // Direct User Selection: 1. Skin Tone (தோல் நிறம்) & 2. Face Reaction (முகபாவனை)
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(14.dp))
                    .background(Color(0x80F0F9FF))
                    .border(1.dp, Color(0x4D0284C7), RoundedCornerShape(14.dp))
                    .padding(horizontal = 10.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // 1. Skin Tone Selector
                Text(
                    text = "Skin Tone (தோல் நிறம்)",
                    color = Color(0xFF0369A1),
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    val skinModes = listOf(
                        com.example.onnx.SkinToneSourceMode.SOURCE_IDENTITY to "1. Source Skin",
                        com.example.onnx.SkinToneSourceMode.TARGET_SCENE to "2. Target Skin",
                        com.example.onnx.SkinToneSourceMode.BALANCED_BLEND to "50/50 Blend"
                    )
                    skinModes.forEach { (mode, label) ->
                        val selected = uiState.skinToneMode == mode
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(36.dp)
                                .clip(RoundedCornerShape(10.dp))
                                .background(
                                    if (selected) Color(0xFF0C3B78) else Color(0xFF14203E)
                                )
                                .border(
                                    width = if (selected) 1.4.dp else 1.dp,
                                    color = if (selected) Color(0xFF00E5FF) else Color(0xFF2B3E6B),
                                    shape = RoundedCornerShape(10.dp)
                                )
                                .clickable { onSkinToneModeChanged(mode) }
                                .padding(horizontal = 4.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = label,
                                color = if (selected) Color.White else Color(0xFFB0C0E0),
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                                fontSize = 11.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }

                // 2. Face Reaction Selector
                Text(
                    text = "Face Reaction (முகபாவனை / சிரிப்பு)",
                    color = Color(0xFFC4B5FD),
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.sp
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    val reactionModes = listOf(
                        com.example.onnx.FaceReactionSourceMode.TARGET_REACTION to "2. Target Reaction",
                        com.example.onnx.FaceReactionSourceMode.SOURCE_REACTION to "1. Source Reaction"
                    )
                    reactionModes.forEach { (mode, label) ->
                        val selected = uiState.faceReactionMode == mode
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(36.dp)
                                .clip(RoundedCornerShape(10.dp))
                                .background(
                                    if (selected) Color(0xFF3E1F7A) else Color(0xFF14203E)
                                )
                                .border(
                                    width = if (selected) 1.4.dp else 1.dp,
                                    color = if (selected) Color(0xFFA855F7) else Color(0xFF2B3E6B),
                                    shape = RoundedCornerShape(10.dp)
                                )
                                .clickable { onFaceReactionModeChanged(mode) }
                                .padding(horizontal = 6.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = label,
                                color = if (selected) Color.White else Color(0xFFB0C0E0),
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                                fontSize = 11.5.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }
            }

            // Row 3: 3 Quick Action Buttons (Detect Faces | Preview Alignment | Skin Tone & Settings)
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 2.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                SecondaryActionPillButton(
                    icon = Icons.Default.CropFree,
                    label = "Detect Faces",
                    onClick = onDetectFaces,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("btn_detect_head")
                )

                SecondaryActionPillButton(
                    icon = Icons.Default.Face,
                    label = "Preview Alignment",
                    onClick = onPreviewAlignmentPage,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("btn_preview_head")
                )

                SecondaryActionPillButton(
                    icon = Icons.Default.Tune,
                    label = "Skin Tone & Reaction",
                    onClick = onOpenSettingsPage,
                    modifier = Modifier
                        .weight(1f)
                        .testTag("btn_open_settings_page")
                )
            }

            // Row 4: Full-Width Gradient CTA ("Swap Face ->" or "Replace Head ->")
            val ctaShape = RoundedCornerShape(16.dp)
            val ctaLabel = when {
                uiState.isSwapping -> uiState.swapProgress?.stageTitle ?: "Swapping Face..."
                uiState.studioMode == StudioMode.HEAD_REPLACEMENT -> "Replace Head"
                uiState.studioMode == StudioMode.IDENTITY_PRESERVED -> "Swap (Preserve Identity)"
                else -> "Swap Face"
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp)
                    .clip(ctaShape)
                    .background(
                        Brush.horizontalGradient(
                            colors = listOf(
                                Color(0xFF8B46FF),
                                Color(0xFF5A67F2),
                                Color(0xFF00E5FF)
                            )
                        )
                    )
                    .border(
                        width = 1.dp,
                        brush = Brush.horizontalGradient(
                            colors = listOf(
                                Color(0xFFC084FC).copy(alpha = 0.8f),
                                Color(0xFF67E8F9).copy(alpha = 0.9f)
                            )
                        ),
                        shape = ctaShape
                    )
                    .clickable(enabled = !uiState.isSwapping) { onRunSwap() }
                    .testTag("btn_execute_swap"),
                contentAlignment = Alignment.Center
            ) {
                if (uiState.isSwapping) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(22.dp),
                            strokeWidth = 2.5.dp,
                            color = Color.White
                        )
                        Text(
                            text = ctaLabel,
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                } else {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.SwapHoriz,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(24.dp)
                        )
                        Text(
                            text = ctaLabel,
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 18.sp
                        )
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PhotoStepRow(
    stepNumber: String,
    stepCircleBrush: Brush,
    title: String,
    subtitle: String,
    bitmap: Bitmap?,
    isLoading: Boolean,
    thumbnailBorderColor: Color,
    buttonFillBrush: Brush,
    buttonBorderColor: Color,
    buttonTestTag: String,
    onPickPhoto: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Numbered circle badge (1 or 2)
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(stepCircleBrush),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = stepNumber,
                color = Color.White,
                fontWeight = FontWeight.ExtraBold,
                fontSize = 16.sp
            )
        }

        // Title & subtitle
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = title,
                color = Color(0xFF0F172A),
                fontWeight = FontWeight.ExtraBold,
                fontSize = 15.sp,
                lineHeight = 18.sp
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = subtitle,
                color = Color(0xFF334155),
                fontSize = 11.5.sp,
                lineHeight = 14.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }

        // Portrait Thumbnail Preview
        val thumbShape = RoundedCornerShape(12.dp)
        Box(
            modifier = Modifier
                .size(56.dp)
                .clip(thumbShape)
                .background(Color(0xFF162240))
                .border(1.5.dp, thumbnailBorderColor, thumbShape)
                .clickable(onClick = onPickPhoto),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = title,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Image(
                    painter = painterResource(id = R.drawable.img_sample_portrait),
                    contentDescription = "$title Sample Preview",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            }

            if (isLoading) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.55f)),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color = ElectricCyan
                    )
                }
            }
        }

        // Add Photo button
        val btnShape = RoundedCornerShape(12.dp)
        Box(
            modifier = Modifier
                .height(42.dp)
                .clip(btnShape)
                .background(buttonFillBrush)
                .border(1.2.dp, buttonBorderColor, btnShape)
                .clickable(onClick = onPickPhoto)
                .padding(horizontal = 12.dp)
                .testTag(buttonTestTag),
            contentAlignment = Alignment.Center
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.PhotoLibrary,
                    contentDescription = null,
                    tint = Color.White,
                    modifier = Modifier.size(16.dp)
                )
                Text(
                    text = "Add Photo",
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.5.sp
                )
            }
        }
    }
}

@Composable
private fun SecondaryActionPillButton(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val pillShape = RoundedCornerShape(12.dp)
    Box(
        modifier = modifier
            .height(42.dp)
            .clip(pillShape)
            .background(Color(0xFF101B36))
            .border(1.dp, Color(0xFF283B66), pillShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 6.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(5.dp)
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = Color(0xFF38BDF8),
                modifier = Modifier.size(17.dp)
            )
            Text(
                text = label,
                color = Color(0xFFE5EEFF),
                fontWeight = FontWeight.Medium,
                fontSize = 11.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun FrontResultShowcaseCard(
    uiState: FaceSwapUiState,
    onDownloadHd: () -> Unit,
    onSaveToGallery: () -> Unit,
    onInspectResultDetails: () -> Unit,
    onViewResultFullScreen: (Bitmap) -> Unit = {}
) {
    val result = uiState.swapResult
    val currentTimeText = remember(result) {
        SimpleDateFormat("hh:mm a", Locale.getDefault()).format(Date())
    }

    val cardShape = RoundedCornerShape(20.dp)
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = cardShape,
        color = Color(0xE6FFFFFF), // Luminous Frosted Glass (matching Image 1)
        border = BorderStroke(1.2.dp, Color(0x66C084FC))
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Top Header Row: Sparkle + "Result" | "Ready to Download" green badge
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.AutoAwesome,
                        contentDescription = null,
                        tint = Color(0xFF7E22CE),
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = "Result",
                        color = Color(0xFF0F172A),
                        fontWeight = FontWeight.ExtraBold,
                        fontSize = 15.5.sp
                    )
                }

                // Ready to Download emerald pill
                Row(
                    modifier = Modifier
                        .clip(RoundedCornerShape(50))
                        .background(Color(0xFF0A2E26))
                        .border(1.dp, Color(0xFF15694F), RoundedCornerShape(50))
                        .clickable { onInspectResultDetails() }
                        .padding(horizontal = 10.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(5.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = NeonEmerald,
                        modifier = Modifier.size(13.dp)
                    )
                    Text(
                        text = if (result != null) "Ready to Download" else "Preview • Tap to Inspect",
                        color = Color(0xFF6EE7B7),
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 11.sp
                    )
                }
            }

            // Middle Row: [ Source -> Target | Result Image ] + [ Download / Save to Gallery buttons ]
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                // Left visual comparison box
                Row(
                    modifier = Modifier
                        .weight(1.35f)
                        .height(106.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(Color(0xFF131D38))
                        .border(1.dp, Color(0xFF283A66), RoundedCornerShape(14.dp))
                        .clickable { onInspectResultDetails() }
                        .padding(horizontal = 8.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    // Small Source Thumbnail
                    Box(
                        modifier = Modifier
                            .width(48.dp)
                            .height(62.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .border(1.dp, Color(0xFF3B4E7A), RoundedCornerShape(8.dp))
                    ) {
                        if (uiState.sourceBitmap != null) {
                            Image(
                                bitmap = uiState.sourceBitmap.asImageBitmap(),
                                contentDescription = "Source Face",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        } else {
                            Image(
                                painter = painterResource(id = R.drawable.img_sample_portrait),
                                contentDescription = "Source Preview",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }

                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(14.dp)
                    )

                    // Small Target Thumbnail
                    Box(
                        modifier = Modifier
                            .width(48.dp)
                            .height(62.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .border(1.dp, Color(0xFF3B4E7A), RoundedCornerShape(8.dp))
                    ) {
                        if (uiState.targetBitmap != null) {
                            Image(
                                bitmap = uiState.targetBitmap.asImageBitmap(),
                                contentDescription = "Target Photo",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        } else {
                            Image(
                                painter = painterResource(id = R.drawable.img_sample_portrait),
                                contentDescription = "Target Preview",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }

                    Spacer(modifier = Modifier.width(2.dp))

                    // Larger Output Result Portrait (tap to view full-screen with zoom)
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(94.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .border(1.2.dp, Color(0xFF00D4FF).copy(alpha = 0.6f), RoundedCornerShape(10.dp))
                            .clickable {
                                if (result != null) {
                                    onViewResultFullScreen(result.outputBitmap)
                                } else {
                                    onInspectResultDetails()
                                }
                            }
                    ) {
                        if (result != null) {
                            Image(
                                bitmap = result.outputBitmap.asImageBitmap(),
                                contentDescription = "Swapped Result Output",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        } else {
                            Image(
                                painter = painterResource(id = R.drawable.img_sample_portrait),
                                contentDescription = "Result Preview",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }
                }

                // Right Stacked Action Buttons: Download + Save to Gallery
                Column(
                    modifier = Modifier.weight(0.95f),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    val btnShape = RoundedCornerShape(12.dp)
                    // 1. Download Button (Cyan to Purple gradient)
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(44.dp)
                            .clip(btnShape)
                            .background(
                                Brush.horizontalGradient(
                                    colors = listOf(
                                        Color(0xFF00D4FF),
                                        Color(0xFF5B6BF9),
                                        Color(0xFF8B46FF)
                                    )
                                )
                            )
                            .clickable {
                                if (result != null) onDownloadHd() else onInspectResultDetails()
                            }
                            .testTag("btn_save_hd_result"),
                        contentAlignment = Alignment.Center
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Download,
                                contentDescription = null,
                                tint = Color.White,
                                modifier = Modifier.size(18.dp)
                            )
                            Text(
                                text = "Download",
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                fontSize = 13.5.sp
                            )
                        }
                    }

                    // 2. Save to Gallery Button (Dark Navy with glowing border)
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(44.dp)
                            .clip(btnShape)
                            .background(Color(0xFF101932))
                            .border(
                                width = 1.2.dp,
                                brush = Brush.horizontalGradient(
                                    colors = listOf(
                                        Color(0xFF38BDF8).copy(alpha = 0.6f),
                                        Color(0xFF8B5CF6).copy(alpha = 0.8f)
                                    )
                                ),
                                shape = btnShape
                            )
                            .clickable {
                                if (result != null) onSaveToGallery() else onInspectResultDetails()
                            }
                            .testTag("btn_save_result"),
                        contentAlignment = Alignment.Center
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.PhotoLibrary,
                                contentDescription = null,
                                tint = Color.White,
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                text = "Save to Gallery",
                                color = Color.White,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 12.5.sp
                            )
                        }
                    }
                }
            }

            // Bottom Footer Status Row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onInspectResultDetails() }
                    .padding(top = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(18.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF00D4FF)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Person,
                            contentDescription = null,
                            tint = Color(0xFF071022),
                            modifier = Modifier.size(13.dp)
                        )
                    }
                    Text(
                        text = if (result != null) {
                            "Face Swap Completed (${result.stageTimings.totalMs} ms)"
                        } else {
                            "Face Swap Completed"
                        },
                        color = Color(0xFFD0DCF5),
                        fontSize = 11.5.sp,
                        fontWeight = FontWeight.Medium
                    )
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Schedule,
                        contentDescription = null,
                        tint = Color(0xFF60A5FA),
                        modifier = Modifier.size(14.dp)
                    )
                    Text(
                        text = "Completed • $currentTimeText",
                        color = Color(0xFF98A8C8),
                        fontSize = 11.sp
                    )
                }
            }
        }
    }
}
