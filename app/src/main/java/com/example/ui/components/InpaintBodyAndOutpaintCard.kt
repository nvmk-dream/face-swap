package com.example.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccessibilityNew
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Height
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.onnx.OutpaintExtendPreset
import kotlin.math.roundToInt

/**
 * Prominent 1-Tap Folder Auto-Pick Banner for InpaintPage:
 * Lets user select their downloaded model folder (e.g. stable-diffusion-v1-5-inpainting-onnx-fp16
 * or main ONNX folder) and automatically detects + imports all .onnx, weights & tokenizer files.
 */
@Composable
fun InpaintFolderAutoPickButton(
    isBusy: Boolean,
    installedPacksCount: Int,
    onAutoPickFolder: () -> Unit,
    hasRealVisFp16Installed: Boolean = false,
    onUploadRealVisFp16Folder: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(14.dp)
    Surface(
        shape = shape,
        color = Color(0xEBF0F9FF),
        border = BorderStroke(1.3.dp, Color(0xFF0284C7)),
        modifier = modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .clickable(enabled = !isBusy, onClick = onAutoPickFolder)
                    .padding(horizontal = 12.dp, vertical = 10.dp)
                    .testTag("inpaint_folder_autopick_btn"),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Box(
                        modifier = Modifier
                            .size(34.dp)
                            .clip(CircleShape)
                            .background(
                                Brush.linearGradient(
                                    listOf(Color(0xFF0284C7), Color(0xFF6366F1))
                                )
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        if (isBusy) {
                            CircularProgressIndicator(
                                color = Color.White,
                                strokeWidth = 2.dp,
                                modifier = Modifier.size(18.dp)
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.FolderOpen,
                                contentDescription = "Auto-Pick ONNX & SD Model Folder",
                                tint = Color.White,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                    Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
                        Text(
                            text = if (isBusy) "Scanning & Auto-Uploading Models..." else "Folder Auto-Pick (.onnx & SD 1.5 Pack)",
                            color = Color(0xFF0F172A),
                            fontWeight = FontWeight.ExtraBold,
                            fontSize = 13.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = if (installedPacksCount > 0) {
                                "$installedPacksCount SD pack ready • Tap to add or switch folder"
                            } else {
                                "Folder select செய்தால் போதும் — ONNX + Tokenizer auto upload ஆகும்"
                            },
                            color = Color(0xFF0369A1),
                            fontWeight = FontWeight.Medium,
                            fontSize = 11.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
                Spacer(modifier = Modifier.width(6.dp))
                Surface(
                    shape = RoundedCornerShape(50),
                    color = Color(0xFF0284C7)
                ) {
                    Text(
                        text = "Pick",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 11.5.sp,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                    )
                }
            }

            if (onUploadRealVisFp16Folder != null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(1.dp)
                        .background(Color(0x330284C7))
                )
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .background(
                            if (hasRealVisFp16Installed) Color(0x80ECFDF5) else Color(0x66EEF2FF)
                        )
                        .clickable(enabled = !isBusy, onClick = onUploadRealVisFp16Folder)
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                        .testTag("inpaint_upload_realvis_fp16_btn"),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(1.dp)
                    ) {
                        Text(
                            text = "Realistic_Vision_V5.1 (FP16) — தனி Slot",
                            color = Color(0xFF0F172A),
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = if (hasRealVisFp16Installed) {
                                "✓ Installed separately from SD 1.5 (FP16) • Tap to Replace"
                            } else {
                                "SD 1.5 (FP16) உடன் சேராமல் தனி folder-ஆக upload செய்ய தட்டவும்"
                            },
                            color = if (hasRealVisFp16Installed) Color(0xFF047857) else Color(0xFF4338CA),
                            fontWeight = FontWeight.Medium,
                            fontSize = 10.5.sp,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    Surface(
                        shape = RoundedCornerShape(50),
                        color = if (hasRealVisFp16Installed) Color(0xFF059669) else Color(0xFF4F46E5)
                    ) {
                        Text(
                            text = if (hasRealVisFp16Installed) "✓ Replace" else "+ Upload",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
                        )
                    }
                }
            }
        }
    }
}

/**
 * 100% Offline Body Contour Reshape (Lean -> Curvy) & Half-Body -> Full-Body Outpaint Panel.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InpaintBodyAndOutpaintCard(
    hasSourceImage: Boolean,
    curvyAmount: Float,
    hipPositionRatio: Float,
    outpaintPreset: OutpaintExtendPreset,
    hasSdPackInstalled: Boolean,
    isAdjusting: Boolean,
    onCurvyAmountChanged: (Float) -> Unit,
    onHipPositionChanged: (Float) -> Unit,
    onSelectOutpaintPreset: (OutpaintExtendPreset) -> Unit,
    onResetEdits: () -> Unit,
    modifier: Modifier = Modifier
) {
    val cardShape = RoundedCornerShape(16.dp)
    val darkSlate = Color(0xFF0F172A)
    val mutedSlate = Color(0xFF475569)
    val hasActiveChanges = curvyAmount > 0.01f || outpaintPreset != OutpaintExtendPreset.NONE

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(cardShape)
            .background(Color(0xD9FFFFFF))
            .border(1.2.dp, Color(0x99FFFFFF), cardShape)
            .padding(horizontal = 14.dp, vertical = 10.dp)
            .testTag("inpaint_body_outpaint_card"),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.weight(1f)
            ) {
                Icon(
                    imageVector = Icons.Default.AccessibilityNew,
                    contentDescription = "Body Reshape and Full Body Outpaint",
                    tint = Color(0xFF7C3AED),
                    modifier = Modifier.size(18.dp)
                )
                Column {
                    Text(
                        text = "Body Reshape (Curvy) & Full-Body Extend",
                        color = darkSlate,
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.5.sp
                    )
                    Text(
                        text = "100% Offline • Lean→Curvy & Half→Feet Canvas",
                        color = mutedSlate,
                        fontSize = 10.5.sp
                    )
                }
            }
            if (isAdjusting) {
                CircularProgressIndicator(
                    color = Color(0xFF7C3AED),
                    strokeWidth = 2.dp,
                    modifier = Modifier.size(16.dp)
                )
            } else if (hasActiveChanges) {
                Row(
                    modifier = Modifier
                        .heightIn(min = 32.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color(0xFFFEE2E2))
                        .clickable(onClick = onResetEdits)
                        .padding(horizontal = 10.dp, vertical = 5.dp)
                        .testTag("inpaint_body_reset_btn"),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Reset Body Reshape and Extend",
                        tint = Color(0xFFDC2626),
                        modifier = Modifier.size(13.dp)
                    )
                    Text(
                        text = "Reset",
                        color = Color(0xFFDC2626),
                        fontWeight = FontWeight.Bold,
                        fontSize = 11.sp
                    )
                }
            }
        }

        // 1. Offline Curvy Contour Slider
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Curvy",
                color = darkSlate,
                fontWeight = FontWeight.SemiBold,
                fontSize = 12.5.sp,
                modifier = Modifier.width(62.dp)
            )
            Slider(
                value = curvyAmount.coerceIn(0f, 1f),
                onValueChange = onCurvyAmountChanged,
                valueRange = 0f..1f,
                steps = 19,
                enabled = hasSourceImage,
                modifier = Modifier
                    .weight(1f)
                    .height(34.dp)
                    .testTag("inpaint_curvy_slider")
            )
            Text(
                text = "${(curvyAmount * 100f).roundToInt()}%",
                color = if (curvyAmount > 0.01f) Color(0xFF7C3AED) else mutedSlate,
                fontWeight = FontWeight.Bold,
                fontSize = 12.5.sp,
                modifier = Modifier.width(42.dp)
            )
        }

        // Optional Hip Band Alignment Slider (only shown when Curvy > 0 so card stays compact)
        if (curvyAmount > 0.01f) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Hip Y",
                    color = mutedSlate,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.sp,
                    modifier = Modifier.width(62.dp)
                )
                Slider(
                    value = hipPositionRatio.coerceIn(0.45f, 0.85f),
                    onValueChange = onHipPositionChanged,
                    valueRange = 0.45f..0.85f,
                    steps = 7,
                    enabled = hasSourceImage,
                    modifier = Modifier
                        .weight(1f)
                        .height(32.dp)
                        .testTag("inpaint_hip_y_slider")
                )
                Text(
                    text = "${(hipPositionRatio * 100f).roundToInt()}%",
                    color = mutedSlate,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 11.5.sp,
                    modifier = Modifier.width(42.dp)
                )
            }
        }

        // 2. Half-Body -> Full-Body Outpaint Presets
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Height,
                    contentDescription = "Extend Photo Downwards",
                    tint = Color(0xFF0284C7),
                    modifier = Modifier.size(16.dp)
                )
                Text(
                    text = "Half → Full Body Outpaint:",
                    color = darkSlate,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.sp
                )
            }
            if (!hasSdPackInstalled && outpaintPreset != OutpaintExtendPreset.NONE) {
                Text(
                    text = "Needs SD 1.5 Pack",
                    color = Color(0xFFD97706),
                    fontWeight = FontWeight.Bold,
                    fontSize = 10.5.sp
                )
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutpaintExtendPreset.entries.forEach { preset ->
                val isSelected = outpaintPreset == preset
                Box(
                    modifier = Modifier
                        .heightIn(min = 38.dp)
                        .clip(RoundedCornerShape(50))
                        .background(
                            if (isSelected) Color(0xFF0284C7) else Color(0xFFF1F5F9)
                        )
                        .border(
                            1.dp,
                            if (isSelected) Color(0xFF38BDF8) else Color(0xFFCBD5E1),
                            RoundedCornerShape(50)
                        )
                        .clickable(enabled = hasSourceImage && !isAdjusting) {
                            onSelectOutpaintPreset(preset)
                        }
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                        .testTag("inpaint_outpaint_preset_${preset.name.lowercase()}"),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = preset.shortLabel,
                        color = if (isSelected) Color.White else darkSlate,
                        fontWeight = FontWeight.Bold,
                        fontSize = 12.sp
                    )
                }
            }
        }
    }
}
