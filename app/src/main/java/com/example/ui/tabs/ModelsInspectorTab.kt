package com.example.ui.tabs

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Radar
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material.icons.filled.UploadFile
import com.example.onnx.ClipTokenizerStatus
import com.example.onnx.GhostComponentFeasibility
import com.example.onnx.InstalledSdInpaintPack
import com.example.onnx.ModelRequirementLevel
import com.example.onnx.ModelSlot
import com.example.onnx.OnnxModelInspection
import com.example.onnx.OnnxProtobufInspector
import com.example.ui.theme.AmberWarning
import com.example.ui.theme.CoralError
import com.example.ui.theme.ElectricCyan
import com.example.ui.theme.NeonEmerald
import com.example.onnx.OnnxMemoryServiceState
import com.example.onnx.SdInpaintPackType
import com.example.ui.theme.ObsidianBg
import com.example.ui.theme.RoyalViolet

@Composable
fun ModelsInspectorTab(
    inspections: List<OnnxModelInspection>,
    isInspecting: Boolean,
    memoryServiceState: OnnxMemoryServiceState = OnnxMemoryServiceState(),
    segformerAutoSelectedForAccessory: Boolean = false,
    sdPackDetails: List<InstalledSdInpaintPack> = emptyList(),
    clipTokenizerStatus: ClipTokenizerStatus = ClipTokenizerStatus(),
    onImportModelForSlot: (ModelSlot) -> Unit,
    onRefreshModels: () -> Unit,
    onImportAllFromFolder: () -> Unit = {},
    onUploadClipTokenizer: () -> Unit = {},
    onUploadSingleTokenizerFile: (String) -> Unit = {},
    onUploadRealVisFp16Folder: () -> Unit = {},
    onUploadSd15Fp16Folder: () -> Unit = {},
    onDeleteSdInpaintPack: (SdInpaintPackType) -> Unit = {},
    onPreloadMemoryModels: () -> Unit = {},
    onReleaseMemoryModels: () -> Unit = {},
    modifier: Modifier = Modifier
) {
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
            // 0. ONNX Runtime In-Memory Model Service Card + Tamil Model Usage Guide
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                border = androidx.compose.foundation.BorderStroke(1.2.dp, Color(0x6638BDF8)),
                colors = CardDefaults.cardColors(containerColor = Color(0xE6FFFFFF))
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text = "தற்போது மெமரியில் (RAM) உள்ள மாடல்கள் & அவற்றின் பயன்கள்",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = Color(0xFF0F172A)
                    )
                    Text(
                        text = "ONNX Runtime In-Memory Service • Active in RAM: ${memoryServiceState.activeSessionCount} மாடல்கள்",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0369A1)
                    )
                    Text(
                        text = "கீழே உள்ள பட்டியலில் எந்தெந்த .onnx மாடல்கள் தற்போது நேரடியாக RAM மெமரியில் ஏற்றப்பட்டுள்ளன என்பதையும், ஒவ்வொரு மாடலும் எதற்காகப் பயன்படுகிறது என்பதையும் தமிழில் காணலாம்:",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF334155)
                    )

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(Color(0xB3F1F5F9))
                            .border(1.dp, Color(0x4D38BDF8), RoundedCornerShape(10.dp))
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        MemorySlotRow(
                            label = "1. w600k_r50.onnx (ArcFace 512-D)",
                            tamilPurpose = ModelSlot.RECOGNIZER.tamilTitle + " — " + ModelSlot.RECOGNIZER.tamilRoleSummary,
                            isLoaded = memoryServiceState.w600kLoaded,
                            note = memoryServiceState.loadedSessions[ModelSlot.RECOGNIZER]?.statusNote
                        )
                        MemorySlotRow(
                            label = "2. gfpgan_1.4.onnx (512×512 HD Enhancer)",
                            tamilPurpose = ModelSlot.ENHANCEMENT.tamilTitle + " — " + ModelSlot.ENHANCEMENT.tamilRoleSummary,
                            isLoaded = memoryServiceState.gfpganLoaded,
                            note = memoryServiceState.loadedSessions[ModelSlot.ENHANCEMENT]?.statusNote
                        )
                        MemorySlotRow(
                            label = if (segformerAutoSelectedForAccessory) {
                                "3. segformer_B5_ce.onnx (Auto-Selected: Cap/Glasses)"
                            } else {
                                "3. segformer_B5_ce.onnx (Head/Hair Parser)"
                            },
                            tamilPurpose = ModelSlot.SEGMENTATION.tamilTitle + " — " + ModelSlot.SEGMENTATION.tamilRoleSummary,
                            isLoaded = memoryServiceState.segformerLoaded,
                            isHighlighted = segformerAutoSelectedForAccessory,
                            note = memoryServiceState.loadedSessions[ModelSlot.SEGMENTATION]?.statusNote
                        )
                        MemorySlotRow(
                            label = "4. det_10g.onnx (SCRFD Face Detector)",
                            tamilPurpose = ModelSlot.DETECTOR.tamilTitle + " — " + ModelSlot.DETECTOR.tamilRoleSummary,
                            isLoaded = memoryServiceState.detectorLoaded,
                            note = memoryServiceState.loadedSessions[ModelSlot.DETECTOR]?.statusNote
                        )
                        MemorySlotRow(
                            label = "5. inswapper_128.onnx + emap[512×512]",
                            tamilPurpose = ModelSlot.SWAPPER.tamilTitle + " — " + ModelSlot.SWAPPER.tamilRoleSummary,
                            isLoaded = memoryServiceState.inswapperLoaded,
                            note = if (memoryServiceState.emapLoaded) {
                                "emap[512×512] மெமரியில் உள்ளது • ${memoryServiceState.loadedSessions[ModelSlot.SWAPPER]?.statusNote ?: ""}"
                            } else {
                                memoryServiceState.loadedSessions[ModelSlot.SWAPPER]?.statusNote
                            }
                        )
                        MemorySlotRow(
                            label = "6. mivolo_v2_age.onnx (MiVOLO v2 Age Estimator)",
                            tamilPurpose = ModelSlot.AGE_ESTIMATOR.tamilTitle + " — " + ModelSlot.AGE_ESTIMATOR.tamilRoleSummary,
                            isLoaded = memoryServiceState.mivoloLoaded,
                            note = memoryServiceState.loadedSessions[ModelSlot.AGE_ESTIMATOR]?.statusNote
                        )
                        MemorySlotRow(
                            label = "7. face_reaging.onnx (FRAN Face Re-Aging)",
                            tamilPurpose = ModelSlot.FACE_REAGING.tamilTitle + " — " + ModelSlot.FACE_REAGING.tamilRoleSummary,
                            isLoaded = memoryServiceState.reagingLoaded,
                            note = memoryServiceState.loadedSessions[ModelSlot.FACE_REAGING]?.statusNote
                        )
                        MemorySlotRow(
                            label = "8. facemesh-v2-v1.onnx (3D FaceMesh V2 468-Pt)",
                            tamilPurpose = ModelSlot.FACE_MESH_3D.tamilTitle + " — " + ModelSlot.FACE_MESH_3D.tamilRoleSummary,
                            isLoaded = memoryServiceState.faceMeshLoaded,
                            note = memoryServiceState.loadedSessions[ModelSlot.FACE_MESH_3D]?.statusNote
                        )
                        MemorySlotRow(
                            label = "9. 6DRepNet360_Slim.onnx / SixDRepNet.onnx (6D Head Pose)",
                            tamilPurpose = ModelSlot.HEAD_POSE.tamilTitle + " — " + ModelSlot.HEAD_POSE.tamilRoleSummary,
                            isLoaded = memoryServiceState.headPoseLoaded,
                            note = memoryServiceState.loadedSessions[ModelSlot.HEAD_POSE]?.statusNote
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Button(
                            onClick = onPreloadMemoryModels,
                            enabled = !isInspecting && !memoryServiceState.isLoading,
                            modifier = Modifier
                                .weight(1f)
                                .testTag("preload_memory_models_button"),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text(
                                text = if (memoryServiceState.isLoading) "ஏற்றப்படுகிறது..." else "RAM-ல் ஏற்று (Load to RAM)",
                                fontWeight = FontWeight.Bold
                            )
                        }
                        OutlinedButton(
                            onClick = onReleaseMemoryModels,
                            enabled = memoryServiceState.activeSessionCount > 0 && !memoryServiceState.isLoading,
                            modifier = Modifier.testTag("release_memory_models_button"),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text("Release RAM")
                        }
                    }
                }
            }

            // 1. GHOST 2.0 Feasibility Statement & Directory Structure Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Info,
                                contentDescription = "Model Status",
                                tint = ElectricCyan
                            )
                            Text(
                                text = "Local Model Status & Directory Layout",
                                style = MaterialTheme.typography.titleMedium,
                                color = ElectricCyan
                            )
                        }
                        OutlinedButton(
                            onClick = onRefreshModels,
                            enabled = !isInspecting,
                            modifier = Modifier.testTag("refresh_models_button")
                        ) {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = "Rescan Models",
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Rescan")
                        }
                    }

                    Button(
                        onClick = onImportAllFromFolder,
                        enabled = !isInspecting,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("import_all_folder_button"),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Folder,
                            contentDescription = "Auto-Import All from Folder",
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "1-Tap Auto-Import All .onnx from Folder",
                            fontWeight = FontWeight.Bold
                        )
                    }

                    Text(
                        text = "Tip: Put all your .onnx files in one folder (e.g. Downloads) and tap the button above once. The app will load all models at once and remember the folder!",
                        style = MaterialTheme.typography.labelMedium,
                        color = NeonEmerald
                    )

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
                            text = "FULL GHOST 2.0 ANDROID PORT IS NOT CURRENTLY FEASIBLE AS IMPLEMENTED",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = AmberWarning
                        )
                        Text(
                            text = "Why: Official GHOST 2.0 checkpoints (aligner_1020_gaze_final.ckpt ~1.15 GB and blender_lama.ckpt ~680 MB) are PyTorch Lightning checkpoints coupled to PyTorch3D / Nvdiffrast CUDA mesh rasterizers and EMOCA/FLAME pickle graphs that cannot run in Android ONNX Runtime. Below is the Android-compatible offline architecture using ONNX Runtime Mobile + native CV modules.",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                    // Local Model Directory Tree
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(ObsidianBg)
                            .padding(12.dp),
                        verticalAlignment = Alignment.Top,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Folder,
                            contentDescription = "Local Model Directory",
                            tint = NeonEmerald
                        )
                        Text(
                            text = "Local On-Device Storage Hierarchy (filesDir/models/):\n" +
                                "├── face_swap/      (det_10g.onnx, w600k_r50.onnx, inswapper_128.onnx)\n" +
                                "├── head_swap/      (Android GHOST Head Pose & Cranial Warp Config)\n" +
                                "├── segmentation/   (segformer_B5_ce.onnx — or built-in Geodesic Matte)\n" +
                                "├── matting/        (modnet.onnx — or built-in Joint Bilateral Filter)\n" +
                                "├── inpainting/     (lama_fp32.onnx — or built-in Boundary Marching)\n" +
                                "├── depth/          (Depth Anything V2 Small [P1] / sentis-MiDaS [P2])\n" +
                                "├── age_estimation/ (mivolo_v2_age.onnx — MiVOLO v2 Face+Body Age)\n" +
                                "├── re_aging/       (face_reaging.onnx — FRAN U-Net / CV Fallback)\n" +
                                "├── face_mesh/      (facemesh-v2-v1.onnx — 468-Pt 3D Mesh / PnP-106)\n" +
                                "├── head_pose/      (6DRepNet360_Slim.onnx / SixDRepNet.onnx — 6D Pitch/Yaw/Roll)\n" +
                                "└── enhancement/    (gfpgan_1.4.onnx / gfpgan-v1.4-w16-v1.onnx)",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }

            // 2. Quick Direct Upload Card for 6DRepNet360_Slim.onnx / SixDRepNet.onnx
            val headPoseInspection = inspections.firstOrNull { it.slot == ModelSlot.HEAD_POSE }
            val isHeadPoseInstalled = headPoseInspection?.isValidOnnx == true
            val isHeadPoseRamReady = memoryServiceState.headPoseLoaded
            val headPoseAccent = if (isHeadPoseInstalled) NeonEmerald else ElectricCyan
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("models_sixdrepnet_slim_card"),
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(1.2.dp, headPoseAccent.copy(alpha = 0.6f)),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            modifier = Modifier.weight(1f),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(
                                imageVector = if (isHeadPoseInstalled) Icons.Default.CheckCircle else Icons.Default.Radar,
                                contentDescription = "6DRepNet360_Slim Head Pose ONNX",
                                tint = headPoseAccent,
                                modifier = Modifier.size(22.dp)
                            )
                            Text(
                                text = "6D Head Pose (6DRepNet360_Slim.onnx / SixDRepNet.onnx)",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = headPoseAccent
                            )
                        }
                        Surface(
                            shape = RoundedCornerShape(50),
                            color = headPoseAccent.copy(alpha = 0.16f)
                        ) {
                            Text(
                                text = when {
                                    isHeadPoseRamReady -> "RAM Ready ✓"
                                    isHeadPoseInstalled -> "Installed ✓"
                                    else -> "Optional"
                                },
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = headPoseAccent,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                            )
                        }
                    }
                    Text(
                        text = if (isHeadPoseInstalled) {
                            headPoseInspection?.verificationNote
                                ?: "Installed in models/head_pose/ — Extreme profile (±85°) Pitch/Yaw/Roll active."
                        } else {
                            "6DRepNet360_Slim.onnx (அல்லது SixDRepNet.onnx) upload செய்தால் பக்கவாட்டு முகம் (±85°), தொப்பி, கண்ணாடி மற்றும் சாய்ந்த தலையின் 3D கோணங்கள் துல்லியமாக கணிக்கப்படும்."
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Button(
                            onClick = { onImportModelForSlot(ModelSlot.HEAD_POSE) },
                            enabled = !isInspecting,
                            modifier = Modifier
                                .weight(1f)
                                .heightIn(min = 48.dp)
                                .testTag("models_import_sixdrepnet_slim_button"),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(
                                text = if (isHeadPoseInstalled) {
                                    "Replace 6DRepNet360_Slim.onnx"
                                } else {
                                    "Upload 6DRepNet360_Slim.onnx"
                                },
                                fontWeight = FontWeight.Bold
                            )
                        }
                        OutlinedButton(
                            onClick = onImportAllFromFolder,
                            enabled = !isInspecting,
                            modifier = Modifier
                                .heightIn(min = 48.dp)
                                .testTag("models_import_sixdrepnet_folder_button"),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text("Auto-Folder")
                        }
                    }
                }
            }

            // 3. SD 1.5 Inpaint Packs & CLIP Tokenizer Card (Direct Upload Option)
            SdInpaintAndTokenizerCard(
                sdPackDetails = sdPackDetails,
                clipTokenizerStatus = clipTokenizerStatus,
                isBusy = isInspecting,
                onUploadClipTokenizer = onUploadClipTokenizer,
                onUploadSingleTokenizerFile = onUploadSingleTokenizerFile,
                onUploadRealVisFp16Folder = onUploadRealVisFp16Folder,
                onUploadSd15Fp16Folder = onUploadSd15Fp16Folder,
                onDeleteSdInpaintPack = onDeleteSdInpaintPack,
                onImportSdFolder = onImportAllFromFolder
            )

            // 3. Installed vs Missing Model Slots
            Text(
                text = "On-Device ONNX Model Slots • ஒவ்வொரு மாடலின் பணி (தமிழில்)",
                style = MaterialTheme.typography.titleMedium,
                color = ElectricCyan
            )

            ModelSlot.entries.forEach { slot ->
                val inspection = inspections.firstOrNull { it.slot == slot }
                val isLoadedInRam = memoryServiceState.loadedSessions[slot]?.isLoadedInMemory == true
                val isAutoSelected = slot == ModelSlot.SEGMENTATION && segformerAutoSelectedForAccessory
                ModelInspectionCard(
                    slot = slot,
                    inspection = inspection,
                    isLoadedInRam = isLoadedInRam,
                    isAutoSelected = isAutoSelected,
                    onImportModel = { onImportModelForSlot(slot) }
                )
            }

            // 3. Official GHOST 2.0 Component Feasibility Report
            Text(
                text = "GHOST 2.0 Component-by-Component Android Feasibility Matrix",
                style = MaterialTheme.typography.titleMedium,
                color = NeonEmerald
            )

            OnnxProtobufInspector.GHOST_2_FEASIBILITY_MATRIX.forEach { item ->
                GhostFeasibilityCard(item = item)
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ModelInspectionCard(
    slot: ModelSlot,
    inspection: OnnxModelInspection?,
    isLoadedInRam: Boolean,
    isAutoSelected: Boolean = false,
    onImportModel: () -> Unit
) {
    val isReady = inspection?.isValidOnnx == true
    val statusColor = when {
        isReady -> NeonEmerald
        slot.requirementLevel == ModelRequirementLevel.OPTIONAL_HEAD_SWAP -> ElectricCyan
        else -> CoralError
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(
            width = if (isAutoSelected) 2.dp else 1.2.dp,
            color = if (isAutoSelected) Color(0xFF0284C7) else Color(0x6638BDF8)
        ),
        colors = CardDefaults.cardColors(containerColor = if (isAutoSelected) Color(0xF2EFF6FF) else Color(0xE6FFFFFF))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(
                            imageVector = if (isReady) Icons.Default.CheckCircle else Icons.Default.WarningAmber,
                            contentDescription = slot.displayName,
                            tint = statusColor,
                            modifier = Modifier.size(20.dp)
                        )
                        Text(
                            text = if (isReady) "✓ Installed: ${slot.canonicalFileName}" else "✗ Missing: ${slot.canonicalFileName}",
                            style = MaterialTheme.typography.titleMedium,
                            color = statusColor
                        )
                    }

                    Surface(
                        shape = RoundedCornerShape(50),
                        color = if (isLoadedInRam) NeonEmerald.copy(alpha = 0.2f) else ObsidianBg
                    ) {
                        Text(
                            text = if (isLoadedInRam) "RAM-ல் உள்ளது ●" else "RAM-ல் இல்லை ○",
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            color = if (isLoadedInRam) NeonEmerald else ElectricCyan,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                        )
                    }
                }

                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Surface(
                        shape = RoundedCornerShape(50),
                        color = if (slot.requirementLevel == ModelRequirementLevel.OPTIONAL_HEAD_SWAP) {
                            RoyalViolet.copy(alpha = 0.22f)
                        } else {
                            AmberWarning.copy(alpha = 0.20f)
                        }
                    ) {
                        Text(
                            text = slot.requirementLevel.label,
                            style = MaterialTheme.typography.labelMedium,
                            color = if (slot.requirementLevel == ModelRequirementLevel.OPTIONAL_HEAD_SWAP) {
                                ElectricCyan
                            } else {
                                AmberWarning
                            },
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                        )
                    }
                    if (isAutoSelected) {
                        Surface(
                            shape = RoundedCornerShape(50),
                            color = Color(0xFF0284C7),
                            modifier = Modifier.testTag("segformer_auto_selected_badge")
                        ) {
                            Text(
                                text = "Auto-Selected (Cap/Glasses)",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                            )
                        }
                    }
                }
            }

            // Tamil usage explanation box
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color(0xFFF0FDF4))
                    .border(1.dp, Color(0xFF86EFAC), RoundedCornerShape(12.dp))
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "தமிழ் விளக்கம்: ${slot.tamilTitle}",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF065F46)
                )
                Text(
                    text = slot.tamilRoleSummary,
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFF1E293B)
                )
            }

            Text(
                text = "Directory: ${slot.categoryTitle}${slot.canonicalFileName} • Approx Size: ${slot.approximateSizeLabel}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurface
            )

            Text(
                text = slot.roleSummary,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(ObsidianBg)
                    .border(1.dp, statusColor.copy(alpha = 0.35f), RoundedCornerShape(12.dp))
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "• Input:  ${slot.expectedInputSignature}\n• Output: ${slot.expectedOutputSignature}",
                    style = MaterialTheme.typography.labelMedium
                )
                if (inspection != null && inspection.isPresent) {
                    val mb = "%.2f MB".format(inspection.fileSizeBytes / (1024.0 * 1024.0))
                    Text(
                        text = "• Installed Size: $mb | SHA-256: ${inspection.sha256Prefix ?: "n/a"} | IR v${inspection.irVersion ?: "?"}",
                        style = MaterialTheme.typography.labelMedium,
                        color = NeonEmerald
                    )
                    if (inspection.runtimeInputs.isNotEmpty()) {
                        val inDesc = inspection.runtimeInputs.joinToString("; ") {
                            "${it.name}: ${it.elementType}${it.formattedShape()}"
                        }
                        Text(
                            text = "• Verified Runtime Inputs: $inDesc",
                            style = MaterialTheme.typography.labelMedium,
                            color = ElectricCyan
                        )
                    }
                    if (slot == ModelSlot.SWAPPER) {
                        Text(
                            text = "• Embedded [512, 512] emap Matrix: ${if (inspection.hasEmbeddedEmap512x512) "EXTRACTED" else "NOT FOUND"}",
                            style = MaterialTheme.typography.labelMedium,
                            color = NeonEmerald
                        )
                    }
                } else {
                    Text(
                        text = "• Action: ${inspection?.verificationNote ?: "Place ${slot.canonicalFileName} in ${slot.categoryTitle}"}",
                        style = MaterialTheme.typography.labelMedium,
                        color = AmberWarning
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "License: ${slot.licenseSummary}",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Button(
                    onClick = onImportModel,
                    modifier = Modifier.testTag("import_model_${slot.id}_button"),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.FileOpen,
                        contentDescription = "Import ${slot.canonicalFileName}",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(if (isReady) "Replace .onnx" else "Import .onnx")
                }
            }
        }
    }
}

@Composable
private fun GhostFeasibilityCard(item: GhostComponentFeasibility) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = item.componentName,
                    style = MaterialTheme.typography.titleMedium,
                    color = ElectricCyan,
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = if (item.runsDirectlyInAndroidOrt) "ONNX DIRECT: YES" else "PYTORCH CKPT: NO",
                    style = MaterialTheme.typography.labelMedium,
                    color = if (item.runsDirectlyInAndroidOrt) NeonEmerald else AmberWarning
                )
            }
            Text(
                text = "Purpose: ${item.purpose}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "Framework: ${item.framework} | Size: ${item.approxSize} | RAM: ${item.expectedRam}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "I/O: ${item.inputOutputSpec}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Android Strategy: ${item.conversionAndReplacementStatus}",
                style = MaterialTheme.typography.bodyMedium,
                color = NeonEmerald
            )
            Text(
                text = "Hardware: ${item.hardwareCompatibility} | License: ${item.license}",
                style = MaterialTheme.typography.labelMedium,
                color = AmberWarning
            )
        }
    }
}

@Composable
private fun MemorySlotRow(
    label: String,
    tamilPurpose: String,
    isLoaded: Boolean,
    note: String?,
    isHighlighted: Boolean = false
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(
                if (isHighlighted) Color(0xFF0C2340) else MaterialTheme.colorScheme.surface.copy(alpha = 0.55f)
            )
            .then(
                if (isHighlighted) Modifier.border(1.2.dp, Color(0xFF38BDF8), RoundedCornerShape(8.dp)) else Modifier
            )
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = ElectricCyan,
                modifier = Modifier.weight(1f)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = if (isLoaded) "RAM-ல் உள்ளது ● (IN RAM)" else "டிஸ்க்கில் ○ (ON DISK)",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Bold,
                color = if (isLoaded) NeonEmerald else AmberWarning
            )
        }
        Text(
            text = tamilPurpose,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface
        )
        if (!note.isNullOrBlank()) {
            Text(
                text = note,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun SdInpaintAndTokenizerCard(
    sdPackDetails: List<InstalledSdInpaintPack>,
    clipTokenizerStatus: ClipTokenizerStatus,
    isBusy: Boolean,
    onUploadClipTokenizer: () -> Unit,
    onUploadSingleTokenizerFile: (String) -> Unit = {},
    onUploadRealVisFp16Folder: () -> Unit = {},
    onUploadSd15Fp16Folder: () -> Unit = {},
    onDeleteSdInpaintPack: (SdInpaintPackType) -> Unit = {},
    onImportSdFolder: () -> Unit
) {
    val isTokenizerReady = clipTokenizerStatus.isReady
    val realVisFp16Pack = sdPackDetails.firstOrNull { it.packType == SdInpaintPackType.REALVIS_FP16 }
    val sd15Fp16Pack = sdPackDetails.firstOrNull { it.packType == SdInpaintPackType.SD15_FP16 }
    val cardBorder = if (isTokenizerReady) Color(0xFF10B981) else Color(0xFFF59E0B)
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("sd_inpaint_tokenizer_card"),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.4.dp, cardBorder.copy(alpha = 0.7f)),
        colors = CardDefaults.cardColors(containerColor = Color(0xF2FFFFFF))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    modifier = Modifier.weight(1f),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = if (isTokenizerReady) Icons.Default.CheckCircle else Icons.Default.WarningAmber,
                        contentDescription = "CLIP Tokenizer Status",
                        tint = if (isTokenizerReady) Color(0xFF059669) else Color(0xFFD97706),
                        modifier = Modifier.size(22.dp)
                    )
                    Text(
                        text = "SD 1.5 Inpaint & CLIP Tokenizer",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = Color(0xFF0F172A)
                    )
                }
                Surface(
                    shape = RoundedCornerShape(50),
                    color = if (isTokenizerReady) Color(0xFFDCFCE7) else Color(0xFFFEF3C7)
                ) {
                    Text(
                        text = if (isTokenizerReady) "Tokenizer Ready ✓" else "Tokenizer Missing",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = if (isTokenizerReady) Color(0xFF065F46) else Color(0xFF92400E),
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
                    )
                }
            }

            Text(
                text = "SD 1.5 Inpaint (FP32 / INT8 / FP16) மற்றும் Head Replacement Stage 5-க்கு OpenAI CLIP BPE Tokenizer (vocab.json + merges.txt அல்லது tokenizer.json) தேவை. ஒருமுறை upload செய்தால் எல்லா SD மாடல்களுக்கும் வேலை செய்யும்.",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFF334155)
            )

            // Tokenizer File Chips
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                TokenizerFileBadge(
                    name = "vocab.json",
                    present = clipTokenizerStatus.hasVocabJson
                )
                TokenizerFileBadge(
                    name = "merges.txt",
                    present = clipTokenizerStatus.hasMergesTxt
                )
                TokenizerFileBadge(
                    name = "tokenizer.json (Alt)",
                    present = clipTokenizerStatus.hasTokenizerJson,
                    optionalIfPairReady = clipTokenizerStatus.hasVocabJson && clipTokenizerStatus.hasMergesTxt
                )
            }

            if (sdPackDetails.isNotEmpty()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(Color(0xFFF1F5F9))
                        .border(1.dp, Color(0xFFCBD5E1), RoundedCornerShape(10.dp))
                        .padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "Installed SD Inpaint Packs (${sdPackDetails.size}):",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0F172A)
                    )
                    sdPackDetails.forEach { pack ->
                        val sizeMb = "%.1f MB".format(pack.totalSizeBytes / (1024.0 * 1024.0))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "• ${pack.packType.displayLabel} ($sizeMb)",
                                style = MaterialTheme.typography.labelMedium,
                                color = Color(0xFF1E293B),
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = if (pack.isReadyWithTokenizer) "✓ Ready" else "⚠️ Tokenizer Missing",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = if (pack.isReadyWithTokenizer) Color(0xFF059669) else Color(0xFFB45309)
                            )
                        }
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = onUploadClipTokenizer,
                    enabled = !isBusy,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                        .testTag("upload_clip_tokenizer_button"),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.UploadFile,
                        contentDescription = "Upload CLIP Tokenizer Files",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (isTokenizerReady) {
                            "Update vocab.json / merges.txt"
                        } else {
                            "Upload CLIP Tokenizer"
                        },
                        fontWeight = FontWeight.Bold
                    )
                }
                OutlinedButton(
                    onClick = onImportSdFolder,
                    enabled = !isBusy,
                    modifier = Modifier
                        .heightIn(min = 48.dp)
                        .testTag("import_sd_folder_button"),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Folder,
                        contentDescription = "Import SD Folder",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("SD Folder", fontWeight = FontWeight.SemiBold)
                }
            }

            // Card 1: Inpaint Dedicated Slot — Realistic_Vision_V5.1 (FP16)
            DedicatedSdModelSlotCard(
                title = "Card 1 • Inpaint Slot: Realistic_Vision_V5.1 (FP16)",
                roleDescription = "Inpaint Page (Saree / Curvy / Outpaint / Prompt Edit)-க்கு மட்டுமே பயன்படும் தனி slot.",
                installedPack = realVisFp16Pack,
                accentColor = Color(0xFF4F46E5),
                isBusy = isBusy,
                uploadButtonTag = "models_upload_realvis_fp16_button",
                clearButtonTag = "models_clear_realvis_fp16_button",
                onUploadClick = onUploadRealVisFp16Folder,
                onClearClick = { onDeleteSdInpaintPack(SdInpaintPackType.REALVIS_FP16) }
            )

            // Card 2: Face Swap / Head Seam Bridge Dedicated Slot — SD 1.5 Inpaint (FP16)
            DedicatedSdModelSlotCard(
                title = "Card 2 • Face Swap Seam: SD 1.5 Inpaint (FP16)",
                roleDescription = "Face Swap Option B+C & Head Replacement Stage 5 Seam Bridge-க்கு பயன்படும் தனி slot.",
                installedPack = sd15Fp16Pack,
                accentColor = Color(0xFF0284C7),
                isBusy = isBusy,
                uploadButtonTag = "models_upload_sd15_fp16_button",
                clearButtonTag = "models_clear_sd15_fp16_button",
                onUploadClick = onUploadSd15Fp16Folder,
                onClearClick = { onDeleteSdInpaintPack(SdInpaintPackType.SD15_FP16) }
            )

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = { onUploadSingleTokenizerFile("vocab.json") },
                    enabled = !isBusy,
                    modifier = Modifier.heightIn(min = 40.dp)
                ) {
                    Text(if (clipTokenizerStatus.hasVocabJson) "✓ Replace vocab.json" else "+ Pick vocab.json", fontWeight = FontWeight.SemiBold)
                }
                OutlinedButton(
                    onClick = { onUploadSingleTokenizerFile("merges.txt") },
                    enabled = !isBusy,
                    modifier = Modifier.heightIn(min = 40.dp)
                ) {
                    Text(if (clipTokenizerStatus.hasMergesTxt) "✓ Replace merges.txt" else "+ Pick merges.txt", fontWeight = FontWeight.SemiBold)
                }
                OutlinedButton(
                    onClick = { onUploadSingleTokenizerFile("tokenizer.json") },
                    enabled = !isBusy,
                    modifier = Modifier.heightIn(min = 40.dp)
                ) {
                    Text(if (clipTokenizerStatus.hasTokenizerJson) "✓ Replace tokenizer.json" else "+ Pick tokenizer.json", fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}

@Composable
private fun DedicatedSdModelSlotCard(
    title: String,
    roleDescription: String,
    installedPack: InstalledSdInpaintPack?,
    accentColor: Color,
    isBusy: Boolean,
    uploadButtonTag: String,
    clearButtonTag: String,
    onUploadClick: () -> Unit,
    onClearClick: () -> Unit
) {
    val isReady = installedPack != null
    val sizeLabel = installedPack?.let {
        val gb = it.totalSizeBytes / (1024.0 * 1024.0 * 1024.0)
        if (gb >= 0.8) "%.2f GB".format(gb)
        else "%.0f MB".format(it.totalSizeBytes / (1024.0 * 1024.0))
    }
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        color = if (isReady) Color(0xFFF0FDF4) else Color(0xFFF8FAFC),
        border = BorderStroke(
            1.2.dp,
            if (isReady) Color(0xFF10B981) else accentColor.copy(alpha = 0.5f)
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    modifier = Modifier.weight(1f),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = if (isReady) Icons.Default.CheckCircle else Icons.Default.Folder,
                        contentDescription = title,
                        tint = if (isReady) Color(0xFF059669) else accentColor,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = Color(0xFF0F172A)
                    )
                }
                Surface(
                    shape = RoundedCornerShape(50),
                    color = if (isReady) Color(0xFFDCFCE7) else Color(0xFFE0E7FF)
                ) {
                    Text(
                        text = if (isReady) "Ready ✓ ($sizeLabel)" else "Missing",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = if (isReady) Color(0xFF065F46) else Color(0xFF3730A3),
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            Text(
                text = roleDescription,
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFF334155)
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = onUploadClick,
                    enabled = !isBusy,
                    border = BorderStroke(1.2.dp, if (isReady) Color(0xFF059669) else accentColor),
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                        .testTag(uploadButtonTag),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Folder,
                        contentDescription = "Upload $title",
                        tint = if (isReady) Color(0xFF059669) else accentColor,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (isReady) "Replace Folder" else "Upload Folder",
                        color = if (isReady) Color(0xFF065F46) else accentColor,
                        fontWeight = FontWeight.Bold
                    )
                }

                if (isReady) {
                    OutlinedButton(
                        onClick = onClearClick,
                        enabled = !isBusy,
                        border = BorderStroke(1.dp, Color(0xFFDC2626)),
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .testTag(clearButtonTag),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Clear $title",
                            tint = Color(0xFFDC2626),
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "Clear",
                            color = Color(0xFFDC2626),
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun TokenizerFileBadge(
    name: String,
    present: Boolean,
    optionalIfPairReady: Boolean = false
) {
    val bg = when {
        present -> Color(0xFFDCFCE7)
        optionalIfPairReady -> Color(0xFFF1F5F9)
        else -> Color(0xFFFEE2E2)
    }
    val fg = when {
        present -> Color(0xFF065F46)
        optionalIfPairReady -> Color(0xFF475569)
        else -> Color(0xFF991B1B)
    }
    val prefix = when {
        present -> "✓ "
        optionalIfPairReady -> "○ "
        else -> "✗ "
    }
    Surface(
        shape = RoundedCornerShape(50),
        color = bg,
        border = BorderStroke(1.dp, fg.copy(alpha = 0.35f))
    ) {
        Text(
            text = "$prefix$name",
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.Bold,
            color = fg,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
        )
    }
}
