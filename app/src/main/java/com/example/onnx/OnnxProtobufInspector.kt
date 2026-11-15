package com.example.onnx

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.net.Uri
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

enum class ModelRequirementLevel(val label: String) {
    REQUIRED_CORE("Required (Core)"),
    REQUIRED_IDENTITY("Required for Full Identity"),
    OPTIONAL_HEAD_SWAP("Optional (Has Built-In CV Fallback)")
}

enum class ModelSlot(
    val id: String,
    val subdirectory: String,
    val canonicalFileName: String,
    val displayName: String,
    val categoryTitle: String,
    val approximateSizeLabel: String,
    val requirementLevel: ModelRequirementLevel,
    val tamilTitle: String,
    val tamilRoleSummary: String,
    val roleSummary: String,
    val expectedInputSignature: String,
    val expectedOutputSignature: String,
    val licenseSummary: String,
    val downloadSourceUrl: String
) {
    DETECTOR(
        id = "det_10g",
        subdirectory = "detection",
        canonicalFileName = "det_10g.onnx",
        displayName = "det_10g.onnx (SCRFD-10G_KPS)",
        categoryTitle = "models/detection/ (or models/face_swap/)",
        approximateSizeLabel = "~16.9 MB",
        requirementLevel = ModelRequirementLevel.REQUIRED_CORE,
        tamilTitle = "1. முகம் & 5-புள்ளி கண்டறியும் மாடல் (Face Detector)",
        tamilRoleSummary = "புகைப்படத்தில் உள்ள முகங்கள் மற்றும் இரண்டு கண்கள், மூக்கு, வாய் ஓரங்கள் ஆகிய 5 முக்கிய புள்ளிகளை (5-Point Landmarks) துல்லியமாகக் கண்டறியப் பயன்படுகிறது.",
        roleSummary = "Required for both Mode 1 (Face Swap) and Mode 2 (Full Head Replacement). Detects faces, head pose, and 5-point landmarks across strides 8, 16, and 32.",
        expectedInputSignature = "input.1: FLOAT[1, 3, 640, 640] (RGB normalized (px - 127.5) / 128.0)",
        expectedOutputSignature = "9 Tensors -> Scores: [12800,1], [3200,1], [800,1] | BBoxes: [12800,4], [3200,4], [800,4] | 5-Pt Landmarks: [12800,10], [3200,10], [800,10]",
        licenseSummary = "InsightFace Model Zoo — Non-Commercial Research Only",
        downloadSourceUrl = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip (det_10g.onnx)"
    ),
    RECOGNIZER(
        id = "w600k_r50",
        subdirectory = "recognition",
        canonicalFileName = "w600k_r50.onnx",
        displayName = "w600k_r50.onnx (ArcFace IResNet50)",
        categoryTitle = "models/recognition/ (or models/face_swap/)",
        approximateSizeLabel = "~166 MB",
        requirementLevel = ModelRequirementLevel.REQUIRED_IDENTITY,
        tamilTitle = "2. முக அடையாளப் பிரித்தெடுப்பு மாடல் (Identity Recognizer)",
        tamilRoleSummary = "Source புகைப்படத்தில் உள்ள நபரின் உண்மையான முக சாயலை 512-D ArcFace எண்களாக (Identity Vector) மாற்றித் தருகிறது. 100% அசல் முக அடையாளம் வர இது மிக அவசியம்.",
        roleSummary = "Android ONNX equivalent of GHOST 2.0's backbone50_1.pth. Extracts the 512-D ArcFace identity vector from a 112x112 aligned source face.",
        expectedInputSignature = "input.1: FLOAT[1, 3, 112, 112] (RGB normalized (px - 127.5) / 127.5)",
        expectedOutputSignature = "683 (embedding): FLOAT[1, 512] (512-D ArcFace identity vector)",
        licenseSummary = "InsightFace Buffalo_L / WebFace600K — Non-Commercial Research Only",
        downloadSourceUrl = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip (w600k_r50.onnx)"
    ),
    SWAPPER(
        id = "hyperswap_1b_256",
        subdirectory = "face_swap",
        canonicalFileName = "hyperswap_1b_256.onnx",
        displayName = "hyperswap_1b_256.onnx (HyperSwap 1b 256px Production Generator)",
        categoryTitle = "models/face_swap/",
        approximateSizeLabel = "~256px NCHW ONNX",
        requirementLevel = ModelRequirementLevel.REQUIRED_CORE,
        tamilTitle = "3. முகம் மாற்றும் முக்கிய மாடல் (HyperSwap 1b 256px Production Swap)",
        tamilRoleSummary = "Target புகைப்படத்தின் முக பாவனை மற்றும் வெளிச்சத்திற்கு ஏற்ப Source நபரின் முகத்தை 256×256 துல்லியத்தில் பொருத்தி (Face Swap) புதிய முகத்தை உருவாக்குகிறது.",
        roleSummary = "Primary production FaceFusion Mobile swap model (hyperswap_1b_256.onnx). Synthesizes a 256x256 swapped face from an aligned 256x256 target crop and a 512-D ArcFace identity vector (also supports A/B/C/D benchmark models in developer mode).",
        expectedInputSignature = "target: FLOAT[1, 3, 256, 256] (RGB normalized [-1.0, 1.0]) + source: FLOAT[1, 512] (512-D ArcFace embedding)",
        expectedOutputSignature = "output: FLOAT[1, 3, 256, 256] (Swapped 256x256 RGB face)",
        licenseSummary = "FaceFusion Mobile / InsightFace — Non-Commercial Research Only",
        downloadSourceUrl = "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx"
    ),
    SEGMENTATION(
        id = "segformer_b5_ce",
        subdirectory = "segmentation",
        canonicalFileName = "segformer_B5_ce.onnx",
        displayName = "segformer_B5_ce.onnx (GHOST 2.0 Head/Hair Parser)",
        categoryTitle = "models/segmentation/ (or models/head_replacement/)",
        approximateSizeLabel = "~325 MB (or ~50 MB BiSeNet)",
        requirementLevel = ModelRequirementLevel.OPTIONAL_HEAD_SWAP,
        tamilTitle = "4. தலை, முடி & கழுத்து பிரிக்கும் மாடல் (Head/Hair Parser)",
        tamilRoleSummary = "Mode 2 (Full Head Replacement)-ல் முகம், தலைமுடி (Hair), காதுகள் மற்றும் கழுத்துப் பகுதிகளை பின்னணியிலிருந்து (Background) துல்லியமாகப் பிரிக்கப் பயன்படுகிறது.",
        roleSummary = "Official GHOST 2.0 ONNX semantic parser separating face skin, hair, ears, hat, and neck from clothing and background. Falls back to built-in Chromatic-Geodesic Head/Hair/Neck Parser if absent.",
        expectedInputSignature = "input: FLOAT[1, 3, 512, 512] (ImageNet normalized RGB)",
        expectedOutputSignature = "logits: FLOAT[1, 19, 512, 512] (or [1, 19, 128, 128] semantic classes)",
        licenseSummary = "NVIDIA SegFormer / GHOST 2.0 — Non-Commercial Research Only",
        downloadSourceUrl = "https://github.com/start-life/ghost-2.0 (pretrained/segformer_B5_ce.onnx)"
    ),
    MATTING(
        id = "modnet_matting",
        subdirectory = "matting",
        canonicalFileName = "modnet.onnx",
        displayName = "modnet.onnx (Android Portrait Hair Alpha Matting)",
        categoryTitle = "models/matting/",
        approximateSizeLabel = "~25 MB",
        requirementLevel = ModelRequirementLevel.OPTIONAL_HEAD_SWAP,
        tamilTitle = "5. மென்மையான முடி இழைகள் பிரிக்கும் மாடல் (Hair Alpha Matting)",
        tamilRoleSummary = "தலைமுடியின் நுண்ணிய இழைகளை (Fine Hair Strands) கத்தரித்தது போல் இல்லாமல் இயற்கையாகவும் மென்மையாகவும் பிரிக்க உதவுகிறது.",
        roleSummary = "ONNX replacement for GHOST 2.0's PyTorch stylematte_synth.pth. Computes fine hair-strand alpha mattes at 512x512. Falls back to built-in Joint Bilateral Guided Hair Matting if absent.",
        expectedInputSignature = "input: FLOAT[1, 3, 512, 512] (RGB normalized (px - 127.5) / 127.5)",
        expectedOutputSignature = "output: FLOAT[1, 1, 512, 512] (Alpha matte in [0.0, 1.0])",
        licenseSummary = "Apache-2.0 License (MODNet)",
        downloadSourceUrl = "https://github.com/ZHKKKe/MODNet (modnet_photographic_portrait_matting.onnx)"
    ),
    INPAINTING(
        id = "lama_inpaint",
        subdirectory = "inpainting",
        canonicalFileName = "lama_fp32.onnx",
        displayName = "lama_fp32.onnx (Background-Gap Inpainting)",
        categoryTitle = "models/inpainting/",
        approximateSizeLabel = "~205 MB",
        requirementLevel = ModelRequirementLevel.OPTIONAL_HEAD_SWAP,
        tamilTitle = "6. பின்னணி இடைவெளி நிரப்பும் மாடல் (Background Inpainting)",
        tamilRoleSummary = "பழைய தலைமுடி பெரிதாக இருந்து புதிய தலை சிறியதாக இருக்கும்போது, சுற்றியுள்ள காலி இடங்களை (Background Gaps) இயற்கையாக நிரப்பப் பயன்படுகிறது.",
        roleSummary = "ONNX equivalent of GHOST 2.0's big-lama.pt. Fills disoccluded background gaps when target hair/ears are larger than source hair. Falls back to built-in Multi-Scale Boundary Marching Inpainter if absent.",
        expectedInputSignature = "image: FLOAT[1, 3, 512, 512] + mask: FLOAT[1, 1, 512, 512]",
        expectedOutputSignature = "output: FLOAT[1, 3, 512, 512] (Inpainted background)",
        licenseSummary = "Apache-2.0 License (Samsung AI Center LaMa)",
        downloadSourceUrl = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
    ),
    ENHANCEMENT(
        id = "gfpgan_enhance",
        subdirectory = "enhancement",
        canonicalFileName = "gfpgan_1.4.onnx",
        displayName = "gfpgan_1.4.onnx (Optional Face & Hair Restoration)",
        categoryTitle = "models/enhancement/",
        approximateSizeLabel = "~340 MB",
        requirementLevel = ModelRequirementLevel.OPTIONAL_HEAD_SWAP,
        tamilTitle = "7. 512×512 HD முகம் & கண் மெருகூட்டும் மாடல் (HD Face Restoration)",
        tamilRoleSummary = "மாற்றப்பட்ட முகம், கண்கள் (இமைகள், கருவிழி), புருவங்கள் மற்றும் சருமத் துளைகளை ஆன்லைன் AI ஆப்கள் போல 512×512 HD தரத்திற்குத் தெளிவாக்கப் பயன்படுகிறது.",
        roleSummary = "Optional 512x512 facial detail enhancer after head replacement. Falls back to built-in Tiled Luminance Micro-Contrast & Detail Enhancer if absent.",
        expectedInputSignature = "input: FLOAT[1, 3, 512, 512] (RGB normalized [-1, 1])",
        expectedOutputSignature = "output: FLOAT[1, 3, 512, 512] (Restored RGB [-1, 1])",
        licenseSummary = "Apache-2.0 License (Tencent ARC GFPGAN)",
        downloadSourceUrl = "https://github.com/TencentARC/GFPGAN"
    ),
    DEPTH(
        id = "depth_estimation",
        subdirectory = "depth",
        canonicalFileName = "depth_anything_v2_small.onnx",
        displayName = "Depth Anything V2 Small / sentis-MiDaS",
        categoryTitle = "models/depth/",
        approximateSizeLabel = "~50–280 MB",
        requirementLevel = ModelRequirementLevel.OPTIONAL_HEAD_SWAP,
        tamilTitle = "8. ஆழம் பிரிக்கும் மாடல் (Depth Anything V2 / sentis-MiDaS)",
        tamilRoleSummary = "Inpaint-ல் 1-Click Auto Background/Subject Mask மற்றும் Depth Guard உருவாக்கவும், Face Swap-ல் உண்மையான DSLR Portrait Bokeh & முன்புற பொருள் பாதுகாப்புக்கும் பயன்படுகிறது.",
        roleSummary = "Priority 1: Depth Anything V2 Small (model.onnx / model_fp16.onnx). Priority 2: sentis-MiDaS (dpt_next_vit_large_384.onnx). Powers Inpaint Auto-Mask, Depth Guard, Face Occlusion, and DSLR Bokeh.",
        expectedInputSignature = "pixel_values / input_image: FLOAT[1, 3, 518, 518] or [1, 3, 384, 384]",
        expectedOutputSignature = "predicted_depth / output_depth: FLOAT[1, 518, 518] or [1, 384, 384]",
        licenseSummary = "Apache-2.0 (Depth Anything V2) / MIT (sentis-MiDaS)",
        downloadSourceUrl = "https://huggingface.co/onnx-community/depth-anything-v2-small"
    )
}

data class GhostComponentFeasibility(
    val componentName: String,
    val purpose: String,
    val framework: String,
    val inputOutputSpec: String,
    val approxSize: String,
    val license: String,
    val runsDirectlyInAndroidOrt: Boolean,
    val conversionAndReplacementStatus: String,
    val expectedRam: String,
    val hardwareCompatibility: String
)

data class TensorDescriptor(
    val name: String,
    val elementType: String,
    val shape: List<Long>
) {
    fun formattedShape(): String = shape.joinToString(prefix = "[", postfix = "]") {
        if (it < 0) "dynamic" else it.toString()
    }
}

data class OnnxModelInspection(
    val slot: ModelSlot,
    val isPresent: Boolean,
    val isValidOnnx: Boolean,
    val filePath: String?,
    val fileSizeBytes: Long,
    val sha256Prefix: String?,
    val irVersion: Long?,
    val producerName: String?,
    val opsetVersion: Long?,
    val runtimeInputs: List<TensorDescriptor>,
    val runtimeOutputs: List<TensorDescriptor>,
    val hasEmbeddedEmap512x512: Boolean,
    val verificationNote: String
)

object OnnxProtobufInspector {

    private const val MODELS_ROOT_DIR = "models"
    private const val EMAP_CACHE_FILE = "inswapper_emap_v2_512x512.bin"
    private const val EMAP_FLOATS = 512 * 512
    private const val EMAP_BYTES = EMAP_FLOATS * 4

    val SUBDIRECTORIES = listOf(
        "face_swap",
        "head_replacement",
        "detection",
        "recognition",
        "segmentation",
        "matting",
        "inpainting",
        "enhancement",
        "depth",
        "head_swap"
    )

    /**
     * Complete engineering feasibility matrix for official `start-life/ghost-2.0` components on Android.
     */
    val GHOST_2_FEASIBILITY_MATRIX: List<GhostComponentFeasibility> = listOf(
        GhostComponentFeasibility(
            componentName = "aligner_1020_gaze_final.ckpt",
            purpose = "GHOST 2.0 Aligner: Reenacts source head/hair onto target 3D FLAME pose, expression & gaze.",
            framework = "PyTorch Lightning (.ckpt) + PyTorch3D / Nvdiffrast C++/CUDA mesh renderer",
            inputOutputSpec = "Source RGB + Target Driver + 3D FLAME UV/Normal meshes -> Aligned Head [1, 3, 512, 512]",
            approxSize = "~1.15 GB",
            license = "Non-Commercial Research Only (FLAME / EMOCA / InsightFace)",
            runsDirectlyInAndroidOrt = false,
            conversionAndReplacementStatus = "NOT directly runnable or trivial to export to ONNX due to custom PyTorch3D differentiable mesh rasterization ops. Replaced on Android by SCRFD 5-Point Cranial-Neck Pose Warp + InSwapper-128 Facial Reenactment.",
            expectedRam = "~3.5 - 5.0 GB (Desktop GPU)",
            hardwareCompatibility = "Desktop CUDA GPU; incompatible with Android NNAPI/NPU."
        ),
        GhostComponentFeasibility(
            componentName = "blender_lama.ckpt",
            purpose = "GHOST 2.0 Blender: Fuses aligned source head/hair into inpainted target background and adapts color/lighting.",
            framework = "PyTorch Lightning (.ckpt) with Fast Fourier Convolutions (FFC)",
            inputOutputSpec = "Aligned Head [1,3,512,512] + Inpainted Bg [1,3,512,512] + Masks -> Output [1,3,512,512]",
            approxSize = "~680 MB",
            license = "Non-Commercial Research Only",
            runsDirectlyInAndroidOrt = false,
            conversionAndReplacementStatus = "PyTorch .ckpt cannot run directly in ONNX Runtime. FFC uses 2D rFFT/irFFT ops unsupported by mobile GPU/NPU delegates. Replaced on Android by Multi-Region LAB Illumination/Skin Harmonization + Feathered Cranial-Neck Alpha Compositing.",
            expectedRam = "~2.2 - 2.8 GB",
            hardwareCompatibility = "CPU-only if exported with Opset 17 DFT; replaced by fast native CPU/SIMD blender."
        ),
        GhostComponentFeasibility(
            componentName = "backbone50_1.pth",
            purpose = "ArcFace IResNet-50 identity embedding extractor used to preserve source face identity.",
            framework = "PyTorch (.pth state_dict)",
            inputOutputSpec = "Input: FLOAT[1, 3, 112, 112] -> Output: FLOAT[1, 512]",
            approxSize = "~166 MB",
            license = "InsightFace Non-Commercial Research Only",
            runsDirectlyInAndroidOrt = false,
            conversionAndReplacementStatus = "YES via ONNX equivalent: w600k_r50.onnx (InsightFace Buffalo_L) runs natively in ONNX Runtime Mobile.",
            expectedRam = "~350 MB",
            hardwareCompatibility = "Android CPU, XNNPACK, GPU, and NNAPI supported."
        ),
        GhostComponentFeasibility(
            componentName = "segformer_B5_ce.onnx",
            purpose = "Semantic segmentation of 19 CelebAMask-HQ classes (face skin, hair, ears, neck, hat, cloth, background).",
            framework = "ONNX (.onnx native graph)",
            inputOutputSpec = "Input: FLOAT[1, 3, 512, 512] -> Output: FLOAT[1, 19, 512, 512] (or [1, 19, 128, 128])",
            approxSize = "~325 MB",
            license = "NVIDIA SegFormer Non-Commercial Research License",
            runsDirectlyInAndroidOrt = true,
            conversionAndReplacementStatus = "RUNS DIRECTLY in ONNX Runtime Mobile on Android! Supported as a drop-in model in models/segmentation/segformer_B5_ce.onnx (with built-in Chromatic-Geodesic Head/Hair/Neck Parser fallback).",
            expectedRam = "~850 MB peak",
            hardwareCompatibility = "Android CPU / XNNPACK (Attention ops fallback to CPU on NNAPI)."
        ),
        GhostComponentFeasibility(
            componentName = "stylematte_synth.pth",
            purpose = "High-precision hair-strand alpha matting around the head silhouette.",
            framework = "PyTorch (.pth checkpoint)",
            inputOutputSpec = "Image [1, 3, 512, 512] + Coarse Mask [1, 1, 512, 512] -> Alpha Matte [1, 1, 512, 512]",
            approxSize = "~110 MB",
            license = "MIT / Academic License",
            runsDirectlyInAndroidOrt = false,
            conversionAndReplacementStatus = "PyTorch .pth cannot run directly in ONNX Runtime. Replaced on Android by drop-in ONNX MODNet (models/matting/modnet.onnx, ~25 MB) + built-in Joint Bilateral Edge-Guided Hair Matting.",
            expectedRam = "~400 MB",
            hardwareCompatibility = "Android CPU, GPU, and NNAPI supported via MODNet ONNX."
        ),
        GhostComponentFeasibility(
            componentName = "big-lama.pt",
            purpose = "Inpainting disoccluded background gaps where the target's original hair/ears extend beyond the new source head.",
            framework = "TorchScript (.pt JIT archive)",
            inputOutputSpec = "Image [1, 3, 512, 512] + Gap Mask [1, 1, 512, 512] -> Inpainted Bg [1, 3, 512, 512]",
            approxSize = "~205 MB",
            license = "Apache-2.0 License",
            runsDirectlyInAndroidOrt = false,
            conversionAndReplacementStatus = "TorchScript .pt cannot run in ONNX Runtime directly, but LaMa ONNX (models/inpainting/lama_fp32.onnx) is supported via ONNX Runtime Mobile, with built-in Multi-Scale Boundary Marching Inpainter fallback.",
            expectedRam = "~750 MB",
            hardwareCompatibility = "Android CPU (Opset 17+); built-in inpainter runs in <45ms on CPU."
        ),
        GhostComponentFeasibility(
            componentName = "EMOCA / ResNet50 3DMM Resources",
            purpose = "3D FLAME morphable face/head parameter regression (shape, expression, jaw/neck pose).",
            framework = "PyTorch + Pickle (FLAME_generic.pkl + ResNet50)",
            inputOutputSpec = "Crop [1, 3, 224, 224] -> FLAME shape/exp/pose vectors + 3D mesh vertices",
            approxSize = "~400 MB",
            license = "Max Planck Institute FLAME Non-Commercial License",
            runsDirectlyInAndroidOrt = false,
            conversionAndReplacementStatus = "Cannot run directly in ONNX Runtime due to Python pickle + PyTorch3D dependencies. Replaced on Android by SCRFD-10G 5-Point Keypoint + Cranial Roll/Yaw/Scale Pose Estimation.",
            expectedRam = "~900 MB",
            hardwareCompatibility = "Replaced by lightweight native Kotlin + ONNX SCRFD geometry."
        )
    )

    fun getModelsRootDir(context: Context): File {
        appContextRef = context.applicationContext ?: context
        val root = File(context.filesDir, MODELS_ROOT_DIR)
        if (!root.exists()) root.mkdirs()
        for (sub in SUBDIRECTORIES) {
            val subDir = File(root, sub)
            if (!subDir.exists()) subDir.mkdirs()
        }
        return root
    }

    /**
     * Backward-compatible helper returning the root models dir while ensuring all structured
     * subdirectories (`face_swap/`, `head_swap/`, `segmentation/`, `matting/`, `inpainting/`, `enhancement/`) exist.
     */
    fun getModelsDir(context: Context): File = getModelsRootDir(context)

    private fun isSwapperModelComplete(file: File): Boolean {
        if (!file.exists() || file.length() <= 1024L) return false
        if (file.length() >= 8L * 1024L * 1024L) return true
        val parent = file.parentFile ?: return false
        return parent.listFiles()?.any { f ->
            f.isFile && f.name.endsWith(".data", ignoreCase = true) && f.length() > 1024L * 1024L
        } ?: false
    }

    /**
     * Resolves the file location for a given `ModelSlot`. Checks the primary structured subdirectory first
     * (`filesDir/models/<subdirectory>/<canonicalFileName>`), then checks all other canonical subdirectories
     * (`face_swap/`, `head_replacement/`, `detection/`, `recognition/`, `segmentation/`, `matting/`, `inpainting/`, `enhancement/`)
     * and the root `filesDir/models/<canonicalFileName>` so models placed in any folder are immediately recognized.
     */
    fun resolveModelFile(context: Context, slot: ModelSlot): File {
        if (slot == ModelSlot.DEPTH) {
            return DepthEstimationEngine.resolvePrimaryDepthFile(context)
        }
        val root = getModelsRootDir(context)
        val primaryFile = File(File(root, slot.subdirectory), slot.canonicalFileName)
        if (slot != ModelSlot.SWAPPER && primaryFile.exists() && primaryFile.length() > 1024L) {
            return primaryFile
        }
        val candidateNames = if (slot == ModelSlot.SWAPPER) {
            // Prioritize hyperswap_1b_256.onnx first, then other valid swapper models if user imported them
            listOf(
                "hyperswap_1b_256.onnx",
                "hyperswap_1a_256.onnx",
                "hyperswap_1c_256.onnx",
                "inswapper_128_fp16.onnx",
                "inswapper_128.onnx"
            )
        } else {
            listOf(slot.canonicalFileName)
        }
        if (slot == ModelSlot.SWAPPER) {
            // Pass 1: Prefer complete swapper models (either self-contained >= 8MB or with companion .data weights)
            for (fileName in candidateNames) {
                for (sub in SUBDIRECTORIES) {
                    val candidate = File(File(root, sub), fileName)
                    if (isSwapperModelComplete(candidate)) {
                        return candidate
                    }
                }
                val legacyRootFile = File(root, fileName)
                if (isSwapperModelComplete(legacyRootFile)) {
                    return legacyRootFile
                }
            }
        }
        // Pass 2: Any existing model file > 1 KB
        for (fileName in candidateNames) {
            for (sub in SUBDIRECTORIES) {
                val candidate = File(File(root, sub), fileName)
                if (candidate.exists() && candidate.length() > 1024L) {
                    return candidate
                }
            }
            val legacyRootFile = File(root, fileName)
            if (legacyRootFile.exists() && legacyRootFile.length() > 1024L) {
                return legacyRootFile
            }
        }
        return primaryFile
    }

    private data class CachedOrtSessionEntry(
        val session: OrtSession,
        val lastModified: Long,
        val fileLength: Long
    )

    private val globalSessionCache = java.util.concurrent.ConcurrentHashMap<String, CachedOrtSessionEntry>()
    private val globalSessionLock = Any()

    @Volatile
    private var appContextRef: Context? = null

    private const val CRASH_GUARD_PREFS = "ort_native_crash_guard_v1"
    private const val KEY_PENDING_MODEL = "pending_model_key"
    private const val KEY_PENDING_RANK = "pending_stage_rank"
    private const val PREFIX_MIN_RANK = "min_safe_rank_"
    private const val RANK_XNNPACK = 0
    private const val RANK_CPU_ALL_OPT = 1
    private const val RANK_CPU_BASIC = 2
    private const val RANK_CPU_NO_OPT = 3
    private const val RANK_QUARANTINED = 4

    private fun buildCrashGuardModelKey(file: File): String =
        "${file.name.lowercase()}_${file.length()}_${file.lastModified()}"

    @Synchronized
    internal fun promoteStuckCrashBreadcrumbIfAny(context: Context? = appContextRef) {
        val ctx = context?.applicationContext ?: context ?: return
        appContextRef = ctx
        val prefs = runCatching { ctx.getSharedPreferences(CRASH_GUARD_PREFS, Context.MODE_PRIVATE) }.getOrNull() ?: return
        val pendingKey = prefs.getString(KEY_PENDING_MODEL, null)
        val pendingRank = prefs.getInt(KEY_PENDING_RANK, -1)
        if (!pendingKey.isNullOrBlank() && pendingRank >= 0) {
            val prevMin = prefs.getInt("$PREFIX_MIN_RANK$pendingKey", 0)
            val nextSafeRank = maxOf(prevMin, pendingRank + 1).coerceAtMost(RANK_QUARANTINED)
            prefs.edit()
                .putInt("$PREFIX_MIN_RANK$pendingKey", nextSafeRank)
                .remove(KEY_PENDING_MODEL)
                .remove(KEY_PENDING_RANK)
                .commit()
        }
    }

    internal fun getMinSafeRankForFile(file: File, context: Context? = appContextRef): Int {
        promoteStuckCrashBreadcrumbIfAny(context)
        val ctx = context?.applicationContext ?: context ?: return 0
        val prefs = runCatching { ctx.getSharedPreferences(CRASH_GUARD_PREFS, Context.MODE_PRIVATE) }.getOrNull() ?: return 0
        val key = buildCrashGuardModelKey(file)
        return prefs.getInt("$PREFIX_MIN_RANK$key", 0).coerceIn(0, RANK_QUARANTINED)
    }

    internal fun clearCrashGuardForFile(file: File, context: Context? = appContextRef) {
        val ctx = context?.applicationContext ?: context ?: return
        val prefs = runCatching { ctx.getSharedPreferences(CRASH_GUARD_PREFS, Context.MODE_PRIVATE) }.getOrNull() ?: return
        val key = buildCrashGuardModelKey(file)
        prefs.edit()
            .remove("$PREFIX_MIN_RANK$key")
            .apply()
    }

    private fun markPendingSessionAttempt(file: File, rank: Int) {
        val ctx = appContextRef ?: return
        val prefs = runCatching { ctx.getSharedPreferences(CRASH_GUARD_PREFS, Context.MODE_PRIVATE) }.getOrNull() ?: return
        val key = buildCrashGuardModelKey(file)
        prefs.edit()
            .putString(KEY_PENDING_MODEL, key)
            .putInt(KEY_PENDING_RANK, rank)
            .commit()
    }

    private fun clearPendingSessionAttempt() {
        val ctx = appContextRef ?: return
        val prefs = runCatching { ctx.getSharedPreferences(CRASH_GUARD_PREFS, Context.MODE_PRIVATE) }.getOrNull() ?: return
        if (prefs.contains(KEY_PENDING_MODEL)) {
            prefs.edit()
                .remove(KEY_PENDING_MODEL)
                .remove(KEY_PENDING_RANK)
                .commit()
        }
    }

    /**
     * Loads and caches an [OrtSession] for [file] once and reuses it across subsequent calls.
     * Routes creation through [createSessionWithFallback] (with Native Crash Breadcrumb Guard)
     * so startup preloads and dynamic models never hard-crash in `libonnxruntime.so`.
     */
    fun getOrCreateCachedSession(
        ortEnv: OrtEnvironment?,
        file: File,
        preferHardwareAcceleration: Boolean = false
    ): OrtSession? {
        if (ortEnv == null || !file.exists() || file.length() <= 1024L) return null
        val key = file.absolutePath
        val lastMod = file.lastModified()
        val companionDataLen = file.parentFile?.listFiles()
            ?.filter { it.isFile && it.name.startsWith(file.name, ignoreCase = true) && it.name.endsWith(".data", ignoreCase = true) }
            ?.sumOf { it.length() } ?: 0L
        val effectiveLen = file.length() + companionDataLen
        globalSessionCache[key]?.let { existing ->
            if (existing.lastModified == lastMod && existing.fileLength == effectiveLen) {
                return existing.session
            }
        }
        synchronized(globalSessionLock) {
            globalSessionCache[key]?.let { existing ->
                if (existing.lastModified == lastMod && existing.fileLength == effectiveLen) {
                    return existing.session
                }
                runCatching { existing.session.close() }
                globalSessionCache.remove(key)
            }
            // If loading a heavy model (> 25 MB including .data), evict other heavy sessions first so native C++ heap never spikes
            val isHeavyModel = effectiveLen > 25L * 1024L * 1024L
            if (isHeavyModel) {
                val keysToEvict = globalSessionCache.entries
                    .filter { it.key != key && it.value.fileLength > 25L * 1024L * 1024L }
                    .map { it.key }
                if (keysToEvict.isNotEmpty()) {
                    for (evictKey in keysToEvict) {
                        globalSessionCache.remove(evictKey)?.let { entry ->
                            runCatching { entry.session.close() }
                        }
                    }
                    System.gc()
                }
            }
            return try {
                val (session, _) = createSessionWithFallback(
                    ortEnv = ortEnv,
                    modelFile = file,
                    preferHardwareAcceleration = preferHardwareAcceleration
                )
                globalSessionCache[key] = CachedOrtSessionEntry(session, lastMod, effectiveLen)
                session
            } catch (oom: OutOfMemoryError) {
                clearCachedSessions()
                System.gc()
                null
            } catch (_: Throwable) {
                null
            }
        }
    }

    fun evictCachedSession(file: File?) {
        if (file == null) return
        synchronized(globalSessionLock) {
            globalSessionCache.remove(file.absolutePath)?.let { entry ->
                runCatching { entry.session.close() }
            }
        }
    }

    fun clearCachedSessions() {
        synchronized(globalSessionLock) {
            for ((_, entry) in globalSessionCache) {
                runCatching { entry.session.close() }
            }
            globalSessionCache.clear()
        }
    }

    internal fun resolveEffectiveEpFlags(
        useNnapi: Boolean,
        useXnnpack: Boolean
    ): Pair<Boolean, Boolean> {
        // Never attach both NNAPI and XNNPACK to the same SessionOptions to avoid ORT partitioner native crashes
        return useNnapi to (useXnnpack && !useNnapi)
    }

    /**
     * Creates a memory-safe, optimized `OrtSession.SessionOptions` for Android mobile execution.
     * Never combines NNAPI and XNNPACK in the same session (which causes native SIGSEGV/SIGABRT on
     * Android HALs), keeps NNAPI OFF by default, caps intra-op threads to 2..4 performance cores,
     * and keeps memoryPattern & CPUArenaAllocator OFF to curb native heap growth.
     */
    fun createOptimizedSessionOptions(
        preferHardwareAcceleration: Boolean = false,
        intraOpThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
        useXnnpack: Boolean = !preferHardwareAcceleration,
        useNnapi: Boolean = preferHardwareAcceleration,
        optLevel: OrtSession.SessionOptions.OptLevel = OrtSession.SessionOptions.OptLevel.ALL_OPT
    ): OrtSession.SessionOptions {
        val threads = intraOpThreads.coerceIn(1, 4)
        val (effectiveNnapi, effectiveXnnpack) = resolveEffectiveEpFlags(useNnapi = useNnapi, useXnnpack = useXnnpack)
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        opts.setMemoryPatternOptimization(false)
        opts.setCPUArenaAllocator(false)
        opts.setOptimizationLevel(optLevel)
        if (effectiveNnapi) {
            runCatching { opts.addNnapi() }
        } else if (effectiveXnnpack) {
            val xnnAdded = runCatching {
                opts.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                true
            }.getOrDefault(false)
            if (xnnAdded) {
                // Prevent ORT & XNNPACK threadpool spinning contention on mobile ARM cores
                runCatching { opts.addConfigEntry("session.intra_op.allow_spinning", "0") }
            }
        }
        return opts
    }

    /**
     * Creates an [OrtSession] trying NNAPI alone (only if requested) -> XNNPACK -> CPU ALL_OPT -> CPU BASIC_OPT -> CPU NO_OPT
     * while avoiding unstable NNAPI+XNNPACK mixing and guarding against native crash loops.
     */
    fun createSessionWithFallback(
        ortEnv: OrtEnvironment,
        modelFile: File,
        preferHardwareAcceleration: Boolean = false,
        intraOpThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
    ): Pair<OrtSession, String> {
        val threads = intraOpThreads.coerceIn(1, 4)
        val minSafeRank = getMinSafeRankForFile(modelFile)
        if (minSafeRank >= RANK_QUARANTINED) {
            throw IllegalStateException(
                "Skipped ${modelFile.name}: previous native crash detected even in CPU(NO_OPT). Re-import the model to retry."
            )
        }

        data class Attempt(
            val rank: Int,
            val label: String,
            val nnapi: Boolean,
            val xnnpack: Boolean,
            val optLevel: OrtSession.SessionOptions.OptLevel
        )
        val allAttempts = buildList {
            if (preferHardwareAcceleration) {
                // Keep NNAPI at rank 0 only when user explicitly toggled Hardware Acceleration ON;
                // if NNAPI crashes natively, breadcrumb promotes minSafeRank -> 1 (XNNPACK) so next run skips NNAPI.
                add(Attempt(RANK_XNNPACK, "NNAPI", nnapi = true, xnnpack = false, OrtSession.SessionOptions.OptLevel.BASIC_OPT))
                add(Attempt(RANK_CPU_ALL_OPT, "XNNPACK", nnapi = false, xnnpack = true, OrtSession.SessionOptions.OptLevel.ALL_OPT))
                add(Attempt(RANK_CPU_BASIC, "CPU(ALL_OPT)", nnapi = false, xnnpack = false, OrtSession.SessionOptions.OptLevel.ALL_OPT))
                add(Attempt(RANK_CPU_NO_OPT, "CPU(NO_OPT)", nnapi = false, xnnpack = false, OrtSession.SessionOptions.OptLevel.NO_OPT))
            } else {
                add(Attempt(RANK_XNNPACK, "XNNPACK", nnapi = false, xnnpack = true, OrtSession.SessionOptions.OptLevel.ALL_OPT))
                add(Attempt(RANK_CPU_ALL_OPT, "CPU(ALL_OPT)", nnapi = false, xnnpack = false, OrtSession.SessionOptions.OptLevel.ALL_OPT))
                add(Attempt(RANK_CPU_BASIC, "CPU(BASIC)", nnapi = false, xnnpack = false, OrtSession.SessionOptions.OptLevel.BASIC_OPT))
                add(Attempt(RANK_CPU_NO_OPT, "CPU(NO_OPT)", nnapi = false, xnnpack = false, OrtSession.SessionOptions.OptLevel.NO_OPT))
            }
        }
        val attempts = allAttempts.filter { it.rank >= minSafeRank }.ifEmpty {
            listOf(allAttempts.last())
        }

        var lastError: Throwable? = null
        for (attempt in attempts) {
            markPendingSessionAttempt(modelFile, attempt.rank)
            val res = try {
                runCatching {
                    createOptimizedSessionOptions(
                        preferHardwareAcceleration = attempt.nnapi,
                        intraOpThreads = threads,
                        useXnnpack = attempt.xnnpack,
                        useNnapi = attempt.nnapi,
                        optLevel = attempt.optLevel
                    ).use { opts ->
                        ortEnv.createSession(modelFile.absolutePath, opts) to attempt.label
                    }
                }
            } finally {
                clearPendingSessionAttempt()
            }
            if (res.isSuccess) return res.getOrThrow()
            lastError = res.exceptionOrNull()
            if (lastError is OutOfMemoryError) {
                System.gc()
            }
        }
        throw (lastError ?: IllegalStateException("Unable to initialize ONNX Runtime session for ${modelFile.name}"))
    }

    private const val PREFS_NAME = "onnx_model_folder_prefs"
    private const val KEY_LINKED_TREE_URI = "linked_tree_uri"

    fun getLinkedModelsFolderUri(context: Context): Uri? {
        val raw = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_LINKED_TREE_URI, null)
        return if (!raw.isNullOrBlank()) runCatching { Uri.parse(raw) }.getOrNull() else null
    }

    fun saveLinkedModelsFolderUri(context: Context, treeUri: Uri) {
        runCatching {
            context.contentResolver.takePersistableUriPermission(
                treeUri,
                android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_LINKED_TREE_URI, treeUri.toString())
            .apply()
    }

    fun matchSlotForFileName(fileName: String, parentPathHint: String = ""): ModelSlot? {
        val lower = fileName.trim().lowercase()
        if (!lower.endsWith(".onnx")) return null
        if (DepthEstimationEngine.classifyDepthFile(fileName, parentPathHint) != null) {
            return ModelSlot.DEPTH
        }
        return when {
            lower.contains("det_10g") || lower.contains("scrfd") || lower.contains("det_2.5g") ->
                ModelSlot.DETECTOR
            lower.contains("w600k") || lower.contains("arcface") || lower.contains("backbone50") ->
                ModelSlot.RECOGNIZER
            lower.contains("inswapper") || lower.contains("hyperswap") ->
                ModelSlot.SWAPPER
            lower.contains("segformer") || lower.contains("bisenet") || lower.contains("face_parsing") ||
                lower.contains("face_occluder") || lower.contains("dfl_xseg") ->
                ModelSlot.SEGMENTATION
            lower.contains("modnet") || lower.contains("stylematte") ->
                ModelSlot.MATTING
            lower.contains("lama") ->
                ModelSlot.INPAINTING
            lower.contains("gfpgan") || lower.contains("codeformer") ->
                ModelSlot.ENHANCEMENT
            else -> null
        }
    }

    private fun isAuxiliary3DOrOcclusionModel(fileName: String): Boolean {
        val lower = fileName.trim().lowercase()
        if (!lower.endsWith(".onnx")) return false
        return lower.contains("1k3d68") ||
            lower.contains("2d106det") ||
            lower.contains("face_occluder") ||
            lower.contains("dfl_xseg") ||
            lower.contains("bisenet")
    }

    private data class DiscoveredDocEntry(
        val docId: String,
        val displayName: String,
        val sizeBytes: Long,
        val slot: ModelSlot,
        val priorityScore: Int,
        val parentPathHint: String = ""
    )

    private fun computeSwapCandidatePriority(
        fileName: String,
        sizeBytes: Long,
        hasCompanionData: Boolean
    ): Int {
        val lower = fileName.trim().lowercase()
        val baseScore = when {
            lower == "hyperswap_1b_256.onnx" || lower.contains("hyperswap_1b") -> 100
            lower.contains("hyperswap_1c") -> 90
            lower.contains("hyperswap_1a") -> 85
            lower.contains("hyperswap") -> 80
            lower.contains("inswapper_128_fp16") -> 70
            lower.contains("inswapper_128") -> 60
            else -> 50
        }
        // Self-contained models (>= 8MB) or split models that have their companion .data file get top priority
        val isComplete = sizeBytes >= 8L * 1024L * 1024L || hasCompanionData
        return if (isComplete) baseScore + 1000 else baseScore
    }

    /**
     * Scans a user-selected directory (SAF DocumentTree Uri) for all `.onnx` model files
     * (and any companion `.onnx.data` external weight files), automatically matching and
     * importing them in priority order without exhausting RAM or disk space.
     */
    fun importAllModelsFromTreeUri(
        context: Context,
        treeUri: Uri,
        onlyMissing: Boolean = false,
        onProgress: (String) -> Unit = {}
    ): Result<List<ModelSlot>> {
        return try {
            // Release any active native ONNX sessions before copying large model files
            clearCachedSessions()
            System.gc()

            val resolver = context.contentResolver
            val rootDocId = android.provider.DocumentsContract.getTreeDocumentId(treeUri)
            val rawDiscoveredOnnx = mutableListOf<DiscoveredDocEntry>()
            val rawAuxiliaryOnnx = mutableListOf<Triple<String, String, Long>>() // (docId, displayName, size)
            val dataFilesByName = mutableMapOf<String, Pair<String, Long>>() // lowercase name -> (docId, size)

            fun scanDocumentDirectory(parentDocId: String, parentPathHint: String, depth: Int) {
                if (depth > 4) return
                val childrenUri = android.provider.DocumentsContract.buildChildDocumentsUriUsingTree(
                    treeUri,
                    parentDocId
                )
                val projection = arrayOf(
                    android.provider.DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                    android.provider.DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                    android.provider.DocumentsContract.Document.COLUMN_MIME_TYPE,
                    android.provider.DocumentsContract.Document.COLUMN_SIZE
                )
                runCatching {
                    resolver.query(childrenUri, projection, null, null, null)?.use { cursor ->
                        val idIdx = cursor.getColumnIndex(
                            android.provider.DocumentsContract.Document.COLUMN_DOCUMENT_ID
                        )
                        val nameIdx = cursor.getColumnIndex(
                            android.provider.DocumentsContract.Document.COLUMN_DISPLAY_NAME
                        )
                        val mimeIdx = cursor.getColumnIndex(
                            android.provider.DocumentsContract.Document.COLUMN_MIME_TYPE
                        )
                        val sizeIdx = cursor.getColumnIndex(
                            android.provider.DocumentsContract.Document.COLUMN_SIZE
                        )
                        while (cursor.moveToNext()) {
                            val docId = if (idIdx >= 0) cursor.getString(idIdx) else continue
                            val displayName = if (nameIdx >= 0) cursor.getString(nameIdx).orEmpty() else ""
                            val mimeType = if (mimeIdx >= 0) cursor.getString(mimeIdx).orEmpty() else ""
                            val docSize = if (sizeIdx >= 0 && !cursor.isNull(sizeIdx)) cursor.getLong(sizeIdx) else 0L

                            if (mimeType == android.provider.DocumentsContract.Document.MIME_TYPE_DIR) {
                                val nextHint = if (parentPathHint.isEmpty()) displayName else "$parentPathHint/$displayName"
                                scanDocumentDirectory(docId, nextHint, depth + 1)
                            } else {
                                val lowerName = displayName.trim().lowercase()
                                if (lowerName.endsWith(".onnx.data") || lowerName.endsWith(".data")) {
                                    dataFilesByName[lowerName] = docId to docSize
                                } else {
                                    val matched = matchSlotForFileName(displayName, parentPathHint)
                                    if (matched != null) {
                                        rawDiscoveredOnnx.add(
                                            DiscoveredDocEntry(
                                                docId = docId,
                                                displayName = displayName,
                                                sizeBytes = docSize,
                                                slot = matched,
                                                priorityScore = 0,
                                                parentPathHint = parentPathHint
                                            )
                                        )
                                    }
                                    if (isAuxiliary3DOrOcclusionModel(displayName)) {
                                        rawAuxiliaryOnnx.add(Triple(docId, displayName, docSize))
                                    }
                                }
                            }
                        }
                    }
                }
            }

            onProgress("Scanning folder for .onnx models...")
            val rootHint = rootDocId.substringAfterLast(':')
            scanDocumentDirectory(rootDocId, rootHint, 0)

            val bestBySlot = mutableMapOf<ModelSlot, DiscoveredDocEntry>()
            val bestDepthByFamily = mutableMapOf<DepthModelFamily, DiscoveredDocEntry>()
            for (rawEntry in rawDiscoveredOnnx) {
                val docId = rawEntry.docId
                val displayName = rawEntry.displayName
                val docSize = rawEntry.sizeBytes
                val pathHint = rawEntry.parentPathHint
                val matchedSlot = rawEntry.slot
                val companionLower = "${displayName.trim().lowercase()}.data"
                val hasCompanion = dataFilesByName.containsKey(companionLower)
                val priority = when (matchedSlot) {
                    ModelSlot.SWAPPER -> computeSwapCandidatePriority(displayName, docSize, hasCompanion)
                    ModelSlot.SEGMENTATION -> {
                        val lower = displayName.trim().lowercase()
                        val segScore = when {
                            lower.contains("segformer") -> 95
                            lower.contains("bisenet") -> 90
                            lower.contains("face_occluder") || lower.contains("dfl_xseg") -> 80
                            else -> 70
                        }
                        if (docSize >= 1024L * 1024L || hasCompanion) segScore + 1000 else segScore
                    }
                    ModelSlot.DEPTH -> {
                        val classified = DepthEstimationEngine.classifyDepthFile(displayName, pathHint)
                        val familyBonus = if (classified?.first == DepthModelFamily.DEPTH_ANYTHING_V2) 1000 else 0
                        val variantScore = classified?.second ?: 100
                        val completeness = if (docSize >= 1024L * 1024L || hasCompanion) 2000 else 0
                        completeness + familyBonus + variantScore
                    }
                    else -> if (docSize >= 1024L * 1024L || hasCompanion) 1100 else 100
                }
                if (matchedSlot == ModelSlot.DEPTH) {
                    val fam = DepthEstimationEngine.classifyDepthFile(displayName, pathHint)?.first
                    if (fam != null) {
                        val curFam = bestDepthByFamily[fam]
                        if (curFam == null || priority > curFam.priorityScore) {
                            bestDepthByFamily[fam] = rawEntry.copy(priorityScore = priority)
                        }
                    }
                }
                val currentBest = bestBySlot[matchedSlot]
                if (currentBest == null || priority > currentBest.priorityScore) {
                    bestBySlot[matchedSlot] = rawEntry.copy(priorityScore = priority)
                }
            }

            // Import in priority order: Core Face Swap models first (DETECTOR, RECOGNIZER, SWAPPER), then optional models
            val orderedSlots = listOf(
                ModelSlot.DETECTOR,
                ModelSlot.RECOGNIZER,
                ModelSlot.SWAPPER,
                ModelSlot.ENHANCEMENT,
                ModelSlot.SEGMENTATION,
                ModelSlot.MATTING,
                ModelSlot.INPAINTING
            )
            val entriesToImport = buildList {
                orderedSlots.mapNotNullTo(this) { bestBySlot[it] }
                bestDepthByFamily[DepthModelFamily.DEPTH_ANYTHING_V2]?.let { add(it) }
                bestDepthByFamily[DepthModelFamily.SENTIS_MIDAS]?.let { add(it) }
            }
            val importedSlots = mutableListOf<ModelSlot>()
            val rootDir = getModelsRootDir(context)

            entriesToImport.forEachIndexed { idx, entry ->
                val slot = entry.slot
                // Determine target file name: preserve hyperswap vs inswapper and Depth Anything V2 vs sentis-MiDaS
                val targetFileName = when (slot) {
                    ModelSlot.SWAPPER -> {
                        val lower = entry.displayName.trim().lowercase()
                        when {
                            lower.contains("hyperswap_1b") -> "hyperswap_1b_256.onnx"
                            lower.contains("hyperswap_1a") -> "hyperswap_1a_256.onnx"
                            lower.contains("hyperswap_1c") -> "hyperswap_1c_256.onnx"
                            lower.contains("inswapper_128_fp16") -> "inswapper_128_fp16.onnx"
                            lower.contains("inswapper") -> "inswapper_128.onnx"
                            else -> slot.canonicalFileName
                        }
                    }
                    ModelSlot.DEPTH -> DepthEstimationEngine.canonicalImportedFileName(
                        entry.displayName,
                        entry.parentPathHint
                    )
                    else -> slot.canonicalFileName
                }

                val subDir = File(rootDir, slot.subdirectory).apply { if (!exists()) mkdirs() }
                val destFile = File(subDir, targetFileName)

                val existing = if (slot == ModelSlot.DEPTH) destFile else resolveModelFile(context, slot)
                if (onlyMissing && existing.exists() && existing.length() > 1024L) {
                    return@forEachIndexed
                }

                // Fast-path: if destination file already exists with identical size (and companion .data if any), skip re-copying!
                val companionLower = "${entry.displayName.trim().lowercase()}.data"
                val companionEntry = dataFilesByName[companionLower]
                val companionDest = File(subDir, "${entry.displayName.trim()}.data")

                val mainAlreadyMatches = entry.sizeBytes > 1024L &&
                    destFile.exists() &&
                    destFile.length() == entry.sizeBytes
                val companionAlreadyMatches = companionEntry == null ||
                    (companionDest.exists() && (companionEntry.second <= 0L || companionDest.length() == companionEntry.second))

                if (mainAlreadyMatches && companionAlreadyMatches) {
                    importedSlots.add(slot)
                    return@forEachIndexed
                }

                // Check usable disk space before copying huge files
                val requiredBytes = entry.sizeBytes.coerceAtLeast(0L) + (companionEntry?.second ?: 0L)
                if (requiredBytes > 0L && rootDir.usableSpace < requiredBytes + 64L * 1024L * 1024L) {
                    // Skip optional model if internal storage is nearly full
                    if (slot.requirementLevel == ModelRequirementLevel.OPTIONAL_HEAD_SWAP) {
                        return@forEachIndexed
                    }
                }

                onProgress("Importing (${idx + 1}/${entriesToImport.size}): ${entry.displayName}...")
                val docUri = android.provider.DocumentsContract.buildDocumentUriUsingTree(
                    treeUri,
                    entry.docId
                )
                val importRes = importModelFromUri(
                    context = context,
                    uri = docUri,
                    slot = slot,
                    overrideFileName = targetFileName
                )
                if (importRes.isSuccess) {
                    // If this ONNX model has an external weights .onnx.data file in the folder, copy it alongside
                    if (companionEntry != null) {
                        onProgress("Importing companion weights: ${entry.displayName}.data...")
                        val companionUri = android.provider.DocumentsContract.buildDocumentUriUsingTree(
                            treeUri,
                            companionEntry.first
                        )
                        runCatching {
                            copyUriToFileSafely(context, companionUri, companionDest)
                            val canonicalCompanion = File(subDir, "$targetFileName.data")
                            if (canonicalCompanion.absolutePath != companionDest.absolutePath && !canonicalCompanion.exists()) {
                                copyUriToFileSafely(context, companionUri, canonicalCompanion)
                            }
                        }
                    }
                    importedSlots.add(slot)
                }
            }

            // Also import any auxiliary 3D landmark (1k3d68.onnx, 2d106det.onnx) or occlusion/parsing models
            for ((docId, displayName, docSize) in rawAuxiliaryOnnx) {
                val lower = displayName.trim().lowercase()
                val auxSubDirName = if (lower.contains("1k3d68") || lower.contains("2d106det")) {
                    "landmarks_3d"
                } else {
                    "segmentation"
                }
                val auxDir = File(rootDir, auxSubDirName).apply { if (!exists()) mkdirs() }
                val auxDest = File(auxDir, displayName.trim())
                if (auxDest.exists() && docSize > 1024L && auxDest.length() == docSize) {
                    continue
                }
                if (docSize > 0L && rootDir.usableSpace < docSize + 32L * 1024L * 1024L) {
                    continue
                }
                runCatching {
                    onProgress("Importing 3D/Occlusion model: $displayName...")
                    val docUri = android.provider.DocumentsContract.buildDocumentUriUsingTree(treeUri, docId)
                    copyUriToFileSafely(context, docUri, auxDest)
                }
            }

            // Import Stable Diffusion 1.5 Inpainting ONNX folders (Priority 1: FP16, Priority 2: INT8)
            onProgress("Checking for SD 1.5 Inpaint ONNX folders (FP16 / INT8)...")
            val importedSdPacks = runCatching {
                SdInpaintPackManager.importSdInpaintPacksFromTreeUri(
                    context = context,
                    treeUri = treeUri,
                    onProgress = onProgress
                )
            }.getOrDefault(emptyList())

            if (importedSlots.isNotEmpty() || rawAuxiliaryOnnx.isNotEmpty() || importedSdPacks.isNotEmpty()) {
                saveLinkedModelsFolderUri(context, treeUri)
            }
            Result.success(importedSlots)
        } catch (oom: OutOfMemoryError) {
            clearCachedSessions()
            System.gc()
            Result.failure(IllegalStateException("Low device memory while scanning folder. Please retry."))
        } catch (t: Throwable) {
            Result.failure(t)
        }
    }

    private fun queryDisplayName(context: Context, uri: Uri): String? {
        return runCatching {
            context.contentResolver.query(
                uri,
                arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null
            )?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        }.getOrNull()
    }

    private fun copyUriToFileSafely(context: Context, uri: Uri, destFile: File) {
        val tempFile = File(destFile.parentFile, "${destFile.name}.tmp")
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(tempFile).use { output ->
                val buffer = ByteArray(256 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        } ?: error("Unable to open file stream.")
        if (destFile.exists()) destFile.delete()
        if (!tempFile.renameTo(destFile)) {
            tempFile.copyTo(destFile, overwrite = true)
            tempFile.delete()
        }
    }

    /**
     * Copies any .onnx files bundled in APK assets/ (`assets/models/`, `assets/models/<subdir>/`, or `assets/`)
     * or app-specific external files directory into `filesDir/models/<subdir>/`.
     */
    fun syncBundledAssetsIfPresent(context: Context) {
        val root = getModelsRootDir(context)
        val aliases = mapOf(
            ModelSlot.DETECTOR to listOf("det_10g.onnx", "scrfd_10g_bnkps.onnx", "det_2.5g.onnx"),
            ModelSlot.RECOGNIZER to listOf("w600k_r50.onnx", "w600k_mbf.onnx", "arcface.onnx"),
            ModelSlot.SWAPPER to listOf(
                "hyperswap_1b_256.onnx",
                "hyperswap_1a_256.onnx",
                "hyperswap_1c_256.onnx",
                "inswapper_128.onnx",
                "inswapper_128_fp16.onnx"
            ),
            ModelSlot.SEGMENTATION to listOf("segformer_B5_ce.onnx", "bisenet.onnx", "face_parsing.onnx"),
            ModelSlot.MATTING to listOf("modnet.onnx", "stylematte_synth.onnx"),
            ModelSlot.INPAINTING to listOf("lama_fp32.onnx", "big-lama.onnx", "lama.onnx"),
            ModelSlot.ENHANCEMENT to listOf("gfpgan_1.4.onnx", "codeformer.onnx"),
            ModelSlot.DEPTH to listOf(
                "depth_anything_v2_small.onnx",
                "depth_anything_v2_fp16.onnx",
                "dpt_next_vit_large_384.onnx",
                "dpt_swin2_tiny_256.onnx"
            )
        )

        for ((slot, names) in aliases) {
            val existing = resolveModelFile(context, slot)
            if (existing.exists() && existing.length() > 1024L) continue

            val candidateAssetFolders = listOf(
                "models/${slot.subdirectory}",
                "models",
                ""
            )
            val destFile = File(File(root, slot.subdirectory), slot.canonicalFileName)

            for (assetFolder in candidateAssetFolders) {
                val available = try {
                    context.assets.list(assetFolder)?.toSet() ?: emptySet()
                } catch (_: Exception) {
                    emptySet()
                }
                val match = names.firstOrNull { it in available }
                if (match != null) {
                    val assetPath = if (assetFolder.isEmpty()) match else "$assetFolder/$match"
                    try {
                        context.assets.open(assetPath).use { input ->
                            FileOutputStream(destFile).use { output ->
                                input.copyTo(output, bufferSize = 64 * 1024)
                            }
                        }
                    } catch (_: Exception) {
                    }
                    break
                }
            }
        }

        // Also check app-specific external storage directory (/Android/data/<pkg>/files/models/)
        runCatching {
            val extRoot = context.getExternalFilesDir(null)
            if (extRoot != null && extRoot.exists()) {
                val candidateDirs = listOf(extRoot, File(extRoot, "models"))
                for (dir in candidateDirs) {
                    dir.walkTopDown().maxDepth(2).forEach { file ->
                        if (file.isFile && file.length() > 1024L) {
                            val slot = matchSlotForFileName(file.name)
                            if (slot != null) {
                                val existing = resolveModelFile(context, slot)
                                if (!existing.exists() || existing.length() <= 1024L) {
                                    val dest = File(File(root, slot.subdirectory), slot.canonicalFileName)
                                    runCatching { file.copyTo(dest, overwrite = true) }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Imports an external .onnx file chosen via Android's document picker into `filesDir/models/<subdir>/`.
     */
    fun importModelFromUri(
        context: Context,
        uri: Uri,
        slot: ModelSlot,
        overrideFileName: String? = null
    ): Result<File> {
        return try {
            val root = getModelsRootDir(context)
            val subDir = File(root, slot.subdirectory).apply { if (!exists()) mkdirs() }
            val targetName = overrideFileName ?: if (slot == ModelSlot.DEPTH) {
                val pickedName = queryDisplayName(context, uri)
                if (!pickedName.isNullOrBlank()) {
                    DepthEstimationEngine.canonicalImportedFileName(pickedName, "depth")
                } else {
                    slot.canonicalFileName
                }
            } else {
                slot.canonicalFileName
            }
            val destFile = File(subDir, targetName)
            val tempFile = File(subDir, "$targetName.tmp")

            evictCachedSession(destFile)

            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        val read = input.read(buffer)
                        if (read <= 0) break
                        output.write(buffer, 0, read)
                    }
                    output.flush()
                }
            } ?: error("Unable to open selected file stream.")

            if (tempFile.length() < 256L) {
                tempFile.delete()
                error("Selected file is too small (${tempFile.length()} bytes) to be a valid ONNX model.")
            }

            if (slot == ModelSlot.SWAPPER) {
                // Remove any older swapper variant in this subdirectory so resolveModelFile picks the newly imported one
                subDir.listFiles()?.forEach { f ->
                    if (f.isFile && f.name.endsWith(".onnx") && f.name != tempFile.name && f.name != destFile.name) {
                        f.delete()
                    }
                }
                File(root, EMAP_CACHE_FILE).delete()
            }

            if (destFile.exists()) destFile.delete()
            if (!tempFile.renameTo(destFile)) {
                tempFile.copyTo(destFile, overwrite = true)
                tempFile.delete()
            }

            // Also remove any stale duplicate in root folder to avoid ambiguity
            val legacyRootFile = File(root, slot.canonicalFileName)
            if (legacyRootFile.exists() && legacyRootFile.absolutePath != destFile.absolutePath) {
                legacyRootFile.delete()
            }

            clearCrashGuardForFile(destFile, context)
            Result.success(destFile)
        } catch (oom: OutOfMemoryError) {
            System.gc()
            Result.failure(IllegalStateException("Out of memory while importing ${slot.canonicalFileName}"))
        } catch (t: Throwable) {
            Result.failure(t)
        }
    }

    fun inspectAllModels(context: Context, ortEnv: OrtEnvironment?): List<OnnxModelInspection> {
        appContextRef = context.applicationContext ?: context
        promoteStuckCrashBreadcrumbIfAny(appContextRef)
        syncBundledAssetsIfPresent(context)
        return ModelSlot.entries.map { slot ->
            inspectSlot(context, ortEnv, slot)
        }
    }

    /**
     * Lightweight, zero-native-RAM model inspection.
     * Validates ONNX protobuf headers by streaming only the first 32 KB of each file rather than
     * calling `ortEnv.createSession()` across all 7 models (which would allocate >1.5 GB of native C++ heap).
     */
    fun inspectSlot(context: Context, ortEnv: OrtEnvironment?, slot: ModelSlot): OnnxModelInspection {
        val file = resolveModelFile(context, slot)
        if (!file.exists() || file.length() == 0L) {
            val missingNote = when (slot) {
                ModelSlot.RECOGNIZER ->
                    "Missing ${slot.canonicalFileName} in ${slot.categoryTitle}. Required to extract 512-D ArcFace identity vectors."
                ModelSlot.DETECTOR, ModelSlot.SWAPPER ->
                    "Missing required file ${slot.canonicalFileName} in ${slot.categoryTitle}. Tap 'Import .onnx' to load from device storage."
                else ->
                    "Optional ONNX enhancement (${slot.canonicalFileName} in ${slot.categoryTitle}) not installed. Built-in Android CV fallback will be used automatically."
            }
            return OnnxModelInspection(
                slot = slot,
                isPresent = false,
                isValidOnnx = false,
                filePath = null,
                fileSizeBytes = 0L,
                sha256Prefix = null,
                irVersion = null,
                producerName = null,
                opsetVersion = null,
                runtimeInputs = emptyList(),
                runtimeOutputs = emptyList(),
                hasEmbeddedEmap512x512 = false,
                verificationNote = missingNote
            )
        }

        val headerMeta = parseProtobufHeaderMetadata(file)
        val shaPrefix = computeQuickSha256Prefix(file)

        val cachedEntry = globalSessionCache[file.absolutePath]
        val runtimeInputs: List<TensorDescriptor>
        val runtimeOutputs: List<TensorDescriptor>
        val ortValid: Boolean
        var ortNote: String

        if (cachedEntry != null) {
            runtimeInputs = runCatching { extractDescriptors(cachedEntry.session.inputInfo) }
                .getOrDefault(defaultInputDescriptors(slot))
            runtimeOutputs = runCatching { extractDescriptors(cachedEntry.session.outputInfo) }
                .getOrDefault(defaultOutputDescriptors(slot))
            ortValid = true
            ortNote = "Installed & verified in ONNX Runtime Mobile (${file.name})."
        } else {
            // Fast header validation without allocating hundreds of MBs in native OrtSession
            val validHeader = (headerMeta.irVersion != null && headerMeta.irVersion in 1L..25L) ||
                file.length() >= 4096L
            ortValid = validHeader && file.length() >= 1024L
            runtimeInputs = if (ortValid) defaultInputDescriptors(slot) else emptyList()
            runtimeOutputs = if (ortValid) defaultOutputDescriptors(slot) else emptyList()
            ortNote = if (ortValid) {
                val irStr = headerMeta.irVersion?.let { "IR v$it" } ?: "ONNX Graph"
                val opsetStr = headerMeta.opsetVersion?.let { ", Opset $it" } ?: ""
                "Installed & ready (${file.name}, $irStr$opsetStr)."
            } else {
                "Invalid or truncated ONNX model file."
            }
        }

        val cacheFile = File(getModelsRootDir(context), EMAP_CACHE_FILE)
        val hasEmap = slot == ModelSlot.SWAPPER &&
            !file.name.lowercase().contains("hyperswap") &&
            cacheFile.exists() &&
            cacheFile.length() == EMAP_BYTES.toLong()

        return OnnxModelInspection(
            slot = slot,
            isPresent = true,
            isValidOnnx = ortValid,
            filePath = file.absolutePath,
            fileSizeBytes = file.length(),
            sha256Prefix = shaPrefix,
            irVersion = headerMeta.irVersion,
            producerName = headerMeta.producerName,
            opsetVersion = headerMeta.opsetVersion,
            runtimeInputs = runtimeInputs,
            runtimeOutputs = runtimeOutputs,
            hasEmbeddedEmap512x512 = hasEmap,
            verificationNote = ortNote
        )
    }

    private fun defaultInputDescriptors(slot: ModelSlot): List<TensorDescriptor> {
        return when (slot) {
            ModelSlot.DETECTOR -> listOf(TensorDescriptor("input.1", "FLOAT", listOf(1, 3, 640, 640)))
            ModelSlot.RECOGNIZER -> listOf(TensorDescriptor("input.1", "FLOAT", listOf(1, 3, 112, 112)))
            ModelSlot.SWAPPER -> listOf(
                TensorDescriptor("target", "FLOAT", listOf(1, 3, 256, 256)),
                TensorDescriptor("source", "FLOAT", listOf(1, 512))
            )
            ModelSlot.SEGMENTATION -> listOf(TensorDescriptor("input", "FLOAT", listOf(1, 3, 512, 512)))
            ModelSlot.MATTING -> listOf(TensorDescriptor("input", "FLOAT", listOf(1, 3, 512, 512)))
            ModelSlot.INPAINTING -> listOf(
                TensorDescriptor("image", "FLOAT", listOf(1, 3, 512, 512)),
                TensorDescriptor("mask", "FLOAT", listOf(1, 1, 512, 512))
            )
            ModelSlot.ENHANCEMENT -> listOf(TensorDescriptor("input", "FLOAT", listOf(1, 3, 512, 512)))
            ModelSlot.DEPTH -> listOf(TensorDescriptor("pixel_values", "FLOAT", listOf(1, 3, 518, 518)))
        }
    }

    private fun defaultOutputDescriptors(slot: ModelSlot): List<TensorDescriptor> {
        return when (slot) {
            ModelSlot.DETECTOR -> listOf(
                TensorDescriptor("score_8", "FLOAT", listOf(12800, 1)),
                TensorDescriptor("bbox_8", "FLOAT", listOf(12800, 4)),
                TensorDescriptor("kps_8", "FLOAT", listOf(12800, 10))
            )
            ModelSlot.RECOGNIZER -> listOf(TensorDescriptor("683", "FLOAT", listOf(1, 512)))
            ModelSlot.SWAPPER -> listOf(TensorDescriptor("output", "FLOAT", listOf(1, 3, 256, 256)))
            ModelSlot.SEGMENTATION -> listOf(TensorDescriptor("logits", "FLOAT", listOf(1, 19, 512, 512)))
            ModelSlot.MATTING -> listOf(TensorDescriptor("output", "FLOAT", listOf(1, 1, 512, 512)))
            ModelSlot.INPAINTING -> listOf(TensorDescriptor("output", "FLOAT", listOf(1, 3, 512, 512)))
            ModelSlot.ENHANCEMENT -> listOf(TensorDescriptor("output", "FLOAT", listOf(1, 3, 512, 512)))
            ModelSlot.DEPTH -> listOf(TensorDescriptor("predicted_depth", "FLOAT", listOf(1, 518, 518)))
        }
    }

    private fun extractDescriptors(infoMap: Map<String, NodeInfo>): List<TensorDescriptor> {
        return infoMap.map { (name, nodeInfo) ->
            val info = nodeInfo.info
            if (info is TensorInfo) {
                TensorDescriptor(
                    name = name,
                    elementType = info.type.name,
                    shape = info.shape.toList()
                )
            } else {
                TensorDescriptor(
                    name = name,
                    elementType = info.javaClass.simpleName,
                    shape = emptyList()
                )
            }
        }
    }

    fun loadOrExtractInswapperEmap(context: Context, inswapperFile: File): FloatArray? {
        if (!inswapperFile.exists()) return null
        if (inswapperFile.name.lowercase().contains("hyperswap")) {
            // HyperSwap 1a/1b/1c 256 models consume the L2-normalized 512-D ArcFace vector directly
            return null
        }
        val cacheFile = File(getModelsRootDir(context), EMAP_CACHE_FILE)
        if (cacheFile.exists() && cacheFile.length() == EMAP_BYTES.toLong()) {
            return runCatching {
                val bytes = cacheFile.readBytes()
                val floats = FloatArray(EMAP_FLOATS)
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
                floats
            }.getOrNull()
        }

        val extracted = runCatching {
            scanOnnxFor512x512Initializer(inswapperFile)
        }.getOrNull()

        if (extracted != null && extracted.size == EMAP_FLOATS) {
            runCatching {
                val outBytes = ByteBuffer.allocate(EMAP_BYTES).order(ByteOrder.LITTLE_ENDIAN)
                outBytes.asFloatBuffer().put(extracted)
                FileOutputStream(cacheFile).use { it.write(outBytes.array()) }
            }
        }
        return extracted
    }

    private data class HeaderMeta(
        val irVersion: Long? = null,
        val producerName: String? = null,
        val opsetVersion: Long? = null
    )

    private fun parseProtobufHeaderMetadata(file: File): HeaderMeta {
        var irVersion: Long? = null
        var producerName: String? = null
        var opsetVersion: Long? = null

        runCatching {
            BufferedInputStream(FileInputStream(file), 64 * 1024).use { input ->
                val reader = ProtoStreamReader(input, maxBytesToRead = 32 * 1024L)
                while (!reader.isEof()) {
                    val tag = reader.readVarint32()
                    if (tag <= 0) break
                    val fieldNumber = tag ushr 3
                    val wireType = tag and 0x07
                    when (fieldNumber) {
                        1 -> if (wireType == 0) irVersion = reader.readVarint64() else reader.skipField(wireType)
                        2 -> if (wireType == 2) producerName = reader.readString() else reader.skipField(wireType)
                        8 -> if (wireType == 2) {
                            val len = reader.readVarint32()
                            val subLimit = reader.bytesRead + len
                            while (reader.bytesRead < subLimit && !reader.isEof()) {
                                val subTag = reader.readVarint32()
                                val subField = subTag ushr 3
                                val subWire = subTag and 0x07
                                if (subField == 2 && subWire == 0) {
                                    opsetVersion = reader.readVarint64()
                                } else {
                                    reader.skipField(subWire)
                                }
                            }
                        } else {
                            reader.skipField(wireType)
                        }
                        7 -> break
                        else -> reader.skipField(wireType)
                    }
                }
            }
        }
        return HeaderMeta(irVersion, producerName, opsetVersion)
    }

    private data class CandidateInitializer512(
        val name: String,
        val floats: FloatArray
    )

    private fun scanOnnxFor512x512Initializer(file: File): FloatArray? {
        var namedEmapMatch: FloatArray? = null
        var lastMatch: FloatArray? = null
        BufferedInputStream(FileInputStream(file), 256 * 1024).use { input ->
            val reader = ProtoStreamReader(input, maxBytesToRead = file.length())
            while (!reader.isEof()) {
                val tag = reader.readVarint32()
                if (tag <= 0) break
                val fieldNumber = tag ushr 3
                val wireType = tag and 0x07
                if (fieldNumber == 7 && wireType == 2) {
                    val graphLen = reader.readVarint64()
                    val graphEnd = reader.bytesRead + graphLen
                    while (reader.bytesRead < graphEnd && !reader.isEof()) {
                        val gTag = reader.readVarint32()
                        if (gTag <= 0) break
                        val gField = gTag ushr 3
                        val gWire = gTag and 0x07
                        if (gField == 5 && gWire == 2) {
                            val tensorLen = reader.readVarint64()
                            val candidate = parseTensorIf512x512(reader, tensorLen)
                            if (candidate != null) {
                                lastMatch = candidate.floats
                                val lower = candidate.name.lowercase()
                                if (lower == "buffalo" || lower.contains("emap")) {
                                    namedEmapMatch = candidate.floats
                                }
                            }
                        } else {
                            reader.skipField(gWire)
                        }
                    }
                } else {
                    reader.skipField(wireType)
                }
            }
        }
        return namedEmapMatch ?: lastMatch
    }

    private fun parseTensorIf512x512(reader: ProtoStreamReader, tensorLen: Long): CandidateInitializer512? {
        // A 512x512 float32 TensorProto is either ~1,048,576 bytes (raw_data/packed) or ~1,310,720 bytes (unpacked float_data).
        // Fast-skip any initializer outside this byte size window in O(1) buffered stream skips.
        if (tensorLen < EMAP_BYTES.toLong() || tensorLen > 1_315_000L) {
            reader.skipBytes(tensorLen)
            return null
        }
        val tensorEnd = reader.bytesRead + tensorLen
        val dims = mutableListOf<Long>()
        var dataType = 0
        var tensorName = ""
        var matchedFloats: FloatArray? = null
        var unpackedFloatCursor = 0

        while (reader.bytesRead < tensorEnd && !reader.isEof()) {
            val tag = reader.readVarint32()
            if (tag <= 0) break
            val field = tag ushr 3
            val wire = tag and 0x07
            when (field) {
                1 -> {
                    if (wire == 0) {
                        dims.add(reader.readVarint64())
                    } else if (wire == 2) {
                        val packLen = reader.readVarint32()
                        val packEnd = reader.bytesRead + packLen
                        while (reader.bytesRead < packEnd) {
                            dims.add(reader.readVarint64())
                        }
                    } else {
                        reader.skipField(wire)
                    }
                }
                2 -> {
                    if (wire == 0) dataType = reader.readVarint32() else reader.skipField(wire)
                }
                4 -> {
                    if (wire == 2) {
                        val byteLen = reader.readVarint64()
                        if ((dims.isEmpty() || (dims.size == 2 && dims[0] == 512L && dims[1] == 512L)) &&
                            byteLen == EMAP_BYTES.toLong()
                        ) {
                            val raw = reader.readExactBytes(EMAP_BYTES)
                            val out = FloatArray(EMAP_FLOATS)
                            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
                            matchedFloats = out
                        } else {
                            reader.skipBytes(byteLen)
                        }
                    } else if (wire == 5) {
                        // Proto2 unpacked repeated float float_data = 4
                        val isCandidate = dims.isEmpty() || (dims.size == 2 && dims[0] == 512L && dims[1] == 512L)
                        if (isCandidate) {
                            val arr = matchedFloats ?: FloatArray(EMAP_FLOATS).also { matchedFloats = it }
                            val raw4 = reader.readExactBytes(4)
                            if (unpackedFloatCursor < EMAP_FLOATS) {
                                arr[unpackedFloatCursor++] = ByteBuffer.wrap(raw4).order(ByteOrder.LITTLE_ENDIAN).float
                            }
                        } else {
                            reader.skipBytes(4)
                        }
                    } else {
                        reader.skipField(wire)
                    }
                }
                8 -> {
                    if (wire == 2) {
                        tensorName = reader.readString()
                    } else {
                        reader.skipField(wire)
                    }
                }
                9 -> {
                    if (wire == 2) {
                        val byteLen = reader.readVarint64()
                        if ((dims.isEmpty() || (dims.size == 2 && dims[0] == 512L && dims[1] == 512L)) &&
                            (dataType == 0 || dataType == 1) &&
                            byteLen == EMAP_BYTES.toLong()
                        ) {
                            val raw = reader.readExactBytes(EMAP_BYTES)
                            val out = FloatArray(EMAP_FLOATS)
                            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
                            matchedFloats = out
                        } else {
                            reader.skipBytes(byteLen)
                        }
                    } else {
                        reader.skipField(wire)
                    }
                }
                else -> reader.skipField(wire)
            }
        }
        if (reader.bytesRead < tensorEnd) {
            reader.skipBytes(tensorEnd - reader.bytesRead)
        }
        val validFloats = matchedFloats
        val isComplete = validFloats != null && (unpackedFloatCursor == 0 || unpackedFloatCursor == EMAP_FLOATS)
        return if (dims.size == 2 && dims[0] == 512L && dims[1] == 512L && isComplete && validFloats != null) {
            CandidateInitializer512(tensorName, validFloats)
        } else {
            null
        }
    }

    private fun computeQuickSha256Prefix(file: File): String {
        return runCatching {
            val md = MessageDigest.getInstance("SHA-256")
            FileInputStream(file).use { fis ->
                val buf = ByteArray(64 * 1024)
                var remaining = 1024 * 1024
                while (remaining > 0) {
                    val read = fis.read(buf, 0, minOf(buf.size, remaining))
                    if (read <= 0) break
                    md.update(buf, 0, read)
                    remaining -= read
                }
            }
            md.update(file.length().toString().toByteArray())
            md.digest().take(6).joinToString("") { "%02x".format(it) }
        }.getOrDefault("unknown")
    }

    private class ProtoStreamReader(
        private val input: InputStream,
        private val maxBytesToRead: Long
    ) {
        var bytesRead: Long = 0L
            private set
        private var eof = false

        fun isEof(): Boolean = eof || bytesRead >= maxBytesToRead

        fun readByte(): Int {
            if (isEof()) {
                eof = true
                return -1
            }
            val b = input.read()
            if (b == -1) {
                eof = true
            } else {
                bytesRead++
            }
            return b
        }

        fun readVarint32(): Int = readVarint64().toInt()

        fun readVarint64(): Long {
            var result = 0L
            var shift = 0
            while (shift < 64) {
                val b = readByte()
                if (b == -1) return -1L
                result = result or ((b.toLong() and 0x7FL) shl shift)
                if ((b and 0x80) == 0) return result
                shift += 7
            }
            return result
        }

        fun readString(): String {
            val len = readVarint32()
            if (len <= 0) return ""
            if (len > 4096) {
                skipBytes(len.toLong())
                return ""
            }
            val bytes = readExactBytes(len)
            return String(bytes, Charsets.UTF_8)
        }

        fun readExactBytes(len: Int): ByteArray {
            val out = ByteArray(len)
            var offset = 0
            while (offset < len) {
                val read = input.read(out, offset, len - offset)
                if (read <= 0) {
                    eof = true
                    break
                }
                offset += read
                bytesRead += read
            }
            return out
        }

        fun skipBytes(count: Long) {
            var remaining = count
            while (remaining > 0 && !eof) {
                val skipped = input.skip(remaining)
                if (skipped > 0) {
                    remaining -= skipped
                    bytesRead += skipped
                } else {
                    if (readByte() == -1) break
                    remaining--
                }
            }
        }

        fun skipField(wireType: Int) {
            when (wireType) {
                0 -> readVarint64()
                1 -> skipBytes(8)
                2 -> {
                    val len = readVarint64()
                    if (len > 0) skipBytes(len)
                }
                5 -> skipBytes(4)
                else -> eof = true
            }
        }
    }
}
