package com.example.onnx

import org.json.JSONObject
import java.io.File

/**
 * Pure-Kotlin OpenAI CLIP Byte-Pair Encoding (BPE) Tokenizer (context length = 77)
 * compatible with HuggingFace Optimum / Diffusers Stable Diffusion 1.5 `tokenizer/`
 * (`vocab.json` + `merges.txt`, or `tokenizer.json`).
 */
class ClipBpeTokenizer private constructor(
    private val encoder: Map<String, Int>,
    private val bpeRanks: Map<Pair<String, String>, Int>
) {

    private val byteEncoder: Map<Int, Char> = buildBytesToUnicodeMap()
    private val bpeCache = HashMap<String, List<String>>(256)

    // CLIP regex pattern: contractions | letters | single digit | non-space symbols
    private val tokenPattern = Regex(
        """'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        RegexOption.IGNORE_CASE
    )

    val bosTokenId: Int = encoder[BOS_TOKEN] ?: DEFAULT_BOS_TOKEN_ID
    val eosTokenId: Int = encoder[EOS_TOKEN] ?: DEFAULT_EOS_TOKEN_ID

    fun tokenizeTo77(prompt: String): LongArray {
        val out = LongArray(MAX_LENGTH) { eosTokenId.toLong() }
        out[0] = bosTokenId.toLong()

        val cleaned = prompt
            .replace(WHITESPACE_REGEX, " ")
            .trim()
            .lowercase()

        if (cleaned.isEmpty()) {
            return out
        }

        val tokenIds = ArrayList<Int>(75)
        for (match in tokenPattern.findAll(cleaned)) {
            val rawWord = match.value
            val utf8Bytes = rawWord.toByteArray(Charsets.UTF_8)
            val sb = StringBuilder(utf8Bytes.size)
            for (b in utf8Bytes) {
                val u = b.toInt() and 0xFF
                sb.append(byteEncoder[u] ?: '?')
            }
            val transformed = sb.toString()
            val subwords = bpe(transformed)
            for (sw in subwords) {
                val id = encoder[sw]
                if (id != null) {
                    tokenIds.add(id)
                } else {
                    // Graceful character-level fallback if an unusual unicode symbol isn't in vocab
                    for (ch in sw) {
                        val charId = encoder[ch.toString()] ?: encoder["$ch</w>"]
                        if (charId != null) tokenIds.add(charId)
                    }
                }
                if (tokenIds.size >= MAX_LENGTH - 2) break
            }
            if (tokenIds.size >= MAX_LENGTH - 2) break
        }

        val count = minOf(tokenIds.size, MAX_LENGTH - 2)
        for (i in 0 until count) {
            out[i + 1] = tokenIds[i].toLong()
        }
        out[count + 1] = eosTokenId.toLong()
        return out
    }

    private fun bpe(token: String): List<String> {
        bpeCache[token]?.let { return it }
        if (token.isEmpty()) return emptyList()

        val word = ArrayList<String>(token.length)
        for (i in 0 until token.length - 1) {
            word.add(token[i].toString())
        }
        word.add("${token.last()}</w>")

        if (word.size == 1) {
            bpeCache[token] = word
            return word
        }

        var current = word
        while (current.size > 1) {
            var bestPair: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until current.size - 1) {
                val pair = current[i] to current[i + 1]
                val rank = bpeRanks[pair]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }
            if (bestPair == null) break

            val (first, second) = bestPair
            val merged = ArrayList<String>(current.size)
            var idx = 0
            while (idx < current.size) {
                if (idx < current.size - 1 && current[idx] == first && current[idx + 1] == second) {
                    merged.add(first + second)
                    idx += 2
                } else {
                    merged.add(current[idx])
                    idx += 1
                }
            }
            current = merged
        }

        bpeCache[token] = current
        return current
    }

    companion object {
        const val MAX_LENGTH = 77
        const val BOS_TOKEN = "<|startoftext|>"
        const val EOS_TOKEN = "<|endoftext|>"
        const val DEFAULT_BOS_TOKEN_ID = 49406
        const val DEFAULT_EOS_TOKEN_ID = 49407
        private val WHITESPACE_REGEX = Regex("\\s+")

        fun loadFromDirectory(tokenizerDir: File?): ClipBpeTokenizer {
            if (tokenizerDir != null && tokenizerDir.exists()) {
                val vocabFile = File(tokenizerDir, "vocab.json")
                val mergesFile = File(tokenizerDir, "merges.txt")
                if (vocabFile.exists() && mergesFile.exists()) {
                    val parsed = runCatching { fromVocabAndMerges(vocabFile.readText(), mergesFile.readText()) }.getOrNull()
                    if (parsed != null) return parsed
                }
                val tokJsonFile = File(tokenizerDir, "tokenizer.json")
                if (tokJsonFile.exists()) {
                    val parsed = runCatching { fromTokenizerJson(tokJsonFile.readText()) }.getOrNull()
                    if (parsed != null) return parsed
                }
            }
            return ClipBpeTokenizer(
                encoder = mapOf(BOS_TOKEN to DEFAULT_BOS_TOKEN_ID, EOS_TOKEN to DEFAULT_EOS_TOKEN_ID),
                bpeRanks = emptyMap()
            )
        }

        fun fromVocabAndMerges(vocabJsonText: String, mergesText: String): ClipBpeTokenizer {
            val json = JSONObject(vocabJsonText)
            val vocabMap = HashMap<String, Int>(json.length())
            val keys = json.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                vocabMap[k] = json.getInt(k)
            }

            val rankMap = HashMap<Pair<String, String>, Int>(50_000)
            var rank = 0
            mergesText.lineSequence().forEach { rawLine ->
                val line = rawLine.trim()
                if (line.isEmpty() || line.startsWith("#")) return@forEach
                val spaceIdx = line.indexOf(' ')
                if (spaceIdx > 0 && spaceIdx < line.length - 1) {
                    val p1 = line.substring(0, spaceIdx)
                    val p2 = line.substring(spaceIdx + 1)
                    rankMap[p1 to p2] = rank++
                }
            }
            return ClipBpeTokenizer(vocabMap, rankMap)
        }

        fun fromTokenizerJson(tokenizerJsonText: String): ClipBpeTokenizer {
            val root = JSONObject(tokenizerJsonText)
            val model = root.getJSONObject("model")
            val vocabObj = model.getJSONObject("vocab")
            val vocabMap = HashMap<String, Int>(vocabObj.length())
            val keys = vocabObj.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                vocabMap[k] = vocabObj.getInt(k)
            }

            val rankMap = HashMap<Pair<String, String>, Int>(50_000)
            val mergesArr = model.optJSONArray("merges")
            if (mergesArr != null) {
                for (i in 0 until mergesArr.length()) {
                    val item = mergesArr.getString(i).trim()
                    val spaceIdx = item.indexOf(' ')
                    if (spaceIdx > 0 && spaceIdx < item.length - 1) {
                        rankMap[item.substring(0, spaceIdx) to item.substring(spaceIdx + 1)] = i
                    }
                }
            }
            return ClipBpeTokenizer(vocabMap, rankMap)
        }

        private fun buildBytesToUnicodeMap(): Map<Int, Char> {
            val bs = ArrayList<Int>(256)
            for (b in 33..126) bs.add(b)
            for (b in 161..172) bs.add(b)
            for (b in 174..255) bs.add(b)
            val cs = ArrayList<Int>(bs)
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            val map = HashMap<Int, Char>(256)
            for (i in bs.indices) {
                map[bs[i]] = cs[i].toChar()
            }
            return map
        }
    }
}
