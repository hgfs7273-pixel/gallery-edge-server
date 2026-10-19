/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.edgeserver

import android.util.Log
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.runtime.LlmModelHelper
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.stream.JsonReader
import fi.iki.elonen.NanoHTTPD
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock

private const val TAG = "EdgeServer"

/**
 * On-device HTTP server that exposes an OpenAI-compatible REST API backed by
 * AI Edge Gallery's GPU-accelerated LLM inference.
 *
 * This allows any OpenAI-compatible client (curl, Open WebUI, custom scripts,
 * etc.) to interact with on-device models over localhost without cloud access.
 *
 * Endpoints:
 *   GET  /health                  → server & model status
 *   GET  /v1/models               → list loaded models
 *   POST /v1/chat/completions     → chat completions (streaming & non-streaming)
 */
class EdgeServer(
  hostname: String = DEFAULT_HOST,
  port: Int = DEFAULT_PORT,
  private val timeoutSeconds: Long = DEFAULT_TIMEOUT_SECONDS,
  private val logFile: java.io.File? = null,
) : NanoHTTPD(hostname, port) {

  companion object {
    const val DEFAULT_HOST = "0.0.0.0"
    const val DEFAULT_PORT = 8888
    const val DEFAULT_TIMEOUT_SECONDS = 300L
    private const val MIME_JSON = "application/json"

    // Conservative ceiling for the request prompt (estimated tokens). Measured
    // on-device: ~20k chars OK, ~30k chars crashed the engine, so stay well
    // below that and leave headroom for generation.
    private const val MAX_INPUT_TOKENS_EST = 5000
  }

  /** The active model, set via [EdgeServerManager.bindModel]. */
  @Volatile var activeModel: Model? = null
  @Volatile var activeModelHelper: LlmModelHelper? = null
  @Volatile var activeModelDisplayName: String = ""

  /**
   * Optional callback invoked when a request arrives but no model is bound.
   * The callback should attempt to discover and initialize a downloaded model.
   */
  @Volatile var modelFinder: (() -> Unit)? = null

  private val inferenceLock = ReentrantLock()
  private val gson = Gson()

  /**
   * Appends a diagnostic line to [logFile] (and Logcat). The file survives a
   * process crash, which lets us find out where an oversized request died.
   */
  private fun dbg(message: String) {
    Log.i(TAG, message)
    val file = logFile ?: return
    try {
      if (file.length() > 512 * 1024) file.delete()
      file.appendText("${System.currentTimeMillis()} $message\n")
    } catch (_: Exception) {
    }
  }

  // ───────────────────────────────────────────────────────────────────────
  // Request routing
  // ───────────────────────────────────────────────────────────────────────

  override fun serve(session: IHTTPSession): Response {
    val uri = session.uri ?: ""
    val method = session.method

    // Auto-discover model if not yet bound.
    if (activeModel?.instance == null) {
      tryAutoDiscoverModel()
    }

    return try {
      when {
        uri == "/health" -> handleHealth()
        uri == "/v1/models" && method == Method.GET -> handleListModels()
        uri == "/v1/chat/completions" && method == Method.POST -> handleChatCompletions(session)
        uri == "/debug/log" -> handleDebugLog()
        method == Method.OPTIONS -> newFixedLengthResponse(
          Response.Status.OK, MIME_PLAINTEXT, ""
        ).applyCors()
        else -> newFixedLengthResponse(
          Response.Status.NOT_FOUND, MIME_JSON,
          """{"error":{"message":"Not found: $uri","type":"invalid_request_error"}}"""
        )
      }
    } catch (e: Exception) {
      Log.e(TAG, "Unhandled server error", e)
      errorResponse(500, e.message ?: "Internal server error")
    }
  }

  // ───────────────────────────────────────────────────────────────────────
  // GET /health
  // ───────────────────────────────────────────────────────────────────────

  private fun handleHealth(): Response {
    val loaded = activeModel?.instance != null
    val json = """{"status":"ok","model_loaded":$loaded,"model":"$activeModelDisplayName"}"""
    return newFixedLengthResponse(Response.Status.OK, MIME_JSON, json).applyCors()
  }

  // ───────────────────────────────────────────────────────────────────────
  // GET /v1/models
  // ───────────────────────────────────────────────────────────────────────

  private fun handleListModels(): Response {
    val model = activeModel
    val body = if (model != null) {
      val obj = JsonObject().apply {
        addProperty("id", activeModelDisplayName.ifEmpty { model.name })
        addProperty("object", "model")
        addProperty("owned_by", "ai-edge-gallery")
      }
      """{"object":"list","data":[${gson.toJson(obj)}]}"""
    } else {
      """{"object":"list","data":[]}"""
    }
    return newFixedLengthResponse(Response.Status.OK, MIME_JSON, body).applyCors()
  }

  // ───────────────────────────────────────────────────────────────────────
  // GET /debug/log
  // ───────────────────────────────────────────────────────────────────────

  private fun handleDebugLog(): Response {
    val text = try {
      logFile?.takeIf { it.exists() }?.readText() ?: "(no log)"
    } catch (e: Exception) {
      "(failed to read log: ${e.message})"
    }
    return newFixedLengthResponse(
      Response.Status.OK,
      "text/plain; charset=utf-8",
      text.takeLast(200_000),
    ).applyCors()
  }

  // ───────────────────────────────────────────────────────────────────────
  // POST /v1/chat/completions
  // ───────────────────────────────────────────────────────────────────────

  private fun handleChatCompletions(session: IHTTPSession): Response {
    dbg("chat: begin content-length=${session.headers["content-length"]}")
    // Read the request body as UTF-8. NanoHTTPD's parseBody() falls back to
    // US-ASCII when the client omits a charset, which corrupts non-ASCII
    // (e.g. Chinese) text, so read the body ourselves when possible.
    val bodyStr = try {
      readRequestBodyUtf8(session)
    } catch (e: Exception) {
      return errorResponse(400, "Failed to parse request body: ${e.message}")
    }
    dbg("chat: body read chars=${bodyStr.length}")
    if (bodyStr.isEmpty()) {
      return errorResponse(400, "Empty request body")
    }

    val body: JsonObject = try {
      val reader = JsonReader(java.io.StringReader(bodyStr))
      reader.isLenient = true
      JsonParser.parseReader(reader).asJsonObject
    } catch (e: Exception) {
      return errorResponse(400, "Invalid JSON: ${e.message}")
    }
    dbg("chat: json parsed")

    val model = activeModel
    val helper = activeModelHelper
    if (model == null || helper == null || model.instance == null) {
      return errorResponse(503, "No model loaded. Open the Gallery app and load a model first.")
    }

    // Guard against prompts so large that the native engine aborts/OOMs the
    // whole app. Instead of rejecting outright, keep the system message and
    // the newest turns, dropping the oldest history until it fits.
    val messages = body.getAsJsonArray("messages")
    if (messages == null || messages.size() == 0) {
      return errorResponse(400, "\"messages\" array is required and must not be empty")
    }

    val prompt = buildPromptFitting(messages)
    dbg("chat: prompt chars=${prompt.length}")

    val stream = body.get("stream")?.asBoolean ?: false
    val requestId = "chatcmpl-${UUID.randomUUID().toString().take(12)}"
    val modelId = activeModelDisplayName.ifEmpty { model.name }

    return if (stream) {
      handleStreamingResponse(model, helper, prompt, requestId, modelId)
    } else {
      handleNonStreamingResponse(model, helper, prompt, requestId, modelId)
    }
  }

  // ───────────────────────────────────────────────────────────────────────
  // Prompt builder
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Converts an OpenAI-style messages array into a Gemma prompt string.
   *
   * Handles three `content` shapes:
   *  - String literal
   *  - Array of `{"type":"text","text":"..."}` (multi-modal format)
   *  - Single object with a `text` field
   */
  private fun buildPrompt(messages: com.google.gson.JsonArray): String = buildString {
    for (el in messages) {
      val obj = el.asJsonObject
      val role = obj.get("role")?.asString ?: "user"
      val content = extractContent(obj.get("content"))
      if (content.isNotEmpty()) {
        append("<start_of_turn>$role\n$content<end_of_turn>\n")
      }
    }
    append("<start_of_turn>model\n")
  }

  /**
   * Builds the prompt while keeping it within [MAX_INPUT_TOKENS_EST].
   *
   * Keeps the leading system message (if any) and the most recent turns, and
   * drops older history from the middle until the estimated size fits. The
   * newest user turn is always preserved, even if it alone is oversized, in
   * which case it is truncated with a marker.
   */
  private fun buildPromptFitting(messages: com.google.gson.JsonArray): String {
    val entries = messages.mapNotNull { el ->
      val obj = el.asJsonObject
      val role = obj.get("role")?.asString ?: "user"
      val content = extractContent(obj.get("content"))
      if (content.isEmpty()) null else role to content
    }
    if (entries.isEmpty()) return buildPrompt(messages)

    val systemRaw = entries.firstOrNull { it.first == "system" }
    val rest = if (systemRaw != null) entries.drop(1) else entries
    val overhead = estimateTokens("<start_of_turn>system\n<end_of_turn>\n")

    // A client may send a system prompt larger than the whole budget (agents
    // often do). Truncate it so it cannot alone overflow the engine.
    val system = systemRaw?.let { (role, content) ->
      val maxSystem = (MAX_INPUT_TOKENS_EST * 2) / 3
      role to truncateToTokens(content, maxSystem)
    }

    val kept = ArrayDeque<Pair<String, String>>()
    var total = if (system != null) estimateTokens(system.second) + overhead else 0
    // Walk backwards so the newest turns win.
    for (entry in rest.asReversed()) {
      val cost = estimateTokens(entry.second) + overhead
      if (total + cost > MAX_INPUT_TOKENS_EST) break
      kept.addFirst(entry)
      total += cost
    }

    // If nothing fit, truncate the newest turn so the request still works.
    if (kept.isEmpty() && rest.isNotEmpty()) {
      val newest = rest.last()
      val systemCost = if (system != null) estimateTokens(system.second) + overhead else 0
      // If the system message leaves no room, drop it rather than fail.
      val effectiveSystem = if (MAX_INPUT_TOKENS_EST - systemCost < 128) null else system
      val allowance = (MAX_INPUT_TOKENS_EST -
        (if (effectiveSystem != null) estimateTokens(effectiveSystem.second) + overhead else 0))
        .coerceAtLeast(128)
      val finalList = buildList {
        if (effectiveSystem != null) add(effectiveSystem)
        add(newest.first to truncateToTokens(newest.second, allowance))
      }
      return buildString {
        for ((role, content) in finalList) {
          append("<start_of_turn>$role\n$content<end_of_turn>\n")
        }
        append("<start_of_turn>model\n")
      }
    }

    val finalList = buildList {
      if (system != null) add(system)
      addAll(kept)
    }
    return buildString {
      for ((role, content) in finalList) {
        append("<start_of_turn>$role\n$content<end_of_turn>\n")
      }
      append("<start_of_turn>model\n")
    }
  }

  /**
   * Rough token estimate: ASCII runs average ~4 chars/token, while CJK and
   * other non-ASCII characters are close to 1 token each.
   */
  private fun estimateTokens(s: String): Int {
    var ascii = 0
    var nonAscii = 0
    for (ch in s) {
      if (ch.code < 128) ascii++ else nonAscii++
    }
    return (ascii / 4) + nonAscii + 1
  }

  /**
   * Keeps the beginning and the end of an oversized message, dropping the
   * middle. The end is favoured because user instructions and questions often
   * sit at the tail (e.g. a pasted document followed by "summarize this").
   */
  private fun truncateToTokens(s: String, maxTokens: Int): String {
    if (estimateTokens(s) <= maxTokens) return s
    val headTokens = (maxTokens * 2) / 5
    val tailTokens = maxTokens - headTokens
    val chars = s.toCharArray()

    fun takeFrom(start: Int, forward: Boolean, budget: Int): String {
      val sb = StringBuilder()
      var ascii = 0
      var nonAscii = 0
      var i = start
      while (i in chars.indices) {
        val ch = chars[i]
        if (ch.code < 128) ascii++ else nonAscii++
        if ((ascii / 4) + nonAscii > budget) break
        sb.append(ch)
        i += if (forward) 1 else -1
      }
      return if (forward) sb.toString() else sb.reverse().toString()
    }

    val head = takeFrom(0, true, headTokens)
    val tail = takeFrom(chars.size - 1, false, tailTokens)
    return "$head\n[...truncated...]\n$tail"
  }

  /** Extracts text from an OpenAI `content` field (String | Array | Object | null). */
  private fun extractContent(element: com.google.gson.JsonElement?): String {
    if (element == null || element.isJsonNull) return ""
    if (element.isJsonPrimitive) return element.asString
    if (element.isJsonArray) {
      return buildString {
        for (part in element.asJsonArray) {
          if (part.isJsonObject) {
            part.asJsonObject.get("text")?.asString?.let { append(it) }
          } else if (part.isJsonPrimitive) {
            append(part.asString)
          }
        }
      }
    }
    if (element.isJsonObject) {
      return element.asJsonObject.get("text")?.asString ?: element.toString()
    }
    return ""
  }

  // ───────────────────────────────────────────────────────────────────────
  // Streaming response (SSE)
  // ───────────────────────────────────────────────────────────────────────

  private fun handleStreamingResponse(
    model: Model, helper: LlmModelHelper, prompt: String,
    requestId: String, modelId: String,
  ): Response {
    if (!inferenceLock.tryLock(5, TimeUnit.SECONDS)) {
      return errorResponse(429, "Server busy. Try again later.")
    }

    val pipedOut = PipedOutputStream()
    val pipedIn = PipedInputStream(pipedOut, 64 * 1024)
    val done = AtomicBoolean(false)

    Thread {
      try {
        val latch = CountDownLatch(1)
        helper.runInference(
          model = model,
          input = prompt,
          resultListener = { partial, isDone, _ ->
            try {
              if (partial.isNotEmpty()) {
                val chunk = sseChunk(requestId, modelId, partial, null)
                pipedOut.write("data: $chunk\n\n".toByteArray())
                pipedOut.flush()
              }
              if (isDone) {
                pipedOut.write("data: ${sseChunk(requestId, modelId, "", "stop")}\n\n".toByteArray())
                pipedOut.write("data: [DONE]\n\n".toByteArray())
                pipedOut.flush()
                done.set(true)
                latch.countDown()
              }
            } catch (e: Exception) {
              Log.e(TAG, "SSE write error", e)
              done.set(true); latch.countDown()
            }
          },
          cleanUpListener = { if (!done.get()) { done.set(true); latch.countDown() } },
          onError = { msg ->
            try {
              pipedOut.write("data: {\"error\":{\"message\":\"${escapeJson(msg)}\"}}\n\n".toByteArray())
              pipedOut.write("data: [DONE]\n\n".toByteArray())
              pipedOut.flush()
            } catch (_: Exception) {}
            done.set(true); latch.countDown()
          },
        )
        latch.await(timeoutSeconds, TimeUnit.SECONDS)
      } catch (e: Exception) {
        Log.e(TAG, "Inference error", e)
      } finally {
        try { pipedOut.close() } catch (_: Exception) {}
        inferenceLock.unlock()
      }
    }.start()

    return newChunkedResponse(Response.Status.OK, "text/event-stream", pipedIn).apply {
      addHeader("Cache-Control", "no-cache")
      addHeader("Connection", "keep-alive")
    }.applyCors()
  }

  // ───────────────────────────────────────────────────────────────────────
  // Non-streaming response
  // ───────────────────────────────────────────────────────────────────────

  private fun handleNonStreamingResponse(
    model: Model, helper: LlmModelHelper, prompt: String,
    requestId: String, modelId: String,
  ): Response {
    if (!inferenceLock.tryLock(5, TimeUnit.SECONDS)) {
      return errorResponse(429, "Server busy. Try again later.")
    }
    dbg("inference: begin promptChars=${prompt.length}")
    try {
      val result = StringBuilder()
      val latch = CountDownLatch(1)
      var errorMsg: String? = null

      helper.runInference(
        model = model, input = prompt,
        resultListener = { partial, isDone, _ ->
          if (partial.isNotEmpty()) result.append(partial)
          if (isDone) latch.countDown()
        },
        cleanUpListener = { latch.countDown() },
        onError = { msg -> errorMsg = msg; latch.countDown() },
      )
      dbg("inference: returned, waiting")

      if (!latch.await(timeoutSeconds, TimeUnit.SECONDS)) {
        return errorResponse(504, "Inference timed out after ${timeoutSeconds}s")
      }
      if (errorMsg != null) {
        return errorResponse(500, "Inference error: $errorMsg")
      }

      val json = buildString {
        append("""{"id":"$requestId","object":"chat.completion",""")
        append(""""created":${System.currentTimeMillis() / 1000},"model":"$modelId",""")
        append(""""choices":[{"index":0,"message":{"role":"assistant",""")
        append(""""content":"${escapeJson(result.toString())}"},"finish_reason":"stop"}],""")
        append(""""usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}""")
      }
      return newFixedLengthResponse(Response.Status.OK, MIME_JSON, json).applyCors()
    } finally {
      inferenceLock.unlock()
    }
  }

  // ───────────────────────────────────────────────────────────────────────
  // Helpers
  // ───────────────────────────────────────────────────────────────────────

  /**
   * Reads the request body and decodes it as UTF-8 regardless of whether the
   * client sent a `charset` in the Content-Type header. Falls back to
   * NanoHTTPD's [IHTTPSession.parseBody] for chunked / unknown-length bodies.
   */
  private fun readRequestBodyUtf8(session: IHTTPSession): String {
    val len = session.headers["content-length"]?.trim()?.toIntOrNull() ?: -1
    if (len in 0..(16 * 1024 * 1024)) {
      val buf = ByteArray(len)
      var off = 0
      val input = session.inputStream
      while (off < len) {
        val r = input.read(buf, off, len - off)
        if (r < 0) break
        off += r
      }
      return String(buf, 0, off, Charsets.UTF_8)
    }
    val files = HashMap<String, String>()
    session.parseBody(files)
    return files["postData"] ?: ""
  }

  private fun tryAutoDiscoverModel() {
    val finder = modelFinder ?: return
    Log.i(TAG, "No model bound — invoking modelFinder...")
    finder.invoke()
    // Wait up to 90s for model initialization (GPU init can be slow).
    var waited = 0L
    while (activeModel?.instance == null && waited < 90_000L) {
      Thread.sleep(2_000L)
      waited += 2_000L
      if (waited % 10_000L == 0L) Log.i(TAG, "Waiting for model init... (${waited / 1000}s)")
    }
    if (activeModel?.instance != null) {
      Log.i(TAG, "Model available after ${waited / 1000}s")
    } else {
      Log.w(TAG, "Model not available after ${waited / 1000}s")
    }
  }

  private fun sseChunk(id: String, model: String, content: String, finishReason: String?): String {
    val delta = if (content.isNotEmpty()) """{"content":"${escapeJson(content)}"}""" else "{}"
    val reason = if (finishReason != null) "\"$finishReason\"" else "null"
    return """{"id":"$id","object":"chat.completion.chunk","created":${System.currentTimeMillis() / 1000},"model":"$model","choices":[{"index":0,"delta":$delta,"finish_reason":$reason}]}"""
  }

  private fun escapeJson(s: String): String = s
    .replace("\\", "\\\\").replace("\"", "\\\"")
    .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

  private fun errorResponse(code: Int, message: String): Response {
    val status = when (code) {
      400 -> Response.Status.BAD_REQUEST
      404 -> Response.Status.NOT_FOUND
      429 -> Response.Status.lookup(429) ?: Response.Status.INTERNAL_ERROR
      503 -> Response.Status.lookup(503) ?: Response.Status.INTERNAL_ERROR
      504 -> Response.Status.lookup(504) ?: Response.Status.INTERNAL_ERROR
      else -> Response.Status.INTERNAL_ERROR
    }
    val json = """{"error":{"message":"${escapeJson(message)}","type":"server_error","code":$code}}"""
    return newFixedLengthResponse(status, MIME_JSON, json).applyCors()
  }

  private fun Response.applyCors(): Response {
    addHeader("Access-Control-Allow-Origin", "*")
    addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return this
  }
}
