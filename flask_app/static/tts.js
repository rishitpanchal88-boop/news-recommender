/**
 * News Recommender - Native Web Speech TTS (Text-to-Speech) Reader
 * Robust, cross-browser support (Chrome, Edge, Safari, Firefox), zero API keys.
 */

(function () {
    "use strict";

    if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
        console.warn("TTS Web Speech API is not supported in this browser.");
        return;
    }

    const synth = window.speechSynthesis;
    let voices = [];
    let textChunks = [];
    let currentChunkIndex = 0;
    let isPlaying = false;
    let isPaused = false;
    let currentSpeed = 1.0;
    let selectedVoice = null;
    let keepAliveTimer = null;

    // Prevent Chrome garbage collection bug
    window._activeTTSUtterance = null;

    function populateVoices() {
        voices = synth.getVoices() || [];
        const voiceSelect = document.getElementById("ttsVoiceSelect");
        if (!voiceSelect) return;

        voiceSelect.innerHTML = "";

        if (voices.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "System Default Voice";
            voiceSelect.appendChild(opt);
            return;
        }

        // Sort voices to put English and clear voices first
        const englishVoices = voices.filter(v => (v.lang || "").toLowerCase().startsWith("en"));
        const otherVoices = voices.filter(v => !(v.lang || "").toLowerCase().startsWith("en"));
        const sortedVoices = englishVoices.concat(otherVoices);

        sortedVoices.forEach((v) => {
            const option = document.createElement("option");
            option.value = v.name;
            option.textContent = `${v.name} (${v.lang})`;

            if (!selectedVoice) {
                if (v.name.includes("Natural") || v.name.includes("Google US") || v.default || v.lang === "en-US") {
                    option.selected = true;
                    selectedVoice = v;
                }
            } else if (selectedVoice.name === v.name) {
                option.selected = true;
            }

            voiceSelect.appendChild(option);
        });

        if (!selectedVoice && sortedVoices.length > 0) {
            selectedVoice = sortedVoices[0];
        }

        voiceSelect.onchange = function (e) {
            selectedVoice = voices.find(v => v.name === e.target.value) || null;
            if (isPlaying && !isPaused) {
                playCurrentChunk();
            }
        };
    }

    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = populateVoices;
    }
    populateVoices();

    function extractArticleText() {
        const titleEl = document.querySelector(".article-detail h1, .live-detail h1, h1");
        const titleText = titleEl ? titleEl.innerText.trim() : "";

        // Collect all paragraph texts inside body-text or article-detail
        const bodyParas = document.querySelectorAll(".body-text p, .body-text, .article-detail p.body-text, .live-detail .body-text p");
        let bodyTexts = [];

        if (bodyParas && bodyParas.length > 0) {
            bodyParas.forEach(p => {
                const txt = p.innerText ? p.innerText.trim() : "";
                if (txt) bodyTexts.push(txt);
            });
        }

        // Fallback if no specific paragraph elements matched
        if (bodyTexts.length === 0) {
            const bodyEl = document.querySelector(".body-text, .article-detail, .live-detail");
            if (bodyEl) {
                const raw = bodyEl.innerText ? bodyEl.innerText.trim() : "";
                if (raw) bodyTexts.push(raw);
            }
        }

        const fullText = (titleText ? titleText + ". " : "") + bodyTexts.join(" ");
        if (!fullText || fullText.trim().length === 0) {
            return [];
        }

        // Split text into comfortable sentence chunks for continuous speech
        const sentences = fullText
            .replace(/\s+/g, " ")
            .split(/(?<=[.?!])\s+/)
            .map(s => s.trim())
            .filter(s => s.length > 0);

        return sentences.length > 0 ? sentences : [fullText.trim()];
    }

    function updateUI(state) {
        const playBtn = document.getElementById("ttsPlayBtn");
        const playLabel = document.getElementById("ttsPlayBtnLabel");
        const playIcon = document.getElementById("ttsPlayIcon");
        const pauseIcon = document.getElementById("ttsPauseIcon");
        const waveContainer = document.getElementById("ttsWave");
        const statusText = document.getElementById("ttsStatus");

        if (!playBtn) return;

        if (state === "playing") {
            if (playIcon) playIcon.style.display = "none";
            if (pauseIcon) pauseIcon.style.display = "inline-block";
            if (playLabel) playLabel.textContent = "Pause";
            if (waveContainer) waveContainer.classList.add("active");
            if (statusText) statusText.textContent = "Playing Article...";
        } else if (state === "paused") {
            if (playIcon) playIcon.style.display = "inline-block";
            if (pauseIcon) pauseIcon.style.display = "none";
            if (playLabel) playLabel.textContent = "Resume";
            if (waveContainer) waveContainer.classList.remove("active");
            if (statusText) statusText.textContent = "Paused";
        } else {
            // Stopped / Initial
            if (playIcon) playIcon.style.display = "inline-block";
            if (pauseIcon) pauseIcon.style.display = "none";
            if (playLabel) playLabel.textContent = "Play Audio";
            if (waveContainer) waveContainer.classList.remove("active");
            if (statusText) statusText.textContent = "Listen to Article";
        }
    }

    function startKeepAlive() {
        stopKeepAlive();
        // Chrome bug: SpeechSynthesis pauses after ~15s if no user interaction.
        keepAliveTimer = setInterval(() => {
            if (isPlaying && !isPaused && synth.speaking) {
                synth.pause();
                synth.resume();
            }
        }, 12000);
    }

    function stopKeepAlive() {
        if (keepAliveTimer) {
            clearInterval(keepAliveTimer);
            keepAliveTimer = null;
        }
    }

    function playCurrentChunk() {
        if (!isPlaying || isPaused || currentChunkIndex >= textChunks.length) {
            if (currentChunkIndex >= textChunks.length) {
                stopTTS();
            }
            return;
        }

        const chunk = textChunks[currentChunkIndex];
        if (!chunk || !chunk.trim()) {
            currentChunkIndex++;
            playCurrentChunk();
            return;
        }

        // Cancel previous stuck utterance
        synth.cancel();

        const utterance = new SpeechSynthesisUtterance(chunk);
        window._activeTTSUtterance = utterance; // Prevent garbage collection in Chrome

        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }
        utterance.rate = currentSpeed;
        utterance.pitch = 1.0;

        utterance.onend = function () {
            if (isPlaying && !isPaused) {
                currentChunkIndex++;
                if (currentChunkIndex < textChunks.length) {
                    playCurrentChunk();
                } else {
                    stopTTS();
                }
            }
        };

        utterance.onerror = function (e) {
            console.error("Speech synthesis chunk error:", e);
            if (isPlaying && !isPaused) {
                currentChunkIndex++;
                if (currentChunkIndex < textChunks.length) {
                    playCurrentChunk();
                } else {
                    stopTTS();
                }
            }
        };

        // Small timeout ensures Chrome engine is cleanly unblocked before speaking
        setTimeout(() => {
            try {
                synth.speak(utterance);
                startKeepAlive();
            } catch (err) {
                console.error("Failed to speak:", err);
            }
        }, 40);
    }

    function startTTS() {
        if (isPaused) {
            synth.resume();
            isPaused = false;
            isPlaying = true;
            updateUI("playing");
            startKeepAlive();
            return;
        }

        textChunks = extractArticleText();
        if (textChunks.length === 0) {
            alert("No article text found to read aloud.");
            return;
        }

        currentChunkIndex = 0;
        isPlaying = true;
        isPaused = false;
        updateUI("playing");

        // Force reload voices if empty
        if (!selectedVoice || voices.length === 0) {
            populateVoices();
        }

        playCurrentChunk();
    }

    function pauseTTS() {
        if (isPlaying && !isPaused) {
            synth.pause();
            isPaused = true;
            stopKeepAlive();
            updateUI("paused");
        }
    }

    function stopTTS() {
        synth.cancel();
        stopKeepAlive();
        isPlaying = false;
        isPaused = false;
        currentChunkIndex = 0;
        window._activeTTSUtterance = null;
        updateUI("stopped");
    }

    function initPlayer() {
        const playerCard = document.getElementById("ttsPlayer");
        if (!playerCard) return;

        populateVoices();

        const playBtn = document.getElementById("ttsPlayBtn");
        const stopBtn = document.getElementById("ttsStopBtn");
        const speedBtns = document.querySelectorAll(".tts-speed-btn");

        if (playBtn) {
            playBtn.onclick = function (e) {
                e.preventDefault();
                if (isPlaying && !isPaused) {
                    pauseTTS();
                } else {
                    startTTS();
                }
            };
        }

        if (stopBtn) {
            stopBtn.onclick = function (e) {
                e.preventDefault();
                stopTTS();
            };
        }

        speedBtns.forEach(btn => {
            btn.onclick = function (e) {
                e.preventDefault();
                speedBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                currentSpeed = parseFloat(btn.getAttribute("data-speed") || "1.0");
                if (isPlaying && !isPaused) {
                    playCurrentChunk();
                }
            };
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initPlayer);
    } else {
        initPlayer();
    }

    window.addEventListener("beforeunload", function () {
        synth.cancel();
        stopKeepAlive();
    });
})();
