import io
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

import streamlit as st

try:
    from scipy.io.wavfile import write as wav_write
    SCIPY = True
except Exception:
    import wave, struct
    SCIPY = False

st.set_page_config(page_title="U+ Vibe Studio", page_icon="✨", layout="wide")

# 헤더
st.title("U+ Vibe Studio ✨")
st.caption("대화/프리셋 + 슬라이더로 비주얼/사운드의 바이브를 즉시 체험해보세요.")

mode = st.sidebar.radio("모드 선택", ["Visual Vibes (이미지)", "Sound Vibes (오디오)"])

# 이미지 모드
def apply_visual_vibes(img: Image.Image, brightness=1.0, contrast=1.0, saturation=1.0, warmth=0.0, vignette=0.0, glitch_strength=0.0):
    # RGB만
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 이미지 밝기, 대비 조정
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # HSV : 색상(Hue), 채도(Saturation), 명도(Value)
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    s_arr = np.array(s, dtype=np.float32)
    s_arr = np.clip(s_arr * saturation, 0, 255).astype(np.uint8)
    s = Image.fromarray(s_arr, mode="L")
    img = Image.merge("HSV", (h, s, v)).convert("RGB")

    # 따뜻한 정도(온기) : 색상을 약간 따뜻하게(붉은 쪽, 양수) 또는 차갑게(푸른 쪽, 음수) 밀어주는 효과
    if abs(warmth) > 0.001:
        hue_shift = int(8 * warmth)
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        h_arr = (np.array(h, dtype=np.int16) + hue_shift) % 256
        h = Image.fromarray(h_arr.astype(np.uint8), mode="L")
        img = Image.merge("HSV", (h, s, v)).convert("RGB")

    # 비네트 : 이미지 가장자리를 어둡게 해서 중앙을 강조하는 효과
    if vignette > 0.001:
        w, h = img.size
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_r = np.sqrt((cx**2) + (cy**2))
        mask = 1 - (r / max_r)  # center=1, edge=0
        mask = np.clip(mask, 0, 1) ** (1 + 3 * vignette)
        mask3 = np.dstack([mask]*3)
        arr = np.array(img).astype(np.float32) / 255.0
        out = (arr * mask3 + arr * (1 - mask3) * (1 - 0.3 * vignette))
        img = Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))

    # 글리치 (lightweight): 화면이 깨진 듯, 줄이 밀리거나 색이 어긋나는 디지털 오류
    if glitch_strength > 0.001:
        arr = np.array(img)
        h = arr.shape[0]
        num_bands = int(5 + glitch_strength * 20)
        for _ in range(num_bands):
            y = np.random.randint(0, h)
            band_h = np.random.randint(1, max(2, int(2 + glitch_strength * 8)))
            shift = np.random.randint(-int(20*glitch_strength)-1, int(20*glitch_strength)+1)
            arr[y:y+band_h] = np.roll(arr[y:y+band_h], shift, axis=1)
        img = Image.fromarray(arr)

    return img

# 드럼 소리 발생기
def gen_kick(sr, length_s, strength=1.0):
    t = np.linspace(0, length_s, int(sr*length_s), endpoint=False)
    f0, f1 = 150.0, 40.0
    freq = f0 * (f1/f0) ** (t/length_s)
    env = np.exp(-8*t) * strength
    sig = np.sin(2*np.pi*freq*t) * env
    return sig

def gen_snare(sr, length_s, strength=1.0):
    t = np.linspace(0, length_s, int(sr*length_s), endpoint=False)
    tone = np.sin(2*np.pi*200*t) * np.exp(-15*t)
    noise = (np.random.rand(len(t))*2-1) * np.exp(-25*t)
    sig = (0.3*tone + 0.7*noise) * strength
    return sig

def gen_hat(sr, length_s, strength=1.0):
    t = np.linspace(0, length_s, int(sr*length_s), endpoint=False)
    noise = (np.random.rand(len(t))*2-1) * np.exp(-60*t)
    carrier = np.sin(2*np.pi*6000*t)
    sig = noise * carrier * strength
    return sig

def render_pattern(bpm=90, bars=1, sr=44100, swing=0.0,
                   kick_level=0.8, snare_level=0.7, hat_level=0.5):
    steps_per_bar = 16
    total_steps = steps_per_bar * bars
    sec_per_beat = 60.0 / bpm
    sec_per_step = sec_per_beat / 4.0

    total_len = int(sr * sec_per_step * total_steps) + sr
    mix = np.zeros(total_len, dtype=np.float32)

    for step in range(total_steps):
        step_time = step * sec_per_step
        if (step % 2) == 1:
            step_time += sec_per_step * 0.5 * swing

        idx = int(step_time * sr)

        # 킥, 스네어, 햇 패턴
        if step % 16 in [0, 8]:
            kick = gen_kick(sr, 0.12, kick_level)
            mix[idx:idx+len(kick)] += kick
        if step % 16 in [4, 12]:
            snr = gen_snare(sr, 0.10, snare_level)
            mix[idx:idx+len(snr)] += snr
        hat = gen_hat(sr, 0.05, hat_level * (0.9 if (step % 2) else 1.0))
        mix[idx:idx+len(hat)] += hat

    # 정규화
    peak = np.max(np.abs(mix)) + 1e-9
    mix = mix / peak * 0.9
    return (sr, mix)

def audio_bytes_from_np(sr, wave_np):
    wav_i16 = np.clip(wave_np, -1, 1)
    wav_i16 = (wav_i16 * 32767).astype(np.int16)

    buf = io.BytesIO()
    if SCIPY:
        wav_write(buf, sr, wav_i16)
    else:
        # wave module
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for s in wav_i16:
                wf.writeframes(struct.pack('<h', int(s)))
    buf.seek(0)
    return buf

# 이미지 모드 UI
if mode.startswith("Visual"):
    st.subheader("Visual Vibes")
    colL, colR = st.columns([1,1])

    with colL:
        uploaded = st.file_uploader("이미지 업로드 (PNG/JPG)", type=["png","jpg","jpeg"])
        preset = st.selectbox("프리셋", ["Warm U+", "Cool Neon", "Monochrome+", "Glitch Lite"])

        # Default params
        defaults = {
            "Warm U+": dict(bright=1.05, cont=1.05, sat=1.15, warm=0.25, vign=0.15, glitch=0.0),
            "Cool Neon": dict(bright=1.0, cont=1.1, sat=1.2, warm=-0.15, vign=0.05, glitch=0.05),
            "Monochrome+": dict(bright=1.0, cont=1.15, sat=0.0, warm=0.0, vign=0.10, glitch=0.0),
            "Glitch Lite": dict(bright=1.0, cont=1.0, sat=1.0, warm=0.0, vign=0.05, glitch=0.25),
        }
        d = defaults[preset]

        brightness = st.slider("밝기", 0.5, 1.5, d["bright"], 0.01)
        contrast   = st.slider("대비", 0.5, 1.8, d["cont"], 0.01)
        saturation = st.slider("채도", 0.0, 2.0, d["sat"], 0.01)
        warmth     = st.slider("온기(↔ 차가움)", -1.0, 1.0, d["warm"], 0.01)
        vignette   = st.slider("비네트", 0.0, 1.0, d["vign"], 0.01)
        glitch     = st.slider("글리치 강도", 0.0, 1.0, d["glitch"], 0.01)

        run = st.button("미리보기 생성")

    with colR:
        if uploaded and run:
            img_raw = Image.open(uploaded)
            # EXIF 회전 보정 + RGB 통일
            img = ImageOps.exif_transpose(img_raw).convert("RGB")

            out = apply_visual_vibes(img, brightness, contrast, saturation, warmth, vignette, glitch)

            # 혹시 모를 크기 차이 방지 (현재 함수는 크기 유지하지만 안전장치)
            if out.size != img.size:
                out = out.resize(img.size, Image.BILINEAR)

            st.image(
                np.concatenate([np.array(img), np.array(out)], axis=1),
                caption="좌: 원본 / 우: 적용",
                use_column_width=True
            ) 

            buf = io.BytesIO()
            out.save(buf, format="PNG")
            st.download_button("결과 PNG 다운로드", data=buf.getvalue(), 
                               file_name="uplus_vibe.png", mime="image/png")
        else:
            st.info("이미지를 업로드하고 [미리보기 생성]을 눌러보세요.")

# 오디오 모드 UI
else:
    st.subheader("Sound Vibes")
    preset = st.selectbox("장르 프리셋", ["Lo-Fi", "Chill House", "Retro Wave"])
    if preset == "Lo-Fi":
        bpm = st.slider("BPM", 70, 110, 90)
        swing = st.slider("스윙(그루브)", 0.0, 0.6, 0.15, 0.01)
    elif preset == "Chill House":
        bpm = st.slider("BPM", 110, 128, 122)
        swing = st.slider("스윙(그루브)", 0.0, 0.4, 0.08, 0.01)
    else:
        bpm = st.slider("BPM", 80, 120, 100)
        swing = st.slider("스윙(그루브)", 0.0, 0.5, 0.12, 0.01)

    length_s = st.slider("길이(초)", 2, 16, 8)
    kick_level = st.slider("킥 강도", 0.0, 1.0, 0.8, 0.01)
    snare_level = st.slider("스네어 강도", 0.0, 1.0, 0.7, 0.01)
    hat_level = st.slider("하이햇 강도", 0.0, 1.0, 0.6, 0.01)

    if st.button("비트 생성/재생"):
        sr, wave_np = render_pattern(bpm=bpm, bars=max(1, length_s // 2), swing=swing,
                                     kick_level=kick_level, snare_level=snare_level, hat_level=hat_level)
        buf = audio_bytes_from_np(sr, wave_np)
        st.audio(buf.read(), format="audio/wav")

        # 재다운로드용
        buf.seek(0)
        st.download_button("WAV 다운로드", data=buf.getvalue(), file_name="uplus_vibe.wav", mime="audio/wav")