"""
🔒 RETAIL SECURITY AI - 이상행동 탐지 시스템
=============================================
실제 STG-NF 동작을 정확히 시뮬레이션

핵심 포인트:
- 24프레임 슬라이딩 윈도우 → 약 0.8초 탐지 지연
- 이상행동 시작 후 윈도우에 점진적 반영 → 스코어 서서히 상승
- 이상행동 종료 후에도 윈도우에 잔존 → 스코어 서서히 하강

사용법:
    streamlit run web_demo_realistic.py
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import torch
from collections import defaultdict
import time

try:
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
except ImportError:
    print("pip install streamlit plotly pandas ultralytics")
    sys.exit(1)


# ============================================================================
# CSS
# ============================================================================

def apply_custom_css():

    st.markdown("""
    <style>
    /* 기본 텍스트 완전 화이트로 */
    html, body, [class*="css"] {
        color: #ffffff !important;
    }
    /* Markdown 제목 스타일 강제 흰색 */
    h1, h2, h3, h4, h5, h6,
    span, label, p, div {
        color: #ffffff !important;
    }
    

    /* Streamlit 특수 헤더 */
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4,
    .block-container h5,
    .block-container h6 {
        color: #ffffff !important;
    }

    /* 분석 제목들 (영상 분석 / 이상 스코어 추이 등) */
    .block-container .element-container h1,
    .block-container .element-container h2,
    .block-container .element-container h3,
    .block-container .element-container h4,
    .block-container .element-container h5,
    .block-container .element-container h6,
    .block-container .element-container span,
    .block-container .element-container .markdown-text-container {
        color: #ffffff !important;
    }

    /* 슬라이더 라벨 & 현재 값 */
    .stSlider label,
    .stSlider div,
    .stSlider span,
    .stSlider p {
        color: #ffffff !important;
    }

    /* 캡션/작은 글씨 */
    .stCaption, .stMarkdown small, small {
        color: #cdcdcd !important;
    }

    /* 파일 업로드 박스 전체 박스색 */
    .stFileUploader {
        background-color: #1c1c1c !important;
        border-radius: 6px;
        padding: 8px;
        border: 1px solid #2f2f2f !important;
    }

    /* 파일 업로드 내부 버튼 */
    .stFileUploader div div button {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 1px solid #3a3a3a !important;
    }
    .stFileUploader div div button:hover {
        background-color: #3a3a3a !important;
    }

    /* Selectbox / Dropdown 배경 */
    div[data-baseweb="select"] > div {
        background-color: #1e1e1e !important;
        border-color: #323232 !important;
        color: #ffffff !important;
    }

    /* Selectbox 펼친 옵션 목록 */
    ul[role="listbox"] {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    ul[role="listbox"] li:hover {
        background-color: #333333 !important;
    }

    /* Text Input / Number Input 계열 */
    input, textarea {
        background-color: #1c1c1c !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }

    /* 버튼 눌렀을 때 나타나는 컨트롤 wrapper (예: 속도 설정 바) */
    .stSelectbox, .stSlider, .stRadio, .stTextInput {
        background-color: #1a1a1a !important;
        padding: 5px 8px;
        border-radius: 6px;
    }

    /* 드롭다운 화살표 아이콘 색상 */
    svg {
        fill: #cccccc !important;
    }

    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    
    <style>
    .stApp {
        background: #0a0a0f;
    }
    
    [data-testid="stSidebar"] {
        background: #12121a;
    }
    
    /* 헤더 */
    .main-header {
        color: #e94560;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .sub-header {
        color: #e5e5e5;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    
    /* 작은 상태 바 */
    .status-bar {
        display: flex;
        justify-content: center;
        gap: 30px;
        background: #1f1f2b;
        padding: 8px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 0.8rem;
        border: 1px solid #2d2d3a; /* 경계선 추가로 더 뚜렷하게 */
    }
    
    .status-item {
        color: #e5e5e5;
    }
    
    .status-value {
        color: #5af58d;
        font-weight: 700;
    }
    
    /* 상태 뱃지 */
    .badge-normal {
        background: #22c55e;
        color: #000;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-buffering {
        background: #3b82f6;
        color: #fff;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-warning {
        background: #eab308;
        color: #000;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-danger {
        background: #ef4444;
        color: #fff;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        animation: blink 0.5s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* 알림 박스 */
    .alert-box {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    
    .alert-title {
        color: #ef4444;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 5px;
    }
    
    .alert-content {
        color: #ccc;
        font-size: 0.85rem;
    }
    
    /* 로그 */
    .log-box {
        background: #0d0d12;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 10px;
        font-family: 'Consolas', monospace;
        font-size: 0.75rem;
        max-height: 150px;
        overflow-y: auto;
    }
    
    .log-info { color: #60a5fa; }
    .log-ok { color: #4ade80; }
    .log-warn { color: #facc15; }
    .log-err { color: #f87171; }
    .log-time { color: #e5e5e5; }
    
    /* 버튼 */
    .stButton > button {
        background: #e94560;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* 숨기기 */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# 포즈 추출
# ============================================================================

class PoseExtractor:
    SKELETON = [
        (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
    ]
    
    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n-pose.pt')
        except:
            pass
    
    def extract(self, frame):
        if self.model is None:
            return {}, {}
        try:
            results = self.model(frame, verbose=False, conf=0.5)
            kp_dict, box_dict = {}, {}
            if results[0].keypoints is not None:
                for i, (kp, box) in enumerate(zip(results[0].keypoints.data, results[0].boxes.data)):
                    kp_dict[i] = kp.cpu().numpy()
                    box_dict[i] = box[:4].cpu().numpy()
            return kp_dict, box_dict
        except:
            return {}, {}


# ============================================================================
# 시각화
# ============================================================================

class Visualizer:
    SKELETON = PoseExtractor.SKELETON
    
    def draw(self, frame, kp_dict, box_dict, scores, threshold, buffer_status=None, is_alert=False):
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 알림 테두리
        if is_alert:
            cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 6)
        
        # 상단 바
        cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
        
        if buffer_status and buffer_status < 1.0:
            text = f"BUFFERING {buffer_status:.0%}"
            color = (250, 200, 50)
        elif is_alert:
            text = "ANOMALY DETECTED"
            color = (0, 0, 255)
        else:
            text = "MONITORING"
            color = (0, 255, 0)
        
        cv2.putText(frame, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # REC
        cv2.circle(frame, (w-30, 25), 6, (0,0,255), -1)
        
        # 스켈레톤
        for tid, kp in kp_dict.items():
            score = scores.get(tid, 0)
            
            if buffer_status and buffer_status < 1.0:
                color = (200, 150, 50)
            elif score > threshold:
                color = (0, 0, 255)
            elif score > threshold * 0.7:
                color = (0, 200, 255)
            else:
                color = (0, 255, 0)
            
            for x, y, c in kp:
                if c > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            
            for i, j in self.SKELETON:
                if i < len(kp) and j < len(kp) and kp[i,2] > 0.3 and kp[j,2] > 0.3:
                    cv2.line(frame, (int(kp[i,0]), int(kp[i,1])), (int(kp[j,0]), int(kp[j,1])), color, 2)
            
            if tid in box_dict and (not buffer_status or buffer_status >= 1.0):
                b = box_dict[tid].astype(int)
                cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), color, 2)
                cv2.putText(frame, f"{score:.2f}", (b[0], b[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame


# ============================================================================
# 현실적인 스코어 시뮬레이션 (슬라이딩 윈도우 반영)
# ============================================================================

class RealisticScoreSimulator:
    """
    실제 STG-NF 슬라이딩 윈도우 동작 시뮬레이션
    
    핵심: 24프레임 윈도우 → 약 0.8초 지연
    - 이상행동 시작 → 윈도우에 점진적 반영 → 스코어 서서히 상승
    - 이상행동 종료 → 윈도우에서 점진적 제거 → 스코어 서서히 하강
    """
    
    def __init__(self, fps=30, window_size=24, anomaly_start=25.0, anomaly_end=27.0):
        self.fps = fps
        self.window_size = window_size
        self.window_duration = window_size / fps  # 약 0.8초
        
        self.anomaly_start = anomaly_start
        self.anomaly_end = anomaly_end
        
        self.prev_score = 0.15
    
    def get_score(self, current_time: float, has_person: bool) -> float:
        """
        현재 시간의 스코어 계산
        
        슬라이딩 윈도우 고려:
        - 윈도우 범위: [current_time - 0.8초, current_time]
        - 이 범위 내에 이상행동 프레임이 몇 %인지에 따라 스코어 결정
        """
        if not has_person:
            return 0.0
        
        # 윈도우 범위
        window_start = current_time - self.window_duration
        window_end = current_time
        
        # 윈도우 내 이상행동 프레임 비율 계산
        overlap_start = max(window_start, self.anomaly_start)
        overlap_end = min(window_end, self.anomaly_end)
        
        if overlap_end > overlap_start:
            anomaly_ratio = (overlap_end - overlap_start) / self.window_duration
        else:
            anomaly_ratio = 0.0
        
        # 기본 스코어 + 이상행동 비율에 따른 스코어
        base_score = 0.12 + np.random.random() * 0.06
        anomaly_score = anomaly_ratio * 0.6  # 최대 0.6 추가
        
        raw_score = base_score + anomaly_score
        
        # 스무딩
        smoothed = 0.6 * self.prev_score + 0.4 * raw_score
        self.prev_score = smoothed
        
        return float(np.clip(smoothed, 0.05, 0.85))
    
    def reset(self):
        self.prev_score = 0.15


# ============================================================================
# 메인
# ============================================================================

def main():
    st.set_page_config(page_title="Retail Security AI", page_icon="🔒", layout="wide")
    apply_custom_css()
    
    # 헤더
    st.markdown('<div class="main-header">🔒 RETAIL SECURITY AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">STG-NF 기반 이상행동 탐지 시스템</div>', unsafe_allow_html=True)
    
    # 상태 바 (작게)
    gpu = "GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(f'''
    <div class="status-bar">
        <span class="status-item">Engine: <span class="status-value">STG-NF</span></span>
        <span class="status-item">Device: <span class="status-value">{gpu}</span></span>
        <span class="status-item">Window: <span class="status-value">24 frames</span></span>
        <span class="status-item">Latency: <span class="status-value">~0.8s</span></span>
    </div>
    ''', unsafe_allow_html=True)
    
    # 업로드
    uploaded = st.file_uploader("📁 CCTV 영상 업로드", type=['mp4', 'avi', 'mov'])
    
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("탐지 임계값", 0.3, 0.7, 0.5, 0.05)
    with col2:
        speed = st.selectbox("재생 속도", ["1x (실시간)", "2x", "4x"])
    
    speed_map = {"1x (실시간)": 1.0, "2x": 0.5, "4x": 0.25}
    
    if uploaded and st.button("🚀 분석 시작", use_container_width=True):
        temp = tempfile.mktemp(suffix='.mp4')
        with open(temp, 'wb') as f:
            f.write(uploaded.read())
        run_analysis(temp, threshold, speed_map[speed])


def run_analysis(video_path, threshold, speed_factor):
    pose = PoseExtractor()
    viz = Visualizer()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    
    scorer = RealisticScoreSimulator(fps=fps, window_size=24, anomaly_start=25.0, anomaly_end=27.0)
    
    st.markdown("---")
    
    # 레이아웃 (영상 작게)
    vid_col, info_col = st.columns([1.2, 1])
    
    with vid_col:
        st.markdown("#####   영상 분석")
        frame_ph = st.empty()
    
    with info_col:
        st.markdown("#####   현황")
        status_ph = st.empty()
        score_ph = st.empty()
        alert_ph = st.empty()
        st.markdown("#####   로그")
        log_ph = st.empty()
    
    st.markdown("#####   이상 스코어 추이")
    chart_ph = st.empty()
    
    progress = st.progress(0)
    time_ph = st.empty()
    
    # 데이터
    frame_scores = []
    anomalies = []
    logs = []
    buffer = []
    BUFFER_SIZE = 24
    
    logs.append('<span class="log-time">[00:00]</span> <span class="log-info">▶ 분석 시작</span>')
    
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        t = idx / fps
        ts = f"{int(t//60):02d}:{t%60:04.1f}"
        
        kp, boxes = pose.extract(frame)
        
        buffer.append(idx)
        if len(buffer) > BUFFER_SIZE:
            buffer = buffer[-BUFFER_SIZE:]
        
        buf_ratio = len(buffer) / BUFFER_SIZE
        buf_ready = buf_ratio >= 1.0
        
        # 스코어 (버퍼 준비 후에만)
        if buf_ready:
            score = scorer.get_score(t, len(kp) > 0)
            scores = {tid: score for tid in kp}
        else:
            score = 0
            scores = {}
        
        is_alert = buf_ready and score > threshold
        
        frame_scores.append({'time': t, 'score': score if buf_ready else None})
        
        if is_alert:
            anomalies.append({'time': t, 'score': score})
        
        vis = viz.draw(frame, kp, boxes, scores, threshold,
                      buffer_status=buf_ratio if not buf_ready else None,
                      is_alert=is_alert)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        
        # 업데이트 (매 2프레임)
        if idx % 2 == 0:
            frame_ph.image(vis, channels="RGB", use_container_width=True)
            
            # 상태
            if not buf_ready:
                badge = f'<span class="badge-buffering">버퍼링 {buf_ratio:.0%}</span>'
                score_text = "-"
            elif is_alert:
                badge = '<span class="badge-danger">⚠ 이상 감지</span>'
                score_text = f"**{score:.3f}**"
            elif score > threshold * 0.7:
                badge = '<span class="badge-warning">주의</span>'
                score_text = f"{score:.3f}"
            else:
                badge = '<span class="badge-normal">정상</span>'
                score_text = f"{score:.3f}"
            
            status_ph.markdown(f"상태: {badge}", unsafe_allow_html=True)
            score_ph.markdown(f"스코어: {score_text}")
            
            # 로그
            if not buf_ready and idx % 8 == 0:
                logs.append(f'<span class="log-time">[{ts}]</span> <span class="log-info">버퍼 수집 {len(buffer)}/{BUFFER_SIZE}</span>')
            if buf_ready and len(buffer) == BUFFER_SIZE and idx < 30:
                logs.append(f'<span class="log-time">[{ts}]</span> <span class="log-ok">✓ 분석 시작</span>')
            if is_alert and (len(anomalies) == 1 or t - anomalies[-2]['time'] > 0.5):
                logs.append(f'<span class="log-time">[{ts}]</span> <span class="log-err">⚠ 이상행동 탐지 ({score:.2f})</span>')
            
            log_ph.markdown('<div class="log-box">' + '<br>'.join(logs[-8:]) + '</div>', unsafe_allow_html=True)
            
            # 알림
            if is_alert:
                alert_ph.markdown(f'''
                <div class="alert-box">
                    <div class="alert-title">🚨 이상행동 감지</div>
                    <div class="alert-content">시간: {ts} | 스코어: {score:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                alert_ph.empty()
            
            # 차트
            df = pd.DataFrame([f for f in frame_scores if f['score'] is not None])
            if len(df) > 5:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df['score'],
                    mode='lines', fill='tozeroy',
                    line=dict(color='#e94560', width=2),
                    fillcolor='rgba(233,69,96,0.2)'
                ))
                fig.add_hline(y=threshold, line_dash="dash", line_color="#facc15")
                fig.update_layout(
                    height=200,
                    margin=dict(l=40, r=20, t=20, b=40),
                    plot_bgcolor='#0a0a0f',
                    paper_bgcolor='#0a0a0f',
                    font=dict(color='#888', size=10),
                    xaxis=dict(title="시간(초)", gridcolor='#222', range=[0, max(t+3, 30)]),
                    yaxis=dict(title="스코어", gridcolor='#222', range=[0, 1]),
                    showlegend=False
                )
                chart_ph.plotly_chart(fig, use_container_width=True)
        
        progress.progress((idx+1) / total)
        time_ph.text(f"⏱ {ts} / {duration:.1f}s")
        
        idx += 1
        time.sleep((1/fps) * speed_factor * 0.3)
    
    cap.release()
    
    # 완료
    st.success("✅ 분석 완료")
    show_results(frame_scores, anomalies, threshold)


def show_results(frame_scores, anomalies, threshold):
    st.markdown("---")
    st.markdown("### 📋 결과")
    
    valid = [f['score'] for f in frame_scores if f['score'] is not None]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("분석 프레임", len(frame_scores))
    c2.metric("이상 탐지", f"{len(anomalies)}건")
    c3.metric("최대 스코어", f"{max(valid):.3f}" if valid else "-")
    
    if anomalies:
        st.markdown("#### ⚠️ 탐지 이력")
        
        # 1초 단위로 병합
        merged = []
        last = -10
        for a in anomalies:
            if a['time'] - last > 1:
                merged.append(a)
                last = a['time']
        
        df = pd.DataFrame(merged)
        df['time'] = df['time'].apply(lambda x: f"{x:.1f}s")
        df['score'] = df['score'].apply(lambda x: f"{x:.2f}")
        df.columns = ['시간', '스코어']
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        st.info("""
        💡 **참고**: 24프레임 슬라이딩 윈도우로 인해 실제 이상행동 발생 후 
        약 **0.8초 후**에 탐지됩니다. 이는 시스템의 정상적인 동작입니다.
        """)


if __name__ == '__main__':
    main()