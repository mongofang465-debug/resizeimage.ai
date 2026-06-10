import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
import zipfile

st.set_page_config(page_title="证件照批量调整工具", page_icon="🖼️")

st.title("🖼️ 批量证件照尺寸调整工具（人脸居中裁剪 + 保持比例 + ZIP下载）")

# ===== 固定系统参数（用户不可见）=====
DEFAULT_DPI = 300  # 内部使用，不暴露给用户

# 上传多张图片
uploaded_files = st.file_uploader(
    "上传照片（JPG/PNG, 可多选）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("选择照片尺寸")

    # ---- 中国常见证件照尺寸（cm） ----
    standard_sizes = {
        "一寸照 2.5×3.5cm": (2.5, 3.5),
        "二寸照 3.5×4.9cm": (3.5, 4.9),
        "身份证 3.5×4.4cm": (3.5, 4.4),
        "护照 4.3×5.3cm": (4.3, 5.3),
        "自定义": None
    }

    size_option = st.selectbox("选择证件照尺寸", list(standard_sizes.keys()))

    if size_option == "自定义":
        unit = st.radio("单位", ["像素(px)", "厘米(cm)"])
        if unit == "像素(px)":
            width = st.number_input("宽度(px)", min_value=1, value=358)
            height = st.number_input("高度(px)", min_value=1, value=441)
            target_size = (int(width), int(height))
        else:
            width_cm = st.number_input("宽度(cm)", min_value=0.1, value=3.5, step=0.1)
            height_cm = st.number_input("高度(cm)", min_value=0.1, value=4.4, step=0.1)
            target_size = (
                int(width_cm / 2.54 * DEFAULT_DPI),
                int(height_cm / 2.54 * DEFAULT_DPI)
            )
    else:
        w_cm, h_cm = standard_sizes[size_option]
        target_size = (
            int(w_cm / 2.54 * DEFAULT_DPI),
            int(h_cm / 2.54 * DEFAULT_DPI)
        )

    st.write(f"目标像素尺寸：{target_size[0]} x {target_size[1]} px")

    # 人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if st.button("批量处理并生成 ZIP 下载"):

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:

            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file).convert("RGB")
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # 人脸裁剪
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    margin = 0.4
                    x1 = max(int(x - w * margin), 0)
                    y1 = max(int(y - h * margin), 0)
                    x2 = min(int(x + w * (1 + margin)), img.width)
                    y2 = min(int(y + h * (1 + margin)), img.height)
                    cropped_img = img.crop((x1, y1, x2, y2))
                else:
                    st.warning(f"{uploaded_file.name} 未检测到人脸，使用原图")
                    cropped_img = img

                # ===== 保持比例缩放 + 填充白色背景 =====
                target_w, target_h = target_size
                img_ratio = cropped_img.width / cropped_img.height
                target_ratio = target_w / target_h

                if img_ratio > target_ratio:
                    # 宽图，宽度先满
                    new_w = target_w
                    new_h = int(target_w / img_ratio)
                else:
                    # 高图，高度先满
                    new_h = target_h
                    new_w = int(target_h * img_ratio)

                resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # 创建白底背景
                final_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
                paste_x = (target_w - new_w) // 2
                paste_y = (target_h - new_h) // 2
                final_img.paste(resized_img, (paste_x, paste_y))

                # 写入内存
                img_bytes = io.BytesIO()
                final_img.save(img_bytes, format="JPEG", quality=95)
                img_bytes.seek(0)

                zip_file.writestr(f"resized_{uploaded_file.name}", img_bytes.read())

        zip_buffer.seek(0)
        st.download_button(
            label="📦 下载所有调整后的照片 (ZIP)",
            data=zip_buffer,
            file_name="resized_photos.zip",
            mime="application/zip"
        )