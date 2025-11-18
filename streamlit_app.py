import io
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import streamlit as st

from models.cifar_cnn import CifarCNN

# Те же самые статистики, что и в utils/data.py
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")
st.title("CIFAR-10 Image Classifier")
st.caption("Загрузите изображение (jpg/png), модель предскажет один из 10 классов CIFAR-10.")


@st.cache_resource
def load_model(path: str = "artifacts/cifar_net.pth"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл модели {path}. Сначала обучите модель командой `python train.py`."
        )

    model = CifarCNN(num_classes=len(CLASSES))
    ckpt = torch.load(path, map_location="cpu")

    # мы сохраняли {"state_dict": ...}
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data
def preprocess_image(image_bytes: bytes):
    """
    Преобразуем загруженную картинку:
    - открываем как RGB;
    - ресайзим до 32x32 (как CIFAR-10);
    - нормализуем.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_img = img.copy()

    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    x = tfm(img).unsqueeze(0)  # [1, 3, 32, 32]
    return display_img, x


uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    display_img, x = preprocess_image(uploaded_file.read())
    st.image(display_img, caption="Загруженное изображение", use_column_width=True)

    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
    else:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        top3 = np.argsort(-probs)[:3]

        st.subheader("Предсказание модели")
        main_idx = int(top3[0])
        st.write(f"**Класс:** {CLASSES[main_idx]}  (вероятность: {probs[main_idx]:.3f})")
        st.progress(float(probs[main_idx]))

        st.subheader("Топ-3 классов")
        for idx in top3:
            st.write(f"- {CLASSES[int(idx)]}: `{probs[idx]:.3f}`")
else:
    st.info("Пока файл не выбран. Загрузите изображение формата JPG или PNG.")
