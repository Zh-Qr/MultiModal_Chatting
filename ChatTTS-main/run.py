###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import torch
# torch._dynamo.config.cache_size_limit = 64
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')
import requests
import json
API_KEY = "paSYErMAdyPxNCa6yEmQYOUy"
SECRET_KEY = "gWMidL4x6QhHJern6t4r8IjXbNLUksUU"

import ChatTTS
from IPython.display import Audio

print("done")
chat = ChatTTS.Chat()
# chat.load_models(force_redownload=True)
local_path = '项目2-大模型应用创意设计/ChatTTS-main'
chat.load_models(source='local', local_path=local_path)
def get_mess(s):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + get_access_token()
    # 注意message必须是奇数条
    payload = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": s
        }
    ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
 
    res = requests.request("POST", url, headers=headers, data=payload).json()
    return res['result']

def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))
s = "评价一下美国总统特朗普,请简短一点介绍，不要超过50个字"
result = get_mess(s)
print(result)
params_infer_code = {'prompt':'[speed_3]', 'temperature':.2}
params_refine_text = {'prompt':'[oral_3][laugh_0][break_6]'}

wav = chat.infer(result, \
    params_refine_text=params_refine_text, params_infer_code=params_infer_code)
Audio(wav[0], rate=24_000, autoplay=True)
import soundfile as sf
import numpy as np
import os
print("当前工作目录:", os.getcwd())


def save_audio(audio_data, sample_rate, filename="results/project2_audio/output.wav"):
    # 确保数据是正确的二维形状，对于单声道
    if audio_data.ndim == 2 and audio_data.shape[0] == 1:
        audio_data = audio_data.T  # 转置数据以获得正确的形状

    print("修正后的数据形状:", audio_data.shape)

    # 规范化音频数据到 [-1.0, 1.0]（如果需要）
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val

    # 尝试保存文件
    try:
        sf.write(filename, audio_data, sample_rate, subtype='FLOAT')
        print(f"音频已保存到 {filename}")
    except Exception as e:
        print("保存失败:", e)

# 假设 `wav` 是从你的模型中得到的音频数据列表，采样率为24000 Hz
# 调用 save_audio 函数保存第一个音频
if wav:  # 确保 wav 不为空
    save_audio(wav[0], 24000)