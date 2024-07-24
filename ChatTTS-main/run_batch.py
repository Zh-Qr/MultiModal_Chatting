###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import torch
# torch._dynamo.config.cache_size_limit = 64
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')

import ChatTTS
from IPython.display import Audio

print("done")
chat = ChatTTS.Chat()
# chat.load_models(force_redownload=True)
local_path = '项目2-大模型应用创意设计/ChatTTS-main'
chat.load_models(source='local', local_path=local_path)
filename="results/project2_audio/"
results = ["浙江大学软件学院是浙江大学的一个学院，现属于信息学部。",
       "学院前身是于2001年2月27日在杭州与宁波两地同时挂牌成立的浙江大学软件与网络学院",
       "2001年12月学院成为教育部和国家发展计划委员会批准的首批35所国家示范性软件学院之一，并更名为软件学院。",
       "浙江大学成立于1897年，前身求是书院，是中国人最早自己创办的新式高等学府之一。是首批进入国家211工程和985工程建设的重点大学之一。",
       "浙江大学师资队伍整体力量雄厚。现有中国科学院院士15名，中国工程院院士12名，973项目首席科学家9名，杰出青年基金获得者67名。",
       "高水平的研究生导师队伍和良好的科研实验条件，为开展高水平的研究生教育打下了坚实的基础。",
       "软件学院将始终秉承求是创新的校训，努力建设成为具有国际影响的IT领域专业性学院，成为国家软件人才培养的重要基地之一",
       "作为国家示范性软件学院之一，浙江大学软件学院始终按照教育部示范性软件学院建设的要求，以培养应用型、复合型、国际化高级软件人才为目标，不断探索学生培养新模式",
       "学院充分发挥浙江大学的综合办学优势，汇聚政府、行业、产业等各方资源，共同参与学生培养。",
       "为加强应用型软件人才的培养，学院采用灵活的课程体系和动态的教学计划，实施了理论教学与课程实践、项目实训、企业实习相结合的培养体系。",
       "所有研究生都要求进入企事业单位实习，在实际工作中完成学位论文。学生在企业实习期间，实行双导师制。",
       "学院以杭州、宁波为基地，以长三角地区为重点，积极争取地方政府支持与合作，为地方社会经济建设服务。"]
import soundfile as sf
import numpy as np
import os
print("当前工作目录:", os.getcwd())


def save_audio(audio_data, sample_rate, filename):
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
index = 1
for result in results:
    params_infer_code = {'prompt':'[speed_3]', 'temperature':.2}
    params_refine_text = {'prompt':'[oral_3][laugh_0][break_6]'}
    wav = chat.infer(result, \
        params_refine_text=params_refine_text, params_infer_code=params_infer_code)
    files = filename + "output"+str(index)+".wav"
    save_audio(wav[0], 24000,files)
    index = index+1