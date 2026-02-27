import streamlit as st
import pandas as pd
import pubchempy as pcp
import re
import os
import tempfile
import time
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from padelpy import padeldescriptor
import base64

# ==========================================
# 1. 页面配置与科研风样式 (CSS)
# ==========================================
st.set_page_config(page_title="HOCs-Transformer Predictor", layout="wide", page_icon="🧪")

def local_css():
    st.markdown("""
        <style>
        .main { background-color: #f5f7fa; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #d1d9e6; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .reportview-container .main .block-container { padding-top: 2rem; }
        h1 { color: #0f4c81; font-family: 'Times New Roman', serif; font-weight: 700; border-bottom: 2px solid #0f4c81; padding-bottom: 10px; }
        h2, h3 { color: #1a365d; font-family: 'Arial', sans-serif; }
        .stSidebar { background-color: #e6f0fa; }
        .stSidebar h2 { color: #0f4c81; }
        .stSidebar .stRadio > div { padding: 8px 0; }
        .stSidebar .stTextInput > div > div > input { border: 1px solid #b3d1f1; border-radius: 4px; }
        .status-box { padding: 20px; border-radius: 8px; margin-top: 10px; }
        .highlight { color: #0f4c81; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ==========================================
# 2. 模型架构定义
# ==========================================
class TransformerClassifier(nn.Module):
    def __init__(self, num_feat_dim, vocab_size, max_smiles_length, num_classes,
                 d_model=64, nhead=4, num_encoder_layers=2,
                 dim_feedforward=128, dropout=0.3, activation='relu', embed_size=64):
        super().__init__()
        self.smiles_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_smiles_length, embed_size))
        nn.init.uniform_(self.positional_encoding, -0.1, 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.num_feat_projection = nn.Linear(num_feat_dim, embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(embed_size * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x_num, x_smiles):
        padding_mask = (x_smiles == 0)
        smiles_embedded = self.smiles_embedding(x_smiles) + self.positional_encoding
        transformer_output = self.transformer_encoder(smiles_embedded, src_key_padding_mask=padding_mask)
        
        mask_float = (~padding_mask).unsqueeze(-1).float()
        sum_embeddings = (transformer_output * mask_float).sum(dim=1)
        lengths = mask_float.sum(dim=1).clamp(min=1e-9)
        smiles_features = sum_embeddings / lengths

        num_features = self.num_feat_projection(x_num)
        combined_features = torch.cat((num_features, smiles_features), dim=1)
        return self.classifier(combined_features)

# ==========================================
# 3. 工具函数
# ==========================================
FEATURE_LIST = [
    'AD2D106', 'AD2D111', 'AD2D731', 'ExtFP619', 'FP191', 'FP203', 'FP374', 'FP800', 
    'GraphFP137', 'GraphFP829', 'KRFP98', 'KRFP330', 'KRFP348', 'KRFP349', 'KRFP350', 
    'KRFP354', 'KRFP498', 'KRFP566', 'KRFP589', 'KRFP604', 'KRFP1524', 'KRFP1724', 
    'KRFP1726', 'KRFP3788', 'KRFP4015', 'KRFP4822', 'MACCSFP107', 'MACCSFP123', 
    'MACCSFP153', 'PubchemFP343'
]

FP_TYPE_MAP = {
    'AD2D': 'AP2D', 'ExtFP': 'EFP', 'FP': 'FPR', 'GraphFP': 'GOF', 
    'KRFP': 'KRFC', 'MACCSFP': 'MACCS', 'PubchemFP': 'PCFP'
}

def get_fp_group(feat_name):
    for prefix, group in FP_TYPE_MAP.items():
        if feat_name.startswith(prefix): return group
    return None

@st.cache_resource
def load_model():
    """加载模型包，解决路径定位与依赖加载问题"""
    # 1. 动态获取 app.py 所在的绝对目录，确保能找到模型
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.pt")
    
    # 调试信息：如果找不到文件，打印当前目录下的所有文件协助排查
    if not os.path.exists(model_path):
        st.error(f"⚠️ 路径错误：无法在 {current_dir} 找到 model.pt")
        st.info(f"当前目录下的文件有: {os.listdir(current_dir)}")
        return None
    
    try:
        # 2. 加载模型
        # weights_only=False 是必须的，因为要加载 scaler (sklearn 对象)
        checkpoint = torch.load(
            model_path, 
            map_location=torch.device('cpu'),
            weights_only=False 
        )
        
        config = checkpoint['model_config']
        model = TransformerClassifier(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return {
            "model": model,
            "scaler": checkpoint['scaler'],
            "char_to_idx": checkpoint['char_to_idx'],
            "max_len": checkpoint['max_smiles_length'],
            "threshold": checkpoint.get('threshold', 0.5),
            "class_names": checkpoint.get('class_names', ['Non-RB', 'RB'])
        }
    except ModuleNotFoundError as e:
        st.error(f"❌ 运行环境缺失依赖: {str(e)}")
        st.info("请确保 requirements.txt 中包含 scikit-learn")
        return None
    except Exception as e:
        st.error(f"❌ 模型解析失败: {str(e)}")
        return None

def contains_halogen(smiles):
    smiles_upper = smiles.upper()
    if re.search(r'CL|BR', smiles_upper): return True
    i = 0
    while i < len(smiles_upper):
        if i + 1 < len(smiles_upper):
            if smiles_upper[i:i+2] in ['SI', 'FE', 'TI', 'NI', 'BI', 'LI', 'NA', 'CA', 'AL']:
                i += 2; continue
        if smiles_upper[i] in ['F', 'I']: return True
        i += 1
    return False

def render_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(400, 400))
        return img
    return None

def smiles_to_tensor(smiles, char_to_idx, max_len):
    seq = [char_to_idx.get(ch, 0) for ch in smiles]
    if len(seq) < max_len: seq += [0] * (max_len - len(seq))
    else: seq = seq[:max_len]
    return torch.tensor([seq], dtype=torch.long)

# ==========================================
# 4. 主界面逻辑
# ==========================================
st.markdown("# 🧪 HOCs-BDPred：基于多模态Transformer的卤代有机污染物生物降解性预测平台")
st.markdown("""
*基于 **多模态-Transformer** 架构的深度学习预测平台。通过融合分子拓扑指纹 (PaDEL) 与 SMILES 序列特征，
实现对卤代有机物 (Halogenated Organic Compounds) 可生化性的高精度评估。*
""")

pkg = load_model()
if pkg is None:
    st.error("⚠️ **Model Status:** `model.pt` 未找到。请确保模型文件存放在根目录。")
    st.stop()

# --- 第一部分：数据输入 ---
with st.sidebar:
    st.header("⚙️ 配置与输入")
    input_type = st.radio("输入模式", ["CAS 登录号", "SMILES 字符串"])
    input_val = st.text_input("待测物质标识符", placeholder="例如: C(Cl)Cl 或 107-06-2")
    st.divider()
    st.markdown("### 模型参数摘要")
    st.caption(f"**词表大小:** {len(pkg['char_to_idx'])}")
    st.caption(f"**最大序列长度:** {pkg['max_len']}")
    st.caption(f"**判定阈值 (τ):** {pkg['threshold']}")

target_smiles = None
if input_val:
    if input_type == "CAS 登录号":
        with st.spinner("🔄 正在从 PubChem 检索化学结构..."):
            try:
                results = pcp.get_compounds(input_val, 'name')
                if results: target_smiles = results[0].canonical_smiles
                else: st.sidebar.warning("无法找到对应的 SMILES")
            except: st.sidebar.error("网络连接异常")
    else: target_smiles = input_val

# --- 第二部分：分析面板 ---
if target_smiles:
    main_col1, main_col2 = st.columns([1, 1.5])
    
    with main_col1:
        st.subheader("🖼️ 化学结构可视化")
        img = render_molecule(target_smiles)
        if img:
            st.image(img, use_container_width=True, caption=f"Canonical SMILES: {target_smiles}")
        else:
            st.error("无效的 SMILES 格式")
            st.stop()
        
        # 卤素检测在图下面显示
        st.subheader("🔍 分子属性初筛")
        is_hoc = contains_halogen(target_smiles)
        
        if is_hoc:
            st.success("✅ 卤素检测：符合 (HOC) - 该分子属于卤代有机物")
        else:
            st.warning("⚠️ 卤素检测：不符合 - 该分子不含 F, Cl, Br, I 原子")

    with main_col2:
        # 把预测功能搬到这里
        st.subheader("🧠 预测推理系统")
        if st.button("🚀 执行预测 (Run Prediction)", type="primary"):
            with st.status("正在执行计算推理流水线...", expanded=True) as status:
                start_time = time.time()
                
                # 3.1 特征工程
                st.write("📡 正在调用 PaDEL-Descriptor 计算分子指纹...")
                needed_groups = sorted(list(set([get_fp_group(f) for f in FEATURE_LIST if get_fp_group(f)])))
                
                all_dfs = []
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
                        f.write(f"{target_smiles}\tmol_1\n")
                        temp_smi = f.name
                    
                    for group_code in needed_groups:
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;→ 提取子集: `{group_code}`...")
                        fp_name_map = {
                            'AP2D': 'AtomPairs2DFingerprinter', 'EFP': 'ExtendedFingerprinter',
                            'FPR': 'Fingerprinter', 'GOF': 'GraphOnlyFingerprinter',
                            'KRFC': 'KlekotaRothFingerprinter', 'MACCS': 'MACCSFingerprinter',
                            'PCFP': 'PubchemFingerprinter'
                        }
                        xml_content = f'<Root><Group name="Fingerprint"><Descriptor name="{fp_name_map[group_code]}" value="true"/></Group></Root>'
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f_xml:
                            f_xml.write(xml_content)
                            temp_xml = f_xml.name
                        
                        output_csv = temp_xml + ".csv"
                        padeldescriptor(mol_dir=temp_smi, d_file=output_csv, descriptortypes=temp_xml, 
                                         fingerprints=True, retainorder=True, sp_timeout=None)
                        if os.path.exists(output_csv):
                            all_dfs.append(pd.read_csv(output_csv))
                            os.remove(output_csv)
                        os.remove(temp_xml)
                    os.remove(temp_smi)

                    combined_df = pd.concat(all_dfs, axis=1)
                    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                    extracted_vals = [combined_df[feat].iloc[0] if feat in combined_df.columns else 0 for feat in FEATURE_LIST]
                    
                    # 3.2 模型推理
                    st.write("🧠 正在加载 Transformer 编码器并进行张量映射...")
                    x_num_scaled = pkg['scaler'].transform(np.array([extracted_vals], dtype=np.float32))
                    x_num_tensor = torch.tensor(x_num_scaled, dtype=torch.float32)
                    x_smiles_tensor = smiles_to_tensor(target_smiles, pkg['char_to_idx'], pkg['max_len'])
                    
                    with torch.no_grad():
                        logits = pkg['model'](x_num_tensor, x_smiles_tensor)
                        probs = torch.softmax(logits, dim=1)
                        prob_rb = probs[0, 1].item()
                        prediction = 1 if prob_rb >= pkg['threshold'] else 0
                    
                    status.update(label=f"✅ 推理完成！耗时: {time.time()-start_time:.2f}s", state="complete", expanded=False)

                    # --- 第四部分：结果展示 (Scientific Report) ---
                    st.markdown("### 📊 预测报告 (Research Report)")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        class_name = pkg['class_names'][prediction]
                        label_color = "🟢" if class_name == 'RB' else "🔴"
                        st.metric("预测类别", f"{label_color} {class_name}")
                    
                    with res_col2:
                        desc = "易降解" if class_name == 'RB' else "难降解"
                        st.metric("生化特性定义", desc)
                    
                    # with res_col3:
                    #     confidence = prob_rb if prediction == 1 else (1 - prob_rb)
                    #     st.metric("置信度 (Probability)", f"{round(confidence*100, 2)}%")

                    # 概率分布图
                    # st.progress(prob_rb, text=f"降解概率: {round(prob_rb*100, 2)}%")

                    with st.expander("📝 详细指纹特征向量 (Top-30 Features)"):
                        feat_df = pd.DataFrame([extracted_vals], columns=FEATURE_LIST)
                        st.dataframe(feat_df.style.background_gradient(axis=1, cmap='Blues'), use_container_width=True)
                        
                        csv = feat_df.to_csv(index=False).encode('utf-8')
                        st.download_button("💾 下载特征数据 (CSV)", data=csv, file_name="fingerprints.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"⚠️ 系统异常: {str(e)}")
                    status.update(label="推理中断", state="error")

# --- 页脚 ---
st.divider()
st.caption("""
**Methodology Note:** The model employs a dual-stream architecture: 
1. A **Transformer Encoder** captures the sequential syntax of SMILES strings. 
2. A **Multi-layer Perceptron (MLP)** processes 30 predefined topological fingerprints. 
The final decision is made via late fusion of chemical descriptors and sequential embeddings.
""")
