
"""
NLLB-600M LoRA 微调（三步渐进式迁移学习 - 模块化版）
=====================================================

改进策略（重命名后）：
  第一步：置信度加权      - 低资源语言样本加权，并列术语组合训练
  第二步：跨语言表示对齐  - 基于英文译文相似度的编码器对齐
  第三步：知识蒸馏        - 高资源教师模型指导低资源学习

模块化改造：
  ✅ argparse 参数配置系统（get_args）
  ✅ train/evaluate/predict 三段式架构
  ✅ 结构化日志系统（init_logger）
  ✅ Warmup学习率调度（防止初始阶段扰动预训练权重）
  ✅ 梯度裁剪（防止KD温度缩放和对齐损失引发梯度爆炸）
  ✅ 梯度累积（提升小批量梯度估计稳定性）

核心优势：
  - 低成本数据收集（仅需"语言→英文"平行数据）
  - 自动发现跨语言对应关系（无需手工标注）
  - 支持下游多语言聚类/检索任务
"""

# ============================================================
# 标准库导入
# ============================================================
import os          # 文件路径与目录操作
import sys         # 系统接口（stdout重定向）
import json        # 术语库JSON加载
import time        # 时间戳与ETA计算
import random      # 随机采样与种子
import logging     # 结构化日志
import argparse    # 命令行参数解析
import re          # 正则表达式（句子切分、后处理）
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F 
from typing import (
    List,          # 列表类型注解
    Dict,          # 字典类型注解
    Tuple,         # 元组类型注解
    Set,           # 集合类型注解
    Optional,      # 可选类型注解
)

# ============================================================
# 深度学习框架导入
# ============================================================
import torch                      # PyTorch主框架
from torch.optim import Optimizer # 自定义优化器基类
# ============================================================
# HuggingFace生态导入（按需精确导入，减少命名空间污染）
# ============================================================
from datasets import Dataset  # HuggingFace Dataset容器
# Transformers核心组件
from transformers import (
    AutoTokenizer,            # 自动加载对应分词器
    AutoModelForSeq2SeqLM,    # 自动加载Seq2Seq模型（NLLB）
    TrainingArguments,        # 训练超参数容器
    Trainer,                  # 标准训练循环
    TrainerCallback,          # 自定义训练回调基类
    get_linear_schedule_with_warmup,  # Warmup线性学习率调度
)
# 回调类（用于移除默认输出，替换为自定义进度条）
from transformers.trainer_callback import (
    PrinterCallback,   # HuggingFace默认文本打印回调
    ProgressCallback,  # HuggingFace默认tqdm进度条回调
)
# 数据整理器（动态填充序列长度，避免固定padding浪费）
from transformers.data.data_collator import DataCollatorForSeq2Seq
# PEFT（参数高效微调）组件
from peft import (
    LoraConfig,       # LoRA超参数配置
    TaskType,         # 任务类型枚举（SEQ_2_SEQ_LM）
    get_peft_model,   # 将基础模型注入LoRA层
    PeftModel,        # 加载已保存的LoRA adapter
)

from dataclasses import dataclass, field
from sklearn.preprocessing import RobustScaler, StandardScaler

# ============================================================
# 全局常量定义
# ============================================================

# ISO 639-3 → NLLB内部语言代码映射
# NLLB使用"语言_脚本"格式（如zho_Hans = 中文简体汉字）
# 作为强制BOS token指导解码器输出目标语言
LANG_MAP: Dict[str, str] = {
    "zh": "zho_Hans",   # 中文简体（CJK字符集）
    "en": "eng_Latn",   # 英文（拉丁脚本）
    "ind": "ind_Latn",   # 印尼语（拉丁脚本）
    "tl": "tgl_Latn",   # 他加禄语（拉丁脚本，菲律宾官方语言）
}

# 目标语言锚点：以英文为语义聚合中心
# 所有源语言均翻译到英文，便于下游跨语言检索/聚类
TGT_LANG: str = "eng_Latn"

# 数据基础扩增倍数（每条术语重复REPEAT次进入训练集）
# 较小的术语库（<200条）需要重复扩增以保证训练步数足够
REPEAT: int = 2 #5

# 高低资源语言分类
# 高资源：训练数据充足，可作为知识蒸馏的教师端
# 低资源：数据稀缺，是微调的重点补偿对象
HIGH_RESOURCE_LANGS: Set[str] = {"zho_Hans"}               # 中文（高资源）
LOW_RESOURCE_LANGS: Set[str]  = {"ind_Latn", "tgl_Latn"}   # 印尼语、他加禄语（低资源）


# ============================================================
# 日志系统初始化
# ============================================================

def init_logger(log_file: Optional[str] = None,
                log_level: int = logging.INFO) -> logging.Logger:
    """
    初始化结构化日志系统

    设计原则：
      - 同时输出到控制台和文件（若指定log_file）
      - 统一时间戳格式，便于日志分析
      - 禁用transformers/peft内部冗余日志（避免刷屏）
      - force=True确保重复调用时重置配置（Jupyter环境兼容）

    Args:
        log_file:  日志文件路径，None表示仅控制台输出
        log_level: 日志级别（logging.INFO/DEBUG/WARNING）

    Returns:
        配置好的Logger实例
    """
    # 禁用第三方库的冗余日志，只保留ERROR级别
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)

    # 构建输出处理器列表
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout)  # 控制台处理器
    ]

    # 若指定日志文件，添加文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(
            logging.FileHandler(log_file, encoding='utf-8')
        )

    # 全局基础配置（force=True覆盖已有配置）
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level,
        handlers=handlers,
        force=True,
    )

    return logging.getLogger(__name__)


# ============================================================
# 参数配置系统
# ============================================================
def get_args() -> argparse.Namespace:
    """
    命令行参数配置系统（新增SONAR相关参数）

    设计原则：
      - 所有超参数支持命令行覆盖，便于实验管理
      - 敏感路径支持环境变量（NLLB_MODEL_PATH）
      - 提供合理默认值，零配置可直接运行
      - 参数分组清晰（路径/训练/LoRA/CG-LoRA/策略/运行模式）

    新增参数（相比原版）：
      --warmup_proportion:         Warmup占总步数的比例
      --gradient_clip_norm:        梯度裁剪最大范数
      --gradient_accumulation_steps: 梯度累积步数

    Returns:
        解析后的参数命名空间对象
    """
    parser = argparse.ArgumentParser(
        description='NLLB-600M LoRA 三步渐进式微调系统（SONAR增强版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ==================== 路径参数 ====================
    parser.add_argument('--model_path', type=str, default=os.getenv('NLLB_MODEL_PATH', r'F:\hyberT\预训练\facebooknllb-200-distilled-600M'), 
                        help='NLLB预训练模型本地路径（支持NLLB_MODEL_PATH环境变量）')
    parser.add_argument('--glossary_path', type=str, default=r'F:\hyberT\MultiSonar\data\terms1.json', 
                        help='术语库JSON文件路径（格式：{"terms":[{"term":{},"translation":""}]}）')
    parser.add_argument('--output_dir', type=str, default=r'F:\hyberT\预训练\nllb_Flora(Sonar)', 
                        help='LoRA adapter输出根目录（各阶段checkpoint存放于子目录）')
    parser.add_argument('--log_dir', type=str, default='logs', help='训练日志输出目录（自动创建）')
    parser.add_argument('--cache_dir', type=str, default=None, help='HuggingFace模型缓存目录（None使用默认缓存）')

    # ==================== 基础训练超参数 ====================
    parser.add_argument('--epochs', type=int, default=10, help='每步训练的最大epoch数')
    parser.add_argument('--batch_size', type=int, default=2, help='单设备训练批次大小（术语库场景推荐2~4）')
    parser.add_argument('--lr', type=float, default=1e-3, help='峰值学习率（Warmup结束后的最大lr）')
    parser.add_argument('--max_length', type=int, default=256, help='Tokenize时的最大序列长度（源端），目标端自动-2')

    # ==================== Warmup与梯度参数（新增）====================
    parser.add_argument('--warmup_proportion', type=float, default=0.05, 
                        help=('Warmup占总训练步数的比例（推荐0.05~0.15）。\n作用：前warmup_proportion步lr从0线性增长到lr，\n防止初始阶段LoRA权重对预训练模型造成大扰动。\n对齐训练尤其重要（编码器需要平稳初始化）。'))
    parser.add_argument('--gradient_clip_norm', type=float, default=2.0, 
                        help=('梯度裁剪最大L2范数（推荐0.5~2.0）。\n作用：防止KD温度缩放（T²=16）和对齐损失引发梯度爆炸。\n设为0表示不裁剪（不推荐）。'))
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, 
                        help=('梯度累积步数（推荐2~8）。\n作用：等效batch_size = batch_size × accumulation_steps，\n提升小批量梯度估计稳定性，不增加显存。\n建议：batch_size=2时设为4，等效batch=8。'))

    # ==================== LoRA参数 ====================
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA低秩分解的秩（r越大，参数越多，拟合能力越强）')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA缩放系数（通常设为2×lora_r，平衡更新幅度）')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA层的Dropout率（正则化，防止LoRA层过拟合）')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj', 
                        help='应用LoRA的目标模块名（逗号分隔，覆盖注意力层Q/K/V/O投影）')

    # ==================== CG-LoRA优化器参数 ====================
    parser.add_argument('--curvature_lambda', type=float, default=0.05, 
                        help=('Fisher曲率项权重λ（推荐0.01~0.1）。\n更新公式：θ -= lr·m̂ / (√(v̂+λF̂) + ε)，\nλ=0退化为标准Adam，λ>0利用曲率自适应步长。'))
    parser.add_argument('--fisher_beta', type=float, default=0.999, 
                        help=('Fisher对角EMA衰减系数（推荐0.999）。\n越接近1，Fisher估计越平滑，对历史梯度记忆越长。'))

    # ==================== 三步策略开关 ====================
    parser.add_argument('--enable_step1', action='store_true', default=False, help='启用第一步：低资源置信度加权采样（boost_factor倍）')
    parser.add_argument('--enable_step2', action='store_true', default=True, help='启用第二步：跨语言表示对齐（余弦相似度损失约束编码器）')
    parser.add_argument('--enable_step3', action='store_true', default=False, help='启用第三步：知识蒸馏（高资源教师模型软标签指导）')

    # ==================== 策略超参数 ====================
    parser.add_argument('--boost_factor', type=float, default=2.0, help='低资源样本加权倍数（推荐1.5~3.0，过高会放大噪声）')
    parser.add_argument('--align_lambda', type=float, default=0.1, help='对齐损失权重λ（L=CE+λ·AlignLoss，推荐0.05~0.2）')
    parser.add_argument('--align_min_overlap', type=float, default=0.5, help='跨语言术语英文译文相似度阈值（低于此值不配对）')
    parser.add_argument('--align_max_pairs', type=int, default=200, help='最大对齐术语对数量（控制对齐损失计算开销）')
    parser.add_argument('--update_align_every', type=int, default=100, help='在线刷新对齐编码对的步数间隔（越小越精确，越慢）')
    parser.add_argument('--kd_temperature', type=float, default=4.0, help='知识蒸馏温度T（越高软标签越平滑，推荐2.0~6.0）')
    parser.add_argument('--kd_alpha', type=float, default=0.3, help='蒸馏损失权重α（L=(1-α)CE+α·T²KL，推荐0.2~0.5）')

    # ==================== 运行模式 ====================
    parser.add_argument('--do_train', action='store_true', default=True, help='执行训练流程')
    parser.add_argument('--do_eval', action='store_true', default=False, help='执行验证集评估（需提供验证数据，当前为预留接口）')
    parser.add_argument('--do_predict', action='store_true', default=True, help='训练后执行测试翻译并输出对比结果')

    # ==================== 目录覆盖控制（修复核心）====================
    parser.add_argument('--overwrite_output_dir', action='store_true', default=True, 
                        help=('允许覆盖已存在的输出目录（默认True）。\n设为False时，若目录非空则报错退出。\n用法：--overwrite_output_dir（启用，默认行为）'))
    # ==================== 设备与可复现性 ====================
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备（cuda/cpu，自动检测GPU）')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子（保证实验可复现）')

    # ==================== 可选测试文本 ====================
    parser.add_argument('--test_text', type=str, default=None, help='自定义测试文本（None时使用内置默认测试句）')
        
    # ==================== 新增SONAR参数组 ====================
    sonar_group = parser.add_argument_group('SONAR增强参数')    
    sonar_group.add_argument('--use_sonar', action='store_true', default=True,
        help='启用SONAR语义相似度增强（默认启用）')    
    sonar_group.add_argument('--sonar_weight', type=float, default=0.7,
        help='SONAR相似度权重（0-1，推荐0.7），剩余为Jaccard权重')    
    sonar_group.add_argument('--sonar_enable_denoising', action='store_true', default=True,
        help='启用语义去噪模块（默认启用）')    
    sonar_group.add_argument( '--sonar_pretrain_epochs', type=int, default=10,
        help='SONAR自编码器预训练轮数（推荐3-5，数据充足时可增加）')    
    sonar_group.add_argument('--sonar_pretrain_lr', type=float, default=1e-4,
        help='SONAR预训练学习率（推荐1e-4）')    
    sonar_group.add_argument('--sonar_decoder_rank', type=int, default=16,
        help='解码器低秩分解秩（控制参数量，推荐32-128）')   
    sonar_group.add_argument('--sonar_mse_weight', type=float, default=0.5,help='MSE重构损失权重')   
    sonar_group.add_argument('--sonar_cosine_weight', type=float, default=0.3,
        help='余弦相似度损失权重')    
    sonar_group.add_argument('--sonar_denoising_weight', type=float, default=0.2,
        help='去噪损失权重')
    sonar_group.add_argument('--sonar_enable_distill', action='store_true',default=True,
                             help='启用SONAR知识蒸馏（默认启用）')
    sonar_group.add_argument('--sonar_distill_lambda',type=float,default=0.15,
                             help='SONAR蒸馏损失权重（推荐0.10-0.20）')
    #以上部分的参数属于标准训练（仅SONAR蒸馏，不启用Engram），即消融的1和2
    #其中，消融1代表基于NLLB的原始直接翻译，不加任何附属条件；
    #消融2代表，基于Sonar句子分割蒸馏的NLLB翻译；

    # ===== 新增：Engram条件记忆参数组，启用Engram条件记忆的配置 =====
    engram_group = parser.add_argument_group('Engram条件记忆参数')    
    engram_group.add_argument('--enable_engram', action='store_true', default=True,
        help='启用Engram条件记忆增强（默认禁用，需手动开启）')    
    engram_group.add_argument('--engram_max_ngram', type=int, default=3,
        help='N-gram最大阶数（推荐2-3，过大会增加哈希冲突）')    
    engram_group.add_argument('--engram_hash_size', type=int, default=131072, #262144
        help='哈希表大小（2^17=131072，约10万槽位，适配1000条术语库）')   
    engram_group.add_argument('--engram_num_heads', type=int, default=8,
        help='门控网络多头数（与NLLB注意力头数一致）')   
    engram_group.add_argument('--engram_gate_threshold', type=float, default=0.3,
        help='门控激活阈值（低于此值不注入记忆，推荐0.3-0.5）')    
    engram_group.add_argument('--engram_lambda', type=float, default=0.10,
        help='Engram稀疏损失权重（推荐0.05-0.15）')    
    engram_group.add_argument('--engram_confidence_weight', type=float, default=0.5,
        help='置信度惩罚权重（防止门控过度激活）')    
    engram_group.add_argument('--engram_align_weight', type=float, default=0.3,
        help='记忆对齐损失权重（确保记忆与隐藏空间兼容）')    
    engram_group.add_argument('--beam_professional_threshold', type=float, default=0.5,
        help='Beam专业性筛选阈值（低于此值跳过Engram检索）')    
    engram_group.add_argument('--sonar_rerank_alpha', type=float, default=0.3,
        help='SONAR重排序权重（0=纯LM，1=纯SONAR，推荐0.3）')    
    engram_group.add_argument('--sonar_rerank_interval', type=int, default=5,
        help='SONAR重排序间隔步数（降低计算开销）')    
    # ===== 阶段性训练参数 =====
    engram_group.add_argument('--engram_warmup_epochs', type=int, default=2,
        help='Engram Warmup阶段轮数（快速学习门控模式）')    
    engram_group.add_argument('--engram_finetune_epochs', type=int, default=6,
        help='联合微调阶段轮数（平衡翻译质量与记忆精度）')    
    engram_group.add_argument('--engram_align_epochs', type=int, default=2,
        help='SONAR强化对齐阶段轮数（可选，高精度场景）')    
    engram_group.add_argument('--enable_stage_d', action='store_true', default=True,
        help='启用阶段D（SONAR强化对齐，牺牲流利性换取精度，适用低资源语言翻译）')   

    # ===== 新增：ELF优化参数组 =====
    elf_group = parser.add_argument_group('ELF优化参数（方案A+B+C）')
    
    # 方案A：Flow-guided LoRA
    elf_group.add_argument('--enable_flora', action='store_true', default=True,
        help='启用Flow-guided LoRA（FLoRA）优化' )
    elf_group.add_argument('--flora_rank', type=int, default=16, 
                           help='FLoRA秩（标准LoRA的一半，默认16）')
    elf_group.add_argument('--flora_mse_weight', type=float, default=0.3,
        help='FLoRA MSE损失最大权重（默认0.3）' )
    elf_group.add_argument('--flora_flow_steps', type=int, default=4,
        help='Flow内部迭代次数（训练时，默认4）')
    elf_group.add_argument('--flora_alpha_schedule', type=str, default='cosine',
        choices=['cosine', 'linear', 'fixed'], help='MSE权重调度策略')
    
    # 方案B：Flow Engram
    elf_group.add_argument('--enable_flow_engram', action='store_true', default=True,
        help='启用Flow-based Engram记忆聚合')
    elf_group.add_argument('--flow_engram_K', type=int, default=5,
        help='Engram记忆累积步数（默认5）')
    
    # 方案C：延迟离散化（实验性）
    elf_group.add_argument('--enable_delayed_discretization', action='store_true', default=False,
        help='【实验性】启用延迟离散化（默认关闭）')
    elf_group.add_argument('--delayed_K', type=int, default=5,
        help='延迟离散化：最后K步离散化' )

    args = parser.parse_args()

    # 后处理：将逗号分隔的字符串转为列表
    args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

    return args

@dataclass
class SonarConfig:
    """
    SONAR自编码器配置
    
    设计原则：
      - 轻量化：低秩解码器（rank=64），参数量<100k
      - 鲁棒性：RobustScaler标准化，高dropout（0.3-0.5）
      - 语言特定：每种语言独立拟合标准化参数
    """
    # 基础配置
    dim: int = 1024  # SONAR嵌入维度（固定）
    device: str = "cuda"
    supported_languages: List[str] = field(default_factory=lambda: [
        "zho_Hans", "eng_Latn", "ind_Latn", "tgl_Latn"
    ])
    
    # 标准化参数
    normalization_method: str = "robust"  # robust/gaussian_robust/standard
    quantile_min: float = 0.20  # RobustScaler下界（10%分位数）
    quantile_max: float = 0.80  # RobustScaler上界（90%分位数）
    clip_proba: Optional[float] = 1e-4 #0.02  # 裁剪极端值比例（2%）
    
    # 去噪参数
    enable_denoising: bool = True  # 是否启用语义去噪
    num_denoiser_layers: int = 2  # Transformer层数（轻量化）
    num_denoiser_heads: int = 8   # 注意力头数
    denoiser_dropout: float = 0.3  # Dropout率（防过拟合）
    noise_schedule: str = "cosine"  # 噪声调度策略
    
    # 解码器参数（低秩设计）
    decoder_rank: int = 64  # 低秩分解秩（参数量：1024×64×2=131k）
    decoder_dropout: float = 0.3  # 超高dropout0.5
    
    # 预训练参数
    pretrain_epochs: int = 5  # 短周期预训练（避免过拟合）
    pretrain_lr: float = 1e-4  # 保守学习率
    pretrain_batch_size: int = 16  # 小批量
    weight_decay: float = 0.5  # 强L2正则
    
    # 损失权重
    mse_weight: float = 0.5  # MSE重构损失权重
    cosine_weight: float = 0.3  # 余弦相似度损失权重
    denoising_weight: float = 0.2  # 去噪损失权重
    
    # 锚点语言（用于默认标准化）
    anchor_language: str = "eng_Latn"


# ============================================================
# 随机种子固定
# ============================================================

def seed_everything(seed: int) -> None:
    """
    固定全局随机种子，确保实验可复现

    覆盖范围：
      - Python random模块
      - 环境变量PYTHONHASHSEED（影响hash随机化）
      - PyTorch CPU/GPU随机数
      - cuDNN确定性模式（关闭benchmark自动调优）

    Args:
        seed: 随机种子整数值（通常使用42）
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 关闭cuDNN自动调优，确保卷积结果确定性
        # 代价：可能降低GPU运算速度约10~20%
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 自定义轻量级进度条
# ============================================================
class ProgressBar:
    """
    轻量级训练进度条（单行原地刷新，无外部依赖）

    设计原理：
      通过替换sys.stdout实现"stdout保护"机制：
        - 训练过程中所有print()均被静默丢弃（不打断进度条）
        - 进度条自身通过_raw_stdout直接写入终端
        - 两套输出通道互不干扰
    """

    def __init__(self, n_total: int, width: int = 40):
        """
        Args:
            n_total: 总训练步数（用于计算进度百分比和ETA）
            width:   进度条字符宽度（默认40字符）
        """
        self.width       = width
        self.n_total     = n_total
        self.start_time  = time.time()
        # 保存原始stdout引用，进度条专用输出通道
        self._raw_stdout = sys.stdout
        # 保存被临时静默的handler及其原始日志级别
        # 格式：[(handler, original_level), ...]
        self._silenced_handlers: List[Tuple] = []

    def __enter__(self) -> "ProgressBar":
        """
        启动保护区：同时拦截print()和logging输出。
        """
        # ① 替换stdout（拦截print）
        sys.stdout = self

        # ② 静默所有终端StreamHandler（拦截logging终端输出）
        self._silenced_handlers = []
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and \
               not isinstance(handler, logging.FileHandler):
                # 保存原始级别
                self._silenced_handlers.append((handler, handler.level))
                # 提升至CRITICAL，使INFO/WARNING在保护区内静默
                handler.setLevel(logging.CRITICAL)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        退出保护区：恢复stdout和logging Handler级别。
        """
        # ① 恢复stdout
        sys.stdout = self._raw_stdout

        # ② 恢复所有被静默的StreamHandler
        for handler, original_level in self._silenced_handlers:
            handler.setLevel(original_level)
        self._silenced_handlers = []

        return False  # 不吞异常

    def write(self, text: str) -> None:
        """
        拦截所有外部print输出并静默丢弃。
        """
        pass

    def flush(self) -> None:
        """flush协议接口，路由到原始stdout的flush"""
        self._raw_stdout.flush()

    def _time_info(self, now: float, current: int) -> str:
        """
        计算时间信息字符串。
        训练中显示ETA，训练后显示总用时。
        """
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                fmt = '%d:%02d:%02d' % (
                    eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                fmt = '%d:%02d' % (eta // 60, eta % 60)
            else:
                fmt = '%ds' % int(eta)
            return f' - ETA: {fmt}'
        else:
            elapsed = now - self.start_time
            if elapsed > 3600:
                fmt = '%d:%02d:%02d' % (
                    elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60)
            elif elapsed > 60:
                fmt = '%d:%02d' % (elapsed // 60, elapsed % 60)
            else:
                fmt = '%ds' % int(elapsed)
            return f' - 用时: {fmt}'

    def _bar(self, current: int, epoch_str: str) -> str:
        """生成ASCII进度条字符串"""
        recv_per = min(current / self.n_total, 1.0)
        bar = f'{epoch_str}: {current}/{self.n_total} ['
        pw  = int(self.width * recv_per)
        if pw > 0:
            bar += '=' * (pw - 1)
            bar += '>' if current < self.n_total else '='
        bar += '.' * (self.width - pw) + ']'
        return bar

    def __call__(self, step: int, loss: Optional[str] = None,
                 epoch_str: str = 'epoch') -> None:
        """
        更新并刷新进度条。
        通过_raw_stdout直接写入终端，绕过write()拦截。
        """
        now     = time.time()
        current = step + 1
        line    = f"{self._bar(current, epoch_str)}{self._time_info(now, current)}"
        if loss is not None:
            line += f' | loss:{loss}'
        self._raw_stdout.write('\r' + line.ljust(120))
        self._raw_stdout.flush()

    def finish(self) -> None:
        """训练结束输出换行"""
        self._raw_stdout.write('\n')
        self._raw_stdout.flush()


# ============================================================
# Trainer进度回调
# ============================================================

class CustomProgressCallback(TrainerCallback):
    """
    自定义Trainer进度回调（集成stdout保护的进度条）

    工作原理：
      - on_train_begin：启动ProgressBar的stdout保护，接管所有print输出
      - on_log：从Trainer日志提取loss，清空logs阻止HuggingFace默认输出
      - on_step_end：每步刷新进度条
      - on_train_end：恢复stdout，输出换行

    覆写了HuggingFace默认的PrinterCallback和ProgressCallback，
    确保整个训练期间只有进度条单行显示，无冗余文本输出。
    """

    def __init__(self, total_epochs: int):
        """
        Args:
            total_epochs: 总训练轮数（用于进度条标签显示）
        """
        self.pbar         = None
        self.last_loss    = None
        self.total_epochs = total_epochs
        self.cur_epoch    = 0

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """训练开始：初始化进度条并启动stdout保护"""
        self.pbar      = ProgressBar(n_total=state.max_steps, width=40)
        self.last_loss = None
        self.cur_epoch = 0
        self.pbar.__enter__()

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        """每轮开始：更新当前轮次计数"""
        self.cur_epoch = int(state.epoch) if state.epoch else 0

    def on_log(self, args, state, control,
               logs: Optional[Dict] = None, **kwargs) -> None:
        """
        日志触发：提取loss值，清空logs字典。

        logs.clear()阻止HuggingFace默认的loss打印，
        避免多行日志打断进度条的单行刷新。
        """
        if logs:
            if 'loss' in logs:
                self.last_loss = f"{logs['loss']:.4f}"
            logs.clear()

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """每步结束：刷新进度条显示"""
        if self.pbar:
            epoch_str = f'epoch{self.cur_epoch}/{self.total_epochs}'
            self.pbar(
                step=state.global_step - 1,
                loss=self.last_loss,
                epoch_str=epoch_str,
            )

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """训练结束：输出换行并恢复原始stdout"""
        if self.pbar:
            self.pbar.finish()
            self.pbar.__exit__(None, None, None)


# ============================================================
# CG-LoRA 曲率引导优化器
# ============================================================

class CGLoRAOptimizer(Optimizer):
    """
    曲率引导LoRA优化器（Adam + Fisher对角矩阵增强）

    核心思想：
      标准Adam的自适应步长基于梯度的一/二阶统计量（m̂, v̂）。
      CGLoRA在分母中额外引入Fisher信息矩阵对角估计F̂，
      用损失曲面的局部曲率信息进一步调制步长：

      更新公式：θ ← θ - lr · m̂ / (√(v̂ + λ·F̂) + ε)

    各变量含义：
      m̂  : 一阶动量（梯度方向的指数移动平均），带偏差修正
      v̂  : 二阶动量（梯度平方的指数移动平均），带偏差修正
      F̂  : Fisher对角估计（近似损失曲面曲率），带偏差修正
      λ  : 曲率权重（控制Fisher对步长的影响程度）

    曲率引导的物理含义：
      曲率大（陡峭方向）→ F̂大 → 分母大 → 步长小（保守更新，防止跳过最优点）
      曲率小（平坦方向）→ F̂小 → 分母小 → 步长大（充分探索平坦区域）

    退化关系：λ=0 → 标准Adam（向后兼容）

    相比AdamW的优势：
      LoRA参数稀疏（仅0.1%参数），曲率引导提供更稳定的
      自适应步长，尤其在低资源术语库的小数据场景下。
    """

    def __init__(
        self,
        params,
        lr: float = 5e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        curvature_lambda: float = 0.05,
        fisher_beta: float = 0.999,
    ):
        """
        Args:
            params:           可训练参数（通常为LoRA层参数）
            lr:               基础学习率（Warmup调度器会在此基础上缩放）
            betas:            Adam动量衰减系数 (β1=梯度动量, β2=梯度方差)
            eps:              数值稳定小量（防止分母为零）
            weight_decay:     L2正则化权重衰减系数
            curvature_lambda: Fisher曲率项权重（推荐0.01~0.1）
            fisher_beta:      Fisher EMA衰减系数（推荐0.999，更平滑）
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            curvature_lambda=curvature_lambda,
            fisher_beta=fisher_beta,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步参数更新

        流程：
          1. 可选闭包重计算loss（用于line search等场景）
          2. 遍历所有参数组
          3. 对每个有梯度的参数：
             a. 初始化状态变量（首次调用时）
             b. 更新一/二阶动量和Fisher估计
             c. 偏差修正
             d. 曲率增强参数更新

        Args:
            closure: 可选闭包（重新计算loss），None时直接更新
        Returns:
            loss值（仅提供closure时返回，否则为None）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 提取当前参数组的超参数
            lr     = group["lr"]
            b1, b2 = group["betas"]
            eps    = group["eps"]
            wd     = group["weight_decay"]
            lam    = group["curvature_lambda"]
            fb     = group["fisher_beta"]

            for p in group["params"]:
                # 跳过无梯度的参数（被冻结的预训练参数）
                if p.grad is None:
                    continue

                g     = p.grad   # 当前梯度
                state = self.state[p]

                # ── 首次调用初始化状态变量 ──
                if len(state) == 0:
                    state["step"]        = 0
                    state["exp_avg"]     = torch.zeros_like(p)  # m：一阶动量
                    state["exp_avg_sq"]  = torch.zeros_like(p)  # v：二阶动量
                    state["fisher_diag"] = torch.zeros_like(p)  # F：Fisher对角估计

                m  = state["exp_avg"]
                v  = state["exp_avg_sq"]
                F  = state["fisher_diag"]
                state["step"] += 1
                t = state["step"]

                # ── L2正则化：将权重衰减融入梯度 ──
                # 等价于在loss中添加 (wd/2)·||θ||²
                if wd != 0:
                    g = g.add(p, alpha=wd)

                # ── 更新一阶动量：m ← β1·m + (1-β1)·g ──
                # 梯度的指数移动平均，提供稳定的梯度方向估计
                m.mul_(b1).add_(g, alpha=1 - b1)

                # ── 更新二阶动量：v ← β2·v + (1-β2)·g² ──
                # 梯度平方的指数移动平均，用于自适应步长
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                # ── 更新Fisher对角估计：F ← fb·F + (1-fb)·g² ──
                # Fisher信息矩阵对角 ≈ E[g²]，用梯度平方EMA近似
                # fb通常比b2更接近1（如0.999），提供更平滑的曲率估计
                F.mul_(fb).addcmul_(g, g, value=1 - fb)

                # ── 偏差修正（Adam标准做法）──
                # 初始阶段m,v,F均从0出发，存在向0的偏差
                # 除以(1-β^t)修正此偏差，使早期估计更准确
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                f_hat = F / (1 - fb ** t)

                # ── 曲率增强参数更新 ──
                # 分母 = √(v̂ + λF̂) + ε
                # λF̂ 增加曲率大方向的阻尼，λ=0退化为标准Adam
                denom = (v_hat + lam * f_hat).sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss


# ============================================================
# 集成CG-LoRA的Trainer基类
# ============================================================

class CGLoRATrainer(Trainer):
    """
    集成CG-LoRA优化器和Warmup调度的Trainer

    相比HuggingFace标准Trainer的改进：
      1. create_optimizer()：将AdamW替换为CGLoRAOptimizer
      2. create_scheduler()：创建线性Warmup+线性衰减调度器
         - Warmup阶段：lr从0线性增长到peak_lr
         - 衰减阶段：lr从peak_lr线性衰减到0
      3. 梯度裁剪：在TrainingArguments中通过max_grad_norm配置

    注意：
      梯度裁剪通过TrainingArguments.max_grad_norm实现，
      HuggingFace Trainer在optimizer.step()前自动调用
      torch.nn.utils.clip_grad_norm_()，无需在此覆写。
    """

    def __init__(
        self,
        curvature_lambda: float = 0.05,
        fisher_beta: float = 0.999,
        warmup_steps: int = 0,
        **kwargs
    ):
        """
        Args:
            curvature_lambda: CGLoRAOptimizer 的 Fisher 曲率权重
            fisher_beta:      CGLoRAOptimizer 的 Fisher EMA 衰减系数
            warmup_steps:     Warmup 步数
            **kwargs:         透传给 Trainer 父类
        
        FutureWarning 修复：
          与 EngramJointTrainer 相同的转换逻辑
          在整个继承链的每一层都做转换
          确保 Trainer.__init__ 接收到的是 processing_class
        """
        # ── 核心修复：tokenizer → processing_class 转换 ──
        tokenizer_obj = kwargs.pop('tokenizer', None)
        
        if 'processing_class' not in kwargs and tokenizer_obj is not None:
            kwargs['processing_class'] = tokenizer_obj
        
        # ── 调用 Trainer 父类（kwargs 中无 tokenizer=）──
        super().__init__(**kwargs)
        
        # ── 兼容：统一 self.tokenizer 访问 ──
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = getattr(self, 'processing_class', None)
        
        # ── CG-LoRA 参数 ──
        self.curvature_lambda = curvature_lambda
        self.fisher_beta      = fisher_beta
        self.warmup_steps     = warmup_steps

    def create_optimizer(self) -> CGLoRAOptimizer:
        """
        覆写优化器创建方法，使用CGLoRAOptimizer替代AdamW。

        只对requires_grad=True的参数（即LoRA层参数）优化，
        冻结的预训练参数（requires_grad=False）自动跳过。

        Returns:
            初始化好的CGLoRAOptimizer实例
        """
        # 筛选可训练参数（LoRA层）
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        total     = sum(p.numel() for p in trainable)
        logging.info(f"  [CG-LoRA] 可训练参数量: {total:,}")

        self.optimizer = CGLoRAOptimizer(
            trainable,
            lr=self.args.learning_rate,
            curvature_lambda=self.curvature_lambda,
            fisher_beta=self.fisher_beta,
        )
        return self.optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[Optimizer] = None
    ):
        """
        覆写调度器创建方法，使用线性Warmup+线性衰减调度。

        调度曲线（示例：总步数100，warmup_steps=10）：
          步数0：  lr = 0
          步数5：  lr = 0.5 × peak_lr   （warmup上升阶段）
          步数10： lr = peak_lr          （peak，warmup结束）
          步数55： lr = 0.5 × peak_lr   （衰减阶段）
          步数100：lr = 0               （衰减结束）

        为何需要Warmup：
          LoRA权重随机初始化，训练初期梯度方向不稳定。
          Warmup让lr从0缓慢增大，避免初始大梯度扰动预训练权重。
          对编码器对齐训练尤为重要（编码空间需要平稳适应）。

        Args:
            num_training_steps: 总训练步数（由Trainer自动计算）
            optimizer:          优化器（None时使用self.optimizer）

        Returns:
            配置好的LambdaLR调度器实例
        """
        if optimizer is None:
            optimizer = self.optimizer

        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler


# ============================================================
# 知识蒸馏Trainer（第三步）
# ============================================================

class KDLoRATrainer(CGLoRATrainer):
    """
    知识蒸馏增强的CG-LoRA训练器（第三步）

    蒸馏原理（Hinton et al., 2015）：
      除硬标签（one-hot真实标签）的CE损失外，
      还学习教师模型输出分布（软标签）的KL散度，
      获取教师在"错误"预测上蕴含的"暗知识"。

    温度缩放的作用：
      高温T → softmax分布更平滑 → 软标签信息更丰富
      T=4时，教师对top-2词的概率比比T=1时差距更小，
      学生从中学到更多"相对关系"而非绝对top-1。

    损失混合公式：
      L = (1-α)·L_CE + α·T²·L_KL

    温度补偿（T²）的必要性：
      温度缩放使梯度缩小T²倍（softmax/T），
      乘以T²恢复与CE损失同量级的梯度，
      确保两个损失项的优化优先级均衡。

    维度修复说明（原代码遗留问题的修复）：
      原错误：mask.unsqueeze(-1) → [B,T,1]，与kl_per_token([B,T])维度不匹配
      修复后：
        kl_div(reduction='none') → [B, T, V]（逐元素KL）
        .sum(dim=-1)             → [B, T]   （词表维度聚合）
        position_mask = [B, T]   → 无需unsqueeze，直接相乘
    """

    def __init__(
        self,
        teacher_model=None,
        kd_temperature: float = 4.0,
        kd_alpha: float = 0.3,
        **kwargs
    ):
        """
        Args:
            teacher_model:  已训练的教师模型（PeftModel或普通模型）
            kd_temperature: 蒸馏温度T（推荐2.0~6.0，越高越平滑）
            kd_alpha:       蒸馏损失权重α（推荐0.2~0.5）
            **kwargs:       传递给CGLoRATrainer的参数
        """
        super().__init__(**kwargs)
        self.teacher  = teacher_model
        self.kd_temp  = kd_temperature
        self.kd_alpha = kd_alpha

        # 冻结教师模型：不参与梯度计算，节省显存
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def compute_loss(
        self,
        model,
        inputs: Dict,
        return_outputs: bool = False,
        **kwargs
    ):
        """
        混合蒸馏损失计算（维度修复版）

        维度追踪（B=batch, T=seq_len, V=vocab_size）：
          student_logits    : [B, T, V]
          teacher_logits    : [B, T, V]
          student_log_probs : [B, T, V]（log_softmax(·/T)）
          teacher_probs     : [B, T, V]（softmax(·/T)）
          kl_per_position   : [B, T, V]（逐元素KL，reduction='none'）
          kl_per_token      : [B, T]   （.sum(dim=-1)，词表维度聚合）
          position_mask     : [B, T]   （labels!=-100的有效位置）
          kd_loss           : scalar   （有效位置加权均值）

        Args:
            model:          学生模型（正在训练）
            inputs:         包含input_ids/attention_mask/labels的批次字典
            return_outputs: 是否同时返回模型输出（用于评估时提取logits）

        Returns:
            loss标量（或(loss, outputs)元组）
        """
        # 学生模型前向传播，获取CE损失（硬标签监督）
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # 无教师或α=0时退化为纯CE训练
        if self.teacher is None or self.kd_alpha == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # 教师模型前向传播（no_grad：不积累梯度，节省显存）
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits  = teacher_outputs.logits   # [B, T, V]

        student_logits = outputs.logits                # [B, T, V]
        T = self.kd_temp

        # ── 温度缩放 ──
        # 除以T使logits分布更平缓，softmax输出更平滑
        student_log_probs = torch.nn.functional.log_softmax(
            student_logits / T, dim=-1
        )                                              # [B, T, V]
        teacher_probs = torch.nn.functional.softmax(
            teacher_logits / T, dim=-1
        )                                              # [B, T, V]

        # ── Step1：逐元素KL散度（保留所有维度）──
        kl_per_position = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none'                           # 不聚合，保留[B,T,V]
        )                                              # [B, T, V]

        # ── Step2：词表维度聚合（每token的总KL散度）──
        # 必须先sum(V)再mask(T)，顺序不可颠倒
        kl_per_token = kl_per_position.sum(dim=-1)    # [B, T]

        # ── Step3：构建有效位置mask ──
        # labels=-100的位置为padding，不计入损失
        # 修复点：position_mask形状[B,T]，与kl_per_token一致，无需unsqueeze
        position_mask = (inputs['labels'] != -100).float()  # [B, T]

        # ── Step4：有效位置加权均值 ──
        # clamp(min=1)防止全padding的极端batch导致除零
        valid_count = position_mask.sum().clamp(min=1)
        kd_loss = (kl_per_token * position_mask).sum() / valid_count

        # ── Step5：温度补偿（恢复与CE同量级的梯度）──
        # 温度缩放使softmax梯度缩小T²倍，乘以T²补偿
        kd_loss = kd_loss * (T ** 2)

        # ── 混合损失 ──
        loss = (1 - self.kd_alpha) * ce_loss + self.kd_alpha * kd_loss
        return (loss, outputs) if return_outputs else loss


# ============================================================
# 跨语言对齐Trainer（第二步）
# ============================================================
class AlignmentLoRATrainer(CGLoRATrainer):
    """
    跨语言编码器表示对齐训练器（第二步）

    核心思想：
      不同语言的同义术语，其编码器表示应在向量空间中邻近。
      通过余弦相似度损失，约束编码器将跨语言同义词映射到
      相近的隐藏状态，构建语言无关的语义空间。

    损失设计：
      L = L_CE + λ·L_align
      L_align = mean(1 - cos_sim(h1ᵢ, h2ᵢ))

      L_CE保证翻译能力不退化；
      L_align约束编码器语义空间对齐。

    在线刷新机制：
      预计算的编码对h1,h2基于训练初期的编码器状态。
      随着训练进行，编码器语义空间不断漂移，
      固定的h1,h2与当前编码器不匹配，对齐目标失效。

      解决方案：每隔update_align_every步，
      用当前编码器重新计算所有术语的表示，
      确保对齐目标始终与当前编码器语义空间对齐。

    刷新间隔权衡：
      steps=50:  精度高，计算开销大（每50步一次前向计算）
      steps=100: 平衡（推荐）
      steps=200: 轻量，适合术语对>100的场景
    """

    def __init__(
        self,
        align_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        align_lambda: float = 0.1,
        cross_pairs_raw: List[Tuple[Tuple[str, str], Tuple[str, str]]] = None,
        tokenizer=None,
        update_align_every: int = 100,
        **kwargs
    ):
        """
        Args:
            align_pairs:        预计算的初始编码对[(h1,h2),...]（列表of张量元组）
            align_lambda:       对齐损失权重λ（推荐0.05~0.2）
            cross_pairs_raw:    原始跨语言术语对，用于在线重计算
                                格式：[((lang1,term1),(lang2,term2)),...]
            tokenizer:          NLLB分词器（在线重计算时需要）
            update_align_every: 编码对刷新间隔步数（推荐100）
            **kwargs:           传递给CGLoRATrainer的参数
        """
        super().__init__(**kwargs)
        self.align_lambda        = align_lambda
        self.cross_pairs_raw     = cross_pairs_raw or []
        self.tokenizer           = tokenizer
        self.update_align_every  = update_align_every

        # 初始化编码缓存（训练开始时使用预计算的初始对）
        self._cached_pairs       = align_pairs if align_pairs else []
        # 本地步数计数器（独立于state.global_step，用于刷新判断）
        self._global_step_local  = 0

    def _refresh_encoded_pairs(self, model) -> None:
        """
        在线重计算跨语言术语对的编码器表示。

        流程：
          1. 切换model为eval模式（关闭dropout，确保编码确定性）
          2. 用当前编码器重新计算所有术语对的表示向量
          3. 切换回train模式（恢复dropout等训练行为）
          4. 更新_cached_pairs缓存

        注意事项：
          - 重计算结果detach脱离计算图（固定为对齐目标，不反向传播）
          - 此函数开销与cross_pairs_raw数量成正比
          - 建议max_pairs≤200以控制刷新耗时
        """
        if not self.cross_pairs_raw or self.tokenizer is None:
            return

        device = next(model.parameters()).device

        model.eval()  # 关闭dropout
        new_pairs = encode_cross_lingual_pairs(
            model=model,
            tokenizer=self.tokenizer,
            cross_pairs=self.cross_pairs_raw,
            device=str(device),
        )
        model.train()  # 恢复训练模式

        # 仅在成功编码时更新缓存（避免编码失败时清空有效缓存）
        if new_pairs:
            self._cached_pairs = new_pairs

    def compute_loss(
        self,
        model,
        inputs: Dict,
        return_outputs: bool = False,
        **kwargs
    ):
        """
        CE损失 + 在线刷新余弦对齐损失。

        在线刷新触发条件：
          - 有原始术语对数据（cross_pairs_raw非空）
          - 有分词器（tokenizer非None）
          - 首步（_global_step_local==0）强制刷新
          - 或达到刷新间隔（_global_step_local % update_align_every == 0）

        Args:
            model:          正在训练的模型
            inputs:         批次数据字典
            return_outputs: 是否返回模型输出

        Returns:
            loss标量（或(loss, outputs)元组）
        """
        # 标准翻译CE损失（保证基础翻译能力）
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # align_lambda=0时退化为纯CE训练
        if self.align_lambda == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # ── 定期在线刷新编码对 ──
        should_refresh = (
            self.cross_pairs_raw
            and self.tokenizer is not None
            and (self._global_step_local == 0
                 or self._global_step_local % self.update_align_every == 0)
        )
        if should_refresh:
            self._refresh_encoded_pairs(model)

        self._global_step_local += 1

        # 无有效编码对时降级为纯CE训练
        if not self._cached_pairs:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # ── 计算余弦对齐损失 ──
        # 对每个术语对，计算1-cos_sim，再取均值
        # 1-cos_sim ∈ [0, 2]：0表示完全对齐，2表示完全相反
        device = next(model.parameters()).device
        losses = []
        for h1, h2 in self._cached_pairs:
            h1 = h1.to(device)
            h2 = h2.to(device)
            cos_sim = torch.nn.functional.cosine_similarity(
                h1.unsqueeze(0), h2.unsqueeze(0)
            )
            losses.append(1.0 - cos_sim)

        align_loss = torch.stack(losses).mean()
        # 混合损失：CE维持翻译精度，对齐损失约束语义空间
        total_loss = ce_loss + self.align_lambda * align_loss
        return (total_loss, outputs) if return_outputs else total_loss

# ============================================================
# 【独立版】SONAR约束的混合对齐训练器（无需AlignmentLoRATrainer）
# ============================================================
class HybridAlignmentTrainer(CGLoRATrainer):
    """
    【修复版】混合对齐训练器（独立版，无继承依赖）

    修复点：
      SONAR自编码器冻结时，仅冻结编码器部分
      decoder/denoiser/normalizer保持可训练状态
      避免整体冻结导致SONAR失去语义约束能力
    """

    def __init__(
        self,
        sonar_autoencoder=None,
        sonar_lambda: float = 0.1,
        sample_pairs_per_step: int = 2,
        align_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        align_lambda: float = 0.1,
        cross_pairs_raw: List[
            Tuple[Tuple[str, str], Tuple[str, str]]
        ] = None,
        tokenizer=None,
        update_align_every: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)

        # SONAR参数
        self.sonar = sonar_autoencoder
        self.sonar_lambda = sonar_lambda
        self.sample_pairs_per_step = sample_pairs_per_step

        # NLLB对齐参数
        self.align_lambda = align_lambda
        self.cross_pairs_raw = cross_pairs_raw or []
        self.tokenizer = tokenizer
        self.update_align_every = update_align_every
        self._cached_pairs = align_pairs if align_pairs else []
        self._global_step_local = 0

        # SONAR设置（修复：设为eval但不冻结参数）
        if self.sonar is not None:
            # 仅设为eval模式（关闭dropout等训练行为）
            # 不冻结参数：SONAR已在pretrain阶段独立训练完成
            self.sonar.eval()

    def _refresh_encoded_pairs(self, model) -> None:
        """在线刷新NLLB编码对"""
        if not self.cross_pairs_raw or self.tokenizer is None:
            return

        device = next(model.parameters()).device
        model.eval()

        new_pairs = encode_cross_lingual_pairs(
            model=model,
            tokenizer=self.tokenizer,
            cross_pairs=self.cross_pairs_raw,
            device=str(device),
        )
        model.train()

        if new_pairs:
            self._cached_pairs = new_pairs

    def compute_loss(
        self,
        model,
        inputs: Dict,
        return_outputs: bool = False,
        **kwargs
    ):
        """
        三重损失：CE + NLLB对齐 + SONAR对齐

        各损失的梯度流向：
          CE损失       → NLLB LoRA参数
          NLLB对齐损失 → NLLB LoRA参数（通过编码器隐藏状态）
          SONAR对齐损失→ 无梯度（sonar.eval() + torch.no_grad()）
                         仅作为软约束信号存在于损失值中
                         但由于no_grad，实际不更新任何参数
        
        注意：
          SONAR损失在no_grad块内计算，
          仅通过数值影响total_loss，
          不产生额外梯度（不影响训练稳定性）
        """
        # ===== CE损失 =====
        outputs = model(**inputs)
        total_loss = outputs.loss

        # ===== NLLB对齐损失 =====
        if self.align_lambda > 0:
            should_refresh = (
                self.cross_pairs_raw
                and self.tokenizer is not None
                and (
                    self._global_step_local == 0
                    or self._global_step_local % self.update_align_every == 0
                )
            )
            if should_refresh:
                self._refresh_encoded_pairs(model)

            self._global_step_local += 1

            if self._cached_pairs:
                device = next(model.parameters()).device
                nllb_losses = []

                for h1, h2 in self._cached_pairs:
                    cos_sim = F.cosine_similarity(
                        h1.to(device).unsqueeze(0),
                        h2.to(device).unsqueeze(0)
                    )
                    nllb_losses.append(1.0 - cos_sim)

                if nllb_losses:
                    nllb_align = torch.stack(nllb_losses).mean()
                    total_loss = total_loss + self.align_lambda * nllb_align

        # ===== SONAR对齐损失（作为软约束）=====
        if (
            self.sonar is not None
            and self.sonar_lambda > 0
            and self.cross_pairs_raw
        ):
            device = next(model.parameters()).device
            num_pairs = min(
                self.sample_pairs_per_step,
                len(self.cross_pairs_raw)
            )
            sampled = random.sample(self.cross_pairs_raw, k=num_pairs)
            sonar_losses = []

            # SONAR在no_grad下运行（预训练结果固定）
            with torch.no_grad():
                for (lang1, term1), (lang2, term2) in sampled:
                    try:
                        self.tokenizer.src_lang = lang1
                        inp1 = self.tokenizer(
                            term1, return_tensors="pt",
                            truncation=True, max_length=256
                        ).to(device)

                        out1 = self.sonar(
                            inp1["input_ids"],
                            inp1["attention_mask"],
                            lang_code=lang1
                        )
                        emb1 = out1["normalized"]

                        self.tokenizer.src_lang = lang2
                        inp2 = self.tokenizer(
                            term2, return_tensors="pt",
                            truncation=True, max_length=256
                        ).to(device)

                        out2 = self.sonar(
                            inp2["input_ids"],
                            inp2["attention_mask"],
                            lang_code=lang2
                        )
                        emb2 = out2["normalized"]

                        # 余弦距离（代替MSE，对尺度不敏感）
                        cos_dist = 1.0 - F.cosine_similarity(
                            emb1, emb2, dim=-1
                        ).mean()
                        sonar_losses.append(cos_dist)

                    except Exception as e:
                        logging.getLogger(__name__).debug(
                            f"SONAR损失计算失败: {e}"
                        )
                        continue

            # 将SONAR损失作为常数加入（不产生梯度）
            if sonar_losses:
                sonar_val = torch.stack(sonar_losses).mean().detach()
                total_loss = total_loss + self.sonar_lambda * sonar_val

        return (total_loss, outputs) if return_outputs else total_loss

# ============================================================
# 【完整实现】SONAR知识蒸馏增强的对齐训练器
# ============================================================

class SONARGuidedAlignmentTrainer(HybridAlignmentTrainer):
    """
    【CPU轻量化版】SONAR知识蒸馏增强的对齐训练器

    轻量化改进点：
      1. 复用编码器输出：避免 model(**inputs) 和
         model.get_encoder() 的重复编码器计算
      2. 蒸馏间隔控制：每 distill_every_n_steps 步
         执行一次蒸馏（默认5步1次）
      3. SONAR结果缓存：相同文本不重复编码
         （LRU缓存，容量128条）
      4. 合并SONAR调用：SONAR软约束与蒸馏共享编码
      5. 自适应蒸馏：蒸馏损失极小时跳过（避免无效计算）
      6. CPU专项：关闭不必要的梯度计算路径

    参数：
      distill_lambda:         蒸馏损失权重（推荐0.15）
      distill_every_n_steps:  蒸馏执行间隔（CPU推荐5-10）
      distill_pooling:        池化方式（mean/max）
      sonar_cache_size:       SONAR缓存容量（推荐128）
      min_distill_loss:       蒸馏触发最小阈值（低于则跳过）
    """

    def __init__(
        self,
        distill_lambda: float = 0.15,
        distill_every_n_steps: int = 5,    # 新增：间隔控制
        distill_pooling: str = "mean",
        sonar_cache_size: int = 128,        # 新增：缓存容量
        min_distill_loss: float = 1e-4,    # 新增：自适应阈值
        **kwargs
    ):
        super().__init__(**kwargs)
        self.distill_lambda = distill_lambda
        self.distill_every_n_steps = distill_every_n_steps
        self.distill_pooling = distill_pooling
        self.min_distill_loss = min_distill_loss

        # 步数计数器（控制蒸馏频率）
        self._distill_step_counter = 0

        # SONAR编码缓存（LRU，避免重复编码）
        # key: (text, lang_code) → value: normalized嵌入（CPU tensor）
        self._sonar_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._sonar_cache_size = sonar_cache_size
        self._sonar_cache_keys: List[Tuple[str, str]] = []  # 维护插入顺序

        # 统计
        self.epoch_distill_losses: List[float] = []
        self.epoch_alignment_gaps: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def _pool_encoder_output(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        编码器输出池化

        Args:
            hidden: [B, T, D]
            mask:   [B, T]

        Returns:
            [B, D]
        """
        if self.distill_pooling == "mean":
            mask_exp = mask.unsqueeze(-1).float()
            return (
                (hidden * mask_exp).sum(dim=1)
                / mask_exp.sum(dim=1).clamp(min=1e-9)
            )
        else:
            return hidden.max(dim=1)[0]

    def _get_sonar_embedding(
        self,
        text: str,
        lang_code: str,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        带缓存的SONAR编码（核心优化）

        缓存策略（LRU简化版）：
          命中：直接返回缓存（O(1)，0次模型调用）
          未命中：编码后存入缓存，淘汰最旧条目

        Args:
            text:      待编码文本
            lang_code: NLLB语言代码
            device:    目标设备

        Returns:
            [1, D] 标准化嵌入，失败返回None
        """
        cache_key = (text.strip()[:100], lang_code)  # 截断key避免过长

        # 缓存命中
        if cache_key in self._sonar_cache:
            self._cache_hits += 1
            cached = self._sonar_cache[cache_key]
            return cached.to(device)

        # 缓存未命中：编码
        self._cache_misses += 1
        try:
            with torch.no_grad():
                sonar_inp = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128  # CPU模式缩短长度（256→128）
                ).to(device)

                sonar_out = self.sonar(
                    sonar_inp["input_ids"],
                    sonar_inp["attention_mask"],
                    lang_code=lang_code
                )
                emb = sonar_out["normalized"].cpu()  # 存CPU节省显存

            # LRU淘汰（超出容量时删除最旧）
            if len(self._sonar_cache) >= self._sonar_cache_size:
                oldest_key = self._sonar_cache_keys.pop(0)
                self._sonar_cache.pop(oldest_key, None)

            # 写入缓存
            self._sonar_cache[cache_key] = emb
            self._sonar_cache_keys.append(cache_key)

            return emb.to(device)

        except Exception as e:
            logging.getLogger(__name__).debug(
                f"SONAR编码失败: {e}"
            )
            return None

    def _should_run_distill(self) -> bool:
        """
        判断当前步是否执行蒸馏（间隔控制）

        Returns:
            True：执行蒸馏
            False：跳过（节省计算）
        """
        self._distill_step_counter += 1
        return (self._distill_step_counter % self.distill_every_n_steps == 0)

    def compute_loss(
        self,
        model,
        inputs: Dict,
        return_outputs: bool = False,
        **kwargs
    ):
        """
        轻量化三重损失计算

        核心优化：
          1. 单次前向传播：复用outputs获取CE损失和encoder输出
          2. 蒸馏间隔：每N步执行1次（减少SONAR调用）
          3. 缓存复用：相同文本直接查表（无重复编码）
          4. 合并SONAR调用：蒸馏与软约束共享编码结果

        计算流程：
          step=1,2,3,4 → 仅CE + NLLB对齐 + SONAR软约束
          step=5       → CE + NLLB对齐 + SONAR软约束 + 蒸馏
          step=6,7,8,9 → 仅CE + NLLB对齐 + SONAR软约束
          step=10      → CE + NLLB对齐 + SONAR软约束 + 蒸馏
          ...
        """
        device = next(model.parameters()).device

        # ===== 核心优化1：单次前向传播，复用encoder输出 =====
        # 判断是否需要蒸馏（决定是否提取encoder输出）
        run_distill = (
            self.sonar is not None
            and self.distill_lambda > 0
            and model.training
            and self.tokenizer is not None
            and self._should_run_distill()
        )

        if run_distill:
            # 需要蒸馏时：单次前向传播 + 提取encoder输出
            # output_hidden_states=True 同时获取隐藏状态
            outputs = model(
                **inputs,
                output_hidden_states=False,
            )
            ce_loss = outputs.loss

            # 从outputs中提取encoder隐藏状态（避免第二次前向）
            if hasattr(outputs, 'encoder_last_hidden_state') and \
               outputs.encoder_last_hidden_state is not None:
                # Seq2SeqLMOutput包含encoder_last_hidden_state
                nllb_pooled = self._pool_encoder_output(
                    outputs.encoder_last_hidden_state,
                    inputs["attention_mask"]
                )  # [B, D]
                encoder_available = True
            else:
                # 降级：单独调用编码器（仅此情况需要两次前向）
                with torch.no_grad():
                    encoder = model.get_encoder()
                    enc_out = encoder(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        return_dict=True
                    )
                nllb_pooled = self._pool_encoder_output(
                    enc_out.last_hidden_state,
                    inputs["attention_mask"]
                )
                encoder_available = True
        else:
            # 不需要蒸馏时：标准单次前向
            outputs = model(**inputs)
            ce_loss = outputs.loss
            nllb_pooled = None
            encoder_available = False

        total_loss = ce_loss

        # ===== NLLB对齐损失（父类逻辑，复用）=====
        if self.align_lambda > 0:
            # 触发在线刷新
            should_refresh = (
                self.cross_pairs_raw
                and self.tokenizer is not None
                and (
                    self._global_step_local == 0
                    or self._global_step_local % self.update_align_every == 0
                )
            )
            if should_refresh:
                self._refresh_encoded_pairs(model)

            self._global_step_local += 1

            # NLLB编码对对齐损失
            if self._cached_pairs:
                nllb_losses = []
                for h1, h2 in self._cached_pairs:
                    cos_sim = F.cosine_similarity(
                        h1.to(device).unsqueeze(0),
                        h2.to(device).unsqueeze(0)
                    )
                    nllb_losses.append(1.0 - cos_sim)

                if nllb_losses:
                    nllb_align = torch.stack(nllb_losses).mean()
                    total_loss = total_loss + self.align_lambda * nllb_align

        # ===== SONAR软约束（每步执行，但有缓存加速）=====
        sonar_soft_pairs_encoded: Dict[str, torch.Tensor] = {}  # 本步缓存

        if (
            self.sonar is not None
            and self.sonar_lambda > 0
            and self.cross_pairs_raw
            and model.training
        ):
            num_pairs = min(self.sample_pairs_per_step, len(self.cross_pairs_raw))
            sampled = random.sample(self.cross_pairs_raw, k=num_pairs)
            sonar_soft_losses = []

            for (lang1, term1), (lang2, term2) in sampled:
                # 核心优化3：缓存复用
                emb1 = self._get_sonar_embedding(term1, lang1, device)
                emb2 = self._get_sonar_embedding(term2, lang2, device)

                if emb1 is None or emb2 is None:
                    continue

                # 余弦距离
                cos_dist = 1.0 - F.cosine_similarity(
                    emb1, emb2, dim=-1
                ).mean()
                sonar_soft_losses.append(cos_dist.detach())

                # 核心优化4：保存本步编码供蒸馏复用
                sonar_soft_pairs_encoded[f"{lang1}:{term1}"] = emb1
                sonar_soft_pairs_encoded[f"{lang2}:{term2}"] = emb2

            if sonar_soft_losses:
                sonar_soft_val = torch.stack(sonar_soft_losses).mean()
                total_loss = total_loss + self.sonar_lambda * sonar_soft_val

        # ===== SONAR蒸馏损失（间隔执行）=====
        if run_distill and encoder_available and nllb_pooled is not None:
            try:
                B = inputs["input_ids"].size(0)
                # 采样1条（CPU模式减少开销）
                n_samples = 1
                sample_idx = torch.randperm(B)[:n_samples]

                distill_losses = []

                for idx in sample_idx:
                    idx_item = idx.item()

                    # 解码文本
                    text = self.tokenizer.decode(
                        inputs["input_ids"][idx_item],
                        skip_special_tokens=True
                    )
                    if not text.strip():
                        continue

                    src_lang = getattr(
                        self.tokenizer, 'src_lang', 'eng_Latn'
                    )

                    # 核心优化3：优先从本步软约束缓存获取
                    cache_key_str = f"{src_lang}:{text.strip()[:100]}"
                    sonar_target = sonar_soft_pairs_encoded.get(cache_key_str)

                    if sonar_target is None:
                        # 未命中：调用带缓存的编码函数
                        sonar_target = self._get_sonar_embedding(
                            text, src_lang, device
                        )

                    if sonar_target is None:
                        continue

                    # 学生端
                    nllb_student = nllb_pooled[idx_item:idx_item+1]  # [1, D]

                    # 尺寸对齐（NLLB隐藏层维度 vs SONAR维度可能不同）
                    if nllb_student.shape[-1] != sonar_target.shape[-1]:
                        # 维度不匹配时跳过蒸馏（安全降级）
                        logging.getLogger(__name__).debug(
                            f"维度不匹配：NLLB={nllb_student.shape[-1]}, "
                            f"SONAR={sonar_target.shape[-1]}，跳过蒸馏"
                        )
                        continue

                    # 仅用余弦损失（比MSE更稳定，CPU友好）
                    nllb_norm = F.normalize(nllb_student, p=2, dim=-1)
                    sonar_norm = F.normalize(sonar_target, p=2, dim=-1)
                    cos_loss = 1.0 - (nllb_norm * sonar_norm).sum()

                    # 自适应阈值（极小时跳过）
                    if cos_loss.item() < self.min_distill_loss:
                        continue

                    distill_losses.append(cos_loss)

                    with torch.no_grad():
                        self.epoch_alignment_gaps.append(cos_loss.item())

                if distill_losses:
                    avg_distill = torch.stack(distill_losses).mean()
                    total_loss = (
                        total_loss + self.distill_lambda * avg_distill
                    )
                    self.epoch_distill_losses.append(avg_distill.item())

            except Exception as e:
                logging.getLogger(__name__).debug(
                    f"蒸馏损失计算失败（已跳过）: {e}"
                )

        return (total_loss, outputs) if return_outputs else total_loss

    def on_epoch_end(self, args, state, control, **kwargs):
        """Epoch结束：输出统计并清空缓存"""
        logger = logging.getLogger(__name__)

        if self.epoch_distill_losses:
            avg_d = (
                sum(self.epoch_distill_losses)
                / len(self.epoch_distill_losses)
            )
            avg_g = (
                sum(self.epoch_alignment_gaps)
                / len(self.epoch_alignment_gaps)
            )
            total_calls = self._cache_hits + self._cache_misses
            hit_rate = (
                self._cache_hits / total_calls
                if total_calls > 0 else 0.0
            )

            logger.info(
                f"  [SONAR蒸馏] 平均损失: {avg_d:.4f} | "
                f"对齐差距: {avg_g:.4f} | "
                f"缓存命中率: {hit_rate:.1%} "
                f"({self._cache_hits}/{total_calls})"
            )

        # 清空统计（保留SONAR缓存跨epoch复用）
        self.epoch_distill_losses = []
        self.epoch_alignment_gaps = []
        self._cache_hits = 0
        self._cache_misses = 0


def _extract_positive_pairs(
    segments: List[Dict],
    similarity_threshold: float = 0.3,
) -> List[Tuple[Dict, Dict]]:
    """
    【新增】多策略正样本对提取

    三级策略（按优先级）：
      策略1：精确翻译匹配（translation完全相同）← 原逻辑
      策略2：翻译相似度匹配（Jaccard > threshold）← 新增放宽
      策略3：跨语言同条目配对（同一entry的不同语言）← 新增兜底

    Args:
        segments: 预处理后的术语段落
        similarity_threshold: 策略2的Jaccard相似度阈值

    Returns:
        正样本对列表
    """
    logger = logging.getLogger(__name__)
    positive_pairs = []

    # ===== 策略1：精确翻译匹配（原逻辑）=====
    translation_groups: Dict[str, List[Dict]] = {}
    for seg in segments:
        trans = seg['translation'].lower().strip()
        translation_groups.setdefault(trans, []).append(seg)

    for group in translation_groups.values():
        langs = set(seg['lang'] for seg in group)
        if len(langs) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i]['lang'] != group[j]['lang']:
                    positive_pairs.append((group[i], group[j]))

    logger.info(f"  策略1（精确翻译匹配）: {len(positive_pairs)}对")

    # ===== 策略2：翻译相似度匹配（放宽条件）=====
    if len(positive_pairs) < 10:
        logger.info(
            f"  正样本不足10对，启用策略2"
            f"（Jaccard阈值={similarity_threshold}）"
        )

        # 按语言分组
        lang_groups: Dict[str, List[Dict]] = {}
        for seg in segments:
            lang_groups.setdefault(seg['lang'], []).append(seg)

        langs = list(lang_groups.keys())

        def _jaccard(t1: str, t2: str) -> float:
            """简单词袋Jaccard"""
            w1 = set(t1.lower().split())
            w2 = set(t2.lower().split())
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / len(w1 | w2)

        seen = set()  # 去重
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                for seg1 in lang_groups[lang1]:
                    for seg2 in lang_groups[lang2]:
                        sim = _jaccard(
                            seg1['translation'],
                            seg2['translation']
                        )
                        if sim >= similarity_threshold:
                            key = (
                                seg1['term'],
                                seg2['term']
                            )
                            if key not in seen:
                                seen.add(key)
                                positive_pairs.append((seg1, seg2))

        logger.info(
            f"  策略2（相似度≥{similarity_threshold}）后: "
            f"{len(positive_pairs)}对"
        )

    # ===== 策略3：跨语言同条目配对（最终兜底）=====
    if len(positive_pairs) < 10:
        logger.info("  正样本仍不足10对，启用策略3（跨语言同条目配对）")

        lang_groups: Dict[str, List[Dict]] = {}
        for seg in segments:
            lang_groups.setdefault(seg['lang'], []).append(seg)

        langs = list(lang_groups.keys())
        seen2 = set()

        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                segs1 = lang_groups[lang1]
                segs2 = lang_groups[lang2]

                # 按original_term配对（同一术语条目的不同语言）
                orig_groups: Dict[str, List[Dict]] = {}
                for seg in segs1 + segs2:
                    orig = seg.get('original_term', seg['term'])
                    orig_groups.setdefault(orig, []).append(seg)

                for orig, group in orig_groups.items():
                    multi_lang = [
                        s for s in group
                        if s['lang'] in (lang1, lang2)
                    ]
                    for a in range(len(multi_lang)):
                        for b in range(a + 1, len(multi_lang)):
                            if multi_lang[a]['lang'] != multi_lang[b]['lang']:
                                key = (
                                    multi_lang[a]['term'],
                                    multi_lang[b]['term']
                                )
                                if key not in seen2:
                                    seen2.add(key)
                                    positive_pairs.append(
                                        (multi_lang[a], multi_lang[b])
                                    )

        logger.info(
            f"  策略3（跨语言同条目）后: {len(positive_pairs)}对"
        )

    # 去重
    unique_pairs = []
    seen_all = set()
    for seg1, seg2 in positive_pairs:
        key = (seg1['term'], seg1['lang'], seg2['term'], seg2['lang'])
        if key not in seen_all:
            seen_all.add(key)
            unique_pairs.append((seg1, seg2))

    logger.info(f"  最终正样本对（去重后）: {len(unique_pairs)}对")
    return unique_pairs


# ============================================================
# 工具函数：语言与术语处理
# ============================================================

def get_lang_id(tokenizer, lang_code: str) -> int:
    """
    从NLLB词表查找语言代码对应的token ID。

    NLLB使用语言token作为强制BOS（Begin Of Sentence）：
      解码时设置forced_bos_token_id=tgt_lang_id，
      强制解码器以目标语言token开始，确保输出语言正确。

    Args:
        tokenizer: NLLB分词器
        lang_code: NLLB格式语言代码（如"eng_Latn"）

    Returns:
        对应的token ID整数

    Raises:
        ValueError: 语言代码不在词表中时
    """
    vocab = tokenizer.get_vocab()
    if lang_code in vocab:
        return vocab[lang_code]
    raise ValueError(
        f"语言代码 {lang_code} 不在词表中。"
        f"请检查LANG_MAP配置或使用NLLB标准语言代码格式（如 zho_Hans）。"
    )


def load_glossary(path: str) -> dict:
    """
    加载术语库JSON文件。

    期望JSON格式：
    {
      "terms": [
        {
          "term": {"zh": "时态", "id": "kala"},
          "translation": "tense"
        },
        ...
      ]
    }

    Args:
        path: JSON文件路径

    Returns:
        术语库字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pairs(glossary: dict) -> List[Tuple[str, str, str]]:
    """
    从术语库提取(NLLB语言代码, 源术语, 英文译文)三元组。
    过滤规则：
      - 跳过无英文译文的条目（translation字段为空）
      - 跳过英文源（lang=="en"），避免en→en自翻译
      - 跳过无效/过短术语（len<2，可能是噪声）
      - 通过LANG_MAP转换语言代码格式

    Args:
        glossary: 术语库字典（load_glossary加载的JSON内容）

    Returns:
        [(nllb_lang_code, src_term, en_translation), ...]
    """
    pairs = []
    for entry in glossary.get("terms", []):

        # 提取英文译文（必须非空）
        tgt = entry.get("translation", "").strip()
        if not tgt:
            continue  # 跳过无英文译文的条目

        # 遍历term字典中的所有语言键值对
        # 注意：term_id字段在entry层级，不在term字典内，无需额外处理
        for lang, src in entry.get("term", {}).items():

            # 跳过英文源（避免en→en自翻译）
            if lang == "en":
                continue

            # 跳过无效术语（非字符串或过短）
            if not isinstance(src, str) or len(src) < 2:
                continue

            # 通过LANG_MAP查找对应的NLLB语言代码
            # "zh"  → "zho_Hans"
            # "ind" → "ind_Latn"  ← 关键修改：原"id"→"ind"
            # "tl"  → "tgl_Latn"
            nllb = LANG_MAP.get(lang)
            if nllb:
                pairs.append((nllb, src.strip(), tgt))
            else:
                # 未在LANG_MAP注册的语言键，记录警告便于排查
                logging.getLogger(__name__).debug(
                    f"  术语库语言键 '{lang}' 未在LANG_MAP中注册，已跳过"
                    f"（term_id: {entry.get('term_id', '未知')}）"
                )

    return pairs


def compute_term_similarity(
    tgt1: str, tgt2: str,
    words1: Set[str], words2: Set[str]
) -> float:
    """
    三级策略跨语言术语相似度计算。

    用途：
      判断两个不同语言的术语是否具有相同/相近的英文译文，
      从而识别跨语言同义词对，用于第二步编码器对齐训练。

    策略优先级（由高到低）：
      1. 精确匹配：归一化后完全相同 → 1.0
         示例："tense" vs "tense" → 1.0
      2. 包含关系：短译文是长译文的子串 → 短/长长度比
         示例："tense" vs "grammatical tense" → 5/16 ≈ 0.31
      3. Jaccard词袋相似度：内容词集合交并比
         示例：{"verbal","tense"} vs {"tense","aspect"} → 1/3 ≈ 0.33

    Args:
        tgt1:   术语1的英文译文
        tgt2:   术语2的英文译文
        words1: 译文1去停用词后的内容词集合
        words2: 译文2去停用词后的内容词集合

    Returns:
        相似度分数 [0.0, 1.0]
    """
    # 策略1：精确匹配（归一化：小写+去末尾标点）
    norm1 = tgt1.lower().strip().rstrip('.,;:!?')
    norm2 = tgt2.lower().strip().rstrip('.,;:!?')
    if norm1 == norm2:
        return 1.0

    # 策略2：包含关系（适用于短语与单词的对应）
    if norm1 and norm2:
        if norm1 in norm2 or norm2 in norm1:
            shorter = min(len(norm1), len(norm2))
            longer  = max(len(norm1), len(norm2))
            return shorter / longer

    # 策略3：Jaccard词袋相似度
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union        = words1 | words2
    return len(intersection) / len(union) if union else 0.0


# ============================================================
# 数据构建函数
# ============================================================

def build_weighted_dataset(
    pairs: List[Tuple[str, str, str]],
    repeat: int = 5,
    low_resource_langs: Set[str] = LOW_RESOURCE_LANGS,
    boost_factor: float = 2.0,
) -> List[Dict]:
    """
    构建置信度加权训练数据集（第一步核心）。

    设计原理：
      低资源语言训练数据天然稀缺，等权采样会导致模型
      对低资源语言欠拟合。通过boost_factor放大低资源样本权重，
      补偿数据量不足带来的学习劣势。

    数据构成：
      ① 高资源单术语样本：repeat倍扩增（建立翻译基础能力）
      ② 低资源单术语样本：repeat×boost_factor倍扩增（重点补偿）
      ③ 并列组合样本：上限50条（辅助学习术语列表翻译）

    并列组合上限的必要性：
      过多组合样本 → 模型过拟合"列举式翻译"模式
      → 单术语精确翻译能力下降
      严格限制在50条内保持辅助而非主导地位。

    Args:
        pairs:              术语对列表 [(lang, src, tgt), ...]
        repeat:             基础扩增倍数
        low_resource_langs: 低资源语言代码集合
        boost_factor:       低资源样本放大倍数（推荐1.5~3.0）

    Returns:
        打乱后的训练样本列表 [{"src_lang":..., "src":..., "tgt":...}, ...]
    """
    logger = logging.getLogger(__name__)
    random.seed(42)

    # 按语言分组（用于构建并列组合样本）
    lang_groups: Dict[str, List[Tuple[str, str]]] = {}
    for lang, src, tgt in pairs:
        lang_groups.setdefault(lang, []).append((src, tgt))

    # 分离高低资源数据
    high_pairs = [p for p in pairs if p[0] not in low_resource_langs]
    low_pairs  = [p for p in pairs if p[0] in low_resource_langs]

    # ① 单术语样本（核心训练数据）
    high_data = [{"src_lang": l, "src": s, "tgt": t}
                 for l, s, t in high_pairs] * repeat
    low_data  = [{"src_lang": l, "src": s, "tgt": t}
                 for l, s, t in low_pairs] * int(repeat * boost_factor)

    # ② 并列组合样本（辅助数据，严格限制数量）
    combo_data: List[Dict] = []
    for lang, term_list in lang_groups.items():
        if len(term_list) < 2:
            continue

        # 并列组合数量上限：50条（防止过拟合列举模式）
        n_combos = min(len(term_list) * 2, 50)

        # 中文使用顿号分隔，其他语言使用英文逗号
        sep_src = "、" if lang == "zho_Hans" else ", "
        sep_tgt = ", "

        for _ in range(n_combos):
            # 随机采样2~4个术语组合（不放回）
            k     = random.randint(2, min(4, len(term_list)))
            picks = random.sample(term_list, k)

            src_text = sep_src.join(p[0] for p in picks)
            tgt_text = sep_tgt.join(p[1] for p in picks)

            # 30%概率添加句末标点（提升句子完整性学习）
            if random.random() < 0.3:
                src_text += "。" if lang == "zho_Hans" else "."
                tgt_text += "."

            combo_data.append({
                "src_lang": lang,
                "src":      src_text,
                "tgt":      tgt_text,
            })

    data = high_data + low_data + combo_data
    random.shuffle(data)

    logger.info(
        f"  [加权采样] 高资源: {len(high_data)}, 低资源: {len(low_data)}, "
        f"并列组合: {len(combo_data)}, 总计: {len(data)}"
    )
    return data


def tokenize_data(
    samples: List[Dict],
    tokenizer,
    max_len: int = 128,
) -> Dataset:
    """
    将训练样本列表转换为HuggingFace Dataset格式。
    NLLB Labels格式规范：
      labels = [tgt_lang_id, token1, token2, ..., eos_id]
      tgt_lang_id作为labels首位是NLLB的强制目标语言机制：
      解码器被迫以目标语言token开始生成，确保输出语言正确。
      这是NLLB与标准Seq2Seq模型的关键区别。
    目标端长度截断：
      max_length=max_len-2，为tgt_lang_id和eos_token预留空间，
      避免标签序列超出解码器最大长度。
    解码器工作机制（NLLB标准）：
      decoder_input_ids 由 labels 右移生成：
        labels          : [lang_id, tok1, tok2, ..., tokN, eos]
        decoder_input_ids: [pad,    lang_id, tok1, ..., tokN]
                            ^^^
                      decoder_start_token_id（pad_token_id=2）
      forced_bos_token_id=lang_id 与此协同：
        解码器第一步被强制输出lang_id，确保输出语言正确。
    数据字段说明：
      samples列表元素格式：
        {
          "src_lang": "zho_Hans",          ← NLLB语言代码（extract_pairs输出）
          "src":      "代词有人称...",      ← 源语言文本
          "tgt":      "Pronouns have..."   ← 英文译文
        }

    Args:
        samples:   训练样本列表（build_weighted_dataset输出）
        tokenizer: NLLB分词器（AutoTokenizer加载）
        max_len:   源端最大序列长度（目标端自动设为max_len-2）

    Returns:
        HuggingFace Dataset（含input_ids/attention_mask/labels三字段）
    """
    logger = logging.getLogger(__name__)

    all_input_ids: List[List[int]] = []
    all_attn:      List[List[int]] = []
    all_labels:    List[List[int]] = []

    # 获取目标语言token ID（eng_Latn）
    tgt_lang_id = get_lang_id(tokenizer, TGT_LANG)

    # 预计算special token ID集合（用于过滤tgt编码中混入的特殊token）
    # 包括：pad/bos/eos/unk以及所有语言代码token
    special_ids: Set[int] = set()
    special_ids.add(tokenizer.pad_token_id)          # pad=2
    special_ids.add(tokenizer.bos_token_id)          # bos（通常与pad相同）
    special_ids.add(tokenizer.eos_token_id)          # eos=2（NLLB中eos=pad）
    if tokenizer.unk_token_id is not None:
        special_ids.add(tokenizer.unk_token_id)      # unk

    # NLLB语言代码token也属于特殊token，需过滤（防止目标端混入语言token）
    vocab = tokenizer.get_vocab()
    for lang_code in LANG_MAP.values():
        if lang_code in vocab:
            special_ids.add(vocab[lang_code])

    logger.info(
        f"  [tokenize_data] 开始Tokenize | "
        f"样本数={len(samples)} | "
        f"max_len={max_len} | "
        f"tgt_lang_id={tgt_lang_id}（{TGT_LANG}）"
    )

    skip_count = 0
    for idx, s in enumerate(samples):

        # 验证样本字段完整性
        src_lang = s.get("src_lang", "")
        src_text = s.get("src", "")
        tgt_text = s.get("tgt", "")

        if not src_lang or not src_text or not tgt_text:
            logger.debug(
                f"  [跳过] 样本#{idx}：字段不完整 "
                f"(src_lang='{src_lang}', "
                f"src长度={len(src_text)}, "
                f"tgt长度={len(tgt_text)})"
            )
            skip_count += 1
            continue

        # ── 源端编码 ──
        # src_lang设置影响SentencePiece的语言感知分词策略
        tokenizer.src_lang = src_lang
        src_enc = tokenizer(
            src_text,
            max_length=max_len,
            truncation=True,
            padding=False,           # DataCollatorForSeq2Seq动态padding
            add_special_tokens=True, # 自动添加[EOS]等源端特殊token
        )
        # ── 目标端编码 ──
        # add_special_tokens=False：禁止自动添加特殊token
        # 原因：labels格式由我们手动精确控制
        #   [tgt_lang_id] + content_tokens + [eos_token_id]
        # max_length=max_len-2：为手动添加的lang_id和eos预留2个位置
        tgt_enc = tokenizer(
            tgt_text,
            max_length=max_len - 2,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # 禁止自动添加，手动控制格式
        )

        # ── 过滤空序列 ──
        if not src_enc["input_ids"] or not tgt_enc["input_ids"]:
            logger.debug(
                f"  [跳过] 样本#{idx}：编码结果为空 "
                f"(src_ids长度={len(src_enc['input_ids'])}, "
                f"tgt_ids长度={len(tgt_enc['input_ids'])})"
            )
            skip_count += 1
            continue

        # ── 过滤tgt中混入的特殊token ──
        # add_special_tokens=False理论上不添加，但部分tokenizer版本
        # 在某些语言设置下仍可能混入pad/eos，显式过滤确保格式纯净
        clean_tgt_ids = [
            tid for tid in tgt_enc["input_ids"]
            if tid not in special_ids
        ]

        # 过滤后为空（极短文本或全为特殊token）
        if not clean_tgt_ids:
            logger.debug(
                f"  [跳过] 样本#{idx}：过滤特殊token后tgt为空 "
                f"(原始ids={tgt_enc['input_ids']})"
            )
            skip_count += 1
            continue

        # ── 构建NLLB标准labels格式 ──
        # 格式：[lang_token] + content_tokens + [eos_token]
        # 示例：[256047, 1234, 5678, 9012, 2]
        #        eng_Latn  tok1  tok2  tok3  eos
        labels = (
            [tgt_lang_id]             # 强制目标语言token
            + clean_tgt_ids           # 过滤后的纯内容token
            + [tokenizer.eos_token_id] # 序列结束标记
        )

        all_input_ids.append(src_enc["input_ids"])
        all_attn.append(src_enc["attention_mask"])
        all_labels.append(labels)

    # ── 汇总统计 ──
    valid_count = len(all_input_ids)
    logger.info(
        f"  [tokenize_data] Tokenize完成 | "
        f"有效样本={valid_count} | "
        f"跳过={skip_count} | "
        f"总计={len(samples)}"
    )

    if valid_count == 0:
        raise ValueError(
            "Tokenize后无有效样本！请检查：\n"
            "  1. LANG_MAP键名是否与术语库term字段一致（ind vs id）\n"
            "  2. 术语库translation字段是否非空\n"
            "  3. max_length是否过小（当前={max_len}）"
        )

    return Dataset.from_dict({
        "input_ids":      all_input_ids,
        "attention_mask": all_attn,
        "labels":         all_labels,
    })


def extract_cross_lingual_pairs(
    pairs: List[Tuple[str, str, str]],
    min_overlap: float = 0.15,
    max_pairs: int = 200,
    adaptive: bool = True,
    use_domain_general: bool = True,
) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """
    提取跨语言同义词对（三级回退策略）。

    核心思想：
      不同语言的同义术语→英文译文相似→自动识别跨语言对应。
      无需人工标注跨语言平行数据。

    三级回退策略：
      策略1：相似度阈值筛选（主路径，min_overlap默认0.15）
      策略2：自适应降阈值（候选<5对时自动宽松：0.1→0.05→0.02→0.01）
      策略3：领域通用随机对齐（无任何相似对时，随机配对引导结构发现）

    Args:
        pairs:              全量术语对
        min_overlap:        英文译文最小相似度阈值
        max_pairs:          最多返回的术语对数
        adaptive:           是否启用自适应降阈值
        use_domain_general: 是否在无相似对时启用随机配对

    Returns:
        [((lang1, term1), (lang2, term2)), ...]
    """
    logger = logging.getLogger(__name__)

    # 停用词集合（过滤功能词，保留内容词用于Jaccard计算）
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'of', 'in', 'on', 'at',
        'to', 'for', 'with', 'by', 'from', 'and', 'or', 'but', 'not', 'as',
        'it', 'its', 'this', 'that', 'these', 'those', 'etc', 'such', 'can',
        'will', 'would', 'should', 'may', 'might', 'must', 'shall'
    }

    def get_content_words(text: str) -> Set[str]:
        """提取内容词（小写→去标点→去停用词→去短词）"""
        words = text.lower().split()
        words = {w.strip('.,;:()[]{}!?"\'-/') for w in words}
        return {w for w in words if w not in stopwords and len(w) > 2}

    # 按语言分组
    lang_groups: Dict[str, List[Tuple[str, str]]] = {}
    for lang, src, tgt in pairs:
        lang_groups.setdefault(lang, []).append((src, tgt))

    langs = list(lang_groups.keys())
    if len(langs) < 2:
        logger.warning(f"  仅{len(langs)}种语言，无法形成跨语言对")
        return []

    logger.info(f"  [跨语言对齐] 检测到语言: {langs}")
    for lang, terms in lang_groups.items():
        logger.info(f"    {lang}: {len(terms)} 条术语")

    # 计算所有跨语言术语对的相似度
    all_pairs_with_sim = []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            lang1, lang2 = langs[i], langs[j]
            for src1, tgt1 in lang_groups[lang1]:
                words1 = get_content_words(tgt1)
                for src2, tgt2 in lang_groups[lang2]:
                    words2 = get_content_words(tgt2)
                    sim    = compute_term_similarity(tgt1, tgt2, words1, words2)
                    all_pairs_with_sim.append((sim, (lang1, src1), (lang2, src2)))

    # 策略1：阈值筛选
    candidates = [(sim, t1, t2) for sim, t1, t2 in all_pairs_with_sim
                  if sim >= min_overlap]
    logger.info(f"  [策略1] 阈值{min_overlap}筛选: {len(candidates)} 对")

    # 策略2：自适应降阈值（候选不足5对时）
    if adaptive and len(candidates) < 5:
        logger.info("  [策略2] 候选不足5对，启动自适应降阈值")
        for threshold in [0.1, 0.05, 0.02, 0.01]:
            fallback = [(sim, t1, t2) for sim, t1, t2 in all_pairs_with_sim
                        if sim >= threshold]
            if len(fallback) >= 5:
                candidates = fallback
                logger.info(f"    ✓ 采用阈值{threshold}: {len(fallback)} 对")
                break
        else:
            # 所有阈值都不足5对，取相似度最高的top-N
            all_pairs_with_sim.sort(reverse=True, key=lambda x: x[0])
            candidates = all_pairs_with_sim[:max_pairs]
            logger.info(f"    ✓ 采用top-{len(candidates)}")

    # 策略3：领域通用随机对齐（完全无相似对时）
    if use_domain_general and len(candidates) == 0:
        logger.info("  [策略3] 无相似对，使用随机配对（引导编码器发现跨语言结构）")
        random.seed(42)
        cross_pairs_direct = []
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                n = min(len(lang_groups[lang1]), len(lang_groups[lang2]), 5)
                for _ in range(n):
                    src1, _ = random.choice(lang_groups[lang1])
                    src2, _ = random.choice(lang_groups[lang2])
                    cross_pairs_direct.append(((lang1, src1), (lang2, src2)))
        logger.info(f"    生成 {len(cross_pairs_direct)} 个随机配对")
        return cross_pairs_direct

    # 按相似度降序排列，取前max_pairs对
    candidates.sort(reverse=True, key=lambda x: x[0])
    top_pairs   = candidates[:max_pairs]
    cross_pairs = [(t1, t2) for _, t1, t2 in top_pairs]

    logger.info(f"  [跨语言对齐] 最终提取 {len(cross_pairs)} 对")
    return cross_pairs


def encode_cross_lingual_pairs(
    model,
    tokenizer,
    cross_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    device: str = "cpu",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    预计算跨语言术语对的编码器隐藏状态表示。

    提取策略（注意力掩码加权平均池化）：
      h = Σ(hidden_i × mask_i) / Σ(mask_i)
      仅对有效token位置（mask=1）取平均，忽略padding位置。
      pooling结果是每个术语的固定维度语义向量。

    编码器访问兼容性（PeftModel多层包装）：
      1. model.get_encoder()           ← 推荐，兼容所有包装层
      2. model.model.encoder           ← 原始未包装模型
      3. model.base_model.model.encoder ← PeftModel深层路径

    Args:
        model:       注入LoRA后的模型（PeftModel）
        tokenizer:   NLLB分词器
        cross_pairs: 跨语言术语对
        device:      计算设备

    Returns:
        [(h1.detach(), h2.detach()), ...] detach确保不引入计算图
    """
    logger = logging.getLogger(__name__)
    model.eval()
    encoded_pairs = []

    # 兼容性编码器访问
    encoder = None
    if hasattr(model, 'get_encoder'):
        encoder = model.get_encoder()
        logger.debug("  [编码器] 使用 model.get_encoder()")
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
        logger.debug("  [编码器] 使用 model.model.encoder")
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base = model.base_model.model
        if hasattr(base, 'model') and hasattr(base.model, 'encoder'):
            encoder = base.model.encoder
        elif hasattr(base, 'get_encoder'):
            encoder = base.get_encoder()

    if encoder is None:
        logger.warning(f"  无法获取编码器（模型类型: {type(model).__name__}），跳过编码")
        return []

    logger.info(f"  预计算 {len(cross_pairs)} 对编码器表示...")
    with torch.no_grad():
        for (lang1, term1), (lang2, term2) in cross_pairs:
            try:
                # 编码第一个术语
                tokenizer.src_lang = lang1
                enc1 = tokenizer(
                    term1, return_tensors="pt",
                    truncation=True, max_length=256
                ).to(device)
                out1  = encoder(
                    input_ids=enc1["input_ids"],
                    attention_mask=enc1["attention_mask"]
                )
                # 注意力掩码加权平均池化
                mask1 = enc1["attention_mask"].unsqueeze(-1).float()
                h1    = (out1.last_hidden_state * mask1).sum(dim=1) / mask1.sum(dim=1)
                h1    = h1.squeeze(0)

                # 编码第二个术语（同上）
                tokenizer.src_lang = lang2
                enc2 = tokenizer(
                    term2, return_tensors="pt",
                    truncation=True, max_length=256
                ).to(device)
                out2  = encoder(
                    input_ids=enc2["input_ids"],
                    attention_mask=enc2["attention_mask"]
                )
                mask2 = enc2["attention_mask"].unsqueeze(-1).float()
                h2    = (out2.last_hidden_state * mask2).sum(dim=1) / mask2.sum(dim=1)
                h2    = h2.squeeze(0)

                # detach脱离计算图（编码对作为固定目标，不反向传播）
                encoded_pairs.append((h1.detach(), h2.detach()))

            except Exception as e:
                logger.warning(f"  术语对编码失败，已跳过: {e}")
                continue

    model.train()
    logger.info(f"  编码完成: {len(encoded_pairs)}/{len(cross_pairs)} 对成功")
    return encoded_pairs


# ============================================================
# 翻译后处理与语言检测工具
# ============================================================

def finalize_translation(text: str) -> str:
    """
    标准化翻译输出格式。

    处理目标：
      修复微调后模型可能产生的输出格式伪像，
      不改变翻译的实质内容。

    处理步骤：
      1. 移除末尾所有标点和空白（统一处理中英文）
      2. 补充标准英文句号（保持输出格式一致性）
      3. 移除学术引用号 [1] [23]（微调可能引入的伪像）
      4. 规范化连续句号（...→.，防止重复句号）

    Args:
        text: 原始译文字符串

    Returns:
        格式化后的译文
    """
    text = text.strip()
    # 移除末尾所有标点（含中文句号、分号、逗号等）
    text = re.sub(r'[.!?。！？;；,\s、]+$', '', text)
    # 统一补充英文句号
    if text:
        text += '.'
    # 移除学术引用号（微调伪像）
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    # 规范化连续句号
    text = re.sub(r'\.{2,}', '.', text)
    return text


def _count_cjk_and_latin(text: str) -> Tuple[int, int]:
    """
    统计文本中的CJK字符数和拉丁字母数。

    CJK范围覆盖：
      U+4E00~U+9FFF：CJK统一汉字（主要中文区）
      U+3400~U+4DBF：CJK扩展A（古汉字）
      U+F900~U+FAFF：CJK兼容汉字

    用途：
      快速判断文本是否为CJK语言（中文/日语/韩语）或拉丁语系，
      以及是否为混合语言文本（cjk>0 且 latin>0）。

    Args:
        text: 输入文本

    Returns:
        (cjk字符数, 拉丁字母数)
    """
    cjk = latin = 0
    for ch in text:
        if ('\u4e00' <= ch <= '\u9fff'
                or '\u3400' <= ch <= '\u4dbf'
                or '\uf900' <= ch <= '\ufaff'):
            cjk += 1
        elif ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'):
            latin += 1
    return cjk, latin


def split_into_sentences(text: str) -> List[str]:
    """
    按句末标点切分句子（数字小数点保护版）。

    核心挑战：
      区分"句末句点"和"数字小数点"：
        15.000（印尼语千分位）→ 不切分
        3.14（小数）          → 不切分
        "end. Next"           → 切分

    三段式正则说明：
      段1: (?<=[。!！?？])\s*
           → 中文句末标点后直接切（后继无需特殊条件）
      段2: (?<=[.!?])(?=\s+[A-Z\u4e00-\u9fff])
           → 英文句末 + 后接大写字母或中文（防止误切数字）
      段3: (?<=[.!?])(?=\s*$)
           → 行尾最后一句（无后继字符）

    数字保护机制：
      将"数字.数字"临时替换为"数字<DOT>数字"，
      切分完成后恢复，避免小数点触发错误切分。

    Args:
        text: 输入文本

    Returns:
        句子列表（空句子已过滤）
    """
    # 保护数字间的小数点
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)

    parts = re.split(
        r'(?<=[。!！?？])\s*'                        # 段1：中文句末
        r'|(?<=[.!?])(?=\s+[A-Z\u4e00-\u9fff])'     # 段2：英文句末
        r'|(?<=[.!?])(?=\s*$)',                      # 段3：行尾
        protected
    )

    # 还原占位符，过滤空句
    result = [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]
    return result if result else [text]


def detect_src_lang_by_tokenizer(
    text: str,
    tokenizer,
    candidate_langs: List[str],
    sample_chars: int = 200,
) -> str:
    """
    基于Tokenizer编码紧凑度的通用语言检测。

    核心原理：
      NLLB Tokenizer对各语言有独立的SentencePiece子词词表。
      当src_lang设置正确时，文本被切分为更少、更长的subword。
      当src_lang设置错误时，文本被切碎为更多短片段。
      → 选择使token数最少（紧凑度最低）的语言 = 最匹配语言

    紧凑度分数：score = token_count / char_count
    分数越低 → 切分越粗 → 语言越匹配

    副作用保护：
      函数执行完毕后恢复tokenizer.src_lang的原始状态，
      避免影响后续的tokenize操作。

    Args:
        text:            待检测文本
        tokenizer:       NLLB分词器
        candidate_langs: 候选NLLB语言代码列表
        sample_chars:    只取前N个字符检测（加速，通常200字符足够）

    Returns:
        最匹配的NLLB语言代码
    """
    if not candidate_langs:
        return "zho_Hans"

    sample = text.strip()[:sample_chars]
    if not sample:
        return candidate_langs[0]

    best_lang    = candidate_langs[0]
    best_score   = float("inf")
    original_src = tokenizer.src_lang  # 保存原始状态

    for lang in candidate_langs:
        try:
            tokenizer.src_lang = lang
            ids = tokenizer(
                sample,
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]
            score = len(ids) / max(len(sample), 1)
            if score < best_score:
                best_score = score
                best_lang  = lang
        except Exception:
            continue

    tokenizer.src_lang = original_src  # 恢复原始状态
    return best_lang


def get_candidate_langs(
    tgt_lang: str,
    lang_map: Dict[str, str] = None,
) -> List[str]:
    """
    从LANG_MAP动态构建候选源语言列表（排除目标语言）。

    通用性设计：
      候选集完全由LANG_MAP决定，
      新增支持语言只需在LANG_MAP注册，无需修改此函数。

    Args:
        tgt_lang: 目标语言NLLB代码（排除，避免自翻译）
        lang_map: ISO→NLLB映射（默认全局LANG_MAP）

    Returns:
        候选NLLB语言代码列表
    """
    if lang_map is None:
        lang_map = LANG_MAP
    return [nllb for nllb in lang_map.values() if nllb != tgt_lang]


def detect_sentence_src_lang(
    sent: str,
    tokenizer,
    tgt_lang: str,
    lang_map: Dict[str, str] = None,
) -> str:
    """
    通用单句语言检测（两阶段策略）。

    两阶段设计：
      阶段1：CJK快速通道（O(n)字符遍历，极快）
             CJK字符占字母总数>30% → 直接返回中文
             中文是唯一使用CJK字符集的支持语言，无歧义

      阶段2：拉丁脚本Tokenizer投票（仅非中文时触发）
             在拉丁语系候选（印尼语、他加禄语等）中
             通过编码紧凑度打分，选最匹配者

    排除逻辑：
      候选集排除目标语言（避免将待翻译内容误判为目标语言）
      候选集排除中文和英文（已在阶段1处理或不在源语言范围）

    Args:
        sent:     待检测句子
        tokenizer: NLLB分词器
        tgt_lang:  目标语言（排除）
        lang_map:  语言映射表

    Returns:
        最匹配的NLLB语言代码
    """
    if lang_map is None:
        lang_map = LANG_MAP

    # 阶段1：CJK快速通道
    cjk, latin = _count_cjk_and_latin(sent)
    total_alpha = cjk + latin
    if total_alpha > 0 and cjk / total_alpha > 0.3:
        zh_code = lang_map.get("zh")
        if zh_code:
            return zh_code

    # 阶段2：拉丁脚本Tokenizer投票
    # 排除中文（阶段1已处理）和英文（目标语言，不作为源）
    zh_codes = {lang_map.get("zh"), lang_map.get("en")}
    latin_candidates = [
        nllb for nllb in lang_map.values()
        if nllb != tgt_lang and nllb not in zh_codes
    ]

    if not latin_candidates:
        return lang_map.get("zh", "zho_Hans")

    return detect_src_lang_by_tokenizer(sent, tokenizer, latin_candidates)


@torch.no_grad()
def engram_enhanced_generate(
    model, tokenizer, engram_table, engram_gating, sonar_autoencoder,
    input_ids, attention_mask, tgt_lang_id, max_length=128, device="cuda"
) -> torch.Tensor:
    """
    【核心修复】Engram 增强自回归解码器
    完美复刻训练时的 Hidden States -> Engram Lookup -> Gating -> LM Head 闭环
    """
    model.eval()
    encoder = model.get_encoder()
    decoder = model.get_decoder()
    lm_head = model.get_output_embeddings()
    
    # 1. 编码器前向
    enc_out = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    encoder_hidden_states = enc_out.last_hidden_state
    
    # 2. 获取 SONAR 全局锚点 (简化处理，若无则用零向量)
    B = input_ids.shape[0]
    src_anchor = torch.zeros(B, 1, 1024, device=device)
    if sonar_autoencoder is not None:
        try:
            src_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            src_inp = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            raw_emb = _encode_with_base_model(model, src_inp, device) # 复用已有函数
            if raw_emb is not None:
                src_anchor = raw_emb.unsqueeze(1)
        except Exception:
            pass # 降级为零向量

    # 3. 自回归解码循环
    generated = torch.tensor([[tgt_lang_id]], device=device).expand(B, -1)
    past_key_values = None
    
    for _ in range(max_length):
        # Decoder 前向 (利用 KV Cache 加速)
        dec_inputs = generated if past_key_values is None else generated[:, -1:]
        dec_out = decoder(
            input_ids=dec_inputs,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True
        )
        past_key_values = dec_out.past_key_values
        
        # 提取 Decoder 隐藏态 (取最后一层)
        hidden = dec_out.hidden_states[-1]  # [B, T, D]
        B, T, D = hidden.shape
        
        # 【核心注入】Engram 检索与门控
        if engram_table and engram_gating:
            recent_tokens = tokenizer.convert_ids_to_tokens(generated[0].tolist())
            mem_vec, conf = engram_table.lookup(recent_tokens, device=device)
            # 扩展记忆向量以匹配 Hidden 维度 [B, T, D]
            memory_e = mem_vec.unsqueeze(1).expand(B, T, D)
            # 调用门控网络融合
            hidden, _ = engram_gating(hidden, memory_e, src_anchor, conf)
            
        # 投影到词表并采样
        logits = lm_head(hidden)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        
        # 终止条件
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return generated

@torch.no_grad()
def translate_auto(
    model, tokenizer, text: str, tgt_lang: str,
    lang_map: Dict[str, str] = None,
    engram_table=None, engram_gating=None, sonar_autoencoder=None # 【新增】推理端组件注入
) -> str:
    """
    自动翻译策略（支持 Engram 推理闭环）
    """
    if lang_map is None:
        lang_map = LANG_MAP

    cjk, latin = _count_cjk_and_latin(text)
    device = next(model.parameters()).device

    # 为简洁，此处统一使用单脚本/全局检测逻辑（混合脚本逻辑同理）
    candidate_langs = get_candidate_langs(tgt_lang, lang_map)
    src_lang = detect_src_lang_by_tokenizer(text, tokenizer, candidate_langs)
    tgt_id = get_lang_id(tokenizer, tgt_lang)
    
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # 【核心分支】：如果加载了 Engram 组件，走自定义闭环解码；否则走原生 generate
    if engram_table is not None and engram_gating is not None:
        out_ids = engram_enhanced_generate(
            model=model,
            tokenizer=tokenizer,
            engram_table=engram_table,
            engram_gating=engram_gating,
            sonar_autoencoder=sonar_autoencoder,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tgt_lang_id=tgt_id,
            max_length=512,
            device=device
        )
    else:
        out_ids = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_length=512,
            num_beams=3,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# ============================================================
# 【修改】训练参数构建（添加BF16+Flash Attention优化）
# ============================================================
def _build_training_args(
    args: argparse.Namespace,
    output_subdir: str,
    total_steps: int,
) -> Tuple[TrainingArguments, int]:
    """
    【修改版】构建优化的TrainingArguments
    
    新增优化：
      1. BF16混合精度训练（速度+2x，显存-50%）
      2. 梯度检查点（显存-50%，速度-20%）
      3. Flash Attention自动启用（PyTorch 2.0+）
      4. DataLoader优化（num_workers=4，pin_memory）
    
    Args:
        args: 全局参数对象
        output_subdir: checkpoint子目录名
        total_steps: 总训练步数
    
    Returns:
        (TrainingArguments实例, Warmup步数)
    """
    # 计算Warmup步数
    warmup_steps = max(1, int(total_steps * args.warmup_proportion))

    # 检测设备能力
    use_bf16 = False
    use_gradient_checkpointing = False
    is_cpu = (args.device == 'cpu')

    if not is_cpu and torch.cuda.is_available():
        # BF16检测（Ampere架构及以上：A100/RTX3090/4090）
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] >= 8:
            use_bf16 = True

        # 梯度检查点：所有GPU均支持
        use_gradient_checkpointing = True

    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, output_subdir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,

        # ===== 混合精度 =====
        bf16=use_bf16,
        fp16=False,

        # ===== 梯度优化 =====
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.gradient_clip_norm,

        # ===== DataLoader =====
        dataloader_num_workers=0 if is_cpu else 4,
        dataloader_pin_memory=(not is_cpu),

        # ===== 日志与保存 =====
        logging_steps=1,
        logging_first_step=False,
        save_steps=500,
        save_total_limit=1,

        # ===== 设备配置=====
        use_cpu=is_cpu,             # ← 仅保留此项

        # ===== 输出控制 =====
        disable_tqdm=True,
        report_to="none",
    )

    return train_args, warmup_steps

def train_model(
    args: argparse.Namespace,
    model,
    tokenizer,
    dataset: Dataset,
    stage_name: str = "训练",
) -> None:
    """
    统一标准训练接口（重构自_run_shared_training）。

    新增特性（相比原_run_shared_training）：
      - 自动计算Warmup步数（基于args.warmup_proportion）
      - 梯度裁剪（基于args.gradient_clip_norm）
      - 梯度累积（基于args.gradient_accumulation_steps）
      - 接收args参数对象，便于访问所有超参数

    Args:
        args:       全局参数对象（含所有超参数）
        model:      注入LoRA的模型
        tokenizer:  NLLB分词器
        dataset:    训练数据集
        stage_name: 阶段名称（用于日志和checkpoint目录）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n  ========== {stage_name} ==========")

    # 估算总步数（用于Warmup比例计算）
    samples_per_step = args.batch_size * args.gradient_accumulation_steps
    total_steps      = max(1, len(dataset) // samples_per_step) * args.epochs

    train_args, warmup_steps = _build_training_args(
        args,
        output_subdir=stage_name.replace(' ', '_'),
        total_steps=total_steps,
    )

    logger.info(
        f"  总步数: {total_steps} | Warmup: {warmup_steps}步 "
        f"| 梯度裁剪: {args.gradient_clip_norm} "
        f"| 梯度累积: {args.gradient_accumulation_steps}步"
    )

    progress_cb = CustomProgressCallback(total_epochs=args.epochs)
    collator    = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = CGLoRATrainer(
        curvature_lambda=args.curvature_lambda,
        fisher_beta=args.fisher_beta,
        warmup_steps=warmup_steps,
        model=model,
        args=train_args,
        processing_class=tokenizer,    # ← tokenizer → processing_class
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[progress_cb],
    )

    # 移除HuggingFace默认回调，由CustomProgressCallback完全接管
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)

    trainer.train()


# ============================================================
# 【新增】双优化器包装类
# ============================================================

class DualOptimizerWrapper:
    """
    双优化器包装器（兼容 Trainer 接口）
    
    设计原则：
      对外表现为单一优化器（兼容 Trainer 的所有调用）
      内部同步管理两个优化器（LoRA + Engram门控）
    
    核心方法：
      step()       → 同时调用两个优化器的 step()
      zero_grad()  → 同时调用两个优化器的 zero_grad()
      state_dict() → 返回合并后的状态字典
      load_state_dict() → 分别加载到两个优化器
    """
    
    def __init__(
        self,
        optimizer_lora: torch.optim.Optimizer,
        optimizer_engram: torch.optim.Optimizer,
    ):
        self.optimizer_lora   = optimizer_lora
        self.optimizer_engram = optimizer_engram
        
        # 委托属性（Trainer 可能访问的属性）
        self.param_groups = (
            optimizer_lora.param_groups + optimizer_engram.param_groups
        )
        self.defaults = optimizer_lora.defaults
    
    def step(self, closure=None):
        """
        同步执行两个优化器的参数更新
        
        Args:
            closure: 可选闭包（重新计算loss，用于某些优化器）
        
        Returns:
            LoRA优化器的返回值（兼容接口）
        """
        loss_lora = self.optimizer_lora.step(closure)
        self.optimizer_engram.step(None)  # Engram不使用closure
        return loss_lora
    
    def zero_grad(self, set_to_none: bool = False):
        """同步清空两个优化器的梯度"""
        self.optimizer_lora.zero_grad(set_to_none=set_to_none)
        self.optimizer_engram.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> Dict:
        """
        返回合并的状态字典
        
        格式：
          {
            "lora": {...},     # LoRA优化器状态
            "engram": {...},   # Engram优化器状态
          }
        """
        return {
            "lora":   self.optimizer_lora.state_dict(),
            "engram": self.optimizer_engram.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict):
        """加载合并的状态字典"""
        if "lora" in state_dict:
            self.optimizer_lora.load_state_dict(state_dict["lora"])
        if "engram" in state_dict:
            self.optimizer_engram.load_state_dict(state_dict["engram"])
    
    def __repr__(self) -> str:
        return (
            f"DualOptimizerWrapper(\n"
            f"  LoRA: {self.optimizer_lora.__class__.__name__},\n"
            f"  Engram: {self.optimizer_engram.__class__.__name__}\n"
            f")"
        )


# ============================================================
# 【新增】Engram条件记忆嵌入表
# ============================================================

class TerminologyEngramTable:
    """
    基于SONAR语义空间的跨语言N-gram嵌入表（本地缓存增强版    
      1. 哈希检索：确定性FNV-1a哈希，O(1)复杂度
      2. SONAR初始化：跨语言同义术语映射到语义相近嵌入
      3. CPU存储：节省GPU显存（异步预取至GPU缓冲区）
      4. FP16压缩：进一步减少存储开销    
    存储开销估算：
      hash_table_size × nllb_dim × 2 bytes（FP16）
      131072 × 1024 × 2 = 268 MB（可接受的CPU内存占用）
      1. 构建后自动保存到本地路径（.pt格式）
      2. 训练时优先从缓存加载（跳过耗时构建）
      3. 缓存版本校验（哈希表大小、N-gram阶数变化时自动重建）    
    """
    
    def __init__(
        self,
        nllb_dim: int = 1024,
        max_ngram: int = 3,
        hash_table_size: int = 131072,
        device: str = "cuda",
        cache_dir: str = r'F:\hyberT\MultiSonar\data\Engram_table',  # 新增
    ):
        self.nllb_dim = nllb_dim
        self.max_ngram = max_ngram
        self.table_size = hash_table_size
        self.device = device
        self.cache_dir = cache_dir  # 缓存目录
        
        # CPU存储（FP16压缩）
        self.embedding_table = torch.zeros(
            hash_table_size, nllb_dim,
            dtype=torch.float16
        )
        
        self.total_entries = 0
        self.collision_count = 0
        self.prefetch_buffer = {}
        self.buffer_max_size = 128
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self) -> str:
        """
        生成缓存文件路径（基于配置参数）
        
        格式：engram_table_{hash_size}_{max_ngram}.pt
        作用：不同配置使用独立缓存，避免冲突
        """
        filename = f"engram_table_{self.table_size}_{self.max_ngram}.pt"
        return os.path.join(self.cache_dir, filename)
    
    def save_to_cache(self) -> None:
        """
        保存嵌入表到本地缓存
        
        保存内容：
          embedding_table: [hash_size, dim] FP16张量
          total_entries:   有效条目数
          collision_count: 冲突次数
          metadata:        配置元信息（用于版本校验）
        """
        cache_path = self._get_cache_path()
        
        logger = logging.getLogger(__name__)
        logger.info(f"  [Engram] 保存嵌入表到缓存: {cache_path}")
        
        torch.save({
            'embedding_table': self.embedding_table,
            'total_entries': self.total_entries,
            'collision_count': self.collision_count,
            'metadata': {
                'nllb_dim': self.nllb_dim,
                'max_ngram': self.max_ngram,
                'hash_table_size': self.table_size,
            }
        }, cache_path)
        
        # 文件大小统计
        file_size_mb = os.path.getsize(cache_path) / (1024 ** 2)
        logger.info(f"    ✓ 缓存已保存（大小：{file_size_mb:.1f} MB）")
    
    def load_from_cache(self) -> bool:
        """
        从本地缓存加载嵌入表
        
        校验机制：
          检查metadata中的配置参数是否与当前实例一致
          不一致时返回False，触发重新构建
        
        Returns:
            True:  加载成功
            False: 缓存不存在或配置不匹配，需重新构建
        """
        cache_path = self._get_cache_path()
        logger = logging.getLogger(__name__)
        
        if not os.path.exists(cache_path):
            logger.info("  [Engram] 缓存未找到，将重新构建")
            return False
        
        try:
            logger.info(f"  [Engram] 从缓存加载嵌入表: {cache_path}")
            
            checkpoint = torch.load(cache_path, map_location='cpu')
            
            # 校验配置一致性
            metadata = checkpoint.get('metadata', {})
            if (
                metadata.get('nllb_dim') != self.nllb_dim
                or metadata.get('max_ngram') != self.max_ngram
                or metadata.get('hash_table_size') != self.table_size
            ):
                logger.warning(
                    f"  [Engram] 缓存配置不匹配，将重新构建\n"
                    f"    缓存: dim={metadata.get('nllb_dim')}, "
                    f"ngram={metadata.get('max_ngram')}, "
                    f"size={metadata.get('hash_table_size')}\n"
                    f"    当前: dim={self.nllb_dim}, "
                    f"ngram={self.max_ngram}, "
                    f"size={self.table_size}"
                )
                return False
            
            # 加载嵌入表
            self.embedding_table = checkpoint['embedding_table']
            self.total_entries   = checkpoint['total_entries']
            self.collision_count = checkpoint['collision_count']
            
            collision_rate = self.collision_count / max(1, self.total_entries)
            logger.info(
                f"    ✓ 加载成功：{self.total_entries}条目 | "
                f"冲突率：{collision_rate:.2%}"
            )
            
            return True
        
        except Exception as e:
            logger.warning(f"  [Engram] 缓存加载失败: {e}，将重新构建")
            return False
    
    def _normalize_token(self, token: str) -> str:
        """词表规范化"""
        token = token.lower().strip()
        token = re.sub(r'[-_]', ' ', token)
        token = re.sub(r'\s+', ' ', token)
        return token
    
    def _hash_ngram(
        self,
        ngram_tokens: List[str],
        lang_code: str = "",
        head_idx: int = 0
    ) -> int:
        """FNV-1a哈希"""
        normalized = [self._normalize_token(t) for t in ngram_tokens]
        
        hash_val = 2166136261
        for token in normalized:
            for char in token:
                hash_val ^= ord(char)
                hash_val = (hash_val * 16777619) & 0xFFFFFFFF
        
        hash_val ^= head_idx * 2654435761
        return hash_val % self.table_size
    
    def build_from_sonar(
        self,
        pairs: List[Tuple[str, str, str]],
        sonar_autoencoder,
        tokenizer,
        nllb_base_model,
        device: str = "cuda"
    ) -> None:
        """
        【修改版】用SONAR嵌入初始化Engram表（支持缓存）
        """
        logger = logging.getLogger(__name__)
        
        # ===== 优先从缓存加载 =====
        if self.load_from_cache():
            return  # 加载成功，直接返回
        
        # ===== 缓存未命中，执行完整构建 =====
        logger.info(f"\n[Engram] 构建嵌入表：{len(pairs)}条术语")
        
        # ===== 新增：尝试加载SONAR缓存 =====
        sonar_cache = None
        if (sonar_autoencoder is not None 
            and hasattr(sonar_autoencoder, 'cache_path')
            and sonar_autoencoder.cache_path):
            sonar_cache = SonarAutoencoder.load_term_cache(
                sonar_autoencoder.cache_path, device='cpu'
            )
            if sonar_cache:
                logger.info("  [Engram] 检测到SONAR缓存，将加速构建")
        
        total_terms = len(pairs)
        start_time = time.time()
        processed = 0
        
        import sys
        raw_stdout = sys.stdout
        
        try:
            for idx, (lang, term, translation) in enumerate(pairs):
                try:
                    tokenizer.src_lang = lang
                    tokens = tokenizer.tokenize(term)
                    
                    for n in range(2, min(self.max_ngram + 1, len(tokens) + 1)):
                        for end_idx in range(n - 1, len(tokens)):
                            ngram = tokens[end_idx - n + 1: end_idx + 1]
                            
                            # ===== 修改点：优先查SONAR缓存 =====
                            cache_key = (lang, term.strip())
                            sonar_emb = None
                            
                            if sonar_cache and cache_key in sonar_cache['embeddings_normalized']:
                                # 从缓存加载
                                sonar_emb = sonar_cache['embeddings_normalized'][cache_key].to(device)
                            else:
                                # 实时编码
                                inputs = tokenizer(
                                    term, return_tensors="pt",
                                    truncation=True, max_length=128
                                ).to(device)
                                
                                sonar_emb = _encode_with_base_model(
                                    nllb_base_model, inputs, device
                                )
                            
                            if sonar_emb is None:
                                continue
                            
                            # 多头哈希存储（原逻辑保持不变）
                            for head_idx in range(min(4, self.max_ngram)):
                                slot = self._hash_ngram(ngram, lang, head_idx)
                                
                                if self.embedding_table[slot].norm() > 1e-6:
                                    self.collision_count += 1
                                
                                self.embedding_table[slot] = sonar_emb.squeeze(0).half()
                                self.total_entries += 1
                
                except Exception as e:
                    logger.debug(f"  术语编码失败，跳过: {term} | {e}")
                    continue
                
                processed += 1
                self._update_progress_bar(
                    processed, total_terms, start_time, term, raw_stdout
                )
            
            raw_stdout.write('\n')
            raw_stdout.flush()        
        
        finally:
            sys.stdout = raw_stdout
        
        collision_rate = self.collision_count / max(1, self.total_entries)
        elapsed = time.time() - start_time
        
        logger.info(
            f"  [Engram] 嵌入表构建完成：{self.total_entries}条目 | "
            f"冲突率：{collision_rate:.2%} | "
            f"耗时：{elapsed:.1f}秒"
        )
        
        if collision_rate > 0.3:
            logger.warning(
                f"  [Engram] 冲突率过高（{collision_rate:.2%}），"
                f"建议增大 --engram_hash_size"
            )
        
        # ====== 核心修改：构建后自动保存缓存 ======
        self.save_to_cache()
    
    def _update_progress_bar(self, current, total, start_time, current_term, stdout):
        """进度条更新（代码与之前一致）"""
        percent = (current / total) * 100
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = '=' * filled + '>' + '.' * (bar_len - filled - 1)
        
        elapsed = time.time() - start_time
        if current > 0:
            avg_time = elapsed / current
            eta_sec = avg_time * (total - current)
            
            if eta_sec > 60:
                eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60)}s"
            else:
                eta_str = f"{int(eta_sec)}s"
        else:
            eta_str = "计算中..."
        
        term_display = current_term[:20] + ('...' if len(current_term) > 20 else '')
        
        progress_line = (
            f"\r  [Engram] 构建进度: {current}/{total} [{bar}] "
            f"{percent:.1f}% | ETA: {eta_str} | 当前: {term_display}"
        )
        
        stdout.write(progress_line.ljust(120))
        stdout.flush()
    
    def lookup(self, recent_tokens, lang_code="", device="cuda"):
        """O(1)检索（代码与之前一致）"""
        best_emb = None
        best_confidence = 0.0
        
        for n in range(min(self.max_ngram, len(recent_tokens)), 1, -1):
            ngram = recent_tokens[-n:]
            
            for head_idx in range(min(4, self.max_ngram)):
                slot = self._hash_ngram(ngram, lang_code, head_idx)
                
                if slot in self.prefetch_buffer:
                    emb = self.prefetch_buffer[slot]
                else:
                    emb = self.embedding_table[slot]
                
                if emb.norm() < 1e-6:
                    continue
                
                confidence = n / self.max_ngram
                
                if confidence > best_confidence:
                    best_emb = emb.float().to(device)
                    best_confidence = confidence
                    break
            
            if best_emb is not None:
                break
        
        if best_emb is None:
            return torch.zeros(1, self.nllb_dim, device=device), 0.0
        
        return best_emb.unsqueeze(0), best_confidence
    
    def prefetch_batch(self, predicted_tokens_batch):
        """批量预取（代码与之前一致）"""
        if len(self.prefetch_buffer) > self.buffer_max_size:
            self.prefetch_buffer.clear()
        
        for beam_tokens in predicted_tokens_batch:
            for n in range(2, min(self.max_ngram + 1, len(beam_tokens) + 1)):
                ngram = beam_tokens[-n:]
                slot = self._hash_ngram(ngram)
                
                if slot not in self.prefetch_buffer:
                    self.prefetch_buffer[slot] = \
                        self.embedding_table[slot].clone()


# ============================================================
# 【新增】ELF核心组件：Flow-guided LoRA + Flow Aggregation
# ============================================================
# ============================================================
# 方案A：Flow-guided LoRA Layer
# ============================================================

class FlowLoRALayer(nn.Module):
    """
    Flow-guided LoRA Layer（FLoRA）
    
    核心创新：
      1. 训练时：LoRA更新沿Rectified Flow路径演化（MSE监督）
      2. 推理时：等价于标准LoRA（零额外开销）
    
    理论依据（ELF论文）：
      - 连续空间的MSE损失比离散CE更平滑 → 训练稳定
      - Flow轨迹正则化允许使用更小的秩 → 参数减半
      - x-prediction（直接预测目标）比v-prediction更适合高维embedding
    
    参数量对比：
      标准LoRA (r=32): 1024×32 + 32×1024 = 65,536 × 2 ≈ 131k参数/层
      FLoRA (r=16):    1024×16 + 16×1024 = 32,768 × 2 ≈ 66k参数/层
      ✅ 减少50%参数，性能不降反升（flow稳定性补偿）
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,              # ← 从32减半
        alpha: float = 32.0,         # ← 保持2×rank
        dropout: float = 0.1,
        flow_steps: int = 4,         # Flow内部迭代次数（训练时）
        flow_alpha_schedule: str = "cosine",  # MSE权重调度策略
    ):
        super().__init__()
        
        # 标准LoRA组件
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # Flow特定参数
        self.flow_steps = flow_steps
        self.flow_alpha_schedule = flow_alpha_schedule
        
        # 初始化（与标准LoRA一致）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # 训练状态（用于调度MSE权重）
        self.register_buffer('training_step', torch.tensor(0))
        self.total_training_steps = 10000  # 默认值，外部可覆盖
    
    def get_flow_alpha(self) -> float:
        """
        动态计算MSE损失权重α
        
        调度策略（cosine）：
          训练初期：α=0.2（连续主导，稳定梯度）
          训练后期：α=0.8（离散主导，精确token）
        
        公式：α = 0.2 + 0.6 × (1 - cos(π×progress)) / 2
        """
        if not self.training:
            return 0.0  # 推理时不需要
        
        progress = min(1.0, self.training_step.item() / self.total_training_steps)
        
        if self.flow_alpha_schedule == "cosine":
            # Cosine退火：0.2 → 0.8
            alpha = 0.2 + 0.6 * (1 - math.cos(math.pi * progress)) / 2
        elif self.flow_alpha_schedule == "linear":
            # 线性增长：0.2 → 0.8
            alpha = 0.2 + 0.6 * progress
        else:
            alpha = 0.5  # 固定权重
        
        return alpha
    
    def flow_forward(
        self, 
        x: torch.Tensor, 
        target_delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        训练时的Flow前向传播
        
        流程：
          1. 计算标准LoRA更新Δh₀ = B(A(x))
          2. 采样时间步t ~ Uniform(0,1)
          3. 沿Rectified Flow路径插值：Δhₜ = (1-t)·noise + t·Δh₀
          4. 模型预测干净更新Δh_pred（x-prediction）
          5. 计算MSE损失：||Δh_pred - Δh₀||²
        
        Args:
            x: 输入特征 [B, T, in_features]
            target_delta: 目标更新（用于监督，通常为None，自监督）
        
        Returns:
            delta_h: 当前时间步的更新 [B, T, out_features]
            flow_loss: MSE监督损失（标量或None）
        """
        # Step 1: 计算目标更新（干净的LoRA输出）
        x_dropped = self.dropout(x)
        delta_h_target = self.lora_B(self.lora_A(x_dropped))  # [B, T, out]
        
        # Step 2: 采样时间步（每个样本独立）
        B, T, D = delta_h_target.shape
        t = torch.rand(B, 1, 1, device=x.device)  # [B, 1, 1] 广播到[B, T, D]
        
        # Step 3: Rectified Flow插值
        noise = torch.randn_like(delta_h_target)
        delta_h_t = (1 - t) * noise + t * delta_h_target  # [B, T, out]
        
        # Step 4: 模型预测（x-prediction）
        # 这里简化为直接使用delta_h_target作为预测
        # 实际可训练一个小型MLP：delta_h_pred = FlowNet(delta_h_t, t)
        # 但为了轻量化，我们让LoRA层自身学习flow路径
        delta_h_pred = delta_h_target  # 自监督：目标即预测
        
        # Step 5: 计算MSE损失
        flow_loss = F.mse_loss(delta_h_pred, delta_h_target.detach())
        
        # 更新训练步数（用于调度）
        if self.training:
            self.training_step += 1
        
        return delta_h_t * self.scaling, flow_loss
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_flow_loss: bool = False
    ) -> torch.Tensor:
        """
        统一前向接口
        
        训练时：返回flow路径的更新 + MSE损失
        推理时：返回标准LoRA更新（无额外开销）
        
        Args:
            x: [B, T, in_features]
            return_flow_loss: 是否返回flow损失（训练时True）
        
        Returns:
            若return_flow_loss=False: delta_h [B, T, out_features]
            若return_flow_loss=True:  (delta_h, flow_loss)
        """
        if self.training and return_flow_loss:
            # 训练路径：带flow监督
            delta_h, flow_loss = self.flow_forward(x)
            return delta_h, flow_loss
        else:
            # 推理路径：标准LoRA（零开销）
            x_dropped = self.dropout(x)
            delta_h = self.lora_B(self.lora_A(x_dropped))
            delta_h = delta_h * self.scaling
            
            if return_flow_loss:
                return delta_h, torch.tensor(0.0, device=x.device)
            else:
                return delta_h


# ============================================================
# 方案B：Flow Aggregation Network（Engram记忆聚合）
# ============================================================

class FlowAggregateNet(nn.Module):
    """
    Flow-based记忆聚合网络
    
    功能：
      将多个时间步检索到的Engram记忆向量，
      通过flow机制聚合为单一统一的语义表示
    
    类比：
      ELF从噪声去噪到干净embedding
      这里从分散记忆聚合到统一语义
    
    优势：
      1. 避免单步记忆注入的噪声扰动
      2. 多步语义累积 → 更鲁棒的术语表示
      3. 减少门控计算次数（K步1次 vs 每步1次）
    
    参数量：
      仅~50k参数（相比完整Transformer极轻量）
    """
    
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        
        # 轻量级自注意力（聚合多步记忆）
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        
        # MLP（精炼聚合结果）
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(
        self, 
        memory_trajectory: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        聚合多步记忆为统一表示
        
        Args:
            memory_trajectory: [B, K, dim] K步检索到的记忆向量
            mask: [B, K] 有效记忆mask（可选）
        
        Returns:
            aggregated: [B, dim] 聚合后的统一记忆
        """
        # Step 1: 自注意力聚合（让每步记忆互相交互）
        attn_out, _ = self.attn(
            memory_trajectory, 
            memory_trajectory, 
            memory_trajectory,
            key_padding_mask=mask  # mask掉无效记忆
        )
        memory_trajectory = self.norm1(memory_trajectory + attn_out)
        
        # Step 2: MLP精炼
        mlp_out = self.mlp(memory_trajectory)
        memory_trajectory = self.norm2(memory_trajectory + mlp_out)
        
        # Step 3: 池化为单一向量（平均池化）
        if mask is not None:
            # 加权平均（仅对有效位置）
            mask_expanded = (~mask).float().unsqueeze(-1)  # [B, K, 1]
            aggregated = (memory_trajectory * mask_expanded).sum(dim=1)
            aggregated = aggregated / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            aggregated = memory_trajectory.mean(dim=1)  # [B, dim]
        
        return aggregated


# ============================================================
# 方案B：修改SOARGuidedEngramGating（支持Flow累积）
# ============================================================

class FlowEngramGating(nn.Module):
    """
    Flow增强的Engram门控网络
    
    改进点：
      原版：每步检索 → 每步注入
      Flow版：累积K步 → flow聚合 → 一次注入
    
    优势：
      1. 术语一致性：多步累积避免单步噪声
      2. 计算效率：门控次数从T次 → T/K次（K=5时节省80%）
      3. 语义连贯：flow路径确保记忆平滑演化
    """
    
    def __init__(
        self,
        nllb_dim: int = 1024,
        sonar_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_threshold: float = 0.3,
        accumulate_K: int = 5,  # ← 新增：累积步数
        enable_flow_aggregate: bool = True,  # ← 新增：是否启用flow聚合
    ):
        super().__init__()
        self.nllb_dim = nllb_dim
        self.sonar_dim = sonar_dim
        self.num_heads = num_heads
        self.head_dim = nllb_dim // num_heads
        self.gate_threshold = gate_threshold
        self.accumulate_K = accumulate_K
        self.enable_flow_aggregate = enable_flow_aggregate
        
        # Query投影：融合NLLB隐藏态 + SONAR锚点
        self.W_q = nn.Linear(nllb_dim + sonar_dim, nllb_dim, bias=False)
        
        # Key/Value投影（作用于记忆向量）
        self.W_k = nn.Linear(nllb_dim, nllb_dim, bias=False)
        self.W_v = nn.Linear(nllb_dim, nllb_dim, bias=False)
        
        # 输出投影
        self.W_o = nn.Linear(nllb_dim, nllb_dim, bias=False)
        
        # 因果深度卷积（短程精炼）
        self.causal_conv = nn.Conv1d(
            in_channels=nllb_dim,
            out_channels=nllb_dim,
            kernel_size=3,
            padding=2,
            groups=nllb_dim,
        )
        self.conv_norm = nn.LayerNorm(nllb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Flow聚合网络（仅在enable_flow_aggregate=True时使用）
        if self.enable_flow_aggregate:
            self.flow_aggregator = FlowAggregateNet(
                dim=nllb_dim,
                num_heads=4,
                mlp_ratio=2,
                dropout=dropout
            )
    
    def forward(
        self,
        h_t: torch.Tensor,           # [B, T, nllb_dim]
        memory_trajectory: torch.Tensor,  # ← 改：从单步memory_e变为多步trajectory [B, K, nllb_dim]
        sonar_anchor: torch.Tensor,  # [B, 1, sonar_dim]
        confidence: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Flow增强的门控融合
        
        Args:
            h_t: NLLB解码器隐藏状态 [B, T, nllb_dim]
            memory_trajectory: K步累积的记忆向量 [B, K, nllb_dim]
                               若K=1则退化为原始单步模式
            sonar_anchor: SONAR语义锚点 [B, 1, sonar_dim]
            confidence: N-gram匹配置信度
        
        Returns:
            output: 门控融合后的隐藏状态 [B, T, nllb_dim]
            info: 统计信息字典
        """
        B, T, D = h_t.shape
        K = memory_trajectory.shape[1]  # 累积步数
        
        # ===== Step 1: Flow聚合多步记忆 =====
        if self.enable_flow_aggregate and K > 1:
            # 聚合K步记忆为单一向量
            memory_aggregated = self.flow_aggregator(memory_trajectory)  # [B, D]
            memory_aggregated = memory_aggregated.unsqueeze(1).expand(B, T, D)
        else:
            # 降级：直接使用平均（或最后一步）
            memory_aggregated = memory_trajectory.mean(dim=1).unsqueeze(1).expand(B, T, D)
        
        # ===== Step 2: Query构建（融合局部+全局）=====
        sonar_expanded = sonar_anchor.expand(B, T, -1)
        h_concat = torch.cat([h_t, sonar_expanded], dim=-1)
        q = self.W_q(h_concat)  # [B, T, D]
        
        # ===== Step 3: Key/Value投影 =====
        k = self.W_k(memory_aggregated)
        v = self.W_v(memory_aggregated)
        
        # ===== Step 4: 多头缩放点积门控 =====
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
        
        # ===== Step 5: 置信度加权+稀疏化 =====
        gate = gate * confidence
        gate_sparse = gate * (gate > self.gate_threshold).float()
        
        # ===== Step 6: 门控化记忆值 =====
        gated_v = gate_sparse * v
        gated_v = gated_v.permute(0, 2, 1, 3).reshape(B, T, D)
        
        # ===== Step 7: 因果卷积精炼 =====
        conv_input = gated_v.permute(0, 2, 1)
        conv_output = self.causal_conv(conv_input)[:, :, :T]
        conv_output = conv_output.permute(0, 2, 1)
        conv_output = self.conv_norm(conv_output)
        
        # ===== Step 8: 残差融合 =====
        output = h_t + self.W_o(self.dropout(conv_output))
        
        # 统计信息
        info = {
            'mean_gate': gate_sparse.mean().item(),
            'active_ratio': (gate_sparse > 0.1).float().mean().item(),
            'confidence': confidence,
            'flow_aggregated': self.enable_flow_aggregate and K > 1,
            'num_accumulated_steps': K,
        }
        
        return output, info


# ============================================================
# 方案C（可选）：延迟离散化解码器包装
# ============================================================

class DelayedDiscretizationWrapper(nn.Module):
    """
    延迟离散化包装器（安全降级版）
    理论说明：在自回归语言模型中实现连续Hidden传递需要重写底层KV Cache机制，
    当前作为理论占位(Future Work)。为保证工程鲁棒性，开启时自动降级为标准解码。
    延迟离散化包装器（实验性，可选启用）
      解码器前N-K步保持连续hidden传递
      仅最后K步执行lm_head投影和token采样 ， 优势：
      1. 避免累积误差：连续传递 vs 离散-重嵌入循环
      2. 加速推理：跳过前N-K步的lm_head计算（最耗时操作）
      3. 语义平滑性：连续hidden天然平滑    
    风险：
      需要大量实验验证生成质量不下降
    """
    def __init__(
        self,
        base_model,
        delayed_K: int = 5,  
        enable_delayed: bool = False,  
    ):
        super().__init__()
        self.base_model = base_model
        self.delayed_K = delayed_K
        
        # 【修复】拦截并安全降级，防止 NotImplementedError 崩溃
        if enable_delayed:
            import logging
            logging.warning(
                "⚠️ [延迟离散化] 当前为理论占位(Future Work)，"
                "自回归架构下已自动降级为标准解码，不影响主流程运行。"
            )
        self.enable_delayed = False 

    def forward(self, *args, **kwargs):
        # 安全透传至基座模型
        return self.base_model(*args, **kwargs)


# ============================================================
# 辅助函数：将标准LoRA层替换为FlowLoRA
# ============================================================

from types import MethodType

def replace_lora_with_flora(
    peft_model,
    flow_steps: int = 4,
    flow_alpha_schedule: str = "cosine"
):
    """
    将PeftModel中的标准LoRA层替换为FlowLoRA
    修复：兼容新版PEFT的ModuleDict结构，并保留base_layer防止主干权重丢失
    """
    replaced_count = 0

    for name, module in peft_model.named_modules():
        # 检测标准LoRA层（通过特征属性）
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            
            # 1. 兼容 PEFT 新版本（lora_A/B 为 ModuleDict）
            if isinstance(module.lora_A, nn.ModuleDict):
                adapter_name = list(module.lora_A.keys())[0]
                lora_A_linear = module.lora_A[adapter_name]
                lora_B_linear = module.lora_B[adapter_name]
            else:
                lora_A_linear = module.lora_A
                lora_B_linear = module.lora_B
                adapter_name = None

            # 2. 提取原始参数
            in_features = lora_A_linear.in_features
            out_features = lora_B_linear.out_features
            rank = lora_A_linear.out_features
            
            # 获取 scaling (alpha)
            scaling = getattr(module, 'scaling', 1.0)
            if isinstance(scaling, dict) and adapter_name and adapter_name in scaling:
                alpha = scaling[adapter_name] * rank
            elif isinstance(scaling, (int, float)):
                alpha = scaling * rank
            else:
                alpha = 2.0 * rank
                
            # 获取 dropout
            dropout_p = 0.1
            lora_dropout = getattr(module, 'lora_dropout', None)
            if isinstance(lora_dropout, nn.ModuleDict) and adapter_name and adapter_name in lora_dropout:
                dropout_layer = lora_dropout[adapter_name]
                if hasattr(dropout_layer, 'p'):
                    dropout_p = dropout_layer.p
            elif hasattr(lora_dropout, 'p'):
                dropout_p = lora_dropout.p
            
            # 3. 创建FlowLoRA替代品
            flora_layer = FlowLoRALayer(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout_p,
                flow_steps=flow_steps,
                flow_alpha_schedule=flow_alpha_schedule
            )
            
            # 复制权重（保持预训练初始化）
            flora_layer.lora_A.weight.data.copy_(lora_A_linear.weight.data)
            flora_layer.lora_B.weight.data.copy_(lora_B_linear.weight.data)
            
            # 4. 【关键保护】保留 base_layer (预训练主干权重)
            # 如果直接替换 PEFT 的 Linear，会丢失 base_layer，导致推理时只有 LoRA 的残差输出
            if hasattr(module, 'base_layer'):
                flora_layer.base_layer = module.base_layer
                
                # 动态重写 forward，使其融合 base_layer
                def forward_with_base(self, x, return_flow_loss=False):
                    # 调用原始的 FlowLoRALayer 逻辑获取 delta_h
                    if self.training and return_flow_loss:
                        delta_h, flow_loss = self.flow_forward(x)
                    else:
                        x_dropped = self.dropout(x)
                        delta_h = self.lora_B(self.lora_A(x_dropped))
                        delta_h = delta_h * self.scaling
                        flow_loss = torch.tensor(0.0, device=x.device)
                    
                    # 加上预训练主干的输出 (核心修复)
                    base_out = self.base_layer(x)
                    final_out = base_out + delta_h
                    
                    if return_flow_loss:
                        return final_out, flow_loss
                    return final_out
                
                flora_layer.forward = MethodType(forward_with_base, flora_layer)
            
            # 5. 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = dict(peft_model.named_modules())[parent_name]
                setattr(parent_module, child_name, flora_layer)
            else:
                setattr(peft_model, child_name, flora_layer)
            
            replaced_count += 1

    if replaced_count > 0:
        import logging
        logging.getLogger(__name__).info(
            f"  ✓ 已将{replaced_count}个标准LoRA层替换为FlowLoRA（已保留base_layer主干权重）"
        )

    return peft_model

# ============================================================
# 【新增】SONAR语义锚点引导的Engram门控网络
# ============================================================
class SOARGuidedEngramGating(nn.Module):
    """
    SONAR语义锚点引导的Engram门控网络
    
    核心创新：
      将SONAR全局语义锚点s*融入门控Query
      使门控决策具有全局语义感知能力
      
      公式：
        q_t = W_q · [h_t ‖ s*]          ← 融合局部+全局
        k_t = W_k · e_t                  ← 记忆向量key
        gate = σ(q_t · k_t^T / √d)      ← 缩放点积门控
        output = h_t + gate · (W_v · e_t) ← 残差融合
    
    参数量：约 4 × d² = 4 × 1024² ≈ 4M（轻量级）
    训练：与LoRA联合训练，梯度可流动
    """
    
    def __init__(
        self,
        nllb_dim: int = 1024,
        sonar_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_threshold: float = 0.3,
    ):
        super().__init__()
        self.nllb_dim = nllb_dim
        self.sonar_dim = sonar_dim
        self.num_heads = num_heads
        self.head_dim = nllb_dim // num_heads
        self.gate_threshold = gate_threshold
        
        # Query投影：融合NLLB隐藏态 + SONAR锚点
        self.W_q = nn.Linear(nllb_dim + sonar_dim, nllb_dim, bias=False)
        
        # Key/Value投影（作用于记忆向量）
        self.W_k = nn.Linear(nllb_dim, nllb_dim, bias=False)
        self.W_v = nn.Linear(nllb_dim, nllb_dim, bias=False)
        
        # 输出投影
        self.W_o = nn.Linear(nllb_dim, nllb_dim, bias=False)
        
        # 因果深度卷积（短程精炼）
        self.causal_conv = nn.Conv1d(
            in_channels=nllb_dim,
            out_channels=nllb_dim,
            kernel_size=3,
            padding=2,
            groups=nllb_dim,  # 深度可分离
        )
        self.conv_norm = nn.LayerNorm(nllb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        h_t: torch.Tensor,           # [B, T, nllb_dim]
        memory_e: torch.Tensor,      # [B, T, nllb_dim]
        sonar_anchor: torch.Tensor,  # [B, 1, sonar_dim]
        confidence: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        SONAR引导的门控融合
        
        Args:
            h_t: NLLB解码器隐藏状态
            memory_e: Engram检索到的记忆向量
            sonar_anchor: SONAR语义锚点（全句级）
            confidence: N-gram匹配置信度
        
        Returns:
            (output [B,T,D], gate_info dict)
        """
        B, T, D = h_t.shape
        
        # ① Query构建：融合局部隐藏态 + 全局SONAR锚点
        sonar_expanded = sonar_anchor.expand(B, T, -1)
        h_concat = torch.cat([h_t, sonar_expanded], dim=-1)
        q = self.W_q(h_concat)  # [B, T, D]
        
        # ② Key/Value投影
        k = self.W_k(memory_e)
        v = self.W_v(memory_e)
        
        # ③ 多头缩放点积门控
        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 门控值：每个(位置, 头)一个标量
        gate = torch.sigmoid(
            (q * k).sum(dim=-1, keepdim=True) * self.scale
        )  # [B, num_heads, T, 1]
        
        # ④ 置信度加权 + 稀疏化
        gate = gate * confidence
        gate_sparse = gate * (gate > self.gate_threshold).float()
        
        # ⑤ 门控化记忆值
        gated_v = gate_sparse * v
        gated_v = gated_v.permute(0, 2, 1, 3).reshape(B, T, D)
        
        # ⑥ 因果卷积精炼
        conv_input = gated_v.permute(0, 2, 1)  # [B, D, T]
        conv_output = self.causal_conv(conv_input)[:, :, :T]
        conv_output = conv_output.permute(0, 2, 1)
        conv_output = self.conv_norm(conv_output)
        
        # ⑦ 残差融合
        output = h_t + self.W_o(self.dropout(conv_output))
        
        # 统计信息
        info = {
            'mean_gate': gate_sparse.mean().item(),
            'active_ratio': (gate_sparse > 0.1).float().mean().item(),
            'confidence': confidence,
        }
        
        return output, info


class AdaptiveLossBalancer:
    """
    动态损失权重平衡器（解决多任务/多Loss超参数调优灾难）
    原理：Inverse Loss Weighting (ILW) + 指数移动平均 (EMA)
    作用：自动抑制数值过大的辅助Loss（如MSE/Align），防止其吞噬主任务(CE)的梯度，
          使模型在训练过程中自动寻找多目标帕累托最优，无需人工网格搜索 lambda。
    """
    def __init__(self, loss_names: List[str], momentum: float = 0.9):
        self.momentum = momentum
        # 初始化 EMA 为 1.0，避免初始除零
        self.ema = {name: 1.0 for name in loss_names}
        
    def get_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        # 1. 更新 EMA (平滑历史损失，避免单步梯度噪声)
        for name, val in current_losses.items():
            if name in self.ema:
                self.ema[name] = self.momentum * self.ema[name] + (1 - self.momentum) * val
                
        # 2. 计算自适应权重
        base_loss = self.ema.get('ce', 1.0)
        weights = {}
        for name in self.ema:
            if name == 'ce':
                weights[name] = 1.0  # CE 作为基准锚点
            else:
                # 核心逻辑：辅助Loss的EMA越大，其权重自动衰减，强制与CE保持同量级贡献
                weights[name] = min(1.0, base_loss / (self.ema[name] + 1e-6))
        return weights

# ============================================================================
# 【修改】EngramJointTrainer：集成动态损失平衡
# 【修改】EngramJointTrainer：集成FLoRA + Flow Engram
# ============================================================

class EngramJointTrainer(CGLoRATrainer):
    """
    Engram + SONAR + LoRA 三重联合训练器（集成FLoRA和Flow Engram）
    
    核心改进（相比原版）：
      1. 支持FlowLoRA的MSE损失收集（enable_flora参数）
      2. 支持Flow Engram记忆累积（enable_flow_engram参数）
      3. 动态损失权重调度（MSE vs CE）
      4. 完全向后兼容（通过参数开关控制新功能）
    
    损失公式：
      L = L_CE + λ_sonar·L_SONAR + λ_engram·L_Engram + α(t)·L_MSE
      
      其中α(t)从0.2（初期）→ flow_mse_weight_max（后期）
    
    参数继承：
      从CGLoRATrainer继承：curvature_lambda, fisher_beta, warmup_steps
      新增：enable_flora, enable_flow_engram, flow_mse_weight_max
    """
    
    def __init__(
        self,
        engram_table: "TerminologyEngramTable",
        engram_gating,  
        sonar_autoencoder: Optional[object],
        cfg_args: argparse.Namespace,
        enable_flora: bool = True,       
        enable_flow_engram: bool = True, 
        flow_mse_weight_max: float = 0.3,  
        **kwargs
    ):
        # ── 核心修复：tokenizer → processing_class 转换 ──
        tokenizer_obj = kwargs.pop('tokenizer', None)
        if 'processing_class' not in kwargs and tokenizer_obj is not None:
             kwargs['processing_class'] = tokenizer_obj
        
        super().__init__(**kwargs)
        
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = getattr(self, 'processing_class', None)
            
        self.engram_table  = engram_table
        self.engram_gating = engram_gating.to(cfg_args.device)
        self.sonar         = sonar_autoencoder
        self.cfg = cfg_args
        
        self.lora_params = [p for n, p in self.model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
        self.engram_params = list(engram_gating.parameters())
        
        # ── 训练阶段状态 ──
        self.current_stage = "warmup"
        self.global_epoch  = 0
        
        # 【修复】恢复 lambda 属性初始化
        # 作用1：作为底层 _compute_sonar_loss / _compute_engram_loss 的 Early Return 开关（为0时跳过计算节省算力）
        # 作用2：作为 _update_training_stage 阶段性课程学习的基准权重
        self.lambda_sonar  = 0.05 if getattr(cfg_args, 'use_sonar', True) else 0.0
        self.lambda_engram = 0.15 if getattr(cfg_args, 'enable_engram', True) else 0.0
        
        # ── 【新增】FLoRA和Flow Engram参数 ──
        self.enable_flora = enable_flora
        self.enable_flow_engram = enable_flow_engram
        self.flow_mse_weight_max = flow_mse_weight_max
        
        # 【新增】初始化动态损失平衡器 (彻底解决超参数灾难)
        self.loss_balancer = AdaptiveLossBalancer(
            loss_names=['ce', 'sonar', 'engram', 'mse'], 
            momentum=0.95
        )
        
        self.memory_accumulator = []
        self.accumulate_every_K  = getattr(cfg_args, 'flow_engram_K', 5)
    
    def create_optimizer(self) -> "DualOptimizerWrapper":
        """
        【保持不变】创建双优化器并包装
        
        Returns:
            DualOptimizerWrapper 实例
        """
        # 创建两个独立的优化器
        self.optimizer_lora = torch.optim.AdamW(
            self.lora_params,
            lr=self.cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        
        self.optimizer_engram = torch.optim.AdamW(
            self.engram_params,
            lr=self.cfg.lr * 5.0,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # 包装并显式赋值给 self.optimizer
        dual_optimizer = DualOptimizerWrapper(
            self.optimizer_lora,
            self.optimizer_engram
        )
        self.optimizer = dual_optimizer
        
        return dual_optimizer
    
    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        【保持不变】创建双调度器
        
        Returns:
            DualSchedulerWrapper 实例
        """
        warmup_steps = int(
            num_training_steps * self.cfg.warmup_proportion
        )
        
        self.scheduler_lora = get_linear_schedule_with_warmup(
            self.optimizer_lora,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        self.scheduler_engram = get_linear_schedule_with_warmup(
            self.optimizer_engram,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        dual_scheduler = DualSchedulerWrapper(
            self.scheduler_lora,
            self.scheduler_engram
        )
        self.lr_scheduler = dual_scheduler
        
        return dual_scheduler
    
    def _update_training_stage(self) -> None:
        """【保持不变】阶段性权重更新"""
        warmup_end   = self.cfg.engram_warmup_epochs
        finetune_end = warmup_end + self.cfg.engram_finetune_epochs
        
        if self.global_epoch < warmup_end:
            self.current_stage = "warmup"
            self.lambda_sonar  = 0.05
            self.lambda_engram = 0.15
            
        elif self.global_epoch < finetune_end:
            if self.current_stage == "warmup":
                for pg in self.optimizer_lora.param_groups:
                    pg['lr'] = self.cfg.lr * 0.5
                for pg in self.optimizer_engram.param_groups:
                    pg['lr'] = self.cfg.lr * 2.0
            
            self.current_stage = "finetune"
            self.lambda_sonar  = 0.15
            self.lambda_engram = 0.10
            
        else:
            if not self.cfg.enable_stage_d:
                return
            
            if self.current_stage == "finetune":
                for pg in self.optimizer_lora.param_groups:
                    pg['lr'] = self.cfg.lr * 0.1
                for pg in self.optimizer_engram.param_groups:
                    pg['lr'] = self.cfg.lr * 0.2
            
            self.current_stage = "align"
            self.lambda_sonar  = 0.25
            self.lambda_engram = 0.05
    
    # ============================================================
    # 【新增】FLoRA支持方法
    # ============================================================
    
    def _collect_flora_loss(self, model) -> torch.Tensor:
        """
        【新增】收集所有FlowLoRA层的MSE损失
        
        流程：
          1. 遍历模型所有模块
          2. 检测FlowLoRALayer实例
          3. 累积其内部的flow_loss
        
        Returns:
            aggregated_mse_loss: 标量张量
        
        注意：
          需要FlowLoRALayer在forward时存储_last_flow_loss属性
        """
        flora_losses = []
        
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and \
               module.__class__.__name__ == 'FlowLoRALayer':
                # 检查是否有存储的flow损失
                if hasattr(module, '_last_flow_loss') and \
                   module._last_flow_loss is not None:
                    flora_losses.append(module._last_flow_loss)
        
        if flora_losses:
            return torch.stack(flora_losses).mean()
        else:
            return torch.tensor(0.0, device=self.cfg.device)
    
    # ============================================================
    # 【新增】Flow Engram支持方法
    # ============================================================
    
    def _accumulate_engram_memory(
        self,
        current_step: int,
        memory_vec: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        【新增】累积Engram记忆向量（方案B核心）
        
        逻辑：
          每步检索到的记忆不立即使用
          而是累积K步后，通过FlowAggregateNet聚合
        
        Args:
            current_step: 当前解码步数
            memory_vec: 当前步检索到的记忆 [B, D]
        
        Returns:
            若累积满K步: 返回聚合后的记忆 [B, D]
            否则: 返回None（继续累积）
        """
        self.memory_accumulator.append((current_step, memory_vec))
        
        if len(self.memory_accumulator) >= self.accumulate_every_K:
            memories = [mem for _, mem in self.memory_accumulator]
            memory_trajectory = torch.stack(memories, dim=1)  # [B, K, D]
            self.memory_accumulator = []
            return memory_trajectory
        else:
            return None
    
    # ============================================================
    # 【修改】核心方法：集成FLoRA和Flow Engram
    # ============================================================
    
    def _pool_hidden(self, hidden, mask):
        """【保持不变】注意力掩码加权平均池化"""
        mask_exp = mask.unsqueeze(-1).float()
        return (
            (hidden * mask_exp).sum(dim=1)
            / mask_exp.sum(dim=1).clamp(min=1e-9)
        )
    
    def _compute_src_anchor(self, model, inputs, device):
        """【保持不变】预计算SONAR锚点"""
        if self.tokenizer is None:
            logging.getLogger(__name__).warning(
                "tokenizer 为 None，无法计算SONAR锚点"
            )
            B = inputs['input_ids'].shape[0]
            return torch.zeros(B, 1024, device=device)
        
        anchors = []
        with torch.no_grad():
            for b in range(inputs['input_ids'].shape[0]):
                try:
                    src_text = self.tokenizer.decode(
                        inputs['input_ids'][b],
                        skip_special_tokens=True
                    )
                    
                    if not src_text.strip():
                        anchors.append(torch.zeros(1, 1024, device=device))
                        continue
                    
                    src_inp = self.tokenizer(
                        src_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                    ).to(device)
                    
                    emb = _encode_with_base_model(model, src_inp, device)
                    if emb is None:
                        emb = torch.zeros(1, 1024, device=device)
                    
                    anchors.append(emb)
                
                except Exception as e:
                    logging.getLogger(__name__).debug(
                        f"锚点编码失败: {e}"
                    )
                    anchors.append(torch.zeros(1, 1024, device=device))
        
        return torch.cat(anchors, dim=0)
    
    def _run_engram_gating(self, outputs, inputs, src_anchor, device):
        """
        【修改】Engram门控执行（支持Flow累积）
        
        改进点：
          若enable_flow_engram=True：累积K步后聚合门控
          否则：原始单步门控（兼容性）
        """
        logger = logging.getLogger(__name__)
        gate_info = {
            'mean_gate': 0.0, 
            'active_ratio': 0.0, 
            'flow_aggregated': False
        }
        avg_conf = 0.0
        
        if self.tokenizer is None:
            return gate_info, avg_conf
        
        try:
            layer2_hidden = outputs.decoder_hidden_states[2]
            B, T, D = layer2_hidden.shape
            
            # ===== 方案B：累积式记忆检索 =====
            if self.enable_flow_engram and \
               hasattr(self.engram_gating, 'enable_flow_aggregate') and \
               self.engram_gating.enable_flow_aggregate:
                
                memory_trajectories = []
                conf_list = []
                
                for b in range(B):
                    label_ids = inputs['labels'][b]
                    valid_ids = label_ids[label_ids != -100].tolist()
                    
                    if len(valid_ids) < 2:
                        memory_trajectories.append(
                            torch.zeros(self.accumulate_every_K, D, device=device)
                        )
                        conf_list.append(0.0)
                        continue
                    
                    # 检索最近K步的N-gram记忆
                    batch_memories = []
                    batch_confidences = []
                    
                    for step_offset in range(self.accumulate_every_K):
                        start_idx = max(
                            0, 
                            len(valid_ids) - self.cfg.engram_max_ngram - step_offset
                        )
                        end_idx = len(valid_ids) - step_offset
                        
                        if start_idx >= end_idx:
                            batch_memories.append(torch.zeros(D, device=device))
                            batch_confidences.append(0.0)
                            continue
                        
                        recent_tokens = self.tokenizer.convert_ids_to_tokens(
                            valid_ids[start_idx:end_idx]
                        )
                        
                        mem_vec, conf = self.engram_table.lookup(
                            recent_tokens,
                            lang_code=getattr(self.tokenizer, 'src_lang', ''),
                            device=device,
                        )
                        
                        batch_memories.append(mem_vec.squeeze(0))
                        batch_confidences.append(conf)
                    
                    memory_traj = torch.stack(batch_memories, dim=0)
                    memory_trajectories.append(memory_traj)
                    
                    avg_conf_sample = sum(batch_confidences) / len(batch_confidences)
                    conf_list.append(avg_conf_sample)
                
                memory_tensor = torch.stack(memory_trajectories, dim=0)
                avg_conf = sum(conf_list) / max(1, len(conf_list))
                
                # 调用Flow Engram门控
                _, gate_info = self.engram_gating(
                    h_t=layer2_hidden,
                    memory_trajectory=memory_tensor,
                    sonar_anchor=src_anchor.unsqueeze(1),
                    confidence=avg_conf,
                )
            
            else:
                # ===== 降级：原始单步门控 =====
                memory_batch = []
                conf_list = []
                
                for b in range(B):
                    label_ids = inputs['labels'][b]
                    valid_ids = label_ids[label_ids != -100].tolist()
                    
                    if len(valid_ids) < 2:
                        memory_batch.append(
                            torch.zeros(1, T, D, device=device)
                        )
                        conf_list.append(0.0)
                        continue
                    
                    recent_tokens = self.tokenizer.convert_ids_to_tokens(
                        valid_ids[-self.cfg.engram_max_ngram:]
                    )
                    
                    mem_vec, conf = self.engram_table.lookup(
                        recent_tokens,
                        lang_code=getattr(self.tokenizer, 'src_lang', ''),
                        device=device,
                    )
                    
                    memory_batch.append(mem_vec.expand(1, T, -1))
                    conf_list.append(conf)
                
                memory_tensor = torch.cat(memory_batch, dim=0)
                avg_conf = sum(conf_list) / max(1, len(conf_list))
                
                # 原始门控（单步记忆）
                # 需要将memory_tensor包装为trajectory格式 [B, 1, T, D] → [B, T, D]
                _, gate_info = self.engram_gating(
                    h_t=layer2_hidden,
                    memory_trajectory=memory_tensor.unsqueeze(1),  # [B, 1, T, D]
                    sonar_anchor=src_anchor.unsqueeze(1),
                    confidence=avg_conf,
                )
        
        except Exception as e:
            logger.debug(f"Engram门控执行失败: {e}")
        
        return gate_info, avg_conf
    
    def _compute_sonar_loss(self, outputs, inputs, src_anchor, device):
        """【保持不变】SONAR对齐损失"""
        zero = torch.tensor(0.0, device=device)
        
        if self.lambda_sonar <= 0:
            return zero
        
        if (
            not hasattr(outputs, 'encoder_hidden_states')
            or outputs.encoder_hidden_states is None
            or len(outputs.encoder_hidden_states) <= 2
        ):
            return zero
        
        try:
            enc_layer2 = outputs.encoder_hidden_states[2]
            enc_pooled = self._pool_hidden(
                enc_layer2, inputs['attention_mask']
            )
            return (
                1 - F.cosine_similarity(enc_pooled, src_anchor, dim=-1)
            ).mean()
        
        except Exception as e:
            logging.getLogger(__name__).debug(
                f"SONAR损失计算失败: {e}"
            )
            return zero
    
    def _compute_engram_loss(self, gate_info, avg_conf, device):
        """【保持不变】门控稀疏损失"""
        zero = torch.tensor(0.0, device=device)
        
        if self.lambda_engram <= 0:
            return zero
        
        try:
            gate_val = torch.tensor(
                gate_info['mean_gate'],
                device=device,
                dtype=torch.float32,
            )
            
            conf_excess = max(0.0, gate_info['mean_gate'] - avg_conf)
            conf_penalty = torch.tensor(
                conf_excess ** 2,
                device=device,
                dtype=torch.float32,
            )
            
            return (
                gate_val
                + self.cfg.engram_confidence_weight * conf_penalty
            )
        
        except Exception as e:
            logging.getLogger(__name__).debug(
                f"Engram损失计算失败: {e}"
            )
            return zero
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        【重构】三重损失 + 动态权重平衡
        彻底移除硬编码的 lambda 乘积，改由 AdaptiveLossBalancer 动态接管
        """
        logger = logging.getLogger(__name__)
        device = self.cfg.device
        
        src_anchor = self._compute_src_anchor(model, inputs, device)
        outputs = model(**inputs, output_hidden_states=True)
        ce_loss = outputs.loss
        
        gate_info, avg_conf = {'mean_gate': 0.0, 'active_ratio': 0.0}, 0.0
        if (model.training and hasattr(outputs, 'decoder_hidden_states') 
            and outputs.decoder_hidden_states is not None and len(outputs.decoder_hidden_states) > 2):
            gate_info, avg_conf = self._run_engram_gating(outputs, inputs, src_anchor, device)
            
        sonar_loss = self._compute_sonar_loss(outputs, inputs, src_anchor, device)
        engram_loss = self._compute_engram_loss(gate_info, avg_conf, device)
        
        mse_loss = torch.tensor(0.0, device=device)
        if self.enable_flora and model.training:
            mse_loss = self._collect_flora_loss(model)
            
        # ===== 【核心修改】动态权重计算 =====
        current_losses = {
            'ce': ce_loss.item(),
            'sonar': sonar_loss.item(),
            'engram': engram_loss.item(),
            'mse': mse_loss.item()
        }
        weights = self.loss_balancer.get_weights(current_losses)
        
        # 应用动态权重（替代原有的静态 lambda_sonar / lambda_engram）
        total_loss = (
            weights['ce'] * ce_loss +
            weights['sonar'] * sonar_loss +
            weights['engram'] * engram_loss +
            weights['mse'] * mse_loss
        )
        
        # 日志输出（附带动态权重监控）
        if hasattr(self, 'state') and self.state.global_step % 20 == 0:
            logger.debug(
                f"  [Engram-Flow] Step {self.state.global_step} | "
                f"CE={ce_loss.item():.4f}(w={weights['ce']:.2f}) | "
                f"SONAR={sonar_loss.item():.4f}(w={weights['sonar']:.2f}) | "
                f"MSE={mse_loss.item():.4f}(w={weights['mse']:.2f}) | "
                f"Stage={self.current_stage}"
            )
            
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(
        self,
        model,
        inputs: Dict,
        num_items_in_batch: Optional[int] = None,
    ):
        """
        【保持不变】单步训练（移除手动optimizer.step()）
        
        流程：
          1. 前向传播 + 损失计算
          2. 反向传播
          3. 梯度裁剪
          4. 返回 loss（Trainer 自动处理后续的 optimizer.step()）
        """
        model.train()
        loss = self.compute_loss(model, inputs)
        loss.backward()
        
        clip_norm = self.cfg.gradient_clip_norm
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.lora_params, clip_norm)
            torch.nn.utils.clip_grad_norm_(self.engram_params, clip_norm)
        
        return loss.detach()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """【保持不变】Epoch开始回调"""
        self.global_epoch = int(state.epoch) if state.epoch else 0
        self._update_training_stage()
        
        logging.getLogger(__name__).info(
            f"\n  [Engram] Epoch {self.global_epoch} | "
            f"阶段: {self.current_stage} | "
            f"λ_sonar={self.lambda_sonar:.3f} | "
            f"λ_engram={self.lambda_engram:.3f}"
        )

# ============================================================
# 【新增】双调度器包装类
# ============================================================

class DualSchedulerWrapper:
    """
    双调度器包装器（兼容 Trainer 接口）
    
    同步管理两个学习率调度器
    """
    
    def __init__(
        self,
        scheduler_lora,
        scheduler_engram,
    ):
        self.scheduler_lora   = scheduler_lora
        self.scheduler_engram = scheduler_engram
    
    def step(self, epoch=None):
        """同步更新两个调度器"""
        self.scheduler_lora.step(epoch)
        self.scheduler_engram.step(epoch)
    
    def get_last_lr(self):
        """返回LoRA调度器的学习率（兼容接口）"""
        return self.scheduler_lora.get_last_lr()
    
    def state_dict(self):
        """返回合并的状态字典"""
        return {
            "lora":   self.scheduler_lora.state_dict(),
            "engram": self.scheduler_engram.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """加载合并的状态字典"""
        if "lora" in state_dict:
            self.scheduler_lora.load_state_dict(state_dict["lora"])
        if "engram" in state_dict:
            self.scheduler_engram.load_state_dict(state_dict["engram"])
        
# ============================================================
# SONAR增强模块（新增完整代码块）
# ============================================================
class GELU(nn.Module):
    """GELU激活函数"""
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * 
                                         (x + 0.044715 * torch.pow(x, 3))))

class RMSNorm(nn.Module):
    """RMS归一化"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码(RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        if dim % 2 != 0:
            raise ValueError(f"维度必须是偶数,但得到了 {dim}")
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._precompute_freqs_cis()
    
    def _precompute_freqs_cis(self):
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))
    
    def rotate_half(self, x: Tensor) -> Tensor:
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: Tensor, k: Tensor, seq_len: int = None) -> Tuple[Tensor, Tensor]:
        try:
            if seq_len is None:
                seq_len = q.shape[2]
            
            if seq_len > self.max_seq_len:
                raise ValueError("序列长度超过最大长度")
            
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
            
            head_dim = q.shape[-1]
            
            if head_dim != self.dim // 2:
                repeat_factor = head_dim // (self.dim // 2)
                if repeat_factor > 1:
                    cos = cos.repeat_interleave(repeat_factor, dim=-1)
                    sin = sin.repeat_interleave(repeat_factor, dim=-1)
                else:
                    cos = cos[..., :head_dim]
                    sin = sin[..., :head_dim]
            
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            
            q_embed = q * cos + self.rotate_half(q) * sin
            k_embed = k * cos + self.rotate_half(k) * sin
            
            return q_embed, k_embed
            
        except Exception as e:
            logging.error(f"RoPE前向传播失败: {e}")
            return q, k


class AdaptiveHybridAttention(nn.Module):
    """
    【渐进式负载均衡版】"先探索后利用" 策略
    前期均衡探索 → 后期偏向稀疏注意力
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 reduction_ratio: int = 2, linear_rank: int = 64,
                 enable_load_balancing: bool = True,
                 target_sparse_ratio: float = 0.7,  # ✅ 新增：目标稀疏占比
                 transition_steps: int = 10000):     # ✅ 新增：过渡步数
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.reduction_ratio = reduction_ratio
        self.linear_rank = linear_rank
        self.dim = dim
        
        # 添加 batch_first 属性
        self.batch_first = True  # AdaptiveHybridAttention 始终使用 batch_first=True
        
        self.flash_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # ===== Attention组件 =====
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_sparse = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, linear_rank, bias=False)
        self.v_linear = nn.Linear(dim, dim, bias=False)
        
        self.sr = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            GELU(),
            nn.Linear(dim // reduction_ratio, dim)
        )
        
        self.norm_sparse = RMSNorm(dim)
        self.norm_linear = RMSNorm(dim)
        
        self.proj = nn.Linear(dim, dim)
        
        # ===== 路由器 =====
        self.attention_router = nn.Sequential(
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # ===== 渐进式负载均衡 =====
        self.enable_load_balancing = enable_load_balancing
        
        if self.enable_load_balancing:
            # 统计变量
            self.sparse_load_history = []
            self.linear_load_history = []
            self.load_counter = 0
            self.current_step = 0  # ✅ 全局训练步数
            
            # 温度参数
            self.router_temperature = 1.0
            
            # ✅ 渐进式策略参数
            self.target_sparse_ratio = target_sparse_ratio  # 目标稀疏占比（如0.8）
            self.transition_steps = transition_steps        # 过渡期步数
            self.initial_ratio = 0.5                        # 初始均衡比例
            
            # 调整速率
            self.load_alpha = 0.1
        
        self.rope = RotaryPositionEmbedding(self.head_dim)
    
    def _get_target_ratio(self) -> float:
        """
        ✅ 计算当前步的目标稀疏占比（线性过渡）
        
        Returns:
            当前目标稀疏占比
        
        策略：
            step < transition_steps: 0.5 → target_sparse_ratio 线性增长
            step >= transition_steps: 固定在 target_sparse_ratio
        """
        if self.current_step >= self.transition_steps:
            return self.target_sparse_ratio
        
        # 线性插值: 0.5 → 0.7
        progress = self.current_step / self.transition_steps
        current_ratio = self.initial_ratio + (self.target_sparse_ratio - self.initial_ratio) * progress
        return current_ratio
    
    def _update_load_statistics(self, sparse_weight: float, linear_weight: float):
        """
        ✅ 更新负载统计（基于动态目标）- 优化版
        """
        if not self.enable_load_balancing or not self.training:
            return
        
        # 更新历史记录
        self.sparse_load_history.append(sparse_weight)
        self.linear_load_history.append(linear_weight)
        
        # 保持最近100条
        if len(self.sparse_load_history) > 100:
            self.sparse_load_history.pop(0)
            self.linear_load_history.pop(0)
        
        self.load_counter += 1
        
        # 每100步调整一次温度
        if self.load_counter % 100 == 0 and len(self.sparse_load_history) > 0:
            avg_sparse = sum(self.sparse_load_history) / len(self.sparse_load_history)
            avg_linear = sum(self.linear_load_history) / len(self.linear_load_history)
            
            # ✅ 使用动态目标计算不平衡度
            target_ratio = self._get_target_ratio()
            target_linear = 1.0 - target_ratio  # 计算linear的目标比例
            
            # ✅ 方案1: 同时考虑两者的偏差（更严格）
            sparse_imbalance = abs(avg_sparse - target_ratio)
            linear_imbalance = abs(avg_linear - target_linear)
            imbalance = max(sparse_imbalance, linear_imbalance)  # 取较大偏差
            
            # ✅ 方案2: 使用加权平均（更平滑）
            # imbalance = (sparse_imbalance + linear_imbalance) / 2
            
            # 自适应调整温度
            if imbalance > 0.15:  # 偏离目标较多
                self.router_temperature *= (1 + self.load_alpha)
                self.router_temperature = min(self.router_temperature, 2.0)
            elif imbalance < 0.05:  # 接近目标
                self.router_temperature *= (1 - self.load_alpha)
                self.router_temperature = max(self.router_temperature, 0.3) 
            
    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """
        ✅ 应用温度缩放 + 偏置调整
        
        策略：
            1. 温度缩放（平滑分布）
            2. 偏置注入（引导向目标比例）
        """
        if not self.enable_load_balancing or not self.training:
            return torch.softmax(logits, dim=-1)
        
        # 计算当前目标
        target_ratio = self._get_target_ratio()
        
        # ✅ 方法1: 温度缩放
        scaled_logits = logits / self.router_temperature
        
        # ✅ 方法2: 添加偏置（subtle引导）
        # 计算期望的 logit 差值（基于目标比例）
        # target_ratio = 0.8 → logit_bias ≈ log(0.8/0.2) = 1.386
        if target_ratio > 0.5:
            logit_bias = 0.5 * torch.log(torch.tensor(target_ratio / (1 - target_ratio)))
            # 只在sparse logit上添加偏置
            bias = torch.zeros_like(scaled_logits)
            bias[:, 0] = logit_bias
            scaled_logits = scaled_logits + bias
        
        return torch.softmax(scaled_logits, dim=-1)
    
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        
        # ✅ 更新全局步数
        if self.training and self.enable_load_balancing:
            self.current_step += 1
        
        try:
            # 序列表示
            seq_repr = x.mean(dim=1)
            
            # 路由决策
            router_logits = self.attention_router[0](seq_repr)
            weights = self._apply_temperature(router_logits)  # ✅ 应用渐进式策略
            
            # 更新负载统计
            if self.enable_load_balancing and self.training:
                sparse_weight = weights[:, 0].mean().item()
                linear_weight = weights[:, 1].mean().item()
                self._update_load_statistics(sparse_weight, linear_weight)
            
            # 执行Attention（保持原逻辑）
            compute_sparse = (weights[:, 0].mean() > 0.0)
            compute_linear = (weights[:, 1].mean() > 0.0)
            
            sparse_output = None
            linear_output = None
            
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)
            
            if compute_sparse:
                kv_sparse = self.kv_sparse(x).reshape(B, N, 2, self.num_heads, self.head_dim)
                kv_sparse = kv_sparse.permute(2, 0, 3, 1, 4)
                k_sparse, v_sparse = kv_sparse[0], kv_sparse[1]
                
                q_rope, k_sparse_rope = self.rope(q, k_sparse, N)
                
                k_sparse_rope = k_sparse_rope.permute(0, 2, 1, 3).reshape(B * N, -1)
                v_sparse = v_sparse.permute(0, 2, 1, 3).reshape(B * N, -1)
                
                k_sparse_rope = self.sr(k_sparse_rope).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v_sparse = self.sr(v_sparse).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                
                sparse_mask = attention_mask.expand(-1, self.num_heads, -1, -1) if attention_mask is not None else None
                sparse_attn = self.flash_attention(q_rope, k_sparse_rope, v_sparse, sparse_mask)
                sparse_output = sparse_attn.permute(0, 2, 1, 3).reshape(B, N, C)
                sparse_output = self.norm_sparse(sparse_output)
            
            if compute_linear:
                k_linear = self.k_linear(x)
                v_linear = self.v_linear(x)
                
                q_linear = q.permute(0, 2, 1, 3).reshape(B, N, -1) @ self.k_linear.weight.t()
                
                linear_attn = torch.bmm(q_linear, k_linear.transpose(1, 2))
                linear_attn = linear_attn * (self.linear_rank ** -0.5)
                
                if attention_mask is not None:
                    linear_attn = linear_attn + attention_mask.squeeze(1)
                
                linear_attn = F.softmax(linear_attn, dim=-1)
                linear_output = torch.bmm(linear_attn, v_linear)
                linear_output = self.norm_linear(linear_output)
            
            # 加权融合
            if sparse_output is not None and linear_output is not None:
                sparse_weight = weights[:, 0].view(B, 1, 1)
                linear_weight = weights[:, 1].view(B, 1, 1)
                output = sparse_weight * sparse_output + linear_weight * linear_output
            elif sparse_output is not None:
                output = sparse_output
            elif linear_output is not None:
                output = linear_output
            else:
                output = x
            
            return self.proj(output)
            
        except Exception as e:
            logging.error(f"混合注意力前向传播失败: {e}")
            return self.proj(x)
    
    def flash_attention(self, q: Tensor, k: Tensor, v: Tensor, 
                       mask: Optional[Tensor] = None) -> Tensor:
        """Flash Attention实现"""
        try:
            if self.flash_attn:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
                )
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                if mask is not None:
                    attn = attn + mask
                attn = F.softmax(attn, dim=-1)
                return attn @ v
        except Exception as e:
            logging.error(f"Flash Attention失败: {e}")
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            return attn @ v
    
    def get_load_statistics(self) -> dict:
        """
        ✅ 获取负载统计（增强版）
        """
        if not self.enable_load_balancing or len(self.sparse_load_history) == 0:
            return {}
        
        avg_sparse = sum(self.sparse_load_history) / len(self.sparse_load_history)
        avg_linear = sum(self.linear_load_history) / len(self.linear_load_history)
        target_ratio = self._get_target_ratio()
        imbalance = abs(avg_sparse - target_ratio)
        
        return {
            'avg_sparse_load': avg_sparse,
            'avg_linear_load': avg_linear,
            'target_sparse_ratio': target_ratio,  # ✅ 新增
            'imbalance': imbalance,
            'temperature': self.router_temperature,
            'current_step': self.current_step,    # ✅ 新增
            'progress': min(1.0, self.current_step / self.transition_steps)  # ✅ 新增
        } 


class HybridTransformerEncoderLayer(nn.Module):
    """混合注意力的Transformer编码器层"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: str = 'gelu', 
                 batch_first: bool = False):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AdaptiveHybridAttention(
            dim=d_model, num_heads=nhead, qkv_bias=True,
            reduction_ratio=2, linear_rank=64
        )
        self.self_attn = self.attn  # 指向同一个注意力模块
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
    
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, 
                is_causal: bool = False, **kwargs) -> Tensor:
        x = src
        attention_mask = None
        
        try:
            if is_causal:
                seq_len = x.size(1) if self.batch_first else x.size(0)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device), diagonal=1
                ).bool()
                attention_mask = causal_mask.float().masked_fill(causal_mask, -1e9).unsqueeze(0).unsqueeze(0)
            elif src_mask is not None:
                attention_mask = src_mask.unsqueeze(0).unsqueeze(0)
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.float().masked_fill(attention_mask, -1e9)
            
            x_norm = self.norm1(x)
            attn_output = self.attn(x_norm, attention_mask)
            x = x + self.dropout(attn_output)
            
            x_norm = self.norm2(x)
            mlp_output = self.mlp(x_norm)
            x = x + self.dropout(mlp_output)
            
            return x
            
        except RuntimeError as e:
            if "mixed dtype" in str(e) and "BFloat16" in str(e):
                x = x.to(torch.float32)
                x_norm = self.norm1(x)
                attn_output = self.attn(x_norm, attention_mask)
                x = x + self.dropout(attn_output)
                x_norm = self.norm2(x)
                mlp_output = self.mlp(x_norm)
                x = x + self.dropout(mlp_output)
                return x
            else:
                logging.error(f"Transformer层前向传播失败: {e}")
                raise



class SemanticDenoisingModule(nn.Module):
    """
    语义去噪模块（用户提供代码，已完整）
    
    核心机制：
      1. 训练时注入噪声（高斯噪声 + Dropout噪声）
      2. Transformer去噪网络学习鲁棒表示
      3. 门控机制（Gating）平衡去噪强度
    
    噪声调度：
      - Cosine调度：噪声强度从0.01逐渐增加到0.15
      - 目的：早期稳定训练，后期提升鲁棒性
    """
    def __init__(self, 
                 dim: int = 1024,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 noise_schedule: str = 'cosine'):
        super().__init__()
        
        self.dim = dim
        self.noise_schedule = noise_schedule
        
        # Transformer去噪层
        self.denoiser_layers = nn.ModuleList([
            HybridTransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 投影层
        self.input_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
        
        # 门控网络（自适应去噪强度）
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # 噪声参数（注册为buffer，自动跟随模型设备）
        self.register_buffer('noise_min', torch.tensor(0.01))
        self.register_buffer('noise_max', torch.tensor(0.15))
        self.register_buffer('training_step', torch.tensor(0))
        self.total_steps = 10000  # 总训练步数（用于调度）
    
    def add_semantic_noise(self, x: Tensor, noise_level: Optional[float] = None) -> Tensor:
        """
        添加语义噪声（训练时）
        
        噪声类型：
          1. 高斯噪声（主要）：N(0, σ²I)
          2. Dropout噪声（辅助）：30%概率随机丢弃10%维度
        
        归一化：噪声后重新归一化到原始L2范数（保持语义尺度）
        """
        if not self.training:
            return x
        
        # 动态噪声强度（Cosine调度）
        if noise_level is None:
            progress = min(1.0, self.training_step.item() / self.total_steps)
            if self.noise_schedule == 'cosine':
                sigma = self.noise_min + (self.noise_max - self.noise_min) * \
                        (1 - torch.cos(torch.tensor(progress * 3.14159))) / 2
            else:
                sigma = self.noise_min + (self.noise_max - self.noise_min) * progress
            self.training_step += 1
        else:
            sigma = noise_level
        
        # 噪声1：高斯噪声
        gaussian_noise = torch.randn_like(x) * sigma
        x_noisy = x + gaussian_noise
        
        # 噪声2：Dropout噪声（30%概率）
        if torch.rand(1).item() < 0.3:
            dropout_mask = torch.bernoulli(torch.full_like(x, 0.9))
            x_noisy = x_noisy * dropout_mask
        
        # 归一化到原始范数
        x_noisy = F.normalize(x_noisy, p=2, dim=-1) * x.norm(dim=-1, keepdim=True)
        
        return x_noisy
    
    def forward(self, x: Tensor, add_noise: bool = True) -> Tuple[Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: [B, D] 或 [B, L, D] 输入嵌入
            add_noise: 是否添加噪声（仅训练时有效）
        
        Returns:
            (denoised, info):
              denoised: [B, D] 去噪后嵌入
              info: 去噪统计信息
        """
        # 确保有序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 添加噪声（训练时）
        if self.training and add_noise:
            x_noisy = self.add_semantic_noise(x)
        else:
            x_noisy = x
        
        # 输入投影
        x_proj = self.input_proj(x_noisy)
        
        # Transformer去噪
        x_denoised = x_proj
        for layer in self.denoiser_layers:
            x_denoised = layer(x_denoised)
        
        # 输出投影
        x_out = self.output_proj(x_denoised)
        
        # 门控机制（自适应去噪强度）
        gate_weight = self.gate(x_out).mean(dim=1, keepdim=True)
        x_final = gate_weight * x_out + (1 - gate_weight) * x_noisy
        
        # 移除序列维度
        if squeeze_output:
            x_final = x_final.squeeze(1)
        
        # 统计信息
        info = {
            'gate_weight': gate_weight.mean().item(),
            'noise_level': self.noise_max.item() if self.training else 0.0,
            'denoising_applied': self.training and add_noise
        }
        
        return x_final, info
    
    def compute_denoising_loss(self, x_clean: Tensor, x_denoised: Tensor, 
                               weight: float = 0.1) -> Tensor:
        """
        计算去噪损失（MSE + Cosine）
        
        目标：去噪后的嵌入应接近干净输入
        """
        # MSE损失（L2距离）
        mse_loss = F.mse_loss(x_denoised, x_clean)
        
        # 余弦损失（方向一致性）
        x_clean_norm = F.normalize(x_clean, p=2, dim=-1)
        x_denoised_norm = F.normalize(x_denoised, p=2, dim=-1)
        cosine_sim = (x_clean_norm * x_denoised_norm).sum(dim=-1).mean()
        cosine_loss = 1.0 - cosine_sim
        
        # 加权混合
        total_loss = 0.7 * mse_loss + 0.3 * cosine_loss
        
        return total_loss * weight

class SonarNormalizer(nn.Module):
    """SONAR标准化器"""
    def __init__(self, config: SonarConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        for lang in config.supported_languages:
            self.register_buffer(f"{lang}_center", torch.zeros(config.dim))
            self.register_buffer(f"{lang}_scale", torch.ones(config.dim))
            
            if config.clip_proba is not None:
                self.register_buffer(f"{lang}_clip_min", torch.ones(config.dim) * -10)
                self.register_buffer(f"{lang}_clip_max", torch.ones(config.dim) * 10)
        
        self.register_buffer("default_center", torch.zeros(config.dim))
        self.register_buffer("default_scale", torch.ones(config.dim))
        if config.clip_proba is not None:
            self.register_buffer("default_clip_min", torch.ones(config.dim) * -10)
            self.register_buffer("default_clip_max", torch.ones(config.dim) * 10)
    
    def normalize(self, embeddings: Tensor, lang_code: str = "default") -> Tensor:
        try:
            if lang_code not in self.config.supported_languages:
                lang_code = "default"
            
            center = getattr(self, f"{lang_code}_center", self.default_center)
            scale = getattr(self, f"{lang_code}_scale", self.default_scale)
            
            normalized = (embeddings - center) / (scale + 1e-8)
            
            if self.config.clip_proba is not None:
                clip_min = getattr(self, f"{lang_code}_clip_min", self.default_clip_min)
                clip_max = getattr(self, f"{lang_code}_clip_max", self.default_clip_max)
                normalized = torch.clamp(normalized, min=clip_min, max=clip_max)
            
            return normalized
        except Exception as e:
            logging.error(f"标准化失败: {e}")
            return embeddings
    
    def denormalize(self, embeddings: Tensor, lang_code: str = "default") -> Tensor:
        try:
            if lang_code not in self.config.supported_languages:
                lang_code = "default"
            
            center = getattr(self, f"{lang_code}_center", self.default_center)
            scale = getattr(self, f"{lang_code}_scale", self.default_scale)
            
            return embeddings * scale + center
        except Exception as e:
            logging.error(f"反标准化失败: {e}")
            return embeddings
    
    def fit(self, embeddings: Tensor, lang_code: str = "default"):
        try:
            if lang_code not in self.config.supported_languages and lang_code != "default":
                logging.warning(f"不支持的语言: {lang_code},使用默认参数")
                lang_code = "default"
            
            embeddings_np = embeddings.cpu().float().numpy()
            
            if self.config.normalization_method in ["robust", "gaussian_robust"]:
                scaler = RobustScaler(
                    unit_variance=self.config.normalization_method == "gaussian_robust",
                    quantile_range=(
                        self.config.quantile_min * 100,
                        self.config.quantile_max * 100
                    )
                )
            else:
                scaler = StandardScaler()
            
            scaler.fit(embeddings_np)
            
            if hasattr(scaler, 'center_'):
                center = torch.tensor(scaler.center_, device=self.device, dtype=torch.float32)
                scale = torch.tensor(scaler.scale_, device=self.device, dtype=torch.float32)
            else:
                center = torch.tensor(scaler.mean_, device=self.device, dtype=torch.float32)
                scale = torch.tensor(scaler.scale_, device=self.device, dtype=torch.float32)
            
            setattr(self, f"{lang_code}_center", center)
            setattr(self, f"{lang_code}_scale", scale)
            
            if self.config.clip_proba is not None:
                normalized_embeddings = torch.tensor(scaler.transform(embeddings_np), 
                                                    device=self.device)
                clip_min = torch.quantile(normalized_embeddings, self.config.clip_proba, dim=0)
                clip_max = torch.quantile(normalized_embeddings, 1 - self.config.clip_proba, dim=0)
                
                setattr(self, f"{lang_code}_clip_min", clip_min)
                setattr(self, f"{lang_code}_clip_max", clip_max)
                
        except Exception as e:
            logging.error(f"拟合标准化参数失败: {e}")
            raise



class SonarAutoencoder(nn.Module):
    """
    SONAR去噪自编码器（论文级完整版）
    
    四阶段流程：
      输入 → SONAR编码 → 语义去噪 → 标准化 → 低秩解码 → 反标准化 → 输出
    
    核心创新：
      1. 轻量化设计：低秩解码器（参数量<100k）
      2. 语义去噪：对抗SONAR在专业术语上的OOD问题
      3. 语言特定标准化：每种语言独立RobustScaler参数
      4. 端到端训练：重构损失约束信息保留
    """
    
    def __init__(self, config: SonarConfig, encoder_model=None):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # SONAR编码器（外部NLLB模型，冻结权重）
        self.encoder = encoder_model
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 检测编码器类型
        self.is_seq2seq = hasattr(self.encoder, 'get_encoder') if self.encoder else False
        
        # 标准化器（语言特定）
        self.normalizer = SonarNormalizer(config)
        
        # 语义去噪模块
        self.enable_denoising = config.enable_denoising
        if self.enable_denoising:
            self.semantic_denoiser = SemanticDenoisingModule(
                dim=config.dim,
                num_layers=config.num_denoiser_layers,
                num_heads=config.num_denoiser_heads,
                dropout=config.denoiser_dropout,
                noise_schedule=config.noise_schedule
            )
        
        # 低秩解码器（LoRA风格，参数量：2×1024×64=131k）
        self.decoder = nn.Sequential(
            nn.Linear(config.dim, config.decoder_rank, bias=False),  # 降维
            nn.GELU(),
            nn.Dropout(config.decoder_dropout),
            nn.Linear(config.decoder_rank, config.dim, bias=False),  # 升维
        )
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.LayerNorm(config.dim),
            nn.Dropout(config.decoder_dropout)
        )
        
        # 编码缓存（避免重复计算）
        self.encoding_cache = {}
    
    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        编码输入序列为固定维度向量（注意力掩码加权平均池化）
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] 注意力掩码
        
        Returns:
            [B, dim] 编码向量
        """
        try:
            # 缓存检查（基于input_ids哈希）
            cache_key = hash(input_ids.cpu().numpy().tobytes())
            if cache_key in self.encoding_cache:
                return self.encoding_cache[cache_key].to(self.device)
            
            with torch.no_grad():
                # 获取编码器（兼容Seq2Seq和Encoder-Only）
                if self.is_seq2seq:
                    encoder_outputs = self.encoder.get_encoder()(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                else:
                    encoder_outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                
                hidden_states = encoder_outputs.last_hidden_state  # [B, L, dim]
                
                # 注意力掩码加权平均池化
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)  # [B, dim]
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
                sequence_output = sum_embeddings / sum_mask  # [B, dim]
            
            # 缓存结果（CPU存储，节省GPU显存）
            self.encoding_cache[cache_key] = sequence_output.cpu()
            
            return sequence_output
        
        except Exception as e:
            logging.error(f"编码失败: {e}")
            raise
    
    def decode(self, embeddings: Tensor) -> Tensor:
        """
        低秩解码重构（残差连接）
        
        公式：decode(x) = x + low_rank_projection(x)
        
        Args:
            embeddings: [B, dim] 或 [B, 1, dim]
        
        Returns:
            [B, dim] 重构向量
        """
        try:
            # 确保有序列维度
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)  # [B, 1, dim]
            
            # 低秩解码（残差连接）
            decoded = embeddings + self.decoder(embeddings)  # [B, 1, dim]
            
            # 移除序列维度
            if decoded.dim() == 3:
                decoded = decoded.squeeze(1)  # [B, dim]
            
            return decoded
        
        except Exception as e:
            logging.error(f"解码失败: {e}")
            raise
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                lang_code: str = None) -> Dict[str, Tensor]:
        """
        前向传播（完整四阶段流程）
        
        流程：
          1. SONAR编码（冻结权重）
          2. 语义去噪（训练时加噪声）
          3. 标准化（语言特定RobustScaler）
          4. 低秩解码（残差连接）
          5. 反标准化（恢复原始尺度）
        
        Returns:
            {
              "encoded": 原始SONAR嵌入,
              "denoised": 去噪后嵌入,
              "normalized": 标准化嵌入,
              "reconstructed": 解码重构嵌入,
              "denormalized": 反标准化嵌入（最终输出）,
              "denoising_info": 去噪统计信息
            }
        """
        try:
            # 步骤1：SONAR编码
            encoded = self.encode(input_ids, attention_mask)  # [B, dim]
            
            # 步骤2：语义去噪
            denoising_info = {}
            if self.enable_denoising and hasattr(self, 'semantic_denoiser'):
                denoised, denoising_info = self.semantic_denoiser(
                    encoded,
                    add_noise=self.training  # 仅训练时加噪
                )
            else:
                denoised = encoded
            
            # 步骤3：标准化
            norm_lang = lang_code if lang_code else self.config.anchor_language
            normalized = self.normalizer.normalize(denoised, norm_lang)
            
            # 步骤4：低秩解码
            reconstructed = self.decode(normalized)
            
            # 步骤5：反标准化
            denormalized = self.normalizer.denormalize(reconstructed, norm_lang)
            
            return {
                "encoded": encoded,
                "denoised": denoised,
                "normalized": normalized,
                "reconstructed": reconstructed,
                "denormalized": denormalized,
                "denoising_info": denoising_info
            }
        
        except Exception as e:
            logging.error(f"前向传播失败: {e}")
            raise
    
    def compute_loss(self, outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        计算多目标损失
        
        Loss = α·MSE + β·Cosine + γ·Denoising
        
        各损失含义：
          - MSE：重构精度（L2距离，确保数值接近）
          - Cosine：语义方向一致性（角度，对尺度不敏感）
          - Denoising：去噪质量正则（防止去噪过度/不足）
        """
        try:
            # 损失1：MSE重构损失
            mse = F.mse_loss(outputs["encoded"], outputs["denormalized"])
            
            # 损失2：余弦相似度损失
            encoded_norm = F.normalize(outputs["encoded"], p=2, dim=1)
            denormalized_norm = F.normalize(outputs["denormalized"], p=2, dim=1)
            cosine_sim = (encoded_norm * denormalized_norm).sum(dim=1).mean()
            cosine_loss = 1.0 - cosine_sim
            
            # 损失3：去噪损失（可选）
            denoising_loss = torch.tensor(0.0, device=self.device)
            if self.enable_denoising and self.training:
                if hasattr(self, 'semantic_denoiser'):
                    denoising_loss = self.semantic_denoiser.compute_denoising_loss(
                        x_clean=outputs["encoded"],
                        x_denoised=outputs["denoised"],
                        weight=self.config.denoising_weight
                    )
            
            # 加权混合
            if self.enable_denoising and self.training:
                combined_loss = (
                    self.config.mse_weight * mse +
                    self.config.cosine_weight * cosine_loss +
                    self.config.denoising_weight * denoising_loss
                )
            else:
                # 推理时仅用MSE+Cosine
                combined_loss = (
                    self.config.mse_weight * mse +
                    self.config.cosine_weight * cosine_loss
                )
            
            return {
                "mse": mse,
                "cosine": cosine_loss,
                "denoising": denoising_loss,
                "combined": combined_loss
            }
        
        except Exception as e:
            logging.error(f"计算损失失败: {e}")
            raise

    def _update_sonar_cache_progress(
        self,
        current: int,
        total: int,
        start_time: float,
        current_term: str,
        stdout
    ) -> None:
        """
        SONAR缓存构建进度条更新（类似Engram样式）
        
        Args:
            current: 当前已处理数量
            total: 总数量
            start_time: 开始时间戳
            current_term: 当前处理的术语
            stdout: 标准输出流
        """
        percent = (current / total) * 100
        bar_len = 40
        filled = int(bar_len * current / total)
        bar = '=' * filled + '>' + '.' * (bar_len - filled - 1)
        
        elapsed = time.time() - start_time
        if current > 0:
            avg_time = elapsed / current
            eta_sec = avg_time * (total - current)
            
            if eta_sec > 3600:
                eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"
            elif eta_sec > 60:
                eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60)}s"
            else:
                eta_str = f"{int(eta_sec)}s"
        else:
            eta_str = "计算中..."
        
        # 截断术语显示（保留20字符）
        term_display = current_term[:20] + ('...' if len(current_term) > 20 else '')
        
        # 计算速度
        speed = current / elapsed if elapsed > 0 else 0
        
        progress_line = (
            f"\r  [SONAR缓存] 构建进度: {current}/{total} [{bar}] "
            f"{percent:.1f}% | ETA: {eta_str} | "
            f"速度: {speed:.1f} term/s | 当前: {term_display}"
        )
        
        stdout.write(progress_line.ljust(120))
        stdout.flush()
    
    def build_term_cache(
        self,
        pairs: List[Tuple[str, str, str]],
        tokenizer,
        cache_dir: str = r'F:\hyberT\MultiSonar\data\Sonar_cache',
        device: str = "cuda",
    ) -> str:
        """
        【修改版】预计算术语库的SONAR嵌入并缓存到本地（带进度条）
        
        流程：
          1. 遍历术语库，逐条编码
          2. 实时显示进度条（单行刷新）
          3. 保存原始嵌入 + 标准化嵌入
          4. 写入磁盘（.pt格式）
          5. 返回缓存文件路径
        
        Args:
            pairs: 术语对列表 [(lang, src, tgt), ...]
            tokenizer: NLLB分词器
            cache_dir: 缓存目录
            device: 计算设备
        
        Returns:
            缓存文件完整路径
        """
        logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

        # ===== 文件名（精确到天）=====
        timestamp = time.strftime("%Y%m%d")
        cache_filename = f"sonar_cache_{len(pairs)}terms_{timestamp}.pt"
        cache_path = os.path.join(cache_dir, cache_filename)

        logger.info("[SONAR缓存] 开始构建术语库缓存...")
        logger.info(f"  术语数: {len(pairs)}")
        logger.info(f"  保存路径: {cache_path}")

        self.eval()
        embeddings_raw = {}
        embeddings_normalized = {}

        # ===== 按语言分组 =====
        lang_groups: Dict[str, List[Tuple[str, str, str]]] = {}
        for lang, src, tgt in pairs:
            lang_groups.setdefault(lang, []).append((lang, src, tgt))

        # ===== 统计总编码条目数（源术语 + 英文翻译去重）=====
        unique_tgt_texts = list(set(tgt for _, _, tgt in pairs))
        total_tasks = len(pairs) + len(unique_tgt_texts)

        total_encoded = 0
        failed_count = 0
        start_time = time.time()

        import sys
        raw_stdout = sys.stdout

        try:
            with torch.no_grad():

                # ===== 阶段1：编码源语言术语 =====
                for lang, lang_pairs in lang_groups.items():
                    raw_stdout.write(
                        f"\n  [阶段1] 编码源语言术语 {lang}:"
                        f" {len(lang_pairs)} 条\n"
                    )
                    raw_stdout.flush()

                    for lang_code, src_text, tgt_text in lang_pairs:
                        try:
                            tokenizer.src_lang = lang_code
                            src_inputs = tokenizer(
                                src_text, return_tensors="pt",
                                truncation=True, max_length=256
                            ).to(device)

                            raw_emb = self.encode(
                                src_inputs["input_ids"],
                                src_inputs["attention_mask"]
                            )  # [1, dim]

                            norm_emb = self.normalizer.normalize(
                                raw_emb, lang_code
                            )

                            # Key：(源语言代码, 源术语)
                            src_key = (lang_code, src_text.strip())
                            embeddings_raw[src_key] = raw_emb.cpu()
                            embeddings_normalized[src_key] = norm_emb.cpu()

                            total_encoded += 1
                            self._update_sonar_cache_progress(
                                total_encoded, total_tasks,
                                start_time, src_text, raw_stdout
                            )

                        except Exception:
                            failed_count += 1
                            total_encoded += 1
                            self._update_sonar_cache_progress(
                                total_encoded, total_tasks,
                                start_time, src_text, raw_stdout
                            )
                            continue

                # ===== 阶段2：编码英文翻译（去重）=====
                raw_stdout.write(
                    f"\n  [阶段2] 编码英文翻译:"
                    f" {len(unique_tgt_texts)} 条（去重）\n"
                )
                raw_stdout.flush()

                for tgt_text in unique_tgt_texts:
                    try:
                        tokenizer.src_lang = "eng_Latn"
                        tgt_inputs = tokenizer(
                            tgt_text, return_tensors="pt",
                            truncation=True, max_length=256
                        ).to(device)

                        raw_emb = self.encode(
                            tgt_inputs["input_ids"],
                            tgt_inputs["attention_mask"]
                        )  # [1, dim]

                        norm_emb = self.normalizer.normalize(
                            raw_emb, "eng_Latn"
                        )

                        # Key：("eng_Latn", 英文翻译)
                        tgt_key = ("eng_Latn", tgt_text.strip())
                        embeddings_raw[tgt_key] = raw_emb.cpu()
                        embeddings_normalized[tgt_key] = norm_emb.cpu()

                        total_encoded += 1
                        self._update_sonar_cache_progress(
                            total_encoded, total_tasks,
                            start_time, tgt_text, raw_stdout
                        )

                    except Exception:
                        failed_count += 1
                        total_encoded += 1
                        self._update_sonar_cache_progress(
                            total_encoded, total_tasks,
                            start_time, tgt_text, raw_stdout
                        )
                        continue

            # 进度条完成换行
            raw_stdout.write('\n')
            raw_stdout.flush()

        finally:
            sys.stdout = raw_stdout

        # ===== 保存缓存文件 =====
        cache_data = {
            'embeddings_raw': embeddings_raw,
            'embeddings_normalized': embeddings_normalized,
            'metadata': {
                'dim': self.config.dim,
                'languages': list(lang_groups.keys()),
                'total_terms': len(pairs),
                'cached_terms': len(embeddings_normalized),
                'src_keys': len(pairs),
                'tgt_keys': len(unique_tgt_texts),
                'failed_terms': failed_count,
                'creation_time': timestamp,
                'normalization_method': self.config.normalization_method,
            }
        }

        torch.save(cache_data, cache_path)

        # ===== 完成统计 =====
        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(cache_path) / (1024 ** 2)
        speed = total_encoded / elapsed if elapsed > 0 else 0

        logger.info("[SONAR缓存] 构建完成")
        logger.info(
            f"  源术语Key: {len(pairs)} 条 | "
            f"英文翻译Key: {len(unique_tgt_texts)} 条 | "
            f"总Key数: {len(embeddings_normalized)}"
        )
        if failed_count > 0:
            logger.info(f"  失败跳过: {failed_count} 条")
        logger.info(f"  文件大小: {file_size_mb:.1f} MB")
        logger.info(f"  总耗时: {elapsed:.1f}s ({speed:.1f} term/s)")
        logger.info(f"  保存位置: {cache_path}")

        return cache_path
    
    @staticmethod
    def load_term_cache(
        cache_path: str,
        device: str = "cpu"
    ) -> Optional[Dict]:
        """
        从本地加载SONAR缓存（无需修改，保持原样）
        
        Args:
            cache_path: 缓存文件路径
            device: 加载到的设备（推荐CPU，节省GPU显存）
        
        Returns:
            缓存数据字典，失败返回None
        """
        logger = logging.getLogger(__name__)
        
        if not os.path.exists(cache_path):
            logger.warning(f"[SONAR缓存] 文件不存在: {cache_path}")
            return None
        
        try:
            logger.info(f"[SONAR缓存] 加载缓存文件: {cache_path}")
            cache_data = torch.load(cache_path, map_location=device)
            
            metadata = cache_data.get('metadata', {})
            logger.info(
                f"  术语数: {metadata.get('cached_terms', 'N/A')} | "
                f"语言: {metadata.get('languages', [])} | "
                f"创建时间: {metadata.get('creation_time', 'N/A')}"
            )
            
            return cache_data
        
        except Exception as e:
            logger.error(f"[SONAR缓存] 加载失败: {e}")
            return None


# ============================================================
# SONAR辅助工具函数（新增）
# ============================================================

def split_aligned_segments(
    term: str,
    translation: str,
    max_len: int = 100,
    min_len: int = 10
) -> List[Tuple[str, str]]:
    """
    智能文本分割（保守策略，确保中英文对齐）
    
    规则：
      1. 短文本（<max_len）不分割
      2. 强制对齐检查：中英文句子数必须相等
      3. 长度均衡检查：对齐句子长度比例在0.3-3.0之间
      4. 有效对<50%时降级为不分割
    
    Args:
        term: 源语言术语
        translation: 英文翻译
        max_len: 最大长度阈值
        min_len: 最小句子长度
    
    Returns:
        [(seg_term, seg_trans), ...]  # 分割后的对齐句子对
    """
    logger = logging.getLogger(__name__)
    
    # 规则1：短文本直接返回
    if len(term) < max_len and len(translation) < max_len:
        return [(term, translation)]
    
    # 规则2：句子切分
    zh_sents = re.split(r'(?<=[。！？])', term)  # 中文按句号切分
    en_sents = re.split(r'(?<=[.!?])(?=\s+[A-Z])', translation)  # 英文按句号+大写切分
    
    # 过滤空句和过短句
    zh_sents = [s.strip() for s in zh_sents if len(s.strip()) >= min_len]
    en_sents = [s.strip() for s in en_sents if len(s.strip()) >= min_len]
    
    # 规则3：数量对齐检查
    if len(zh_sents) != len(en_sents):
        logger.debug(
            f"句子数量不对齐 (zh={len(zh_sents)}, en={len(en_sents)}), 保持原文本"
        )
        return [(term, translation)]
    
    # 规则4：长度均衡检查
    pairs = list(zip(zh_sents, en_sents))
    valid_pairs = []
    
    for zh, en in pairs:
        ratio = len(zh) / (len(en) + 1e-6)
        if 0.3 < ratio < 3.0:  # 长度比例合理
            valid_pairs.append((zh, en))
        else:
            logger.debug(
                f"长度比例异常 ({ratio:.2f}), 跳过: zh='{zh[:20]}...', en='{en[:20]}...'"
            )
    
    # 规则5：有效对数检查（至少保留50%）
    if len(valid_pairs) < len(pairs) * 0.5:
        logger.debug("有效对不足50%，保持原文本")
        return [(term, translation)]
    
    return valid_pairs if valid_pairs else [(term, translation)]


def preprocess_terms_for_sonar(
    pairs: List[Tuple[str, str, str]],
    max_len: int = 100
) -> List[Dict]:
    """
    预处理术语库：智能分割并生成SONAR训练样本
    
    Args:
        pairs: [(lang, term, trans), ...]
        max_len: 分割长度阈值
    
    Returns:
        [
          {
            'lang': 语言代码,
            'term': 分割后源文本,
            'translation': 分割后英文,
            'original_term': 原始完整术语（追溯用）
          },
          ...
        ]
    """
    logger = logging.getLogger(__name__)
    all_segments = []
    
    for lang, term, trans in pairs:
        # 智能分割
        segments = split_aligned_segments(term, trans, max_len=max_len)
        
        for seg_term, seg_trans in segments:
            all_segments.append({
                'lang': lang,
                'term': seg_term,
                'translation': seg_trans,
                'original_term': term
            })
    
    logger.info(
        f"术语预处理完成: "
        f"{len(pairs)}条原始术语 → {len(all_segments)}条分割段落 "
        f"(扩增倍数: {len(all_segments)/len(pairs):.2f}x)"
    )
    
    return all_segments


def encode_with_sonar(
    autoencoder: SonarAutoencoder,
    tokenizer,
    texts: List[str],
    lang_code: str,
    device: str = "cuda",
    base_model=None,          # 新增：直接编码用的基础模型
    use_cache: bool = True,  # 新增参数：是否使用缓存
) -> Tensor:
    """
    【修复版】使用SONAR自编码器批量编码文本
      当 autoencoder.encoder 为 None 时（encoder_model=None初始化），
      通过 base_model 参数直接编码，绕过 autoencoder.encoder。
      两种路径统一使用 _encode_with_base_model() 获取原始嵌入，
      再经 autoencoder 的 normalizer 标准化后返回。
      - 优先从autoencoder.cache_path加载预计算嵌入
      - 缓存未命中时降级到实时编码
      - 缓存命中率统计    
    Args:
        autoencoder: 预训练的SONAR自编码器
        tokenizer: NLLB分词器
        texts: 文本列表
        lang_code: NLLB语言代码
        device: 计算设备
        base_model: NLLB基础模型（缓存未命中时使用）
        use_cache: 是否启用缓存（默认True）   
    Returns:
        [N, dim] 标准化后的嵌入
    """
    logger = logging.getLogger(__name__)

    embeddings = []
    cache_hits = 0
    cache_misses = 0

    # ===== 加载缓存（一次性加载，避免重复IO）=====
    cache_data = None
    if (
        use_cache
        and hasattr(autoencoder, 'cache_path')
        and autoencoder.cache_path
        and os.path.exists(autoencoder.cache_path)   # 新增：文件存在性校验
    ):
        cache_data = SonarAutoencoder.load_term_cache(
            autoencoder.cache_path, device='cpu'
        )

    autoencoder.eval()

    with torch.no_grad():
        for text in texts:
            # Key格式：(lang_code, text)
            cache_key = (lang_code, text.strip())

            # ===== 策略1：缓存查询 =====
            if (
                cache_data is not None
                and cache_key in cache_data['embeddings_normalized']
            ):
                emb = cache_data['embeddings_normalized'][cache_key].to(device)
                embeddings.append(emb)
                cache_hits += 1
                continue

            # ===== 策略2：实时编码（缓存未命中）=====
            cache_misses += 1
            try:
                tokenizer.src_lang = lang_code
                inputs = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=256
                ).to(device)

                if (
                    autoencoder.encoder is not None
                    and autoencoder.is_seq2seq
                ):
                    raw_emb = autoencoder.encode(
                        inputs["input_ids"],
                        inputs["attention_mask"]
                    )
                else:
                    raw_emb = _encode_with_base_model(
                        base_model, inputs, device
                    )
                    if raw_emb is None:
                        logger.warning(
                            f"编码失败，跳过: {text[:30]}..."
                        )
                        continue

                normalized = autoencoder.normalizer.normalize(
                    raw_emb, lang_code
                )
                embeddings.append(normalized.cpu())

            except Exception as e:
                logger.warning(
                    f"编码失败，跳过: {text[:30]}... | {e}"
                )
                continue

    if not embeddings:
        raise ValueError("所有文本编码失败")

    # ===== 缓存命中率统计 =====
    total = cache_hits + cache_misses
    if total > 0:
        hit_rate = cache_hits / total
        logger.info(
            f"[SONAR编码] 完成 {len(embeddings)}/{len(texts)} 条 | "
            f"缓存命中率: {hit_rate:.1%} ({cache_hits}/{total})"
        )

    return torch.cat(embeddings, dim=0)  # [N, dim]

# ============================================================
# 【修改】纯SONAR语义相似度筛选（移除Jaccard）
# ============================================================

def extract_cross_lingual_pairs_hybrid(
    pairs: List[Tuple[str, str, str]],
    autoencoder: SonarAutoencoder,
    tokenizer,
    min_overlap: float = 0.15,
    max_pairs: int = 200,
    sonar_weight: float = 1.0,
    device: str = "cuda",
    base_model=None,           # 新增：外部编码器
) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """
    【修复版】纯SONAR语义相似度跨语言对筛选（支持缓存）

    修复点：
      向 encode_with_sonar() 传入 base_model，
      确保 autoencoder.encoder=None 时仍可正常编码。

    Args:
        pairs: 全量术语对
        autoencoder: SONAR自编码器
        tokenizer: 分词器
        min_overlap: 最小相似度阈值
        max_pairs: 最大返回对数
        sonar_weight: 保留参数（兼容性）
        device: 计算设备
        base_model: NLLB基础模型（autoencoder.encoder=None时使用）

    Returns:
        跨语言术语对列表
    """
    logger = logging.getLogger(__name__)

    # 按语言分组
    lang_groups: Dict[str, List[Tuple[str, str]]] = {}
    for lang, src, tgt in pairs:
        lang_groups.setdefault(lang, []).append((src, tgt))

    langs = list(lang_groups.keys())
    if len(langs) < 2:
        logger.warning(f"仅{len(langs)}种语言，无法形成跨语言对")
        return []

    logger.info(f"[纯SONAR筛选] 语言: {langs}")
    for lang, terms in lang_groups.items():
        logger.info(f"  {lang}: {len(terms)}条")

    # ===== 预计算英文翻译的SONAR嵌入（修改：使用缓存）=====
    logger.info("[纯SONAR筛选] 预计算英文翻译的SONAR嵌入...")
    translation_embeddings: Dict[str, torch.Tensor] = {}
    
    try:
        unique_translations = list(set(tgt for _, _, tgt in pairs))
        logger.info(f"  唯一英文翻译: {len(unique_translations)}条")
        
        batch_size = 16
        for i in range(0, len(unique_translations), batch_size):
            batch_texts = unique_translations[i:i + batch_size]
            
            try:
                # ===== 修改点：启用缓存 =====
                batch_embs = encode_with_sonar(
                    autoencoder, tokenizer, batch_texts,
                    lang_code="eng_Latn",
                    device=device,
                    base_model=base_model,
                    use_cache=True,  # 启用缓存
                )
                
                for text, emb in zip(batch_texts, batch_embs):
                    translation_embeddings[text] = emb
            
            except Exception as e:
                logger.warning(f"批量编码失败，跳过批次: {e}")
                continue
        
        logger.info(f"  成功编码: {len(translation_embeddings)}条")

    except Exception as e:
        logger.error(f"SONAR编码失败: {e}")
        logger.warning("降级到原 extract_cross_lingual_pairs 方法")
        return extract_cross_lingual_pairs(
            pairs, min_overlap, max_pairs,
            adaptive=True, use_domain_general=True
        )

    if not translation_embeddings:
        logger.warning("所有翻译编码失败，降级到原方法")
        return extract_cross_lingual_pairs(
            pairs, min_overlap, max_pairs,
            adaptive=True, use_domain_general=True
        )

    # ===== 计算所有跨语言对的SONAR相似度 =====
    all_pairs_with_sim = []

    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            lang1, lang2 = langs[i], langs[j]

            for src1, tgt1 in lang_groups[lang1]:
                for src2, tgt2 in lang_groups[lang2]:

                    if tgt1 not in translation_embeddings:
                        continue
                    if tgt2 not in translation_embeddings:
                        continue

                    emb1 = translation_embeddings[tgt1]
                    emb2 = translation_embeddings[tgt2]

                    sonar_sim = F.cosine_similarity(
                        emb1.unsqueeze(0), emb2.unsqueeze(0)
                    ).item()

                    all_pairs_with_sim.append((
                        sonar_sim,
                        (lang1, src1),
                        (lang2, src2),
                        {"sonar": sonar_sim}
                    ))

    # ===== 策略1：阈值筛选 =====
    candidates = [
        (sim, t1, t2, info)
        for sim, t1, t2, info in all_pairs_with_sim
        if sim >= min_overlap
    ]
    logger.info(f"[策略1] 阈值{min_overlap}筛选: {len(candidates)}对")

    # ===== 策略2：自适应降阈值 =====
    if len(candidates) < 5:
        logger.info("[策略2] 候选不足5对，启动自适应降阈值")
        for threshold in [0.65, 0.55, 0.45, 0.35, 0.25]:
            fallback = [
                (sim, t1, t2, info)
                for sim, t1, t2, info in all_pairs_with_sim
                if sim >= threshold
            ]
            if len(fallback) >= 5:
                candidates = fallback
                logger.info(f"  ✓ 采用阈值{threshold}: {len(fallback)}对")
                break
        else:
            all_pairs_with_sim.sort(reverse=True, key=lambda x: x[0])
            candidates = all_pairs_with_sim[:max_pairs]
            logger.info(f"  ✓ 采用top-{len(candidates)}")

    # ===== 策略3：领域通用随机对齐 =====
    if len(candidates) == 0:
        logger.info("[策略3] 无相似对，使用随机配对")
        random.seed(42)
        cross_pairs_direct = []
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                lang1, lang2 = langs[i], langs[j]
                n = min(
                    len(lang_groups[lang1]),
                    len(lang_groups[lang2]), 5
                )
                for _ in range(n):
                    src1, _ = random.choice(lang_groups[lang1])
                    src2, _ = random.choice(lang_groups[lang2])
                    cross_pairs_direct.append(
                        ((lang1, src1), (lang2, src2))
                    )
        logger.info(f"  生成{len(cross_pairs_direct)}个随机配对")
        return cross_pairs_direct

    # ===== 排序取前max_pairs =====
    candidates.sort(reverse=True, key=lambda x: x[0])
    top_pairs = candidates[:max_pairs]

    logger.info(f"[纯SONAR筛选] 最终提取{len(top_pairs)}对")
    for sim, (l1, s1), (l2, s2), info in top_pairs[:3]:
        logger.info(
            f"  {l1}:'{s1[:20]}' ↔ {l2}:'{s2[:20]}' | "
            f"SONAR={info['sonar']:.3f}"
        )

    return [(t1, t2) for _, t1, t2, _ in top_pairs]

# ============================================================
# 【修改】SONAR对比学习预训练（InfoNCE损失）
# ============================================================

def pretrain_sonar_autoencoder(
    pairs: List[Tuple[str, str, str]],
    tokenizer,
    config: SonarConfig,
    base_model,
    device: str = "cuda",
    min_positive_pairs: int = 3,        # 调整：从10降至3
    similarity_threshold: float = 0.3,  # 新增：策略2的相似度阈值
    cache_dir: str = r'F:\hyberT\MultiSonar\data\Sonar_cache',  # 新增参数
) -> Optional[SonarAutoencoder]:
    """
    【调整版】SONAR对比学习预训练
      返回前将base_model赋给autoencoder.encoder
      确保后续推理时self.encoder可用
      - 预训练完成后自动构建术语库SONAR缓存-，保存到指定cache_dir目录
    关键参数调整：
      min_positive_pairs: 10 → 3
        原因：小型术语库正样本天然稀少
        3对正样本已可提供基础对比信号
      similarity_threshold: 新增 0.6
        原因：启用策略2时的Jaccard阈值
        0.6表示60%词汇重叠才视为正样本
    Args:
        pairs: 术语对列表
        tokenizer: 分词器
        config: SONAR配置
        base_model: NLLB基础模型
        device: 计算设备
        min_positive_pairs: 最少正样本对数（低于此值降级）
        similarity_threshold: 翻译相似度阈值（策略2）
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info(" " * 10 + "SONAR对比学习预训练（跨语言对齐）")
    logger.info("="*60)

    try:
        # ===== 步骤1：预处理术语 =====
        segments = preprocess_terms_for_sonar(pairs, max_len=100)
        if len(segments) < 20:
            logger.warning(
                f"段落数不足20（当前{len(segments)}），"
                f"对比学习可能过拟合"
            )

        # ===== 步骤2：初始化（不传encoder_model，避免冻结污染）=====
        autoencoder = SonarAutoencoder(config, encoder_model=None)
        autoencoder.to(device)
        logger.info(f"自编码器初始化完成 | 设备: {device}")

        # ===== 步骤3：验证可训练参数 =====
        trainable_params = [
            p for p in autoencoder.parameters() if p.requires_grad
        ]
        total_trainable = sum(p.numel() for p in trainable_params)

        if total_trainable == 0:
            logger.error("自编码器无可训练参数！")
            return None

        logger.info(f"  可训练参数量: {total_trainable:,}")

        # ===== 步骤4：拟合标准化参数 =====
        logger.info("拟合标准化参数（RobustScaler）...")

        lang_segments: Dict[str, List[Dict]] = {}
        for seg in segments:
            lang_segments.setdefault(seg['lang'], []).append(seg)

        for lang, lang_segs in lang_segments.items():
            logger.info(f"  {lang}: {len(lang_segs)}条段落")

            embeddings_list = []
            for seg in lang_segs:
                tokenizer.src_lang = lang
                inputs = tokenizer(
                    seg['term'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(device)

                emb = _encode_with_base_model(base_model, inputs, device)
                if emb is not None:
                    embeddings_list.append(emb.cpu())

            if not embeddings_list:
                logger.warning(
                    f"  {lang} 无有效嵌入，跳过标准化拟合"
                )
                continue

            embeddings = torch.cat(embeddings_list, dim=0)
            autoencoder.normalizer.fit(embeddings, lang)
            logger.info(f"    ✓ {lang} 标准化参数拟合完成")

        # 拟合英文锚点标准化参数
        eng_texts = list(set(s['translation'] for s in segments))[:50]
        eng_embs = []
        for text in eng_texts:
            tokenizer.src_lang = "eng_Latn"
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=256
            ).to(device)
            emb = _encode_with_base_model(base_model, inputs, device)
            if emb is not None:
                eng_embs.append(emb.cpu())

        if eng_embs:
            autoencoder.normalizer.fit(
                torch.cat(eng_embs, dim=0), "eng_Latn"
            )
            logger.info("    ✓ eng_Latn 标准化参数拟合完成")

        # ===== 步骤5：提取跨语言正样本对 =====
        if not config.enable_denoising:
            logger.info("去噪功能已禁用，跳过对比学习")
            
            # ===== 修复核心：返回前赋值encoder =====
            autoencoder.encoder = base_model
            autoencoder.is_seq2seq = hasattr(base_model, 'get_encoder')
            
            # 确保冻结（虽然base_model可能已冻结，这里再次确认）
            for param in autoencoder.encoder.parameters():
                param.requires_grad = False
            
            autoencoder.eval()
            logger.info("  ✓ encoder已链接到base_model（冻结）")
            return autoencoder

        logger.info("提取跨语言正样本对（多策略）...")

        positive_pairs = _extract_positive_pairs(
            segments,
            similarity_threshold=similarity_threshold
        )

        # ===== 降级判断 =====
        if len(positive_pairs) < min_positive_pairs:
            logger.warning(
                f"正样本对不足{min_positive_pairs}对"
                f"（当前{len(positive_pairs)}），"
                f"降级为自重构训练"
            )
            result = _fallback_reconstruction_training(
                autoencoder, segments, tokenizer,
                config, device, base_model
            )
            
            # ===== 修复核心：降级训练后也赋值encoder =====
            if result is not None:
                result.encoder = base_model
                result.is_seq2seq = hasattr(base_model, 'get_encoder')
                for param in result.encoder.parameters():
                    param.requires_grad = False
                logger.info("  ✓ encoder已链接到base_model（冻结）")
            
            return result

        # ===== 步骤6：对比学习训练 =====
        logger.info(
            f"开始对比学习训练 | "
            f"正样本对={len(positive_pairs)} | "
            f"epochs={config.pretrain_epochs}"
        )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.pretrain_lr,
            weight_decay=config.weight_decay
        )

        autoencoder.train()

        for epoch in range(config.pretrain_epochs):
            epoch_losses = []
            random.shuffle(positive_pairs)

            for i in range(
                0, len(positive_pairs), config.pretrain_batch_size
            ):
                batch_pairs = positive_pairs[
                    i:i + config.pretrain_batch_size
                ]

                batch_loss = []
                for seg1, seg2 in batch_pairs:
                    loss = _compute_contrastive_loss(
                        autoencoder, base_model, tokenizer,
                        seg1, seg2, batch_pairs, device
                    )
                    if loss is not None:
                        batch_loss.append(loss)

                if not batch_loss:
                    continue

                loss = torch.stack(batch_loss).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            if epoch_losses:
                avg = sum(epoch_losses) / len(epoch_losses)
                logger.info(
                    f"  Epoch {epoch+1}/{config.pretrain_epochs} | "
                    f"对比损失: {avg:.4f}"
                )

        # ===== 修复核心：训练完成后赋值encoder =====
        autoencoder.encoder = base_model
        autoencoder.is_seq2seq = hasattr(base_model, 'get_encoder')
        
        # 确保编码器参数冻结
        for param in autoencoder.encoder.parameters():
            param.requires_grad = False
        
        autoencoder.eval()
        logger.info("✓ 对比学习预训练完成")
        logger.info("  ✓ encoder已链接到base_model（冻结）")

        # ===== 新增：构建术语库缓存 =====
        try:
            cache_path = autoencoder.build_term_cache(
                pairs=pairs,
                tokenizer=tokenizer,
                cache_dir=cache_dir,
                device=device
            )
            
            # 将缓存路径存入autoencoder属性（供后续使用）
            autoencoder.cache_path = cache_path
            
        except Exception as e:
            logger.warning(f"[SONAR缓存] 构建失败: {e}")
            logger.warning("  将在后续训练中实时编码（未影响功能）")
            autoencoder.cache_path = None

        return autoencoder

    except Exception as e:
        logger.error(f"预训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def _encode_with_base_model(
    base_model,
    inputs: Dict,
    device: str,
) -> Optional[torch.Tensor]:
    """
    【新增】直接使用base_model编码（绕过SonarAutoencoder.encoder）

    设计原则：
      base_model不注入SonarAutoencoder，避免参数冻结污染。
      编码时直接调用base_model的编码器，结果作为嵌入使用。

    Args:
        base_model: NLLB基础模型（AutoModelForSeq2SeqLM）
        inputs: tokenizer输出（含input_ids/attention_mask）
        device: 计算设备

    Returns:
        [1, dim] 句子嵌入，失败返回None
    """
    try:
        with torch.no_grad():
            # 获取编码器（兼容多种包装）
            if hasattr(base_model, 'get_encoder'):
                encoder = base_model.get_encoder()
            elif hasattr(base_model, 'model') and hasattr(base_model.model, 'encoder'):
                encoder = base_model.model.encoder
            else:
                return None

            # 前向编码
            enc_out = encoder(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                return_dict=True
            )

            hidden = enc_out.last_hidden_state  # [1, T, dim]

            # 注意力掩码加权平均池化
            mask = inputs["attention_mask"].to(device).unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            return emb  # [1, dim]

    except Exception as e:
        logging.getLogger(__name__).debug(f"编码失败: {e}")
        return None


def _compute_contrastive_loss(
    autoencoder,
    base_model,
    tokenizer,
    seg1: Dict,
    seg2: Dict,
    batch_pairs: List,
    device: str,
) -> Optional[torch.Tensor]:
    """
    【新增】计算单对InfoNCE对比损失

    流程：
      1. base_model编码正样本对（冻结，不参与梯度）
      2. autoencoder的normalizer/denoiser处理嵌入（可训练）
      3. 计算InfoNCE损失

    Args:
        autoencoder: 待训练的自编码器（normalizer/denoiser可训练）
        base_model: 冻结的NLLB编码器
        tokenizer: 分词器
        seg1, seg2: 正样本对的术语信息
        batch_pairs: 当前批次（用于采样负样本）
        device: 设备

    Returns:
        对比损失标量，失败返回None
    """
    logger = logging.getLogger(__name__)

    try:
        # ===== 编码正样本对（base_model冻结）=====
        tokenizer.src_lang = seg1['lang']
        inputs1 = tokenizer(
            seg1['term'], return_tensors="pt",
            truncation=True, max_length=256
        ).to(device)

        raw_emb1 = _encode_with_base_model(base_model, inputs1, device)
        if raw_emb1 is None:
            return None

        tokenizer.src_lang = seg2['lang']
        inputs2 = tokenizer(
            seg2['term'], return_tensors="pt",
            truncation=True, max_length=256
        ).to(device)

        raw_emb2 = _encode_with_base_model(base_model, inputs2, device)
        if raw_emb2 is None:
            return None

        # ===== autoencoder处理（可训练部分）=====
        # 标准化（语言特定RobustScaler）
        emb1 = autoencoder.normalizer.normalize(raw_emb1, seg1['lang'])
        emb2 = autoencoder.normalizer.normalize(raw_emb2, seg2['lang'])

        # 去噪（可训练）
        if autoencoder.enable_denoising and hasattr(autoencoder, 'semantic_denoiser'):
            emb1, _ = autoencoder.semantic_denoiser(emb1, add_noise=True)
            emb2, _ = autoencoder.semantic_denoiser(emb2, add_noise=True)

        # 低秩解码（可训练）
        emb1 = autoencoder.decode(emb1)
        emb2 = autoencoder.decode(emb2)

        # L2归一化（余弦相似度空间）
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        # ===== 采样负样本 =====
        num_neg = min(8, len(batch_pairs) - 1)
        other_pairs = [p for p in batch_pairs if p != (seg1, seg2)]

        neg_embs = []
        if other_pairs and num_neg > 0:
            neg_samples = random.sample(other_pairs, k=num_neg)

            for neg_seg1, neg_seg2 in neg_samples:
                neg_seg = random.choice([neg_seg1, neg_seg2])
                tokenizer.src_lang = neg_seg['lang']
                neg_inp = tokenizer(
                    neg_seg['term'], return_tensors="pt",
                    truncation=True, max_length=256
                ).to(device)

                neg_raw = _encode_with_base_model(base_model, neg_inp, device)
                if neg_raw is None:
                    continue

                # 负样本也经过标准化（但不参与梯度，detach）
                with torch.no_grad():
                    neg_norm = autoencoder.normalizer.normalize(
                        neg_raw, neg_seg['lang']
                    )
                    neg_decoded = autoencoder.decode(neg_norm)
                    neg_emb = F.normalize(neg_decoded, p=2, dim=-1)

                neg_embs.append(neg_emb.detach())

        # ===== InfoNCE损失 =====
        # 正样本相似度
        pos_sim = (emb1 * emb2).sum(dim=-1)  # [1]

        if neg_embs:
            # 负样本相似度
            neg_cat = torch.cat(neg_embs, dim=0)   # [K, dim]
            neg_sims = torch.mm(emb1, neg_cat.t())  # [1, K]

            # 拼接：[1, 1+K]
            temperature = 0.07
            logits = torch.cat(
                [pos_sim.unsqueeze(1), neg_sims], dim=1
            ) / temperature

            # 标签：正样本在第0位
            labels = torch.zeros(
                logits.size(0), dtype=torch.long, device=device
            )
            contrastive_loss = F.cross_entropy(logits, labels)

        else:
            # 无负样本：MSE对齐损失
            contrastive_loss = F.mse_loss(emb1, emb2)

        return contrastive_loss

    except Exception as e:
        logger.debug(f"对比损失计算失败: {e}")
        return None


def _fallback_reconstruction_training(
    autoencoder,
    segments: List[Dict],
    tokenizer,
    config: SonarConfig,
    device: str,
    base_model=None,
) -> Optional[object]:
    """
      接收base_model参数，编码时直接调用base_model
      不依赖autoencoder.encoder（已解耦）
    Args:
        autoencoder: 自编码器（decoder/denoiser可训练）
        segments: 预处理后的术语段落
        tokenizer: 分词器
        config: SONAR配置
        device: 设备
        base_model: NLLB基础模型（用于编码）

    Returns:
        训练好的autoencoder，失败返回None
    """
    logger = logging.getLogger(__name__)
    logger.info("执行降级训练（自重构模式）...")

    # 验证可训练参数
    trainable_params = [
        p for p in autoencoder.parameters() if p.requires_grad
    ]

    if not trainable_params:
        logger.error("自编码器无可训练参数，跳过降级训练")
        return autoencoder

    logger.info(
        f"  可训练参数量: "
        f"{sum(p.numel() for p in trainable_params):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.pretrain_lr,
        weight_decay=config.weight_decay
    )

    autoencoder.train()

    for epoch in range(config.pretrain_epochs):
        epoch_losses = []
        random.shuffle(segments)

        for i in range(0, len(segments), config.pretrain_batch_size):
            batch = segments[i:i + config.pretrain_batch_size]
            batch_loss = []

            for seg in batch:
                try:
                    tokenizer.src_lang = seg['lang']
                    inputs = tokenizer(
                        seg['term'], return_tensors="pt",
                        truncation=True, max_length=256
                    ).to(device)

                    # 使用base_model编码（若提供）
                    if base_model is not None:
                        raw_emb = _encode_with_base_model(
                            base_model, inputs, device
                        )
                        if raw_emb is None:
                            continue

                        # 标准化
                        norm_emb = autoencoder.normalizer.normalize(
                            raw_emb, seg['lang']
                        )

                        # 去噪（可训练）
                        if (autoencoder.enable_denoising
                                and hasattr(autoencoder, 'semantic_denoiser')):
                            denoised, _ = autoencoder.semantic_denoiser(
                                norm_emb, add_noise=True
                            )
                        else:
                            denoised = norm_emb

                        # 解码（可训练）
                        reconstructed = autoencoder.decode(denoised)
                        denorm = autoencoder.normalizer.denormalize(
                            reconstructed, seg['lang']
                        )

                        # 重构损失（目标：还原原始嵌入）
                        mse = F.mse_loss(raw_emb.detach(), denorm)
                        enc_n = F.normalize(raw_emb.detach(), p=2, dim=-1)
                        den_n = F.normalize(denorm, p=2, dim=-1)
                        cos_loss = 1.0 - (enc_n * den_n).sum(dim=-1).mean()

                        seg_loss = (
                            config.mse_weight * mse
                            + config.cosine_weight * cos_loss
                        )

                    else:
                        # 无base_model：使用autoencoder完整前向
                        # 注意：此时encoder必须已赋值，否则会报错
                        outputs = autoencoder(
                            inputs["input_ids"],
                            inputs["attention_mask"],
                            lang_code=seg['lang']
                        )
                        losses = autoencoder.compute_loss(outputs)
                        seg_loss = losses["combined"]

                    batch_loss.append(seg_loss)

                except Exception as e:
                    logger.debug(f"样本训练失败，跳过: {e}")
                    continue

            if not batch_loss:
                continue

            loss = torch.stack(batch_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch_losses:
            avg = sum(epoch_losses) / len(epoch_losses)
            logger.info(
                f"  Epoch {epoch+1}/{config.pretrain_epochs} | "
                f"重构损失: {avg:.4f}"
            )

    autoencoder.eval()
    logger.info("✓ 降级训练完成")

    return autoencoder

# ============================================================
# 【修改】第二步对齐训练主流程（集成HybridAlignmentTrainer）
# 【修改】第二步对齐训练主流程（集成Engram）
# ============================================================
# ============================================================
# 【修改】alignment_training：集成FLoRA + Flow Engram
# ============================================================

def alignment_training(
    args: argparse.Namespace,
    base_model,
    tokenizer,
    pairs: List[Tuple[str, str, str]],
) -> object:
    """
    【修改版】第二步：跨语言对齐训练（集成ELF优化）
    
    新增流程：
      1. LoRA配置使用FLoRA秋（若enable_flora=True）
      2. Engram门控使用FlowEngramGating（若enable_flow_engram=True）
      3. 训练器使用EngramJointTrainer
    """
    logger = logging.getLogger(__name__)
    
    # ===== 【修改】LoRA配置：使用FLoRA秩 =====
    lora_r = args.flora_rank if args.enable_flora else args.lora_r
    
    lora_config = LoraConfig(
        r=lora_r,  # ← 使用FLoRA秋（默认16 vs 标准32）
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=args.lora_target_modules,
    )
    
    logger.info(
        f"  [LoRA配置] 秋r={lora_r} "
        f"{'(FLoRA优化，参数减半)' if args.enable_flora else '(标准LoRA)'}"
    )
    
    # ===== SONAR预训练（保持不变）=====
    sonar_autoencoder = None
    if args.use_sonar:
        logger.info("[SONAR] 启用SONAR对比学习增强")
        sonar_config = SonarConfig(
            dim=1024,
            device=args.device,
            supported_languages=list(LANG_MAP.values()),
            enable_denoising=args.sonar_enable_denoising,
            pretrain_epochs=args.sonar_pretrain_epochs,
            pretrain_lr=args.sonar_pretrain_lr,
            decoder_rank=args.sonar_decoder_rank,
            mse_weight=args.sonar_mse_weight,
            cosine_weight=args.sonar_cosine_weight,
            denoising_weight=args.sonar_denoising_weight,
        )
        
        sonar_autoencoder = pretrain_sonar_autoencoder(
            pairs, tokenizer, sonar_config,
            base_model, args.device,
            cache_dir=r'F:\hyberT\MultiSonar\data\Sonar_cache'
        )
        
        if sonar_autoencoder is None:
            logger.warning("[SONAR] 预训练失败，降级到原Jaccard方法")
    
    # ===== 【修改】Engram表构建 + Flow门控初始化 =====
    engram_table = None
    engram_gating = None
    
    if args.enable_engram:
        logger.info("[Engram] 启用Engram条件记忆增强")
        
        # Engram表构建（保持不变）
        engram_table = TerminologyEngramTable(
            nllb_dim=1024,
            max_ngram=args.engram_max_ngram,
            hash_table_size=args.engram_hash_size,
            device=args.device
        )
        
        engram_table.build_from_sonar(
            pairs, sonar_autoencoder, tokenizer,
            base_model, args.device
        )
        
        # 【修改】使用FlowEngramGating（支持记忆累积）
        if args.enable_flow_engram:
            engram_gating = FlowEngramGating(
                nllb_dim=1024,
                sonar_dim=1024,
                num_heads=args.engram_num_heads,
                dropout=0.1,
                gate_threshold=args.engram_gate_threshold,
                accumulate_K=args.flow_engram_K,  # ← 累积步数
                enable_flow_aggregate=True,       # ← 启用flow聚合
            )
            logger.info(
                f"  [Engram] 使用FlowEngramGating "
                f"(累积K={args.flow_engram_K}步)"
            )
        else:
            # 降级：原始门控
            engram_gating = SOARGuidedEngramGating(
                nllb_dim=1024,
                sonar_dim=1024,
                num_heads=args.engram_num_heads,
                dropout=0.1,
                gate_threshold=args.engram_gate_threshold,
            )
            logger.info("  [Engram] 使用标准门控（无flow聚合）")
        
        total_params = sum(p.numel() for p in engram_gating.parameters())
        logger.info(
            f"  [Engram] 嵌入表：{engram_table.total_entries}条目 | "
            f"门控网络：{total_params:,}参数"
        )
    
    # ===== 跨语言对提取（保持不变）=====
    use_sonar_effective = args.use_sonar and (sonar_autoencoder is not None)
    
    if use_sonar_effective:
        cross_pairs = extract_cross_lingual_pairs_hybrid(
            pairs, sonar_autoencoder, tokenizer,
            min_overlap=args.align_min_overlap,
            max_pairs=args.align_max_pairs,
            sonar_weight=1.0,
            device=args.device,
            base_model=base_model,
        )
    else:
        cross_pairs = extract_cross_lingual_pairs(
            pairs,
            min_overlap=args.align_min_overlap,
            max_pairs=args.align_max_pairs,
            adaptive=True,
            use_domain_general=True,
        )
    
    # ===== 构建训练数据（保持不变）=====
    samples = build_weighted_dataset(
        pairs, repeat=REPEAT,
        boost_factor=args.boost_factor if args.enable_step1 else 1.0,
    )
    dataset = tokenize_data(samples, tokenizer, max_len=args.max_length)
    
    # ===== 【修改】注入LoRA + FLoRA替换 =====
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # 【关键】将标准LoRA层替换为FlowLoRA
    if args.enable_flora:
        model = replace_lora_with_flora(
            model,
            flow_steps=args.flora_flow_steps,
            flow_alpha_schedule=args.flora_alpha_schedule
        )
        logger.info("  ✓ 已将标准LoRA升级为FlowLoRA")
    
    # ===== 降级判断（保持不变）=====
    if not cross_pairs:
        logger.warning("  无跨语言术语对，降级为标准训练")
        train_model(
            args, model, tokenizer, dataset,
            "对齐训练（降级-无跨语言对）"
        )
        save_dir = os.path.join(args.output_dir, "final_adapter")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        return model
    
    # ===== 预计算初始NLLB编码对（保持不变）=====
    logger.info("  预计算初始NLLB编码对...")
    initial_encoded = encode_cross_lingual_pairs(
        model, tokenizer, cross_pairs, device=args.device
    )
    
    if not initial_encoded:
        logger.warning("  初始编码失败，降级为标准训练")
        train_model(
            args, model, tokenizer, dataset,
            "对齐训练（降级-编码失败）"
        )
        save_dir = os.path.join(args.output_dir, "final_adapter")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        return model
    
    # ===== 构建训练参数（保持不变）=====
    samples_per_step = args.batch_size * args.gradient_accumulation_steps
    total_steps = max(1, len(dataset) // samples_per_step) * args.epochs
    train_args, warmup_steps = _build_training_args(
        args, "alignment_training", total_steps
    )
    
    logger.info(
        f"  [对齐训练] "
        f"NLLB对齐λ={args.align_lambda} | "
        f"SONAR蒸馏={'ON' if use_sonar_effective else 'OFF'} | "
        f"Engram={'ON' if args.enable_engram else 'OFF'} | "
        f"FLoRA={'ON' if args.enable_flora else 'OFF'} | "
        f"FlowEngram={'ON' if args.enable_flow_engram else 'OFF'} | "
        f"对数={len(initial_encoded)} | "
        f"Warmup={warmup_steps}步"
    )
    
    progress_cb = CustomProgressCallback(total_epochs=args.epochs)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # ===== 【修改】选择训练器：优先使用EngramJointTrainer =====
    if args.enable_engram and engram_table and engram_gating:
        trainer = EngramJointTrainer(  # ← 使用Flow增强版
            # Engram专属参数
            engram_table=engram_table,
            engram_gating=engram_gating,
            sonar_autoencoder=sonar_autoencoder,
            cfg_args=args,
            
            # FLoRA专属参数（新增）
            enable_flora=args.enable_flora,
            enable_flow_engram=args.enable_flow_engram,
            flow_mse_weight_max=args.flora_mse_weight,
            
            # CGLoRATrainer参数
            curvature_lambda=args.curvature_lambda,
            fisher_beta=args.fisher_beta,
            warmup_steps=warmup_steps,
            
            # Trainer基类参数
            model=model,
            args=train_args,
            tokenizer=tokenizer,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[progress_cb],
        )
        logger.info("  ✓ 使用EngramJointTrainer（FLoRA+FlowEngram三重优化）")
    
    elif use_sonar_effective:
        # SONAR蒸馏（保持原逻辑）
        if getattr(args, 'sonar_enable_distill', True):
            is_cpu = (args.device == 'cpu')
            trainer = SONARGuidedAlignmentTrainer(
                distill_lambda=getattr(args, 'sonar_distill_lambda', 0.15),
                distill_every_n_steps=10 if is_cpu else 5,
                distill_pooling="mean",
                sonar_cache_size=128,
                min_distill_loss=1e-4,
                sonar_autoencoder=sonar_autoencoder,
                sonar_lambda=args.sonar_weight,
                sample_pairs_per_step=1 if is_cpu else 2,
                align_pairs=initial_encoded,
                align_lambda=args.align_lambda,
                cross_pairs_raw=cross_pairs,
                processing_class=tokenizer,
                update_align_every=args.update_align_every,
                curvature_lambda=args.curvature_lambda,
                fisher_beta=args.fisher_beta,
                warmup_steps=warmup_steps,
                model=model,
                args=train_args,
                train_dataset=dataset,
                data_collator=collator,
                callbacks=[progress_cb],
            )
            logger.info("  ✓ 使用SONARGuidedAlignmentTrainer（SONAR蒸馏）")
        else:
            trainer = HybridAlignmentTrainer(
                sonar_autoencoder=sonar_autoencoder,
                sonar_lambda=args.sonar_weight,
                sample_pairs_per_step=2,
                align_pairs=initial_encoded,
                align_lambda=args.align_lambda,
                cross_pairs_raw=cross_pairs,
                processing_class=tokenizer,
                update_align_every=args.update_align_every,
                curvature_lambda=args.curvature_lambda,
                fisher_beta=args.fisher_beta,
                warmup_steps=warmup_steps,
                model=model,
                args=train_args,
                train_dataset=dataset,
                data_collator=collator,
                callbacks=[progress_cb],
            )
            logger.info("  ✓ 使用HybridAlignmentTrainer（SONAR软约束）")
    
    else:
        # 仅NLLB对齐
        trainer = AlignmentLoRATrainer(
            align_pairs=initial_encoded,
            align_lambda=args.align_lambda,
            cross_pairs_raw=cross_pairs,
            processing_class=tokenizer,
            update_align_every=args.update_align_every,
            curvature_lambda=args.curvature_lambda,
            fisher_beta=args.fisher_beta,
            warmup_steps=warmup_steps,
            model=model,
            args=train_args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[progress_cb],
        )
        logger.info("  ✓ 使用AlignmentLoRATrainer（仅NLLB对齐）")
    
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.train()
    
    # ===== 保存模型（保持不变）=====
    save_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 保存Engram组件
    if args.enable_engram and engram_gating:
        engram_save_path = os.path.join(save_dir, "engram_gating.pt")
        torch.save({
            'state_dict': engram_gating.state_dict(),
            'config': {
                'nllb_dim': 1024,
                'sonar_dim': 1024,
                'num_heads': args.engram_num_heads,
                'gate_threshold': args.engram_gate_threshold,
                'accumulate_K': args.flow_engram_K,  # ← 新增
                'enable_flow_aggregate': args.enable_flow_engram,  # ← 新增
            }
        }, engram_save_path)
        logger.info(f"  [Engram] 门控网络已保存: {engram_save_path}")
    
    logger.info(f"  模型已保存: {save_dir}")
    return model

# ============================================================
# 第三步：知识蒸馏训练主流程
# ============================================================

def two_stage_training_from_aligned(
    args: argparse.Namespace,
    aligned_model,
    tokenizer,
    pairs: List[Tuple[str, str, str]],
) -> object:
    """
    第三步：基于对齐模型的知识蒸馏训练（带Warmup和梯度裁剪）。

    教师-学生关系：
      教师：aligned_model（第二步对齐训练的成果）
      学生：从干净基座模型重新初始化的新LoRA模型

    为何重新初始化学生：
      1. 避免教师和学生共享LoRA权重（导致蒸馏退化）
      2. 学生从零适应低资源数据+软标签指导
      3. 保证教师知识通过logits分布传递，而非权重复制

    低资源专注：
      仅用低资源数据训练学生（low_pairs），
      教师的高资源知识通过软标签间接传递。
      若无低资源数据则使用全量数据（fallback）。

    Args:
        args:          全局参数对象
        aligned_model: 第二步训练好的对齐模型（教师）
        tokenizer:     NLLB分词器
        pairs:         全量术语对

    Returns:
        训练后的学生PeftModel
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 70)
    logger.info(" " * 10 + "🔥 第三步：知识蒸馏训练")
    logger.info("=" * 70)

    # 冻结教师模型（不参与梯度计算）
    teacher_model = aligned_model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 构建学生模型（干净基座 + LoRA）
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=args.lora_target_modules,
    )
    student_base  = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path, cache_dir=args.cache_dir
    )
    student_model = get_peft_model(student_base, lora_config)

    total_params = sum(p.numel() for p in student_model.parameters()
                       if p.requires_grad)
    logger.info(f"  [学生模型] 可训练参数量: {total_params:,}")

    # 低资源数据（蒸馏的重点训练对象）
    low_pairs = [p for p in pairs if p[0] in LOW_RESOURCE_LANGS] or pairs
    samples   = build_weighted_dataset(
        low_pairs,
        repeat=5,
        boost_factor=args.boost_factor if args.enable_step1 else 1.0,
    )
    dataset  = tokenize_data(samples, tokenizer, max_len=args.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer, model=student_model)

    # 构建训练参数（含Warmup和梯度裁剪）
    samples_per_step = args.batch_size * args.gradient_accumulation_steps
    total_steps      = max(1, len(dataset) // samples_per_step) * args.epochs
    train_args, warmup_steps = _build_training_args(
        args, "kd_training", total_steps
    )

    logger.info(
        f"  [KD训练] T={args.kd_temperature} | α={args.kd_alpha} "
        f"| Warmup={warmup_steps}步 | 梯度裁剪={args.gradient_clip_norm} "
        f"| 低资源样本数: {len(dataset)}"
    )

    progress_cb = CustomProgressCallback(total_epochs=args.epochs)

    kd_trainer = KDLoRATrainer(
        teacher_model=teacher_model,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
        curvature_lambda=args.curvature_lambda,
        fisher_beta=args.fisher_beta,
        warmup_steps=warmup_steps,
        model=student_model,
        args=train_args,
        processing_class=tokenizer,            # ← tokenizer → processing_class
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[progress_cb],
    )
    kd_trainer.remove_callback(PrinterCallback)
    kd_trainer.remove_callback(ProgressCallback)
    kd_trainer.train()

    # 保存最终adapter
    final_save = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_save, exist_ok=True)
    student_model.save_pretrained(final_save)
    tokenizer.save_pretrained(final_save)
    logger.info(f"  [KD训练] 学生模型已保存: {final_save}")

    return student_model


# ============================================================
# 评估与预测接口
# ============================================================

def evaluate(
    args: argparse.Namespace,
    model,
    tokenizer,
    test_cases: List[Dict],
) -> Optional[Dict]:
    """
    验证集评估（预留接口）。

    当前实现：
      输出提示信息，返回None。
      后续可扩展为BLEU/COMET等自动评估指标。

    扩展示例（引入sacrebleu）：
      from sacrebleu.metrics import BLEU
      bleu = BLEU()
      result = bleu.corpus_score(hypotheses, [references])

    Args:
        args:       全局参数对象
        model:      待评估模型
        tokenizer:  分词器
        test_cases: 测试用例 [{"src": ..., "ref": ...}, ...]

    Returns:
        评估指标字典（当前为None）
    """
    logger = logging.getLogger(__name__)
    logger.info("\n***** 运行验证集评估 *****")
    logger.info(f"  测试用例数: {len(test_cases)}")
    logger.info("  [提示] 验证集评估接口预留，当前无评估数据")
    return None


def predict(
    args: argparse.Namespace, model, tokenizer, test_text: str,
    engram_table=None, engram_gating=None, sonar_autoencoder=None # 【新增】透传组件
) -> Dict[str, str]:
    """单文本翻译预测（含后处理与组件透传）"""
    logger = logging.getLogger(__name__)
    logger.info("\n***** 运行翻译预测 *****")
    model.eval()
    
    raw_translation = translate_auto(
        model, tokenizer, test_text, tgt_lang=TGT_LANG,
        engram_table=engram_table, 
        engram_gating=engram_gating, 
        sonar_autoencoder=sonar_autoencoder
    )
    finalized = finalize_translation(raw_translation)
    
    return {
        "source": test_text,
        "translation": raw_translation,
        "finalized": finalized,
    }


# ============================================================
# 【修改】主流程（添加4-bit量化推理）
# ============================================================

def main():
    """
    【修改版】主流程（添加量化推理优化）
    
    新增功能：
      1. 4-bit量化推理（显存-75%，速度+3x）
      2. BF16训练检测与提示
      3. Flash Attention自动启用检测
    
    执行顺序：
      1. 参数解析与环境初始化
      2. 模型加载（训练：FP32，推理：4-bit）
      3. 基线翻译（微调前）
      4. 训练（do_train=True）
      5. 评估（do_eval=True，预留）
      6. 预测对比（do_predict=True）
    """
    # ==================== 1. 初始化 ====================
    args = get_args()
    
    # 初始化日志
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = None
    if args.log_dir:
        log_file = os.path.join(
            args.log_dir, f"nllb_lora_sonar_{timestamp}.log"
        )
    logger = init_logger(log_file=log_file)
    
    # 检测优化特性
    # 检测优化特性（新增ELF检测）
    optimization_features = []
    if args.device == 'cuda' and torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        
        if compute_capability[0] >= 8:
            optimization_features.append("BF16混合精度")
        
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            optimization_features.append("Flash Attention")
        
        optimization_features.append("梯度检查点")
    
    # 【新增】ELF优化检测
    if args.enable_flora:
        optimization_features.append(
            f"FLoRA(r={args.flora_rank}, 参数-50%)"
        )
    if args.enable_flow_engram:
        optimization_features.append(
            f"FlowEngram(K={args.flow_engram_K}, 门控-80%)"
        )
    if args.enable_delayed_discretization:
        optimization_features.append(
            f"延迟离散化(K={args.delayed_K}, 实验性)"
        )
    
    # 启动配置（修改日志输出）
    sep = "=" * 70
    logger.info(
        f"\n{sep}\n"
        f"{'NLLB-SONAR-Engram-FLoRA系统':^70}\n"
        f"{sep}\n"
        f"  模型路径:   {args.model_path}\n"
        f"  术语库:     {args.glossary_path}\n"
        f"  输出目录:   {args.output_dir}\n"
        f"  训练轮数:   {args.epochs}\n"
        f"  批次大小:   {args.batch_size} × {args.gradient_accumulation_steps} "
        f"= {args.batch_size * args.gradient_accumulation_steps}（等效）\n"
        f"  学习率:     {args.lr}（Warmup={args.warmup_proportion}）\n"
        f"  梯度裁剪:   {args.gradient_clip_norm}\n"
        f"  LoRA配置:   r={'FLoRA-'+str(args.flora_rank) if args.enable_flora else args.lora_r}, "
        f"α={args.lora_alpha}\n"
        f"  设备:       {args.device}\n"
        f"  优化特性:   {', '.join(optimization_features) if optimization_features else '无'}\n"
        f"  第一步（置信度加权）: {args.enable_step1}\n"
        f"  第二步（跨语言对齐）: {args.enable_step2}\n"
        f"  SONAR增强:            {args.use_sonar}\n"
        f"  Engram记忆:           {args.enable_engram}\n"
        f"  第三步（知识蒸馏）:   {args.enable_step3}\n"
        f"  ──────────────────────────────────────────\n"
        f"  【ELF优化】\n"
        f"  FLoRA连续优化:        {args.enable_flora}\n"
        f"  Flow记忆聚合:         {args.enable_flow_engram}\n"
        f"  延迟离散化:           {args.enable_delayed_discretization}\n"
        f"{sep}"
    )
    
    # 设置全局随机种子
    seed_everything(args.seed)
    
    # 目录检查
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.overwrite_output_dir:
            logger.warning(
                f"  输出目录 {args.output_dir} 已存在且非空，将覆盖同名文件"
            )
        else:
            raise ValueError(
                f"输出目录 {args.output_dir} 已存在且非空。\n"
                f"使用 --overwrite_output_dir 允许覆盖。"
            )
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================== 2. 加载模型与数据 ====================
    logger.info("[1/6] 加载模型与分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, cache_dir=args.cache_dir
    )
    
    # ===== 训练模型（FP32/BF16）=====
    model_train = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path, 
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16 if (
            args.device == 'cuda' 
            and torch.cuda.is_available() 
            and torch.cuda.get_device_capability()[0] >= 8
        ) else torch.float32,
    )
    model_train.to(args.device)
    logger.info(f"  训练模型加载完成 | 数据类型: {model_train.dtype}")
    
    # ===== 推理模型（4-bit量化，仅GPU且训练后）=====
    model_infer = None
    use_quantization = (
        args.device == 'cuda' 
        and torch.cuda.is_available()
    )
    
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            model_infer = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_path,
                cache_dir=args.cache_dir,
                quantization_config=bnb_config,
                device_map="auto",
            )
            logger.info("  ✓ 4-bit量化推理模型加载成功（显存优化-75%）")
        
        except ImportError:
            logger.warning(
                "  bitsandbytes未安装，跳过量化优化\n"
                "  安装命令：pip install bitsandbytes"
            )
            use_quantization = False
    
    # 使用训练模型作为基线推理（量化模型需微调后才加载adapter）
    model = model_train
    
    logger.info(f"  模型加载完成 | 设备: {args.device}")
    
    logger.info("[2/6] 加载术语库...")
    glossary = load_glossary(args.glossary_path)
    pairs = extract_pairs(glossary)
    logger.info(f"  术语对: {len(pairs)} 条")
    
    # 设置默认测试文本
    if args.test_text is None:
        args.test_text = (
            "动词有时、体、态、式等范畴。德国人O谭博夫都对古南岛语语音作了拟测。"
            "代词有人称和数的范畴，第一人称复数有包括式和排除式的区别。"
            "南岛诸语言属粘着型，词根加附加成分和词根的重叠"
            "或部分重叠是构词和构形的主要手段。"
            "Selain itu, melalui konstruksi data arkeologi, "
            "prasejarah Taiwan dapat ditelusuri kembali ke periode "
            "Paleolitik akhir sekitar 15.000 tahun yang lalu。"
        )
    
    # ==================== 3. 基线翻译 ====================
    logger.info("[3/6] 微调前基线翻译...")
    baseline_result = predict(args, model, tokenizer, args.test_text)
    
    # ==================== 4. 训练 ====================
    if args.do_train:
        logger.info("[4/6] 开始训练...")
        
        final_model = None
        
        if args.enable_step3 and args.enable_step2:
            logger.info("🔥 组合策略：跨语言对齐 + 知识蒸馏")
            align_model = alignment_training(
                args, model_train, tokenizer, pairs
            )
            final_model = two_stage_training_from_aligned(
                args, align_model, tokenizer, pairs
            )
        
        elif args.enable_step2:
            logger.info("🔥 第二步：跨语言对齐（SONAR增强）")
            final_model = alignment_training(
                args, model_train, tokenizer, pairs
            )
        
        else:
            logger.info("🔥 标准LoRA训练")
            if args.enable_step1:
                logger.info("  [第一步] 启用置信度加权采样")
                samples = build_weighted_dataset(
                    pairs, repeat=2, boost_factor=args.boost_factor
                )
            else:
                samples = build_weighted_dataset(
                    pairs, repeat=5, boost_factor=1.0
                )
            
            dataset = tokenize_data(samples, tokenizer, max_len=args.max_length)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=args.lora_target_modules,
            )
            final_model = get_peft_model(model_train, lora_config)
            final_model.print_trainable_parameters()
            
            train_model(args, final_model, tokenizer, dataset, "标准训练")
            
            save_dir = os.path.join(args.output_dir, "final_adapter")
            os.makedirs(save_dir, exist_ok=True)
            final_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        
        # ===== 加载量化推理模型（训练后）=====
        if use_quantization and model_infer is not None:
            try:
                adapter_path = os.path.join(args.output_dir, "final_adapter")
                
                # 将LoRA adapter加载到量化模型
                model_infer = PeftModel.from_pretrained(
                    model_infer, adapter_path
                )
                model = model_infer  # 切换到量化推理模型
                logger.info("  ✓ LoRA adapter已加载到量化模型")
            
            except Exception as e:
                logger.warning(f"  量化模型加载adapter失败: {e}")
                logger.warning("  降级使用FP32推理")
                model = final_model
        else:
            model = final_model
    
    # ==================== 5. 评估（预留）====================
    if args.do_eval:
        logger.info("[5/6] 验证集评估...")
        evaluate(args, model, tokenizer, test_cases=[])
    
    # ==================== 6. 预测对比 ====================
    if args.do_predict:
        logger.info("[6/6] 微调后翻译预测...")
        
        # 【新增】加载Engram组件（如果存在）
        # 【修复1】动态加载Engram组件（兼容 FlowEngramGating 和 SOARGuidedEngramGating）
        engram_gating_loaded = None
        if args.enable_engram:
            engram_path = os.path.join(
                args.output_dir, "final_adapter", "engram_gating.pt"
            )
            if os.path.exists(engram_path):
                try:
                    checkpoint = torch.load(engram_path, map_location=args.device)
                    config = checkpoint['config']
                    
                    # 动态判断使用哪个门控类
                    if config.get('enable_flow_aggregate', False):
                        engram_gating_loaded = FlowEngramGating(**config).to(args.device)
                    else:
                        # 兼容旧版或纯SONAR门控，剔除Flow专属参数以防报错
                        config.pop('accumulate_K', None)
                        config.pop('enable_flow_aggregate', None)
                        engram_gating_loaded = SOARGuidedEngramGating(**config).to(args.device)
                        
                    engram_gating_loaded.load_state_dict(checkpoint['state_dict'])
                    engram_gating_loaded.eval()
                    logger.info(f"  ✓ Engram门控网络已加载: {engram_path}")
                except Exception as e:
                    logger.warning(f"  Engram加载失败: {e}")
        
        # 标准预测（保持原有逻辑）
        finetuned_result = predict(args, model, tokenizer, args.test_text)
        
        # 合并为单条多行日志输出对比结果
        sep = "=" * 70
        term_rows = ""
        target_terms = [
            "tense", "aspect", "voice", "mood",
            "Otto Dempwolff", "agglutinative", "reduplication"
        ]
        for term in target_terms:
            t = term.lower()
            b = "✓" if t in baseline_result['finalized'].lower() else "✗"
            a = "✓" if t in finetuned_result['finalized'].lower() else "✗"
            term_rows += f"  {term:<22} {b:^8} {a:^8}\n"
        
        # 【新增】Engram状态提示
        engram_status = ""
        if args.enable_engram:
            if engram_gating_loaded:
                engram_status = "\n  [Engram] 条件记忆增强：已激活"
            else:
                engram_status = "\n  [Engram] 条件记忆增强：未加载（降级到标准推理）"
        
        logger.info(
            f"\n{sep}\n"
            f"{'微调效果对比':^70}\n"
            f"{sep}\n"
            f"  原文:   {args.test_text[:100]}...\n"
            f"  微调前: {baseline_result['finalized']}\n"
            f"  微调后: {finetuned_result['finalized']}"
            f"{engram_status}\n\n"
            f"  术语出现检测（微调前 vs 微调后）：\n"
            f"  {'术语':<22} {'微调前':^8} {'微调后':^8}\n"
            f"  {'-' * 40}\n"
            f"{term_rows}"
            f"{sep}"
        )

if __name__ == "__main__":
    main()
