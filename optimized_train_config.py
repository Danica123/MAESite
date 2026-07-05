import argparse

def get_optimized_config():
    parser = argparse.ArgumentParser()

    # =====================================================
    # 1. 模型架构参数 (适配新版 MultiScaleGVP)
    # =====================================================
    parser.add_argument('--num_layers', type=int, default=3)  # 每个尺度的 GVP 层数
    parser.add_argument('--hidden_nf', type=int, default=256)  # 隐藏层维度 (建议 256 或 512，太大显存吃不消)
    # 向量投影维度 (3 -> target_v_dim)
    parser.add_argument('--target_v_dim', type=int, default=16)

    # 【关键修改】scales 现在代表距离阈值 (Angstrom)
    # 5A (局部), 10A (中程), 16A (全局)
    parser.add_argument('--scales', type=str, default="5,10,20")

    # 输入维度 (取决于 dataloader)
    # 请务必根据 dataloader 的实际输出调整 (例如是否加了 Entropy/PosEnc)
    # 当前 dataloader 输出约为 1885
    parser.add_argument('--orig_s_dim', type=int, default=1889)
    parser.add_argument('--orig_v_dim', type=int, default=3)

    # 边特征维度 (16 RBF + 16 Pos = 32)
    parser.add_argument('--edge_s_dim', type=int, default=32)
    parser.add_argument('--edge_v_dim', type=int, default=3)

    # =====================================================
    # 2. 训练超参数
    # =====================================================
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)  # 蛋白质通常一个batch放不下太多
    parser.add_argument('--gradient_accumulation', type=int, default=8)  # 显存小就加大这个16

    parser.add_argument('--lr', type=float, default=1e-4)  # GVP 通常不需要太大的 LR
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)  # 0.3

    # 损失函数配置
    parser.add_argument('--use_focal_loss', type=bool, default=True)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.75)  # 正样本权重较高
    parser.add_argument('--use_simple_loss', type=bool, default=False)
    parser.add_argument('--use_combined_loss', type=bool, default=False)

    # 学习率调度 (Warmup + Cosine)
    parser.add_argument('--warmup_epochs', type=int, default=10)  # 10% 的 epoch
    parser.add_argument('--min_lr', type=float, default=1e-6)

    # 早停与正则
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--early_delta', type=float, default=0.0005)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 路径配置
    parser.add_argument('--indir', type=str,
                        default="/pubssd/dhh/GTSite/DNA_train_data/")  # /pubssd/dhh/GTSitedata/DNA_train_data/
    parser.add_argument('--save_dir', type=str,
                        default="/home2/2023/23dhh2/GTSite/model/DNA")  # /home2/dhh/GTSite/MAGGN/model/DNA

    parser.add_argument('--seed', type=int, default=1992)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # 自动计算 cosine decay 的周期
    FLAGS.cosine_epochs = FLAGS.epochs - FLAGS.warmup_epochs

    return FLAGS, UNPARSED_ARGV
