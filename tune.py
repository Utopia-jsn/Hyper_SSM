import optuna
from ultralytics import YOLO

# 定义超参数搜索空间
def objective(trial):
    # 生成超参数建议值
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.001)
    momentum = trial.suggest_float("momentum", 0.9, 0.99)
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)  # 数据增强参数

    # 加载模型
    model = YOLO("hyper-mamba20-B.yaml").load("best.pt")

    # 训练配置
    results = model.train(
        data="WTDataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=lr0,
        weight_decay=weight_decay,
        momentum=momentum,
        hsv_h=hsv_h,
        verbose=False,
        project="optuna_tv_detection",
        name=f"trial_{trial.number}"
    )

    # 获取验证集mAP作为优化目标
    return results.results_dict["metrics/mAP50-95(B)"]

# 运行Optuna优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 输出最佳参数
print("Best trial:")
trial = study.best_trial
print(f"  mAP: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
