import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from pymatgen.core import Structure
from matminer.featurizers.structure import DensityFeatures

# Langmuir模型
def langmuir(P, qmax, b):
    return qmax * b * P / (1 + b * P)

data_dir = 'part_1'   # 你的数据文件夹
output_csv = 'mof_train_77K_100bar_langmuir.csv'
target_temp = 77
target_pressure = 100

rows = []
for fname in os.listdir(data_dir):
    if fname.endswith('.json'):
        json_path = os.path.join(data_dir, fname)
        cif_path = os.path.join(data_dir, fname.replace('.json', '.cif'))
        if not os.path.exists(cif_path):
            print(f"Missing cif for {fname}")
            continue
        # 解析json
        with open(json_path, 'r') as f:
            js = json.load(f)
        # 找到77K的isotherm
        s = Structure.from_str(js['cif'], fmt='cif')
        density_feats = DensityFeatures().featurize(s)
        density = density_feats[0] if density_feats else None
        # density = js.get('density', None)
        if density is None or density == 0:
            print(f"Missing or zero density for {fname}")
            continue

        # 找到77K的isotherm
        target_ads = None
        target_unit = None
        for iso in js.get('isotherms', []):
            if abs(iso.get('temperature', 0) - target_temp) < 1:
                pts = iso.get('isotherm_data', [])
                pressures = np.array([pt.get('pressure', 0) for pt in pts])
                adsorptions = np.array([pt.get('total_adsorption', 0) for pt in pts])
                units = [pt.get('adsorptionUnits', iso.get('adsorptionUnits', 'mol/kg')) for pt in pts]
                # 默认全部单位一样，否则报错提醒
                if len(set(units)) > 1:
                    print(f"Warning: Multiple units in one isotherm for {fname}, will use the first.")
                unit = units[0]
                # 先查找±5 bar范围点
                close_idx = np.where(np.abs(pressures - target_pressure) <= 5)[0]
                if len(close_idx) > 0:
                    target_ads = adsorptions[close_idx[0]]
                    target_unit = unit
                elif len(pressures) >= 3 and pressures.min() < target_pressure < pressures.max():
                    try:
                        p0 = [adsorptions.max(), 0.01]
                        bounds = ([0, 1e-5], [1000, 100])
                        popt, _ = curve_fit(langmuir, pressures, adsorptions, p0=p0, bounds=bounds)
                        target_ads = float(langmuir(target_pressure, *popt))
                        target_unit = unit
                    except Exception as e:
                        print(f"Langmuir拟合失败，{fname}，尝试线性插值：{e}")
                        try:
                            target_ads = float(np.interp(target_pressure, pressures, adsorptions))
                            target_unit = unit
                        except:
                            target_ads = None
                            target_unit = None
                elif pressures.min() <= target_pressure <= pressures.max():
                    try:
                        target_ads = float(np.interp(target_pressure, pressures, adsorptions))
                        target_unit = unit
                    except:
                        target_ads = None
                        target_unit = None
                break

        if target_ads is None:
            print(f"No valid adsorption at {target_temp}K, {target_pressure}bar for {fname}")
            continue

        # 单位转换：全部统一成 mol/kg
        if target_unit == "mol/kg":
            ads_molkg = target_ads
        elif target_unit == "g/l":
            # 1.先转 g/g (g/l / 1000 / density)，再除以2.016得到mol/g，再*1000变成mol/kg
            ads_molkg = (target_ads / density) / 2.016
        elif target_unit == "cm3(STP)/cm3":
            # 2. 1 mol H2 (STP) = 22414 cm³
            ads_molkg = (target_ads / density) / 22414 * 1000
        else:
            print(f"Unknown unit {target_unit} for {fname}, skip.")
            continue

        rows.append({
            "id": js.get('name', os.path.splitext(fname)[0]),
            "cif_path": cif_path,
            "target": ads_molkg
        })

# 合并成DataFrame
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Done! {len(df)} records saved to {output_csv}")

# ----------- 数据集划分 ---------------
trainval, test = train_test_split(df, test_size=0.15, random_state=42)
train, val = train_test_split(trainval, test_size=0.1765, random_state=42)

train.to_csv('mof_train.csv', index=False)
val.to_csv('mof_val.csv', index=False)
test.to_csv('mof_test.csv', index=False)
print(f"训练集: {len(train)}, 验证集: {len(val)}, 测试集: {len(test)} 条样本")