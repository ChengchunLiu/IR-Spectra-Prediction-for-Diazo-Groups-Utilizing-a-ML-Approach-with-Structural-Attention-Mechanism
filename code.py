from rdkit.Chem import AllChem
from rdkit import Chem
from joblib import dump, load
from catboost import CatBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import math
from sklearn.model_selection import train_test_split, KFold
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.pipeline import make_pipeline

excel_file = r'data.xlsx'
df = pd.read_excel(excel_file)

def process_smiles(smiles_str):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        mol = Chem.AddHs(mol)

        # Find the positively charged nitrogen atom (N+)
        nplus_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() == 1]
        if not nplus_atoms:
            return None, None, None, None
        nplus = mol.GetAtomWithIdx(nplus_atoms[0])

        # Find the carbon atom connected to N+
        carbon_neighbors = [neighbor for neighbor in nplus.GetNeighbors() if neighbor.GetSymbol() == "C"]
        if not carbon_neighbors:
            return "NoCarbonFound", None, None, None
        carbon = carbon_neighbors[0]

        # Find other atoms connected to the carbon, excluding N+
        connected_atoms = [neighbor for neighbor in carbon.GetNeighbors() if neighbor.GetIdx() != nplus.GetIdx()]

        # Priority order for connected atoms
        priority_order = [
            ('Cl', 'SINGLE'), ('S', 'AROMATIC'), ('S', 'SINGLE'), ('F', 'SINGLE'),
            ('O', 'AROMATIC'), ('O', 'DOUBLE'), ('O', 'SINGLE'),
            ('N', 'TRIPLE'), ('N', 'AROMATIC'), ('N', 'DOUBLE'), ('N', 'SINGLE'),
            ('C', 'TRIPLE'), ('C', 'AROMATIC'), ('C', 'DOUBLE'), ('C', 'SINGLE'),
            ('H', 'SINGLE')
        ]

        # Create a list to store atom atomic number and connections
        connections_list = []
        for atom in connected_atoms:
            atomic_num = atom.GetAtomicNum()
            connected_atom_symbol = atom.GetSymbol()
            neighbors_info = []
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() != carbon.GetIdx():
                    neighbor_symbol = neighbor.GetSymbol()
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    bond_type_str = str(bond.GetBondType())
                    neighbors_info.append((neighbor_symbol, bond_type_str))

            connections_dict = {'atomic_num': atomic_num, 'connections': {connected_atom_symbol: neighbors_info}}
            connections_list.append(connections_dict)

        # Determine R1 and R2 based on atomic number and connection priority
        connections_list.sort(key=lambda x: x['atomic_num'], reverse=True)
        if len(connections_list) > 1 and connections_list[0]['atomic_num'] == connections_list[1]['atomic_num']:
            # If atomic numbers are the same, sort by connection priority
            connections_list.sort(key=lambda x: min(priority_order.index(y) if y in priority_order else len(priority_order)
                                                    for y in [item for sublist in x['connections'].values() for item in sublist]))

        R1, R2 = connections_list[0], connections_list[1] if len(connections_list) > 1 else None
        atomic_number_R1 = R1['atomic_num']
        atomic_number_R2 = R2['atomic_num'] if R2 else None

        return R1, R2, atomic_number_R1, atomic_number_R2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

def calculate_morgan_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return list(fp)
def Feature_Engineering():

    possible_columns = [
        'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
         'F_SINGLE_R1',  'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2', 'O_SINGLE_R1', 'O_SINGLE_R2',
        'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2', 'N_SINGLE_R1', 'N_SINGLE_R2',
         'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1',  'C_DOUBLE_R1',  'C_DOUBLE_R2',  'C_SINGLE_R1', 'C_SINGLE_R2',
          'H_SINGLE_R1', 'H_SINGLE_R2'

    ]

    results = []

    for index, row in df.iterrows():
        smiles_str = row['SMILES']
        R1, R2, atomic_number_R1, atomic_number_R2 = process_smiles(smiles_str)
        # 初始化计数字典
        counts = {'R1' : R1, 'R2' : R2, 'atomic_number_R1': atomic_number_R1, 'atomic_number_R2': atomic_number_R2, 'SMILES': smiles_str, 'N2_IR_Characteristic_Peak': row['N2_IR_Characteristic_Peak'], 'File Name' : row['File Name'], 'name': row['name'], 'Corresponding Author': row['Corresponding Author']}
        counts.update({col: 0 for col in possible_columns})


        if R1 == "NoCarbonFound":
            continue

        fingerprint = calculate_morgan_fingerprint(smiles_str)
        for i, bit in enumerate(fingerprint):
            counts[f'Fingerprint_{i}'] = bit

        for suffix, connections in [('R1', R1), ('R2', R2)]:
            if connections:
                for connection in connections['connections'].values():
                    for bond in connection:
                        bond_type_str = f'{bond[0]}_{bond[1]}'
                        counts[f'{bond_type_str}_{suffix}'] = counts.get(f'{bond_type_str}_{suffix}', 0) + 1

        results.append(counts)


    results_df = pd.DataFrame(results)
    results_df_filled = results_df.fillna(0)
    electronegativity_dict = {
        1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }

    # Define the covalent radius dictionary
    covalent_radius_dict = {
        1: 37, 6: 77, 7: 75, 8: 73, 9: 71, 14: 111, 15: 106, 16: 102, 17: 99, 35: 114, 53: 133
    }

    # Add electronegativity and covalent radius columns
    results_df_filled['electronegativity_R1'] = results_df_filled['atomic_number_R1'].map(electronegativity_dict)
    results_df_filled['electronegativity_R2'] = results_df_filled['atomic_number_R2'].map(electronegativity_dict)
    results_df_filled['covalent_radius_R1'] = results_df_filled['atomic_number_R1'].map(covalent_radius_dict)
    results_df_filled['covalent_radius_R2'] = results_df_filled['atomic_number_R2'].map(covalent_radius_dict)

    # Replace NaN values if necessary (optional)
    results_df_filled['electronegativity_R1'].fillna(0, inplace=True)
    results_df_filled['electronegativity_R2'].fillna(0, inplace=True)
    results_df_filled['covalent_radius_R1'].fillna(0, inplace=True)
    results_df_filled['covalent_radius_R2'].fillna(0, inplace=True)

    print("FE Processing complete.")
    return results_df_filled

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # Predict on training and test sets
    y_train_pred = model.predict(X_train).ravel()
    y_test_pred = model.predict(X_test).ravel()

    # Compute performance metrics
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_test = r2_score(y_test, y_test_pred)

    return model, y_train_pred, rmse_train, r2_train, y_test_pred, rmse_test, r2_test

def plot_true_vs_pred(y_train, y_train_pred, y_test, y_test_pred, model_name, r2_train, rmse_train, r2_test, rmse_test):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    # Adjust values for plotting
    adjusted_y_train = y_train + 2104
    adjusted_y_train_pred = y_train_pred + 2104
    adjusted_y_test = y_test + 2104
    adjusted_y_test_pred = y_test_pred + 2104

    # Plot train data in deep blue and test data in orange
    plt.scatter(adjusted_y_train, adjusted_y_train_pred, c='#00008B', s=15, alpha=0.4, label='Train')
    plt.scatter(adjusted_y_test, adjusted_y_test_pred, c='#FFA500', s=15, alpha=0.4, label='Test')

    plt.xlabel('True value', fontproperties='Arial', size=15)
    plt.ylabel("Predicted value", fontproperties='Arial', size=15)
    plt.title(f"{model_name}", fontproperties='Arial', size=20)

    # Draw a line for reference
    plt.plot([2000, 2200], [2000, 2200], linewidth=1, linestyle='--', color='black')

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.text(2030, 2180, f'Train R² = {r2_train:.3f}\nTest R² = {r2_test:.3f}', fontsize=10, fontproperties='Arial')
    plt.text(2030, 2160, f'Train RMSE = {rmse_train:.3f}\nTest RMSE = {rmse_test:.3f}', fontsize=10, fontproperties='Arial')

    plt.xlim([2000, 2200])
    plt.ylim([2000, 2200])

    plt.legend()
    plt.tight_layout()
    #plt.savefig(f'onlyMorgan_train and test//True_vs_Predicted_{model_name}.svg', bbox_inches='tight', dpi=300)
    plt.show()
def plot_learning_curve(estimator, X_train, y_train, X_test, y_test, cv, total_data_size, train_sizes):
    plt.style.use('ggplot')  # Apply the ggplot style for consistent styling
    _, axes = plt.subplots(1, 3, figsize=(20, 3), dpi=300)

    titles = ["Learning Curve (MSE)", "Learning Curve (RMSE)", "Learning Curve (R²)"]
    metrics = [mean_squared_error, lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred)), r2_score]

    for i in range(3):
        axes[i].set_title(titles[i], fontsize=15)
        axes[i].set_xlabel("Training examples (%)", fontsize=12)
        axes[i].set_ylabel("Score" if i != 2 else "R² Score", fontsize=12)

        best_train_scores, best_val_scores, best_test_scores = [], [], []

        for m in train_sizes:
            num_examples = int(m * total_data_size)
            X_train_subset, y_train_subset = X_train[:num_examples], y_train[:num_examples]
            estimator.fit(X_train_subset, y_train_subset)

            fold_best_val_scores = []
            for train_index, val_index in cv.split(X_train_subset):
                X_fold_train, X_fold_val = X_train_subset[train_index], X_train_subset[val_index]
                y_fold_train, y_fold_val = y_train_subset[train_index], y_train_subset[val_index]

                estimator.fit(X_fold_train, y_fold_train)
                y_fold_val_pred = estimator.predict(X_fold_val)

                fold_score = metrics[i](y_fold_val, y_fold_val_pred)
                fold_best_val_scores.append(fold_score)

            best_val_score = max(fold_best_val_scores) if i == 2 else min(fold_best_val_scores)

            y_train_pred = estimator.predict(X_train_subset)
            train_score = metrics[i](y_train_subset, y_train_pred)

            y_test_pred = estimator.predict(X_test)
            test_score = metrics[i](y_test, y_test_pred)

            best_train_scores.append(train_score)
            best_val_scores.append(best_val_score)
            best_test_scores.append(test_score)

        train_size_percentage = [m * 100 for m in train_sizes]

        # Enable grid with default ggplot parameters
        axes[i].grid(True)

        axes[i].plot(train_size_percentage, best_train_scores, 'o-', color="#00008B", label="Training score")
        axes[i].plot(train_size_percentage, best_val_scores, 'o-', color="#FFA500", label="Cross-validation score")
        axes[i].plot(train_size_percentage, best_test_scores, 'o-', color="green", label="Test score")
        axes[i].legend(loc="best")

    plt.tight_layout()
    #plt.savefig('learning_curve.svg', bbox_inches='tight', dpi=300)
    plt.show()

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # 只计算那些 y_true 非零的 MRE
    non_zero_mask = y_true != 0
    mre = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) if np.any(non_zero_mask) else np.nan

    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, mre, r2
def calculate_similarity(train_smiles, test_smiles):
    """计算训练集和测试集之间的Tanimoto相似度"""
    train_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in train_smiles]
    test_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smile), 2) for smile in test_smiles]

    similarity_scores = []
    for test_fp in test_fps:
        max_similarity = max(AllChem.DataStructs.TanimotoSimilarity(test_fp, train_fp) for train_fp in train_fps)
        similarity_scores.append(max_similarity)
    return similarity_scores

def plot_metric_scores(similarity_thresholds, scores, metric_name, filename, facecolor='white'):
    plt.style.use('ggplot')  # 使用 ggplot 样式
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # 设置图表背景色
    fig.patch.set_facecolor(facecolor)

    # 转换相似度阈值为百分比格式
    similarity_thresholds_percent = [f'{threshold * 100:.0f}%' for threshold in similarity_thresholds]

    # 绘制分数点
    ax.scatter(similarity_thresholds_percent, scores, color='b')

    # 在每个点上标注数值
    for i, txt in enumerate(scores):
        ax.annotate(f'{txt:.3f}', (similarity_thresholds_percent[i], scores[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # 设置标题和标签
    ax.set_title(f'Model Performance - {metric_name}', fontproperties='Arial', size=20)
    ax.set_xlabel('Similarity Threshold', fontproperties='Arial', size=15)
    ax.set_ylabel(metric_name, fontproperties='Arial', size=15)

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.5)

    # 保存图像
    plt.tight_layout()
    #plt.savefig(f'{filename}.svg', bbox_inches='tight', dpi=300)
    plt.show()
def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise
def plot_metric_scores_noise(noise_levels, scores, metric_name, filename, facecolor='white'):
    plt.style.use('ggplot')  # 使用 ggplot 样式
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # 设置图表背景色
    fig.patch.set_facecolor(facecolor)

    # 转换噪声水平为百分比格式
    noise_levels_percent = [f'{level * 100:.0f}%' for level in noise_levels]

    # 绘制分数点
    ax.scatter(noise_levels_percent, scores, color='b')

    # 在每个点上标注数值
    for i, txt in enumerate(scores):
        ax.annotate(f'{txt:.3f}', (noise_levels_percent[i], scores[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # 设置标题和标签
    ax.set_title(f'Model Performance - {metric_name}', fontproperties='Arial', size=20)
    ax.set_xlabel('Noise Level', fontproperties='Arial', size=15)
    ax.set_ylabel(metric_name, fontproperties='Arial', size=15)

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.5)

    # 保存图像
    plt.tight_layout()
    #plt.savefig(f'{filename}.svg', bbox_inches='tight', dpi=300)

    plt.show()

def Model_construction():
    df = Feature_Engineering()
    print("Total number of rows in the data:", len(df))

    y = df['N2_IR_Characteristic_Peak'].values - 2104

    feature = ['electronegativity_R1', 'electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2',
               'Cl_SINGLE_R1', 'Cl_SINGLE_R2',
               'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
               'F_SINGLE_R1', 'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2',
               'O_SINGLE_R1', 'O_SINGLE_R2',
               'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2',
               'N_SINGLE_R1', 'N_SINGLE_R2',
               'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1', 'C_DOUBLE_R1', 'C_DOUBLE_R2',
               'C_SINGLE_R1', 'C_SINGLE_R2',
               'H_SINGLE_R1', 'H_SINGLE_R2'
               ]
    Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]

    all_features = feature + Morgan_features
    X = df[all_features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgbm', lgb.LGBMRegressor(random_state=42))
    ]

    # Define stacked regressor
    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=BayesianRidge()
    )

    xgb =  XGBRegressor()

    # Define voting regressor
    mixture_model = VotingRegressor(
        estimators=[('stacked', stacked_model), ('xgb', xgb)]
    )

    # Update the models dictionary
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        'Bayesian Ridge Regression':BayesianRidge(),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(random_state=42),
        #"Mixture Model": mixture_model
    }

    all_models = []
    for model_name, model in models.items():
        print(f'\nTraining {model_name}...')
        model, y_train_pred, rmse_train, r2_train, y_test_pred, rmse_test, r2_test = train_evaluate_model(model,
                                                                                                          X_train,
                                                                                                          y_train,
                                                                                                          X_test,
                                                                                                          y_test)
        plot_true_vs_pred(y_train, y_train_pred, y_test, y_test_pred, model_name, r2_train, rmse_train, r2_test,
                          rmse_test)

    #print('Model saving')
    #dump(mixture_model, 'mixture_model.joblib')
    #dump(scaler, 'scaler.joblib')

def Cross_validation():
    df = Feature_Engineering()
    print("Total number of rows in the data:", len(df))

    y = df['N2_IR_Characteristic_Peak'].values - 2104
    feature = ['electronegativity_R1', 'electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2',
               'Cl_SINGLE_R1', 'Cl_SINGLE_R2',
               'S_AROMATIC_R1', 'S_AROMATIC_R2', 'S_SINGLE_R1', 'S_SINGLE_R2',
               'F_SINGLE_R1', 'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2',
               'O_SINGLE_R1', 'O_SINGLE_R2',
               'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2',
               'N_SINGLE_R1', 'N_SINGLE_R2',
               'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1', 'C_DOUBLE_R1', 'C_DOUBLE_R2',
               'C_SINGLE_R1', 'C_SINGLE_R2',
               'H_SINGLE_R1', 'H_SINGLE_R2']
    Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]

    all_features = feature + Morgan_features
    X = df[all_features].values

    # 2/8分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgbm', lgb.LGBMRegressor(random_state=42))
    ]

    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=BayesianRidge()
    )

    xgb =  XGBRegressor()

    mixture_model = VotingRegressor(
        estimators=[('stacked', stacked_model), ('xgb', xgb)]
    )
    model = mixture_model

    kf = KFold(n_splits=15, shuffle=True, random_state=42)

    total_data_size = len(X)
    train_sizes = [0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    plot_learning_curve(model, X_train, y_train, X_test, y_test, cv=kf, total_data_size=total_data_size,
                        train_sizes=train_sizes)

def Similarity_high_to_low():
    df = Feature_Engineering()
    print("Total number of rows in the data:", len(df))
    df['N2_IR_Characteristic_Peak'] -= 2104
    features = ['electronegativity_R1','electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2', 'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1',
                'S_AROMATIC_R2',
                'S_SINGLE_R1', 'S_SINGLE_R2', 'F_SINGLE_R1', 'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2',
                'O_DOUBLE_R1', 'O_DOUBLE_R2', 'O_SINGLE_R1', 'O_SINGLE_R2', 'N_TRIPLE_R2', 'N_TRIPLE_R1',
                'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2', 'N_SINGLE_R1', 'N_SINGLE_R2',
                'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1', 'C_DOUBLE_R1', 'C_DOUBLE_R2',
                'C_SINGLE_R1', 'C_SINGLE_R2', 'H_SINGLE_R1', 'H_SINGLE_R2']
    Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]
    all_features = features + Morgan_features
    X = df[all_features]
    y = df['N2_IR_Characteristic_Peak'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    index_mapping = {index: i for i, index in enumerate(X_test.index)}

    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgbm', lgb.LGBMRegressor(random_state=42))
    ]

    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=BayesianRidge()
    )

    xgb = XGBRegressor()

    mixture_model = VotingRegressor(
        estimators=[('stacked', stacked_model), ('xgb', xgb)]
    )
    model = mixture_model

    model.fit(X_train_scaled, y_train)

    train_smiles = df.loc[X_train.index, 'SMILES']
    test_smiles = df.loc[X_test.index, 'SMILES']
    similarity_scores = calculate_similarity(train_smiles, test_smiles)
    test_df = pd.DataFrame({'SMILES': test_smiles, 'Similarity': similarity_scores, 'Index': X_test.index})
    test_df.set_index('Index', inplace=True)

    thresholds = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.65, 0.50]
    metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'MRE': [], 'R2': []}

    total_test_samples = len(X_test_scaled)
    print("Total number of test samples:", total_test_samples)

    for threshold in thresholds:

        group_indices = test_df[test_df['Similarity'] >= threshold].index
        mapped_indices = [index_mapping[idx] for idx in group_indices if idx in index_mapping]

        X_group_scaled = X_test_scaled[mapped_indices]
        y_group = y_test[mapped_indices]

        y_pred = model.predict(X_group_scaled)

        mse, rmse, mae, mre, r2 = evaluate_model(y_group, y_pred)

        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['MRE'].append(mre)
        metrics['R2'].append(r2)

        group_size = len(X_group_scaled)
        group_percentage = (group_size / total_test_samples) * 100
        print(f"Threshold: {threshold}, Group Size: {group_size}, Percentage: {group_percentage:.2f}%")
        print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MRE: {mre}, R2: {r2}")

    for metric_name, values in metrics.items():
        plot_metric_scores(thresholds, values, metric_name, f'{metric_name}_similarity_high_to_low')

def Similarity_low_to_high():
    df = Feature_Engineering()
    print("Total number of rows in the data:", len(df))
    df['N2_IR_Characteristic_Peak'] -= 2104

    features = ['electronegativity_R1','electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2', 'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1',
                'S_AROMATIC_R2',
                'S_SINGLE_R1', 'S_SINGLE_R2', 'F_SINGLE_R1', 'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2',
                'O_DOUBLE_R1', 'O_DOUBLE_R2', 'O_SINGLE_R1', 'O_SINGLE_R2', 'N_TRIPLE_R2', 'N_TRIPLE_R1',
                'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2', 'N_SINGLE_R1', 'N_SINGLE_R2',
                'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1', 'C_DOUBLE_R1', 'C_DOUBLE_R2',
                'C_SINGLE_R1', 'C_SINGLE_R2', 'H_SINGLE_R1', 'H_SINGLE_R2']
    Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]
    all_features = features + Morgan_features
    X = df[all_features]
    y = df['N2_IR_Characteristic_Peak'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    index_mapping = {index: i for i, index in enumerate(X_test.index)}

    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgbm', lgb.LGBMRegressor(random_state=42))
    ]

    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=BayesianRidge()
    )

    xgb = XGBRegressor()

    mixture_model = VotingRegressor(
        estimators=[('stacked', stacked_model), ('xgb', xgb)]
    )
    model = mixture_model

    model.fit(X_train_scaled, y_train)

    train_smiles = df.loc[X_train.index, 'SMILES']
    test_smiles = df.loc[X_test.index, 'SMILES']
    similarity_scores = calculate_similarity(train_smiles, test_smiles)
    test_df = pd.DataFrame({'SMILES': test_smiles, 'Similarity': similarity_scores, 'Index': X_test.index})
    test_df.set_index('Index', inplace=True)

    thresholds = [0.50, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90]
    metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'MRE': [], 'R2': []}
    total_test_samples = len(X_test_scaled)
    print("Total number of test samples:", total_test_samples)

    accumulated_indices = []

    for threshold in thresholds:

        new_indices = test_df[test_df['Similarity'] <= threshold].index
        mapped_new_indices = [index_mapping[idx] for idx in new_indices if idx in index_mapping]

        accumulated_indices.extend(mapped_new_indices)

        accumulated_indices = list(set(accumulated_indices))

        X_group_scaled = X_test_scaled[accumulated_indices]
        y_group = y_test[accumulated_indices]

        y_pred = model.predict(X_group_scaled)

        mse, rmse, mae, mre, r2 = evaluate_model(y_group, y_pred)

        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['MRE'].append(mre)
        metrics['R2'].append(r2)

        group_size = len(X_group_scaled)
        group_percentage = (group_size / total_test_samples) * 100
        print(f"Threshold: {threshold}, Group Size: {group_size}, Percentage: {group_percentage:.2f}%")
        print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MRE: {mre}, R2: {r2}")

    for metric_name, values in metrics.items():
        plot_metric_scores(thresholds, values, metric_name, f'{metric_name}_low_to_high_similarity')

def noise():
    df = Feature_Engineering()
    print("Total number of rows in the data:", len(df))
    df['N2_IR_Characteristic_Peak'] -= 2104

    # 特征选择
    features = ['electronegativity_R1','electronegativity_R2', 'covalent_radius_R1', 'covalent_radius_R2', 'Cl_SINGLE_R1', 'Cl_SINGLE_R2', 'S_AROMATIC_R1', 'S_AROMATIC_R2',
               'S_SINGLE_R1', 'S_SINGLE_R2',
               'F_SINGLE_R1', 'F_SINGLE_R2', 'O_AROMATIC_R1', 'O_AROMATIC_R2', 'O_DOUBLE_R1', 'O_DOUBLE_R2',
               'O_SINGLE_R1', 'O_SINGLE_R2',
               'N_TRIPLE_R2', 'N_TRIPLE_R1', 'N_AROMATIC_R1', 'N_AROMATIC_R2', 'N_DOUBLE_R1', 'N_DOUBLE_R2',
               'N_SINGLE_R1', 'N_SINGLE_R2',
               'C_TRIPLE_R1', 'C_TRIPLE_R2', 'C_AROMATIC_R2', 'C_AROMATIC_R1', 'C_DOUBLE_R1', 'C_DOUBLE_R2',
               'C_SINGLE_R1', 'C_SINGLE_R2',
               'H_SINGLE_R1', 'H_SINGLE_R2'
               ]
    Morgan_features = [f'Fingerprint_{i}' for i in range(2048)]
    all_features = features + Morgan_features
    X = df[all_features]
    y = df['N2_IR_Characteristic_Peak'].values

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgbm', lgb.LGBMRegressor(random_state=42))
    ]

    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=BayesianRidge()
    )

    xgb = XGBRegressor()

    mixture_model = VotingRegressor(
        estimators=[('stacked', stacked_model), ('xgb', xgb)]
    )
    model = mixture_model

    model.fit(X_train_scaled, y_train)

    noise_levels = np.arange(0, 1.01, 0.1)  # 从 0% 到 100%
    metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'MRE': [], 'R2': []}
    for noise_level in noise_levels:
        X_train_noisy = add_noise(X_train_scaled, noise_level)
        X_test_noisy = add_noise(X_test_scaled, noise_level)

        model.fit(X_train_noisy, y_train)
        y_pred_noisy = model.predict(X_test_noisy)

        mse, rmse, mae, mre, r2 = evaluate_model(y_test, y_pred_noisy)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['MAE'].append(mae)
        metrics['MRE'].append(mre)
        metrics['R2'].append(r2)

        print(f"Noise Level: {noise_level * 100:.0f}%")
        print(f"  MSE: {mse:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  MRE: {mre:.3f}")
        print(f"  R2: {r2:.3f}\n")

    for metric_name, values in metrics.items():
        plot_metric_scores_noise(noise_levels, values, metric_name, f"{metric_name}_noise")

if __name__ == '__main__':
    #Model_construction()
    #Cross_validation()
    #Similarity_high_to_low()
    #Similarity_low_to_high()
    #noise()