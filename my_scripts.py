# Вспомогательная функция для подсчета общего числа значений в парах
def pair_counts_df(dataframe, columne_name_1, columne_name_2):
    import pandas as pd
    if columne_name_1 not in dataframe.columns or columne_name_2 not in dataframe.columns:
        print('Wrong column name')
        return
    aid_df = dataframe[[columne_name_1, columne_name_2]].value_counts()
    unique_values_1, unique_values_2 = dataframe[columne_name_1].unique(), dataframe[columne_name_2].unique()
    return pd.DataFrame([[aid_df[(un_value_1, un_value_2)] if (un_value_1, un_value_2) in aid_df.index else 0
                          for un_value_2 in unique_values_2]
                         for un_value_1 in unique_values_1], index=unique_values_1, columns=unique_values_2)


# Функция позволяющая выбирать данные из датасета для построения графиков.
def get_subset_for_graphs(metal, halogen, df, bg_type=None, x_name='aver Hal...Hal', y_name='Band gap',
                          for_plotly=False):
    if metal not in ['Bi', 'Sb', 'Bi Sb']:
        print('Wrong metal!!!')
        return get_subset_for_graphs(input('Metal: '), halogen, df=df, bg_type=bg_type, x_name=x_name, y_name=y_name)
    if halogen not in ['Cl', 'Br', 'I', 'Cl Br', 'Br I', 'Cl I']:
        print('Wrong halogen!!!')
        return get_subset_for_graphs(metal, input('Halogen: '), df=df, bg_type=bg_type, x_name=x_name, y_name=y_name)
    if (bg_type is not None) and (bg_type not in ['d', 'i', 'cht', 'unk']):
        print('Wrong BG type!!!')
        return get_subset_for_graphs(metal, halogen, df=df, bg_type=input('BG Type: '), x_name=x_name, y_name=y_name)
    if x_name not in df.columns:
        print('Wrong name of columns for x axis!!!')
        return get_subset_for_graphs(metal, halogen, df=df, bg_type=bg_type, x_name=input('x name: '), y_name=y_name)
    if y_name not in df.columns:
        print('Wrong name of columns for y axis!!!')
        return get_subset_for_graphs(metal, halogen, df=df, bg_type=bg_type, x_name=x_name, y_name=input('y name: '))
    if bg_type is None:
        x = df[(df[y_name].notna()) & (df[x_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen)][x_name]
        y = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen)][y_name]
        t = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen)]['REFCODE']
    elif 'unknown' == bg_type:
        x = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen) & (df['BG Type'].isna())][x_name]
        y = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen) & (df['BG Type'].isna())][y_name]
        t = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen) & (df['BG Type'].isna())]['REFCODE']
    else:
        x = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen) & (df['BG Type'] == bg_type)][x_name]
        y = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen) & (df['BG Type'] == bg_type)][y_name]
        t = df[(df[x_name].notna()) & (df[y_name].notna()) & (df['Type MHal4'] == 'a') & (df['M(III)'] == metal) & (
                df['Hal'] == halogen) & (df['BG Type'] == bg_type)]['REFCODE']
    if for_plotly:
        return dict(x=x, y=y, text=t)
    return x, y


# Словарь цветов для раскраски графиков
style_dict = {('Bi', 'I', 'd'): '#300072',
              ('Bi', 'I', 'i'): '#000031',
              ('Bi', 'I', 'cht'): '#9b2222',
              ('Bi', 'I', 'unk'): '#ff00ff',
              ('Bi', 'Cl', 'd'): '#00ffff',
              ('Bi', 'Cl', 'cht'): '#fa8072',
              ('Bi', 'Br', 'd'): '#006400',
              ('Bi', 'Br', 'i'): '#22cc22',
              ('Bi', 'Br', 'cht'): '#8b0000',
              ('Bi', 'Br I', 'd'): '#a400d3',
              ('Sb', 'I', 'd'): '#ff5510',
              ('Sb', 'I', 'cht'): '#b22222',
              ('Sb', 'I', 'unk'): '#f71585',
              ('Sb', 'Cl', 'd'): '#adff2f',
              ('Sb', 'Cl', 'i'): '#0000cd',
              ('Sb', 'Cl', 'cht'): '#ff4500',
              ('Sb', 'Cl', 'unk'): '#2e8b57',
              ('Sb', 'Br', 'd'): '#ffd700',
              ('Sb', 'Br', 'cht'): '#ff0000',
              ('Sb', 'Br', 'unk'): '#009090',
              ('Sb', 'Br I', 'd'): '#fdfd00',
              ('Bi Sb', 'I', 'd'): '#808000',
              ('Bi Sb', 'Br I', 'd'): '#a9a9a9'}


# Функция для отрисовки графиков:
def plotly_graphs(l, x_name, y_name, df, leg_pos=None, save=False, name_to_save=None, sd=None,
                  x_range=(3.4, 5.3), y_range=(1.5, 3.5), dx0=0., dx=0.1, dy0=0., dy=0.1):
    if sd is None:
        sd = style_dict
    from plotly import graph_objs as go

    def make_axis_names(df):
        # Правильные подписи для осей на графиках
        axis_name_dict = {'Band gap': 'Band gap, eV',
                          'aver Hal...Hal': 'Average Hal...Hal distance, Å',
                          'min Hal...Hal': 'Minimal Hal...Hal distance, Å',
                          'Temperature': 'Temperature °',
                          'delta d': r'$\Delta\text{d}$',
                          'sigma^2': r'$\sigma^{2}$',
                          'N/aver-d': r'$\frac{\text{N(Hal...Hal)}}{\bar{d}\left(\text{Hal...Hal}\right)}$'}
        axis_name_dict = dict(
            list(axis_name_dict.items()) +
            [(d, 'M—' + d + ' distance, Å') for d in df.columns[19:25]] +
            [(a, 'Hal-' + a.split('-')[0] + '—M—' + a.split('-')[1] + '-Hal angle, °') for a in df.columns[25:40]]
        )
        return axis_name_dict

    axis_name_dict = make_axis_names(df)

    if leg_pos is None:
        leg_pos = dict(x=0.8, y=0.5)
    fig = go.Figure()
    for m, h, t, name in l:
        if t == 'cht' and m == 'Sb':
            marker_ring = sd[(m, h, t)]
            marker_color = '#ffffff'
        elif t == 'i':
            marker_ring = sd[(m, h, t)]
            marker_color = '#ffffff'
        else:
            marker_ring = sd[(m, h, t)]
            marker_color = sd[(m, h, t)]
        fig.add_trace(go.Scatter(
            get_subset_for_graphs(m, h, df=df, bg_type=t, x_name=x_name, y_name=y_name, for_plotly=True),
            name=name,
            mode='markers',
            marker=dict(size=8, color=marker_color, line=dict(color=marker_ring, width=3))))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='#000000', mirror=True,
                     title=dict(text=axis_name_dict[x_name], font=dict(size=26)),
                     range=x_range, tick0=dx0, dtick=dx)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#000000', mirror=True,
                     title=dict(text=axis_name_dict[y_name], font=dict(size=26)),
                     range=y_range, tick0=dy0, dtick=dy)
    fig.update_layout(template='simple_white', width=1200, height=700, legend=leg_pos)
    fig.update_traces(hoverinfo='all',
                      hovertemplate=f"{axis_name_dict[x_name].split(', ')[0]}" +
                                    " %{x}<br>" + f"{axis_name_dict[y_name].split(', ')[0]}"
                                    + " %{y}<br>%{text}")
    if save:
        if name_to_save is None:
            name_to_save = input()
        fig.write_image('images/' + name_to_save + '.png', scale=5)
        fig.write_html('images/' + name_to_save + '.html')
    else:
        fig.show()


# Функция для выбора дескрипторов (не обязательны):
def get_descriptors_by_ent(data, metal_le=True, halogen_le=False, number_of_geom=21, names_of_geom=None,
                           hh_min=True, hh_av=False, hh_num=False, hh_nd=False, del_d=False, sigma=False):
    descriptors_names = ['Band gap', 'Temperature']
    if metal_le:
        descriptors_names.append('M')
    else:
        descriptors_names.extend(['Bi', 'Sb'])
    if halogen_le:
        descriptors_names.append('X')
    else:
        descriptors_names.extend(['I', 'Br', 'I'])
    if hh_min:
        descriptors_names.append('min Hal...Hal')
    if hh_av:
        descriptors_names.append('aver Hal...Hal')
    if hh_num:
        descriptors_names.append('Number if Hal...Hal contacts')
    if hh_nd:
        descriptors_names.append('N/aver-d')
    if del_d:
        descriptors_names.append('delta d')
    if sigma:
        descriptors_names.append('sigma^2')
    all_geom = ['Hal-t1', 'Hal-t2', 'Hal-d1', 'Hal-d2', 'Hal-d3', 'Hal-d4', 't1-t2', 't1-d1', 't1-d2', 't1-d3', 't1-d4',
                't2-d1', 't2-d2', 't2-d3', 't2-d4', 'd1-d2', 'd1-d3', 'd1-d4', 'd2-d3', 'd2-d4', 'd3-d4']
    if number_of_geom == 21:
        descriptors_names.extend(all_geom)
    elif number_of_geom > 21 or number_of_geom <= 0:
        print('Wrong!')
        return
    elif 1 <= number_of_geom < 21:
        if names_of_geom == None:
            names_of_geom = input('string if descriptors designers or indesies:  ')
        if (type(names_of_geom) == str) and names_of_geom.isdigit():
            names_of_geom = list(map(int, names_of_geom.split()))
            if (len(names_of_geom) == number_of_geom) and all([14 <= x < 35 for x in names_of_geom]):
                descriptors_names.extend(data.columns[:names_of_geom].to_list())
            else:
                print('Wrong!')
                return
        elif (type(names_of_geom) == str) and all([x in all_geom] for x in names_of_geom.split()):
            descriptors_names.extend(names_of_geom.split())
        elif type(names_of_geom) == list or type(names_of_geom) == tuple:
            if all([14 <= x < 35 for x in names_of_geom]):
                descriptors_names.extend(data.columns[:names_of_geom].to_list())
            elif all([x in all_geom] for x in names_of_geom.split()):
                descriptors_names.extend(list(names_of_geom))
            else:
                print('Wrong!')
                return
    if len(set(descriptors_names) & set(all_geom)) != 0:  # Тут не забываем#
        return data[data['Hal-t1'].notna()][descriptors_names].copy()
    return data[descriptors_names].copy()


# Функция для разбиения на трейн и тест и нормализации
def train_test_split_and_normolize(data, drop=True, scaler=None, train_size=0.85, random_state=7,
                                   shuffle=True, stratify=None):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

    if scaler is None:
        scaler = MinMaxScaler()
    else:
        scaler = scaler()
    data = data.copy()

    if drop:
        data = data.dropna()
    x, y = data.drop(columns=['REFCODE', 'Band gap']), data['Band gap']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state,
                                                        shuffle=shuffle, stratify=stratify)
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    print('In train set', x_train.shape[0], 'In test set', x_test.shape[0])
    return x_train, x_test, y_train, y_test, scaler


# Один цикл обучения и предсказания
def train_val_loop(x, y, train_indices, val_index, model, models, metrics_dict, dict_to_save, estmator_dict):
    import pandas as pd

    idx = y.iloc[val_index].index[0]
    x_tr, x_val = x.iloc[train_indices], x.iloc[val_index]
    y_tr, y_val = y.iloc[train_indices], y.iloc[val_index]
    regressor = models[model]
    regressor.fit(x_tr.to_numpy(), y_tr.to_numpy())
    y_tr_pred = regressor.predict(x_tr.to_numpy())
    y_tr_pred = pd.Series(y_tr_pred, name=y_tr.name, index=y_tr.index)
    y_val_pred = regressor.predict(x_val.to_numpy())
    y_val_pred = pd.Series(y_val_pred, name=y_val.name, index=y_val.index)
    aid_dict = {}
    for name, metric in metrics_dict.items():
        if name == 'RMSE':
            aid_dict[name + ' on train'] = metric(y_tr, y_tr_pred, squared=False)
            aid_dict[name + ' on validate'] = metric(y_val, y_val_pred, squared=False)
        elif name == 'R2':
            aid_dict[name + ' on train'] = metric(y_tr, y_tr_pred)
        else:
            aid_dict[name + ' on train'] = metric(y_tr, y_tr_pred)
            aid_dict[name + ' on validate'] = metric(y_val, y_val_pred)
    dict_to_save[idx] = aid_dict
    estmator_dict[idx] = regressor


# Полный цикл LOO
def lvo_cv(x, y, model, models, metrics_dict):
    import pandas as pd
    from sklearn.model_selection import LeaveOneOut

    results = {}
    estimators = {}
    loo = LeaveOneOut()
    for train_indicies, val_index in loo.split(x):
        train_val_loop(x, y, train_indicies, val_index, model, models, metrics_dict, results, estimators)
    return pd.DataFrame(results), estimators


# Функция для последовательной тренировки нескольких моделей из словаря
def full_cv(x, y, models_dict, metrics_dict):
    main_dict = {}
    for model in models_dict.keys():
        print(model)
        results, estimators = lvo_cv(x, y, model, models_dict, metrics_dict)
        main_dict[model] = {'results': results, 'estimators': estimators}
        print()
    return main_dict

# Для расчетного средних метрик при валидации
def calculate_mean_metrics_on_validation(result, save=False, name_to_save=None):
    import pandas as pd

    helper = pd.concat(
        [pd.concat([result[model]['results'].mean(axis=1), result[model]['results'].std(axis=1)], axis=1) for model in
         result.keys()], axis=1)
    new_column_names = []
    for model in result.keys():
        new_column_names.append((model, 'mean'))
        new_column_names.append((model, 'std'))
    result_columns = pd.MultiIndex.from_tuples(new_column_names)
    helper.columns = result_columns
    if save:
        if name_to_save is None:
            name_to_save = input() + '.xlsx'
        helper.to_excel(name_to_save + '.xlsx')
    return helper

# Функция для обучения на всем трейне и предсказании на тесте
def predicts(x_train, y_train, x_test, y_test, models_dict, metrics_dict):
    import pandas as pd
    res_dict ={}
    estimators_dict ={}
    for model in models_dict.keys():
        regressor = models_dict[model]
        regressor.fit(x_train.to_numpy(), y_train.to_numpy())
        y_tr_pred = regressor.predict(x_train.to_numpy())
        y_tr_pred = pd.Series(y_tr_pred, name=y_train.name, index=y_train.index)
        y_pred = regressor.predict(x_test.to_numpy())
        y_pred = pd.Series(y_pred, name=y_test.name, index=y_test.index)
        aid_dict = {}
        for name, metric in metrics_dict.items():
            if name == 'RMSE':
                aid_dict[name+' on all train'] = metric(y_train, y_tr_pred, squared=False)
                aid_dict[name+' on test'] = metric(y_test, y_pred, squared=False)
            else:
                aid_dict[name+' on all train'] = metric(y_train, y_tr_pred)
                aid_dict[name+' on test'] = metric(y_test, y_pred)
        res_dict[model] = aid_dict
        estimators_dict[model] = regressor
    return pd.DataFrame(res_dict), estimators_dict

# Функция для нахождения объектов по индексам с наибольшей ошибкой при валидации
def find_biggest_error(results):
    import pandas as pd

    d = {}
    for model in results.keys():
        aid = {}
        for metric in results[model]['results'].index:
            if 'validate' in metric and 'MSE' in metric:
                helper = results[model]['results'].loc[metric].sort_values()[-6:][::-1]
                aid[metric] = ' '.join([str(x)+'  '+str(y)[:5] for x, y in zip(helper.index, helper)])

        d[model] = aid
    return pd.DataFrame(d)

# Pipeline обучения
def pipline_true(x_train, x_test, y_train, y_test,  descriptors_names, models, metrics, save_cv=False, name_to_save_cv=None):
    x_train, x_test = x_train[descriptors_names].copy(), x_test[descriptors_names].copy()
    cv_res = full_cv(x_train, y_train, models, metrics)
    metrics_on_val = calculate_mean_metrics_on_validation(cv_res, save=save_cv, name_to_save=name_to_save_cv)
    test_results = predicts(x_train, y_train, x_test, y_test, models, metrics)
    return metrics_on_val, test_results[0], cv_res, test_results[1]

# # Для отрисовки boxplot для MAE, MSE, RMSE on train & val
def score_plots(result, save=False, scale=False):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    fig, ax = plt.subplots(3, 2, figsize=(20, 17))

    fin = {}
    for model in result[2].keys():
        aid_dict = {}
        for x in result[2][model]['results'].index:
            aid_dict[x] = result[2][model]['results'].loc[x].to_numpy()
        fin[model] = aid_dict
    res = pd.DataFrame(fin).T

    for i, col_name in enumerate(res.columns[:-1]):
        sns.boxplot(y=np.concatenate([x for x in res[col_name]]),
                    x=np.array([[x]*len(res.iloc[0, 0]) for x in list(res.index)]).ravel(),
                    ax=ax[i//2, i%2])
        ax[i//2, i%2].set_ylabel(col_name, size=14)

    if scale:
        mae_min = -float(input('For MAE: '))
        mse_min = -float(input('For MSE: '))
        rmse_min = -float(input('For RMSE: '))
        ax[0, 0].set_ylim([mae_min, 0.01])
        ax[1, 0].set_ylim([mse_min, 0.01])
        ax[2, 0].set_ylim([rmse_min, 0.01])

    if not save:
        plt.show()
    else:
        name = input()
        plt.savefig(name + '.png', dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()

# Для отрисовки boxplot для R^2 при валидации
def r2_plot(result, save=False):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    plt.figure(figsize=(12, 6))

    fin = {}
    for model in result[2].keys():
        aid_dict = {}
        for x in result[2][model]['results'].index:
            aid_dict[x] = result[2][model]['results'].loc[x].to_numpy()
        fin[model] = aid_dict
    res = pd.DataFrame(fin).T

    sns.boxplot(y=np.concatenate([x for x in res['R2 on train']]),
                x=np.array([[x]*len(res.iloc[0, 0]) for x in list(res.index)]).ravel())
    plt.xlabel("Model", size=20)
    plt.ylabel("R$^2$ on train", size=20)
    #plt.grid(None)
    if not save:
        plt.show()
    else:
        name = input()
        plt.savefig(name + '.png', dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()