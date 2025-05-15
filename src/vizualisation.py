import matplotlib.pyplot as plt
import seaborn as sns

def plot_pie(train, test, col, size=(14, 6)):
    if test is None:
        fig, ax = plt.subplots(figsize=(size[0] // 2, size[1]))
        values = train[col].value_counts()
        labels = values.index
        ax.pie(values, labels=labels, autopct='%.1f%%', startangle=90,
               shadow=True, textprops={'fontsize': 14}, explode=[0.05]*len(labels))
        ax.set_title(f'{col} - Répartition', fontsize=18)
    else:
        fig, axes = plt.subplots(1, 2, figsize=size)
        for ax, data, title in zip(axes, [train, test], ['Train', 'Test']):
            values = data[col].value_counts()
            labels = values.index
            ax.pie(values, labels=labels, autopct='%.1f%%', startangle=90,
                   shadow=True, textprops={'fontsize': 14}, explode=[0.05]*len(labels))
            ax.set_title(f'{col} - {title}', fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_stat(data, feature, title, size=(14, 8), palette="husl"):
    fig, ax = plt.subplots(figsize=size)
    order = data[feature].value_counts().index
    sns.countplot(x=feature, hue=feature, data=data, order=order, ax=ax, palette=palette, legend=False)
    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Nombre de clients')
    ax.set_xlabel(feature)
    total = len(data[feature])
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{100 * height / total:.1f}%',
                    (p.get_x() + p.get_width() / 2, height + total*0.005),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

def plot_percent_target1(data, feature, title, size=(14, 8)):
    cat_perc = data.groupby(feature)['TARGET'].mean().reset_index()
    cat_perc.sort_values('TARGET', ascending=False, inplace=True)
    fig, ax = plt.subplots(figsize=size)
    sns.barplot(
        x=feature, y='TARGET', data=cat_perc, hue=feature, palette='viridis',
        ax=ax, legend=False, errorbar=None, order=cat_perc[feature])
    ax.set_title(title, fontsize=20)
    ax.set_ylabel('Taux de TARGET=1', fontsize=14)
    ax.set_xlabel(feature, fontsize=14)
    plt.grid(axis='y')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{100 * height:.1f}%',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')