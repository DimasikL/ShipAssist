import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_tsne(
        data_filename: str,
        output_filename: str = None,
        embedding_prefix: str = 'w2v2_emb',
        group_column: str = 'audio_group',
        included_groups: list = None,
        color_mapping: dict = None,
        default_color_group: str = 'Цвет 3: Остальные',
        tsne_n_components: int = 2,
        tsne_perplexity: int = 30,
        tsne_max_iter: int = 1000,
        random_state: int = 42,
        figure_size: tuple = (12, 10),
        scatter_alpha: float = 0.8,
        title: str = 't-SNE Визуализация',
        xlabel: str = 'Компонента 1',
        ylabel: str = 'Компонента 2',
        legend_title: str = 'Цветовые Группы'
):
    try:
        df = pd.read_csv(data_filename)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{data_filename}' не найден.")
        return

    if included_groups:
        df_filtered = df[df[group_column].isin(included_groups)].copy()
    else:
        df_filtered = df.copy() # Если группы не указаны, используем все данные

    if df_filtered.empty:
        print("Ошибка: После фильтрации не осталось данных.")
        return

    if color_mapping:
        df_filtered['color_group'] = df_filtered[group_column].map(color_mapping)
        df_filtered['color_group'] = df_filtered['color_group'].fillna(default_color_group)
    else:
        df_filtered['color_group'] = df_filtered[group_column]

    embeddings = df_filtered.filter(like=embedding_prefix)
    color_labels = df_filtered['color_group']

    if embeddings.empty:
        print(f"Ошибка: Столбцы с префиксом '{embedding_prefix}' не найдены или пусты.")
        return

    tsne = TSNE(n_components=tsne_n_components, perplexity=tsne_perplexity,
                max_iter=tsne_max_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=figure_size)

    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=color_labels,
        palette=sns.color_palette("hsv", n_colors=color_labels.nunique()),
        legend="full",
        alpha=scatter_alpha
    )

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=legend_title)

    if output_filename:
        try:
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            print(f"График сохранен в {output_filename}")
        except Exception as e:
            print(f"Ошибка при сохранении графика: {e}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    my_included_groups = [
        'train user 2', 'train user 2-aug', 'drug slova2', 'drug slova2-aug',
        'drug slova3', 'drug slova3-aug', 'train user 3', 'train user 3-aug',
        'drug slova', 'test user 1'
    ]

    my_color_mapping = {
        'train user 2': 'Цвет 1: user 2 & drug slova 2',
        'train user 2-aug': 'Цвет 1: user 2 & drug slova 2',
        'drug slova2': 'Цвет 1: user 2 & drug slova 2',
        'drug slova2-aug': 'Цвет 1: user 2 & drug slova 2',

        'drug slova3': 'Цвет 2: user 3 & drug slova 3',
        'drug slova3-aug': 'Цвет 2: user 3 & drug slova 3',
        'train user 3': 'Цвет 2: user 3 & drug slova 3',
        'train user 3-aug': 'Цвет 2: user 3 & drug slova 3'
    }

    visualize_tsne(
        data_filename='dset_embedder_name=w2v2,do_prep=True,norm_duration=True,output_hidden_states=True.csv',
        included_groups=my_included_groups,
        color_mapping=my_color_mapping,
        title='t-SNE Визуализация с Объединенными Группами',
        legend_title='Цветовые Группы',
        output_filename='tsne_plot.png' # Сохранить график в файл
        # output_filename=None # Чтобы показать график вместо сохранения
    )

    # Пример использования с другими параметрами (например, без специального маппинга цвета)
    # visualize_tsne(
    #     data_filename='dset_embedder_name=w2v2,do_prep=True,norm_duration=True,output_hidden_states=True.csv',
    #     included_groups=['train user 2', 'test user 1'],
    #     color_mapping=None, # Не использовать специальный маппинг, использовать оригинальные группы
    #     title='t-SNE для избранных групп',
    #     legend_title='Оригинальные Группы',
    #     tsne_perplexity=20,
    #     output_filename='tsne_selected_groups.png'
    # )

    # Пример использования со всеми группами и другим префиксом эмбеддингов
    # visualize_tsne(
    #     data_filename='another_dataset.csv', # Воображаемый другой файл
    #     embedding_prefix='another_emb', # Другой префикс для эмбеддингов
    #     included_groups=None, # Включить все группы из файла
    #     color_mapping=None,
    #     title='t-SNE для другого набора данных (все группы)',
    #     output_filename='tsne_all_groups_another_data.png'
    # )