from exploratory_analysis_constants import ICRConstants
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class ExploratoryAnalysis:

    def __init__(self, df):
        self.df = df

    def plot_heatmap(self, corr_threshold=None):
        constants = ICRConstants().exploratory_params()

        cMap = plt.cm.get_cmap(constants['correlation_matrix']['cmap'], lut=constants['correlation_matrix']['n_colors'])
        corr_matrix = self.df.corr()

        if corr_threshold is not None:
            np.fill_diagonal(corr_matrix.values, 0)
            included_columns_mask = (corr_matrix.abs().max() > corr_threshold)
            corr_matrix = corr_matrix.loc[included_columns_mask, included_columns_mask]

        plt.figure(figsize=constants['correlation_matrix']['fig-size'])
        plt.title('correlations')

        h_map = sns.heatmap(corr_matrix,
                            vmin=constants['correlation_matrix']['min'], vmax=constants['correlation_matrix']['max'],
                            cmap=cMap,
                            annot=True,
                            fmt=constants['correlation_matrix']['fmt'],
                            xticklabels=corr_matrix.columns,
                            yticklabels=corr_matrix.columns)
        plt.xticks(rotation=constants['correlation_matrix']['rotation'])

        cbar = h_map.collections[0].colorbar
        cbar.set_ticks([k / 10. for k in constants['correlation_matrix']['bar_ticks']])

        bottom, top = h_map.get_ylim()
        h_map.set_ylim(bottom + 0.5, top - 0.5)

        plt.show()

    def plot_class_frequency(self, column):
        class_frequency = self.df[column].value_counts()
        plt.figure(figsize=(10, 6))
        sns.set_style('darkgrid')

        sns.barplot(x=class_frequency.index, y=class_frequency.values)

        plt.xlabel('Classes', size=14)
        plt.ylabel('Frequency', size=14)
        plt.title('Class Frequency for column {}'.format(column), size=16)

        plt.show()

    def plot_groups(self, target='Class', num_col_per_plot=6):
        num_cols = self.df.shape[1]

        num_groups = np.ceil(num_cols / num_col_per_plot)

        for i in range(int(num_groups)):
            start_col = i * num_col_per_plot
            end_col = min((i + 1) * num_col_per_plot, num_cols)
            group_cols = self.df.columns[start_col:end_col].tolist()

            if target not in group_cols:
                group_cols.append(target)

            print("Procesando las columnas: ", group_cols)

            pair_plot = sns.pairplot(self.df[group_cols], hue=target, diag_kind="kde", diag_kws=dict(shade=True))
            plt.show()

    def plot_multiclass_histograms(self, target='Class', normalize=False):
        sns.set_color_codes("bright")
        sns.set_style("white")

        params = ICRConstants.exploratory_params()['multiclass_histograms']

        plt.figure(figsize=params['fig-size'])
        plt.title("Histograms by Class")

        classes = self.df[target].unique()

        color_map = plt.get_cmap(params['cmap'])
        colors = [color_map(i) for i in np.linspace(0, 1, len(classes))]

        assert len(self.df.columns) <= params['num_rows'] * params['num_columns'], \
            "Not enough rows and columns to plot all features. Increase rows or columns."

        for i, var in enumerate(self.df.drop([target], axis=1).columns):
            ax = plt.subplot(params['num_rows'], params['num_columns'], i + 1)
            ax.set_title(var)

            plt.xticks(fontsize=params['x-fontsize'])
            plt.yticks(fontsize=params['y-fontsize'])

            for j, class_ in enumerate(classes):
                class_indices = np.where(self.df[target] == class_)[0]
                if normalize:
                    class_data = (self.df[var].iloc[class_indices] - self.df[var].min()) / (self.df[var].max() - self.df[var].min())
                else:
                    class_data = self.df[var].iloc[class_indices]
                sns.distplot(class_data, bins=params['bins'], norm_hist=True, kde=False, color=colors[j], label=str(class_))

            plt.xlabel(var, fontsize=params['x-fontsize'])
            plt.ylabel("Abs. Frequencies", fontsize=params['y-fontsize'])

            plt.ylabel("Abs. Frequencies")
            plt.legend(loc='best')

        plt.tight_layout()
        plt.show()

    def convert_and_rearrange(self, column_to_convert, target):
        df = pd.get_dummies(self.df, columns=[column_to_convert], drop_first=True)
        dummy_column = f"{column_to_convert}_B"
        if dummy_column in df.columns:
            df.rename(columns={dummy_column: column_to_convert}, inplace=True)

        cols = [col for col in df if col != target] + [target]
        df = df[cols]
        return df


