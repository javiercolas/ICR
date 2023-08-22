class ICRConstants:

    @staticmethod
    def exploratory_params():
        return {
            'correlation_matrix': {
                'n_colors': 9,
                'fig-size': (20, 16.5),
                'cmap': "bwr",
                'min': -1.,
                'max': 1.,
                'fmt': '.2f',
                'rotation': 90,
                'bar_ticks': range(-8, 9, 2)
            },
            'multiclass_histograms':{
                'n_colors': 9,
                'fig-size': (30, 200),
                'cmap': "bwr",
                'min': -1.,
                'max': 1.,
                'fmt': '.2f',
                'rotation': 90,
                'bar_ticks': range(-10, 15, 4),
                'num_columns': 3,
                'num_rows': 20,
                'bins': 40,
                'x-fontsize': 25,
                'y-fontsize': 25
            }
        }