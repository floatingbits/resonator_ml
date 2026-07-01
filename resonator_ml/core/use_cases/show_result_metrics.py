from resonator_ml.ports.evaluation_data_provider import EvaluationDataProvider
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from resonator_ml.ports.series_provider import SeriesProvider

def add_relative_fields(corr_df):
    corr_df['spectral/energy'] = corr_df['spectral'] / corr_df['energy']
    corr_df['spectral/phase'] = corr_df['spectral'] / corr_df['phase']
    corr_df['energy/phase'] = corr_df['energy'] / corr_df['phase']
    return corr_df
def show_corr_matrix(corr_df):
    corr_df = corr_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm')

    # Show the plot in Streamlit
    st.pyplot(plt)

class ShowResultMetrics:
    def __init__(self, data_provider: EvaluationDataProvider, loss_provider: SeriesProvider):
        self.data_provider = data_provider
        self.loss_provider = loss_provider

    def execute(self):
        evaluation_data = self.data_provider.provide_data()
        df = pd.DataFrame.from_dict(evaluation_data.data)

        st.title("Experiment Dashboard")
        df['sc_scaled'] = np.minimum(15*(df[evaluation_data.result_fields[1]] -0.3), 20)
        df['lsfft_scaled'] = np.minimum(5*(df[evaluation_data.result_fields[0]] -4.5), 10)
        df['mel_scaled'] = np.minimum(0.8*(df[evaluation_data.result_fields[2]] - 14), 10)
        df['combined_metrics'] = df['lsfft_scaled'] +   df['sc_scaled']    +  df['mel_scaled']
        def iterate_over_series(series:SeriesProvider, func, max_len):
            return [func(series.data_at(index)) for index in range(max_len)]
        df['min_loss'] = iterate_over_series(self.loss_provider, lambda data: np.array(data).min(),
                                             min(len(df), self.loss_provider.num_plots()))
        df['min_loss_index'] = iterate_over_series(self.loss_provider, lambda data: np.array(data).argmin(),
                                             min(len(df), self.loss_provider.num_plots()))
        def normalize_loss(row):
            factor = row['spectral'] + row['phase'] +row['energy']
            return row['min_loss']/factor
        df['normalized_loss'] = df.apply(normalize_loss, axis=1)
        def min_loss_ratio(group_id, min_loss_value):
            group_df = df[df[evaluation_data.group_id_field] == group_id]
            minimum = group_df['min_loss'].min()
            return min_loss_value/minimum

        df['min_loss_ratio'] = df.apply(lambda row: min_loss_ratio(row[evaluation_data.group_id_field],row['min_loss'] ), axis=1)

        # Filter
        param1 = st.selectbox("Config", df[evaluation_data.group_id_field].unique())
        filtered = df[df[evaluation_data.group_id_field] == param1]


        # Tabelle
        st.dataframe(filtered)

        # Stats
        st.write(filtered.describe())
        # Loss Corr
        loss_columns = ['combined_metrics'] + evaluation_data.result_fields + ['min_loss_ratio']
        corr_df = filtered[filtered['min_loss_ratio'] < 1.04][loss_columns]
        show_corr_matrix(corr_df)
        corr_df = filtered[filtered['combined_metrics'] < 22][loss_columns]
        show_corr_matrix(corr_df)

        # Gruppierung
        grouped = df.groupby(evaluation_data.group_id_field)['combined_metrics'].min()
        st.bar_chart(grouped, y_label="metric min")
        grouped = df.groupby(evaluation_data.group_id_field)['combined_metrics'].quantile(0.25)
        st.bar_chart(grouped, y_label="metric quantile 25")
        grouped = df.groupby(evaluation_data.group_id_field)['combined_metrics'].mean()
        st.bar_chart(grouped, y_label="metric mean")
        grouped = df.groupby(evaluation_data.group_id_field)['normalized_loss'].mean()
        st.bar_chart(grouped, y_label="best loss mean")
        grouped = df.groupby(evaluation_data.group_id_field)['normalized_loss'].min()
        st.bar_chart(grouped, y_label="best loss min")
        grouped = df.groupby(evaluation_data.group_id_field)['min_loss_index'].min()
        st.bar_chart(grouped, y_label="best loss index")



        no_outlier_performance_df = df[df['min_loss_ratio'] < 1.04]
        best_combined_metrics = df[df['combined_metrics'] < 22]
        result = (
            best_combined_metrics.groupby(evaluation_data.group_id_field)[['combined_metrics', 'min_loss']]
            .corr()
            .loc[(slice(None), 'combined_metrics'), 'min_loss']
            .reset_index()
        )
        result.columns = ['group_id', 'metric_name', 'correlation']
        result = result[['correlation']]
        st.bar_chart(result, y_label="corr of loss and metrics by param group")

        loss_columns = ['combined_metrics', 'sc_scaled', 'lsfft_scaled', 'mel_scaled'] + evaluation_data.result_fields + evaluation_data.input_fields + ['normalized_loss']




        show_corr_matrix(add_relative_fields(no_outlier_performance_df[loss_columns]))
        show_corr_matrix(add_relative_fields(best_combined_metrics[loss_columns]))