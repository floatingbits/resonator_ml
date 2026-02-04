from app.bootstrap import build_plot_training_data_use_case


if __name__ == "__main__":
    use_case = build_plot_training_data_use_case()
    use_case.execute()
