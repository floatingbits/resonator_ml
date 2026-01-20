from app.bootstrap import build_train_loop_network_use_case, configure_stdout

if __name__ == "__main__":

    train = build_train_loop_network_use_case()

    train.execute()