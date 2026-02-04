from app.bootstrap import build_train_loop_network_use_case, configure_stdout
import sys

if __name__ == "__main__":
    reuse_last_model_file = False if len(sys.argv) < 2 else bool(sys.argv[1])
    train = build_train_loop_network_use_case(reuse_last_model_file)

    train.execute()