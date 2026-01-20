from app.bootstrap import build_generate_sound_file_use_case, configure_stdout


if __name__ == "__main__":
    configure_stdout('generate_soundfile')
    generate_sound_file = build_generate_sound_file_use_case()
    generate_sound_file.execute()



