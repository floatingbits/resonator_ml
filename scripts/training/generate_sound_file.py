from app.bootstrap import build_generate_sound_file_use_case


if __name__ == "__main__":
    generate_sound_file = build_generate_sound_file_use_case()
    generate_sound_file.execute()



