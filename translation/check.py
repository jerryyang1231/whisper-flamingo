import os

def check_translation_completeness(input_dir, output_dir):
    incomplete_files = []

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".trans.txt"):
                input_file = os.path.join(subdir, file)
                output_file = os.path.join(output_dir, os.path.relpath(input_file, input_dir))

                if not os.path.exists(output_file):
                    incomplete_files.append((input_file, "Output file missing"))
                    continue

                # 比較行數
                with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "r", encoding="utf-8") as f_out:
                    input_lines = len(f_in.readlines())
                    output_lines = len(f_out.readlines())

                    if input_lines != output_lines:
                        incomplete_files.append((input_file, f"Incomplete: {output_lines}/{input_lines} lines translated"))

    return incomplete_files

# 路徑設定
input_dir = "/share/nas169/jerryyang/corpus/LibriSpeech/LibriSpeech"  # 替換為輸入檔案的路徑
output_dir = "/share/nas169/jerryyang/corpus/seamlessm4t/LibriSpeech/Spanish"  # 替換為翻譯結果的路徑

incomplete_files = check_translation_completeness(input_dir, output_dir)

if incomplete_files:
    print("Incomplete translations detected:")
    for file, reason in incomplete_files:
        print(f"{file}: {reason}")
else:
    print("All files are fully translated!")
