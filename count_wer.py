import whisper
import jiwer
import json
import csv
import os
from tqdm import tqdm
from pathlib import Path
import librosa
# from whisper_ft_librispeech_load_data_from_hf import LibriSpeechDataset


def count_wer(whisper_model_name, wav_list, sentence_list, output_json="results.json"):
    """Calculate WER for each wav file and save the results to a JSON file.

    Args:
        whisper_model_name (str): Name of the Whisper model, e.g., 'base.en'.
        wav_name_list (list): A list of paths to wav files.
        sentence_list (list): A list of reference sentences corresponding to each wav file.
        output_json (str): The name of the output JSON file to save results. Default is 'results.json'.
    """
    # Load the Whisper model
    model = whisper.load_model(whisper_model_name)

    # Initialize a list to store results
    results = []
    total_wer = 0

    # Process each wav file
    for wav, sentence in tqdm(zip(wav_list, sentence_list)):
        # Transcribe the audio file
        result = model.transcribe(wav)
        hypothesis = result["text"]

        # Calculate WER
        wer = jiwer.wer(sentence, hypothesis)
        total_wer += wer

        # Store the result
        results.append({
            "wav_name": str(wav_name),
            "reference": sentence,
            "hypothesis": hypothesis,
            "wer": wer
        })

    # Calculate the average WER
    average_wer = total_wer / len(wav_name_list)

    # Save results to a JSON file
    results.append({
        "average_wer": average_wer
    })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json}")
    print(f"Average WER: {average_wer}")

if __name__ == '__main__':
    whisper_model = 'tiny'
    base_wav_list = '/datas/store163/wago/wav2vec-vc/convert/cpu'
    #
    base_vctk_txt_path = '/datas/store163/wago/data/vctk/origin/txt'
    output_path = 'result/wav2vec-vc_vctk_convert_result.json'
    #
    #
    vctk(whisper_model, base_wav_list, base_vctk_txt_path, output_path)
